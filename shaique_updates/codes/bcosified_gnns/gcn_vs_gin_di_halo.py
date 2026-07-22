import random
import sys
from pathlib import Path
import os

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool


def find_repo_root(*starts: Path) -> Path:
    search_starts = [s.resolve() for s in starts if s is not None]
    env_root = os.environ.get("BCOSGNN_PROJECT_ROOT")
    if env_root:
        search_starts.append(Path(env_root).resolve())

    seen = set()
    for start in search_starts:
        for path in [start, *start.parents]:
            if path in seen:
                continue
            seen.add(path)
            if (path / "pyproject.toml").exists() and (path / "bcosgnn").is_dir():
                return path

    raise RuntimeError(
        "Could not locate repo root. Set BCOSGNN_PROJECT_ROOT or run from inside the repo.",
    )


PROJECT_ROOT = find_repo_root(Path(__file__).resolve().parent, Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bcosgnn.completeness import run_bcos_multiclass_node_completeness_check
from bcosgnn.explain import explain
from bcosgnn.sanitized_models import BCosGCN, BCosGINE, ReadoutThenAgg


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "results"
COMPLETENESS_DIR = RESULTS_DIR / "completeness_plots"


class ZincProcessedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        data_path = Path(self.processed_dir) / "data.pt"
        print(f"Loading data from: {data_path.resolve()}")
        try:
            self.data, self.slices = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        return None

    def process(self):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val_test(dataset, seed: int, test_size: float = 0.1, val_size: float = 0.1):
    labels = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = labels[train_val_idx]
    val_ratio_on_trainval = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_on_trainval,
        random_state=seed,
        stratify=train_val_labels,
    )

    return train_idx, val_idx, test_idx


def make_loader(indices, dataset, batch_size: int = 64, shuffle: bool = False):
    subset = dataset.index_select(torch.tensor(indices, dtype=torch.long))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


class EvalModelAdapter(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if isinstance(self.base_model, BCosGINE):
            return self.base_model(x, edge_index, edge_attr, batch)
        return self.base_model(x, edge_index, batch)


class VanillaGCN(torch.nn.Module):
    def __init__(self, node_size: int, hidden_dim: int, num_classes: int, num_convs: int = 4):
        super().__init__()
        self.node_encoder = nn.Linear(node_size, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_convs)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        return global_add_pool(self.readout(x), batch)


def build_vanilla_model(node_size: int, hidden_dim: int, num_classes: int):
    return VanillaGCN(node_size=node_size, hidden_dim=hidden_dim, num_classes=num_classes, num_convs=4).to(DEVICE)


def build_bcos_model(node_size: int, hidden_dim: int, num_classes: int, b: float):
    readout = ReadoutThenAgg(
        in_channels=hidden_dim,
        hidden_channels=[hidden_dim],
        out_channels=num_classes,
        b=b,
        max_out=1,
        agg="sum",
    )
    return BCosGCN(
        node_size=node_size,
        hidden_channels=[hidden_dim, hidden_dim],
        num_convs=4,
        readout=readout,
        b=b,
        max_out=1,
        conv_kwargs={"aggr": "sum"},
    ).to(DEVICE)


def build_bcos_gine_model(node_size: int, edge_size: int, hidden_dim: int, num_classes: int, b: float):
    readout = ReadoutThenAgg(
        in_channels=hidden_dim,
        hidden_channels=[hidden_dim],
        out_channels=num_classes,
        b=b,
        max_out=1,
        agg="sum",
    )
    return BCosGINE(
        node_size=node_size,
        edge_size=edge_size,
        hidden_channels=[hidden_dim, hidden_dim],
        num_convs=4,
        readout=readout,
        b=b,
        max_out=1,
        conv_kwargs={"aggr": "sum"},
    ).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = total_graphs = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        if isinstance(model, BCosGINE):
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.view(-1).long()).sum().item())
        total_graphs += int(batch.num_graphs)

    return total_loss / total_graphs, total_correct / total_graphs


@torch.inference_mode()
def evaluate_loader(model, loader, criterion, num_classes: int):
    model.eval()
    total_loss = total_correct = total_graphs = 0
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        if isinstance(model, BCosGINE):
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1).long())
        pred = logits.argmax(dim=-1)

        total_loss += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.view(-1).long()).sum().item())
        total_graphs += int(batch.num_graphs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(batch.y.detach().cpu().long().view(-1))

    probs = torch.softmax(torch.cat(all_logits), dim=-1).numpy()
    labels = torch.cat(all_labels).numpy()
    try:
        clf_auroc = float(roc_auc_score(labels, probs, multi_class="ovr", labels=np.arange(num_classes)))
    except ValueError:
        clf_auroc = float("nan")

    return {
        "loss": total_loss / total_graphs,
        "acc": total_correct / total_graphs,
        "clf_auroc": clf_auroc,
    }


def get_attribution_scores_multiclass(model, data, batch=None):
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None), batch)
    pred_class = int(logits.view(1, -1).argmax(dim=-1).item())

    contrib = explain(
        model,
        data.x,
        data.edge_index,
        getattr(data, "edge_attr", None),
        batch,
    ).detach()
    scores = contrib.abs().sum(1)
    return scores.cpu().numpy(), pred_class


def evaluate_jaccard_multiclass(model, dataset, gt_attr: str = "explanation_mask"):
    jaccards = []
    device = next(model.parameters()).device

    for data in dataset:
        gt = getattr(data, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        k = int(gt_mask.sum())
        if k == 0:
            continue

        try:
            data_t = data.clone().to(device)
            batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)
            scores, _ = get_attribution_scores_multiclass(model, data_t, batch=batch)
            top_k = set(np.argsort(scores)[-k:].tolist())
            gt_set = set(np.where(gt_mask)[0].tolist())
            inter = len(gt_set & top_k)
            union = len(gt_set | top_k)
            jaccards.append(inter / union if union > 0 else 0.0)
        except Exception:
            jaccards.append(0.0)

    return float(np.mean(jaccards)) if jaccards else float("nan")


def evaluate_auroc_multiclass(model, dataset, gt_attr: str = "explanation_mask"):
    aurocs = []
    device = next(model.parameters()).device

    for data in dataset:
        gt = getattr(data, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue

        try:
            data_t = data.clone().to(device)
            batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)
            scores, _ = get_attribution_scores_multiclass(model, data_t, batch=batch)
            aurocs.append(roc_auc_score(gt_mask, scores))
        except Exception:
            pass

    return float(np.mean(aurocs)) if aurocs else float("nan")


def run_experiment(
    experiment_name: str,
    dataset,
    seeds,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    b: float,
    early_stop_patience: int,
    min_delta: float,
):
    criterion = CrossEntropyLoss()
    seed_results = []
    best_result = None
    best_result_score = -float("inf")

    node_size = dataset.num_node_features
    try:
        edge_size = dataset.num_edge_features
    except AttributeError:
        first_graph = dataset[0]
        edge_size = first_graph.edge_attr.shape[1] if hasattr(first_graph, "edge_attr") and first_graph.edge_attr is not None else 1
    num_classes = dataset.num_classes

    for seed in seeds:
        print("=" * 100)
        print(f"{experiment_name} | seed={seed}")
        print("=" * 100)
        set_seed(seed)

        train_idx, val_idx, test_idx = split_train_val_test(dataset, seed=seed)
        train_loader = make_loader(train_idx, dataset, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(val_idx, dataset, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(test_idx, dataset, batch_size=batch_size, shuffle=False)
        test_data = [dataset[int(i)] for i in test_idx]

        if experiment_name == "vanilla_gcn_sum":
            model = build_vanilla_model(node_size=node_size, hidden_dim=hidden_dim, num_classes=num_classes)
            do_expl_metrics = False
        elif experiment_name == "bcos_gcn_sum":
            model = build_bcos_model(node_size=node_size, hidden_dim=hidden_dim, num_classes=num_classes, b=b)
            do_expl_metrics = True
        elif experiment_name == "bcos_gine_sum":
            model = build_bcos_gine_model(node_size=node_size, edge_size=edge_size, hidden_dim=hidden_dim, num_classes=num_classes, b=b)
            do_expl_metrics = True
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        best_state = None
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = SAVED_MODELS_DIR / f"best_{experiment_name}_seed_{seed}.pt"

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_metrics = evaluate_loader(model, val_loader, criterion, num_classes=num_classes)
            scheduler.step(val_metrics["loss"])

            if (best_val_loss - val_metrics["loss"]) > min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                torch.save(
                    {
                        "experiment": experiment_name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_loss": float(best_val_loss),
                        "model_state_dict": best_state,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                    f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                    f"| lr={optimizer.param_groups[0]['lr']:.2e}",
                )

            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})",
                )
                break

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif best_state is not None:
            model.load_state_dict(best_state)

        eval_model = EvalModelAdapter(model).to(DEVICE)
        test_metrics = evaluate_loader(model, test_loader, criterion, num_classes=num_classes)

        if do_expl_metrics:
            test_jaccard = evaluate_jaccard_multiclass(eval_model, test_data)
            test_expl_auroc = evaluate_auroc_multiclass(eval_model, test_data)
        else:
            test_jaccard = float("nan")
            test_expl_auroc = float("nan")

        result = {
            "experiment": experiment_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_model_path": str(checkpoint_path),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_clf_auroc": float(test_metrics["clf_auroc"]),
            "test_expl_auroc": float(test_expl_auroc),
            "test_jaccard": float(test_jaccard),
            "model": model,
            "eval_model": eval_model,
            "test_dataset": test_data,
        }
        seed_results.append(result)

        print(
            f"{experiment_name} | seed={seed} | best_epoch={best_epoch} "
            f"| ckpt={checkpoint_path.name} "
            f"| test_acc={result['test_acc']:.4f} "
            f"| test_clf_auroc={result['test_clf_auroc']:.4f} "
            f"| test_expl_auroc={result['test_expl_auroc']:.4f} "
            f"| test_jaccard={result['test_jaccard']:.4f}",
        )

        rank_score = float(
            np.nan_to_num(result["test_expl_auroc"], nan=-1e9)
            + np.nan_to_num(result["test_jaccard"], nan=-1e9)
            + np.nan_to_num(result["test_acc"], nan=-1e9)
        )
        if (best_result is None) or (rank_score > best_result_score):
            best_result_score = rank_score
            best_result = result

    metrics_keys = ["test_loss", "test_acc", "test_clf_auroc", "test_expl_auroc", "test_jaccard"]
    summary = {}
    for key in metrics_keys:
        values = np.array([r[key] for r in seed_results], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    summary["experiment"] = experiment_name

    return seed_results, best_result, summary


def main():
    print(f"Device: {DEVICE}")

    data_root = PROJECT_ROOT / "shaique_updates" / "codes" / "multi_class_Zinc" / "zinc_di_halo_benzene_data"
    if not data_root.exists():
        raise FileNotFoundError("Cannot find zinc_di_halo_benzene_data.")

    dataset = ZincProcessedDataset(root=str(data_root))
    print(f"Dataset size: {len(dataset)}")
    print(f"Node features: {dataset.num_node_features}")
    print(f"Classes: {dataset.num_classes}")

    seeds = [0, 1, 2]
    experiments = ["vanilla_gcn_sum", "bcos_gcn_sum", "bcos_gine_sum"]

    all_seed_rows = []
    summary_rows = []
    best_by_experiment = {}

    for experiment_name in experiments:
        seed_results, best_result, summary = run_experiment(
            experiment_name=experiment_name,
            dataset=dataset,
            seeds=seeds,
            epochs=200,
            batch_size=64,
            lr=1e-3,
            hidden_dim=128,
            b=2.0,
            early_stop_patience=25,
            min_delta=1e-4,
        )

        rows = [
            {
                key: value
                for key, value in result.items()
                if key in {
                    "experiment",
                    "seed",
                    "best_epoch",
                    "best_val_loss",
                    "test_loss",
                    "test_acc",
                    "test_clf_auroc",
                    "test_expl_auroc",
                    "test_jaccard",
                }
            }
            for result in seed_results
        ]
        all_seed_rows.extend(rows)
        summary_rows.append(summary)
        best_by_experiment[experiment_name] = best_result

    df_seed_results = pl.DataFrame(all_seed_rows)
    df_summary = pl.DataFrame(summary_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = RESULTS_DIR / "bcosGCN_vs_bcosGINE.csv"

    df_summary.write_csv(summary_csv)

    print("\nPer-seed test metrics:")
    print(df_seed_results)
    print("\nMean ± std across seeds:")
    print(df_summary)
    print(f"\nSaved summary CSV: {summary_csv}")

    COMPLETENESS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nCompleteness checks (best B-COS seeds):")
    for experiment_name, best_result in best_by_experiment.items():
        if "bcos" not in experiment_name:
            continue
        plot_path = COMPLETENESS_DIR / f"completeness_{experiment_name}_seed_{best_result['seed']}.pdf"
        completeness = run_bcos_multiclass_node_completeness_check(
            best_result["eval_model"],
            best_result["test_dataset"],
            device=DEVICE,
            title=f"B-COS Completeness ({experiment_name}, seed={best_result['seed']})",
            output_path=plot_path,
        )
        print(
            f"- {experiment_name}: seed={best_result['seed']} "
            f"| completeness_mae={completeness['mae']:.2e} "
            f"| graphs={completeness['num_graphs']} "
            f"| plot={completeness['plot_path']}",
        )


if __name__ == "__main__":
    main()
