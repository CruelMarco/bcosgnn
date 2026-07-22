import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from torch_geometric.datasets import MoleculeNet


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

from bcosgnn.completeness import run_bcos_gine_completeness_check
from bcosgnn.sanitized_models import (
    AggThenReadout,
    BCosGINE,
    ReadoutThenAgg,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "results"
COMPLETENESS_DIR = RESULTS_DIR / "completeness_plots"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(indices, dataset, batch_size: int = 64, shuffle: bool = False):
    subset = [dataset[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

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

def _as_binary_label(y_tensor: torch.Tensor) -> int:
    y = float(y_tensor.view(-1)[0].item())
    return int(y > 0.0)

def load_molhiv_dataset(root: str):
    dataset = MoleculeNet(root=root, name="HIV")
    filtered = []
    for data in dataset:
        y = data.y.view(-1)[0]
        if torch.isfinite(y):
            data.y = torch.tensor([_as_binary_label(data.y)], dtype=torch.long)
            filtered.append(data)
    return filtered


class EvalModelAdapter(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.base_model(x.float(), edge_index, edge_attr.float() if edge_attr is not None else None, batch)


class VanillaGINE(torch.nn.Module):
    def __init__(self, node_size: int, edge_size: int, hidden_dim: int, num_classes: int, num_convs: int = 4):
        super().__init__()
        self.node_encoder = nn.Linear(node_size, hidden_dim)
        self.edge_encoder = nn.Linear(edge_size, hidden_dim)

        convs = []
        for _ in range(num_convs):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            convs.append(GINEConv(mlp, train_eps=False, aggr="mean"))
        self.convs = nn.ModuleList(convs)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        graph_emb = global_mean_pool(x, batch)
        return self.readout(graph_emb)


def build_vanilla_model(node_size: int, edge_size: int, hidden_dim: int, num_classes: int):
    return VanillaGINE(
        node_size=node_size,
        edge_size=edge_size,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_convs=4,
    ).to(DEVICE)


def build_bcos_model(
    node_size: int,
    edge_size: int,
    num_classes: int,
    hidden_dim: int,
    readout_mode: str,
    b: float,
):
    if readout_mode == "readout_then_agg":
        readout = ReadoutThenAgg(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim],
            out_channels=num_classes,
            b=b,
            max_out=1,
            agg="mean",
        )
    elif readout_mode == "agg_then_readout":
        readout = AggThenReadout(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim],
            out_channels=num_classes,
            b=b,
            max_out=1,
            agg="mean",
        )
    else:
        raise ValueError(f"Unsupported readout_mode: {readout_mode}")

    return BCosGINE(
        node_size=node_size,
        edge_size=edge_size,
        hidden_channels=[hidden_dim, hidden_dim],
        num_convs=4,
        readout=readout,
        b=b,
        max_out=1,
        conv_kwargs={"aggr": "mean"},
    ).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        
        # For MolHIV, features are discrete, but model expects floats for linear embedding
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        y = batch.y.view(-1).float()
        is_labeled = y == y # remove NaNs
        
        loss = criterion(logits.view(-1)[is_labeled], y[is_labeled])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (logits.view(-1) > 0).float()
        total_loss += float(loss.item()) * batch.num_graphs
        total_correct += int((pred[is_labeled] == y[is_labeled]).sum().item())
        total_graphs += int(batch.num_graphs)

    return total_loss / total_graphs, total_correct / total_graphs


@torch.inference_mode()
def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(-1).float()
        is_labeled = y == y
        
        if is_labeled.sum() > 0:
            loss = criterion(logits.view(-1)[is_labeled], y[is_labeled])
            total_loss += float(loss.item()) * batch.num_graphs
        
        pred = (logits.view(-1) > 0).float()
        total_correct += int((pred[is_labeled] == y[is_labeled]).sum().item())
        total_graphs += int(batch.num_graphs)

        all_logits.append(logits.detach().cpu().view(-1))
        all_labels.append(batch.y.detach().cpu().float().view(-1))

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(logits_all).numpy()
    
    is_labeled = labels_all == labels_all
    
    try:
        clf_auroc = float(
            roc_auc_score(labels_all[is_labeled], probs[is_labeled]),
        )
    except ValueError:
        clf_auroc = float("nan")

    return {
        "loss": total_loss / max(1, total_graphs),
        "acc": total_correct / max(1, total_graphs),
        "clf_auroc": clf_auroc,
    }


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
    seed_results = []
    best_result = None
    best_result_score = -float("inf")

    sample = dataset[0]
    node_size = int(sample.x.size(-1))
    edge_size = int(sample.edge_attr.size(-1)) if sample.edge_attr is not None else 0
    num_classes = 1 # Binary classification for MolHIV

    for seed in seeds:
        print("=" * 100)
        print(f"{experiment_name} | seed={seed}")
        print("=" * 100)
        set_seed(seed)

        train_idx, val_idx, test_idx = split_train_val_test(
            dataset,
            seed=seed,
            test_size=0.1,
            val_size=0.1,
        )

        # 1. Calculate pos_weight based on the training data for this seed
        train_labels = np.array([int(dataset[i].y.item()) for i in train_idx])
        num_pos = int(train_labels.sum())
        num_neg = len(train_labels) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=DEVICE)
        
        # Instantiate the loss function with the calculated weight
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader = make_loader(train_idx, dataset, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(val_idx, dataset, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(test_idx, dataset, batch_size=batch_size, shuffle=False)
        test_data = [dataset[int(i)] for i in test_idx]

        if experiment_name == "vanilla_gine_mean":
            model = build_vanilla_model(
                node_size=node_size,
                edge_size=edge_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            )
        elif experiment_name == "bcos_gine_mean_readout_then_agg":
            model = build_bcos_model(
                node_size=node_size,
                edge_size=edge_size,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                readout_mode="readout_then_agg",
                b=b,
            )
        elif experiment_name == "bcos_gine_mean_agg_then_readout":
            model = build_bcos_model(
                node_size=node_size,
                edge_size=edge_size,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                readout_mode="agg_then_readout",
                b=b,
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 2. Change LR Scheduler to MAXIMIZE the target metric (AUROC)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        best_state = None
        # 3. Change early stopping trackers to look for highest AUROC
        best_val_auroc = -float("inf") 
        best_epoch = -1
        epochs_without_improvement = 0
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = SAVED_MODELS_DIR / f"best_{experiment_name}_seed_{seed}.pt"

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_metrics = evaluate_loader(model, val_loader, criterion)

            current_lr = optimizer.param_groups[0]["lr"]
            
            # 4. Step scheduler based on AUROC, not loss
            scheduler.step(val_metrics["clf_auroc"])

            # 5. Check if AUROC improved by min_delta
            if (val_metrics["clf_auroc"] - best_val_auroc) > min_delta:
                best_val_auroc = val_metrics["clf_auroc"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                torch.save(
                    {
                        "experiment": experiment_name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_auroc": float(best_val_auroc), 
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
                    f"| val_auroc={val_metrics['clf_auroc']:.4f} "
                    f"| lr={current_lr:.2e}",
                )

            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_auroc={best_val_auroc:.4f})",
                )
                break

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif best_state is not None:
            model.load_state_dict(best_state)

        eval_model = EvalModelAdapter(model).to(DEVICE)
        test_metrics = evaluate_loader(model, test_loader, criterion)

        result = {
            "experiment": experiment_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auroc": float(best_val_auroc),
            "best_model_path": str(checkpoint_path),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_clf_auroc": float(test_metrics["clf_auroc"]),
            "model": model,
            "eval_model": eval_model,
            "test_dataset": test_data,
        }
        seed_results.append(result)

        print(
            f"{experiment_name} | seed={seed} | best_epoch={best_epoch} "
            f"| ckpt={checkpoint_path.name} "
            f"| test_acc={result['test_acc']:.4f} "
            f"| test_clf_auroc={result['test_clf_auroc']:.4f}"
        )

        rank_score = float(
            np.nan_to_num(result["test_clf_auroc"], nan=-1e9)
        )
        if (best_result is None) or (rank_score > best_result_score):
            best_result_score = rank_score
            best_result = result

    metrics_keys = [
        "test_loss",
        "test_acc",
        "test_clf_auroc",
    ]
    summary = {}
    for key in metrics_keys:
        values = np.array([r[key] for r in seed_results], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    summary["experiment"] = experiment_name

    return seed_results, best_result, summary


def main():
    print(f"Device: {DEVICE}")

    # Patch torch.load to avoid errors with PygGraphPropPredDataset on newer PyTorch versions
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    DATASET_ROOT = str(SCRIPT_DIR / "data")
    dataset = load_molhiv_dataset(DATASET_ROOT)
    labels = np.array([int(d.y.item()) for d in dataset], dtype=np.int64)
    
    torch.load = _original_torch_load

    print(f"Loaded MolHIV samples: {len(dataset)}")
    print(f"Positive ratio: {labels.mean():.4f}")

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Node features: {sample.x.size(-1)}")
    print(f"Edge features: {sample.edge_attr.size(-1) if sample.edge_attr is not None else 0}")
    print(f"Classes: 1 (Binary)")

    seeds = [0, 1, 2]
    epochs = 200
    batch_size = 64
    lr = 1e-3
    hidden_dim = 128
    b = 2.0
    early_stop_patience = 25
    min_delta = 1e-4

    experiments = [
        "vanilla_gine_mean",
        "bcos_gine_mean_readout_then_agg",
        "bcos_gine_mean_agg_then_readout",
    ]

    all_seed_rows = []
    summary_rows = []
    best_by_experiment = {}

    for experiment_name in experiments:
        seed_results, best_result, summary = run_experiment(
            experiment_name=experiment_name,
            dataset=dataset,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            b=b,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta,
        )

        rows = [
            {
                key: value
                for key, value in result.items()
                if key
                in {
                    "experiment",
                    "seed",
                    "best_epoch",
                    "best_val_auroc", # Updated from best_val_loss
                    "test_loss",
                    "test_acc",
                    "test_clf_auroc",
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
    seed_results_csv = RESULTS_DIR / "molhiv_all_seed_results.csv"
    summary_csv = RESULTS_DIR / "molhiv_summary_mean_std.csv"
    combined_csv = RESULTS_DIR / "molhiv_all_results_with_summary.csv"

    df_seed_results.write_csv(seed_results_csv)
    df_summary.write_csv(summary_csv)
    df_combined = pl.concat(
        [
            df_seed_results.with_columns(pl.lit("seed").alias("row_type")),
            df_summary.with_columns(pl.lit("summary").alias("row_type")),
        ],
        how="diagonal",
    )
    df_combined.write_csv(combined_csv)

    print("\nPer-seed test metrics:")
    print(df_seed_results)
    print("\nMean ± std across seeds:")
    print(df_summary)
    print(f"\nSaved per-seed CSV: {seed_results_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved combined CSV: {combined_csv}")

    print("\nBest run picked per experiment (by clf auroc rank score):")
    for experiment_name, best_result in best_by_experiment.items():
        print(
            f"- {experiment_name}: seed={best_result['seed']} "
            f"| test_acc={best_result['test_acc']:.4f} "
            f"| test_clf_auroc={best_result['test_clf_auroc']:.4f}"
        )

    COMPLETENESS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nCompleteness checks (best B-COS GINE seeds):")
    for experiment_name, best_result in best_by_experiment.items():
        if "bcos" not in experiment_name:
            continue

        plot_path = COMPLETENESS_DIR / (
            f"completeness_{experiment_name}_seed_{best_result['seed']}.pdf"
        )
        completeness = run_bcos_gine_completeness_check(
            best_result["eval_model"],
            best_result["test_dataset"],
            device=DEVICE,
            title=f"B-COS GINE Completeness ({experiment_name}, seed={best_result['seed']})",
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