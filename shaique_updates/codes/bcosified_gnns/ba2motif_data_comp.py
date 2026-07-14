import functools
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
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import degree


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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_one_hot_degree_clamped(data, max_degree: int = 4):
    _, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.long).clamp(max=max_degree)
    data.x = torch.nn.functional.one_hot(deg, num_classes=max_degree + 1).float()
    if not hasattr(data, "edge_attr"):
        data.edge_attr = None
    return data


def split_train_val_test(dataset, seed: int, test_size: float = 0.2, val_size: float = 0.1):
    labels = np.array([int(d.y.item()) for d in dataset])
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

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    return train_data, val_data, test_data


def make_loader(data_list, transform, batch_size: int = 64, shuffle: bool = False):
    return DataLoader([transform(d.clone()) for d in data_list], batch_size=batch_size, shuffle=shuffle)


class VanillaGCN(torch.nn.Module):
    def __init__(self, node_size: int, hidden_dim: int = 64, num_convs: int = 3):
        super().__init__()
        self.node_encoder = nn.Linear(node_size, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_convs)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch):
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        return global_add_pool(self.readout(x), batch)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch).flatten()
        loss = criterion(logits, batch.y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (logits > 0).long()
        total_loss += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.long()).sum().item())
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
        logits = model(batch.x, batch.edge_index, batch.batch).flatten()
        loss = criterion(logits, batch.y.float())
        pred = (logits > 0).long()

        total_loss += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.long()).sum().item())
        total_graphs += int(batch.num_graphs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(batch.y.detach().cpu().long())

    logits_all = torch.cat(all_logits).numpy()
    labels_all = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    try:
        clf_auroc = float(roc_auc_score(labels_all, probs))
    except ValueError:
        clf_auroc = float("nan")

    return {
        "loss": total_loss / total_graphs,
        "acc": total_correct / total_graphs,
        "clf_auroc": clf_auroc,
    }


def run_experiment(
    experiment_name: str,
    dataset_list,
    transform,
    seeds,
    epochs: int,
    batch_size: int,
    lr: float,
    early_stop_patience: int,
    min_delta: float,
):
    criterion = BCEWithLogitsLoss()
    seed_results = []

    for seed in seeds:
        print("=" * 100)
        print(f"{experiment_name} | seed={seed}")
        print("=" * 100)
        set_seed(seed)

        train_data, val_data, test_data = split_train_val_test(
            dataset_list,
            seed=seed,
            test_size=0.2,
            val_size=0.1,
        )
        
        train_loader = make_loader(train_data, transform, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(val_data, transform, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(test_data, transform, batch_size=batch_size, shuffle=False)

        node_size = train_loader.dataset[0].x.shape[1]
        
        model = VanillaGCN(node_size=node_size, hidden_dim=64, num_convs=3).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

        best_state = None
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = SAVED_MODELS_DIR / f"best_{experiment_name}_seed_{seed}.pt"

        # Sequential training loop ensuring evaluation happens after cycles finish
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_metrics = evaluate_loader(model, val_loader, criterion)
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

            if epoch % 5 == 0 or epoch == 1:
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

        # Load best model for the final test set evaluation
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate_loader(model, test_loader, criterion)

        result = {
            "experiment": experiment_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_clf_auroc": float(test_metrics["clf_auroc"]),
        }
        seed_results.append(result)

        print(
            f"{experiment_name} | seed={seed} | best_epoch={best_epoch} "
            f"| test_acc={result['test_acc']:.4f} "
            f"| test_clf_auroc={result['test_clf_auroc']:.4f}",
        )

    metrics_keys = ["test_loss", "test_acc", "test_clf_auroc"]
    summary = {}
    for key in metrics_keys:
        values = np.array([r[key] for r in seed_results], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    
    summary["experiment"] = experiment_name

    return seed_results, summary


def main():
    print(f"Device: {DEVICE}")

    # ==========================================
    # 1. LOAD CUSTOM DATASET
    # ==========================================
    custom_data_path = PROJECT_ROOT / "shaique_updates" / "data" / "Custom_BA2Motif" / "custom_ba2motif_dataset.pt"
    print(f"\nLoading custom data from: {custom_data_path}")
    if not custom_data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {custom_data_path}")

    custom_dataset_list = torch.load(custom_data_path, weights_only=False)
    for i, data in enumerate(custom_dataset_list):
        data.y = torch.tensor([0]) if i < 500 else torch.tensor([1])
        data.x = torch.ones((data.num_nodes, 1))
        if not hasattr(data, "edge_attr"):
            data.edge_attr = None
    print(f"Successfully loaded {len(custom_dataset_list)} graphs from Custom Dataset.")

    # ==========================================
    # 2. LOAD STANDARD PYG DATASET
    # ==========================================
    pyg_data_dir = PROJECT_ROOT / "data" / "BA2Motif"
    print(f"\nLoading standard PyG BA2MotifDataset into: {pyg_data_dir}")
    pyg_dataset = BA2MotifDataset(root=str(pyg_data_dir))
    
    pyg_dataset_list = []
    for data in pyg_dataset:
        if not hasattr(data, "edge_attr"):
            data.edge_attr = None
        pyg_dataset_list.append(data)
    print(f"Successfully loaded {len(pyg_dataset_list)} graphs from PyG Dataset.")

    # ==========================================
    # 3. SETUP & RUN EXPERIMENTS
    # ==========================================
    transform = functools.partial(get_one_hot_degree_clamped, max_degree=4)
    seeds = [0, 1, 2]
    base_experiment = "vanilla_gcn_readout_agg"
    
    datasets_to_test = {
        "Custom_BA2Motif": custom_dataset_list,
        "PyG_BA2Motif": pyg_dataset_list
    }

    all_seed_rows = []
    summary_rows = []

    for dataset_name, dataset_list in datasets_to_test.items():
        print(f"\n{'='*100}")
        print(f"STARTING EVALUATION: {base_experiment} on {dataset_name}")
        print(f"{'='*100}")
        
        current_exp_name = f"{base_experiment}_{dataset_name}"

        seed_results, summary = run_experiment(
            experiment_name=current_exp_name,
            dataset_list=dataset_list,
            transform=transform,
            seeds=seeds,
            epochs=200,
            batch_size=64,
            lr=3e-4,
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
                }
            }
            for result in seed_results
        ]
        
        for r in rows:
            r["dataset"] = dataset_name
        summary["dataset"] = dataset_name

        all_seed_rows.extend(rows)
        summary_rows.append(summary)

    # ==========================================
    # 4. AGGREGATE AND SAVE RESULTS
    # ==========================================
    df_seed_results = pl.DataFrame(all_seed_rows)
    df_summary = pl.DataFrame(summary_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_results_csv = RESULTS_DIR / "vanilla_gcn_dataset_comparison_seeds.csv"
    summary_csv = RESULTS_DIR / "vanilla_gcn_dataset_comparison_summary.csv"
    combined_csv = RESULTS_DIR / "vanilla_gcn_dataset_comparison_combined.csv"

    columns_order = ["dataset"] + [col for col in df_seed_results.columns if col != "dataset"]
    df_seed_results = df_seed_results.select(columns_order)
    
    summary_cols_order = ["dataset"] + [col for col in df_summary.columns if col != "dataset"]
    df_summary = df_summary.select(summary_cols_order)

    df_seed_results.write_csv(seed_results_csv)
    df_summary.write_csv(summary_csv)
    pl.concat(
        [
            df_seed_results.with_columns(pl.lit("seed").alias("row_type")),
            df_summary.with_columns(pl.lit("summary").alias("row_type")),
        ],
        how="diagonal",
    ).write_csv(combined_csv)

    print("\n================ FINAL COMPARISON RESULTS ================")
    print("\nPer-seed test metrics:")
    print(df_seed_results)
    print("\nMean ± std across seeds:")
    print(df_summary)
    print(f"\nSaved per-seed CSV: {seed_results_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved combined CSV: {combined_csv}")


if __name__ == "__main__":
    main()