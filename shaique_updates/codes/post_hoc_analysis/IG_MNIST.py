import sys
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.explain import Explainer, CaptumExplainer, ModelConfig

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "bcosgnn").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (pyproject.toml + bcosgnn/).")

repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
print("Repo root added:", repo_root)

def _augment_with_normalized_pos(dataset):
    augmented = []
    for data in dataset:
        d = data.clone()
        pos = d.pos.float()
        pos_min = pos.min(dim=0).values
        pos_max = pos.max(dim=0).values
        denom = (pos_max - pos_min).clamp(min=1e-8)
        pos_norm = (pos - pos_min) / denom

        intensity = d.x.float()
        # Create the 2D representation: [x, 1-x]
        reciprocal_intensity = torch.cat([intensity, 1.0 - intensity], dim=-1)
        
        d.x = torch.cat([reciprocal_intensity, pos_norm], dim=-1)

        augmented.append(d)
    return augmented

def load_and_split_data(batch_size=128, add_pos_features=True):
    print("Loading preprocessed sparsified .pt splits...")
    # Use the absolute path provided on the submit machine / shared filesystem.
    split_root = Path("/home/moso00002/bcosgnn/bcosgnn/shaique_updates/codes/MNISTsp/data/MNIST/sparsified_pt_splits")

    if not ((split_root / "train_sparsified.pt").exists() and (split_root / "val_sparsified.pt").exists() and (split_root / "test_sparsified.pt").exists()):
        raise FileNotFoundError(
            f"Could not find saved sparsified split files at absolute path: {split_root}. "
            f"Make sure this path is accessible from the execute node or add the files to transfer_input_files."
        )

    train_dataset = torch.load(split_root / "train_sparsified.pt", map_location="cpu", weights_only=False)
    val_dataset = torch.load(split_root / "val_sparsified.pt", map_location="cpu", weights_only=False)
    test_dataset = torch.load(split_root / "test_sparsified.pt", map_location="cpu", weights_only=False)

    # Balance test_dataset to 1000 datapoints (100 per class)
    class_counts = {i: 0 for i in range(10)}
    balanced_test = []
    for data in test_dataset:
        label = int(data.y.item())
        if class_counts.get(label, 0) < 100:
            balanced_test.append(data)
            class_counts[label] = class_counts.get(label, 0) + 1
    test_dataset = balanced_test

    if add_pos_features:
        train_dataset = _augment_with_normalized_pos(train_dataset)
        val_dataset = _augment_with_normalized_pos(val_dataset)
        test_dataset = _augment_with_normalized_pos(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = torch.tensor([int(d.y.item()) for d in train_dataset + val_dataset + test_dataset])
    num_classes = int(all_labels.unique().numel())
    num_node_features = int(train_dataset[0].num_node_features)

    dataset_info = {
        "num_node_features": num_node_features,
        "num_classes": num_classes,
    }

    print(f"Using split directory: {split_root}")
    print(f"Loaded splits. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Node features: {num_node_features} (pos features added: {add_pos_features})")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, dataset_info

BATCH_SIZE = 128
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, dataset_info = load_and_split_data(batch_size=BATCH_SIZE)

class VanillaGINBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        _ = edge_attr
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(norm(x))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return x

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None), batch=batch.batch)
        y = batch.y.view(-1).long()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1), total_correct / max(total_graphs, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None), batch=batch.batch)
        y = batch.y.view(-1).long()
        loss = criterion(logits, y)
        total_loss += loss.item() * batch.num_graphs
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1), total_correct / max(total_graphs, 1)


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _get_single_graph_batch(data):
    return data.batch if hasattr(data, "batch") else torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_ig_explainer(model):
    return Explainer(
        model=model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type=None,
    )

def evaluate_ig_dynamic_k(model, ig_explainer, test_loader, test_dataset):
    model.eval()
    correct_graphs = 0
    total_graphs = 0
    node_aurocs = []
    node_jaccards = []
    node_f1s = []
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None), batch=batch.batch)
            pred_classes = logits.argmax(dim=1)
            
            all_preds.append(pred_classes.cpu())
            all_labels.append(batch.y.cpu())
            
            correct_graphs += (pred_classes == batch.y).sum().item()
            total_graphs += batch.num_graphs
            
    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    # Macro F1 for multi-class MNIST
    test_f1 = f1_score(all_labels_np, all_preds_np, average='macro')

    for data in test_dataset:
        data = data.to(DEVICE)
        batch = _get_single_graph_batch(data)

        if hasattr(data, "node_mask"):
            gt_mask = data.node_mask.detach().cpu().numpy().astype(int)
        elif hasattr(data, "explanation_mask"):
            gt_mask = data.explanation_mask.detach().cpu().numpy().astype(int)
        else:
            continue

        with torch.no_grad():
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            pred_label = logits.argmax(dim=-1).item()

        explanation = ig_explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch,
            target=pred_label,
        )

        node_attr = explanation.node_mask.abs().sum(dim=1)

        if node_attr.max() > node_attr.min():
            node_scores = (node_attr - node_attr.min()) / (node_attr.max() - node_attr.min())
        else:
            node_scores = torch.zeros_like(node_attr)
        node_scores_np = node_scores.detach().cpu().numpy()

        if len(np.unique(gt_mask)) > 1:
            node_aurocs.append(roc_auc_score(gt_mask, node_scores_np))

        k = int(gt_mask.sum())
        if k > 0:
            _, top_k = torch.topk(node_attr, k=min(k, data.num_nodes))
            pred_binary = np.zeros(data.num_nodes, dtype=int)
            pred_binary[top_k.detach().cpu().numpy()] = 1
            intersect = (pred_binary * gt_mask).sum()
            union = (pred_binary + gt_mask).clip(0, 1).sum()
            node_jaccards.append(intersect / (union + 1e-8))
            
            # F1 Score calculation
            TP = intersect
            FP = pred_binary.sum() - TP
            FN = gt_mask.sum() - TP
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            node_f1s.append(2 * (precision * recall) / (precision + recall + 1e-8))

    acc = correct_graphs / max(total_graphs, 1)
    node_auc = float(np.mean(node_aurocs)) if node_aurocs else float("nan")
    node_jaccard = float(np.mean(node_jaccards)) if node_jaccards else float("nan")
    node_f1 = float(np.mean(node_f1s)) if node_f1s else float("nan")
    return acc, test_f1, node_auc, node_f1, node_jaccard

def time_ig_explanations(model, ig_explainer, dataset, warmup=2):
    model.eval()
    times_ms = []
    graphs = dataset

    for data in graphs[:warmup]:
        data = data.to(DEVICE)
        batch = _get_single_graph_batch(data)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            pred_label = logits.argmax(dim=-1).item()
        _ = ig_explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch,
            target=pred_label,
        )

    for data in graphs:
        data = data.to(DEVICE)
        batch = _get_single_graph_batch(data)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            pred_label = logits.argmax(dim=-1).item()

        _sync_if_cuda()
        start = time.perf_counter()
        _ = ig_explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch,
            target=pred_label,
        )
        _sync_if_cuda()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=float)
    mean_ms = float(arr.mean()) if arr.size else float("nan")
    median_ms = float(np.median(arr)) if arr.size else float("nan")
    p90_ms = float(np.percentile(arr, 90)) if arr.size else float("nan")
    total_s = float(arr.sum() / 1000.0) if arr.size else float("nan")
    graphs_per_s = float(1000.0 / mean_ms) if mean_ms > 0 else float("nan")
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "total_s": total_s,
        "graphs_per_s": graphs_per_s,
        "n_graphs": int(arr.size),
    }

def benchmark_test_inference_total(model, test_loader):
    model.eval()
    _sync_if_cuda()
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            _ = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None), batch=batch.batch)
    _sync_if_cuda()
    return time.perf_counter() - t0


HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 25

def train_ig_one_seed(seed: int, epochs: int = EPOCHS, early_stop_patience: int = EARLY_STOP_PATIENCE):
    set_all_seeds(seed)

    model = VanillaGINBackbone(
        in_channels=dataset_info["num_node_features"],
        hidden_channels=HIDDEN_DIM,
        out_channels=dataset_info["num_classes"],
        num_layers=4,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    _sync_if_cuda()
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        _, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print frequently enough to show training is progressing
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

        if epochs_no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    _sync_if_cuda()
    train_time_s = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_time_s

def run_multi_seed_ig_experiment(
    seeds=(11, 22, 33, 44, 55),
    epochs=EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
 ):
    per_seed = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        model_seed, train_time_s = train_ig_one_seed(
            seed=seed,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
        )

        ig_explainer = build_ig_explainer(model_seed)
        test_acc, test_f1, node_auc, node_f1, node_jaccard = evaluate_ig_dynamic_k(
            model_seed, ig_explainer, test_loader, test_dataset
        )

        test_process_total_s = benchmark_test_inference_total(model_seed, test_loader)
        timing = time_ig_explanations(model_seed, ig_explainer, test_dataset, warmup=2)

        row = {
            "seed": seed,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "node_auroc": node_auc,
            "node_f1": node_f1,
            "node_jaccard": node_jaccard,
            "train_time_s": train_time_s,
            "test_process_total_s": test_process_total_s,
            "ig_explain_total_s": timing["total_s"],
            "ig_mean_ms": timing["mean_ms"],
            "ig_median_ms": timing["median_ms"],
            "ig_p90_ms": timing["p90_ms"],
            "ig_graphs_per_s": timing["graphs_per_s"],
            "n_graphs": timing["n_graphs"],
        }
        per_seed.append(row)

        print(
            f"Seed {seed} | Test Acc {test_acc:.4f} | Test F1 {test_f1:.4f} | Node AUROC {node_auc:.4f} | Node F1 {node_f1:.4f} | Node Jaccard {node_jaccard:.4f} | "
            f"Train {train_time_s:.2f}s | TestProc {test_process_total_s:.2f}s | "
            f"IG {timing['total_s']:.2f}s ({timing['mean_ms']:.2f} ms/graph)"
        )

    print("\n" + "=" * 84)
    print("MULTI-SEED SUMMARY (mean ± std)")
    print("=" * 84)
    metrics = [
        "test_acc",
        "test_f1",
        "node_auroc",
        "node_f1",
        "node_jaccard",
        "train_time_s",
        "test_process_total_s",
        "ig_explain_total_s",
        "ig_mean_ms",
        "ig_median_ms",
        "ig_p90_ms",
        "ig_graphs_per_s",
    ]
    for metric in metrics:
        vals = np.array([row[metric] for row in per_seed], dtype=float)
        print(f"{metric:22s}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("-" * 84)
    print(f"Total IG explanation time over all seeds : {sum(row['ig_explain_total_s'] for row in per_seed):.4f} s")
    print(f"Total test processing time over all seeds: {sum(row['test_process_total_s'] for row in per_seed):.4f} s")
    print(f"Total training time over all seeds       : {sum(row['train_time_s'] for row in per_seed):.4f} s")

    return per_seed

    # Main run (5 seeds)
SEEDS = [11, 22, 33]
ig_multi_seed_results = run_multi_seed_ig_experiment(
    seeds=SEEDS,
    epochs=EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
 )

# Optional: table view
import pandas as pd
ig_result_df = pd.DataFrame(ig_multi_seed_results)
ig_result_df