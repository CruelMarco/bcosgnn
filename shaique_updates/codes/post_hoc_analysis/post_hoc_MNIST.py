import os
import sys
import random
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE =', DEVICE)

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'pyproject.toml').exists() and (p / 'bcosgnn').is_dir():
            return p
    raise RuntimeError('Could not locate repo root (pyproject.toml + bcosgnn/).')

repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
print('Repo root added:', repo_root)

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

def get_gnn_explainer(model, epochs: int = 200, lr: float = 0.01):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type=None,
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

def get_ground_truth_mask(data):
    keys_to_check = ['node_mask', 'explanation_mask']
    for key in keys_to_check:
        if hasattr(data, key):
            mask = getattr(data, key)
            if mask is not None:
                return mask
    return None

def evaluate_custom_jaccard(explainer, model, dataset, max_graphs=None):
    jaccard_scores = []
    graphs = dataset[:max_graphs] if max_graphs else dataset
    print(f"Evaluating Jaccard on {len(graphs)} graphs...")
    
    for data in tqdm(graphs):
        data = data.to(DEVICE)
        
        gt_mask = get_ground_truth_mask(data)
        if gt_mask is None: continue
        
        gt_mask = gt_mask.squeeze().cpu().numpy()
        gt_nodes = set(np.where(gt_mask == 1)[0])
        k = len(gt_nodes)
        if k == 0: continue

        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            target = logits.argmax().item()
            
        explanation = explainer(
            x=data.x, 
            edge_index=data.edge_index, 
            target=torch.tensor([target], device=DEVICE),
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch
        )
        
        pred_mask = explanation.node_mask.detach().cpu().numpy().flatten()
        
        top_k_indices = np.argsort(pred_mask)[-k:]
        pred_nodes = set(top_k_indices)

        intersection = len(gt_nodes.intersection(pred_nodes))
        union = len(gt_nodes.union(pred_nodes))
        jaccard_scores.append(intersection / (union + 1e-8))

    if not jaccard_scores: return 0.0
    return float(np.mean(jaccard_scores))

def evaluate_custom_auroc(explainer, model, dataset, max_graphs=None):
    auroc_scores = []
    graphs = dataset[:max_graphs] if max_graphs else dataset
    print(f"Evaluating AUROC on {len(graphs)} graphs...")
    
    for data in tqdm(graphs):
        data = data.to(DEVICE)
        
        gt_mask = get_ground_truth_mask(data)
        if gt_mask is None: continue
        gt_mask = gt_mask.squeeze().cpu().numpy()
        
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue

        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            target = logits.argmax().item()

        explanation = explainer(
            x=data.x, 
            edge_index=data.edge_index, 
            target=torch.tensor([target], device=DEVICE),
            edge_attr=getattr(data, "edge_attr", None),
            batch=batch
        )
        
        pred_mask = explanation.node_mask.detach().cpu().numpy().flatten()
        
        try:
            score = roc_auc_score(gt_mask, pred_mask)
            auroc_scores.append(score)
        except ValueError: pass

    if not auroc_scores: return 0.0
    return float(np.mean(auroc_scores))


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _get_batch(data):
    return data.batch if hasattr(data, "batch") else torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)

def time_gnnexplainer_per_graph(gnn_explainer, model, dataset, warmup: int = 2, max_graphs: int | None = None):
    model.eval()
    times_ms = []
    graphs = dataset[:max_graphs] if max_graphs is not None else dataset

    # Warmup
    for data in graphs[:warmup]:
        data = data.to(DEVICE)
        batch = _get_batch(data)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            pred_label = logits.argmax(dim=-1).item()
        _ = gnn_explainer(x=data.x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch, target=torch.tensor([pred_label], device=DEVICE))

    # Timing
    for data in graphs:
        data = data.to(DEVICE)
        batch = _get_batch(data)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch)
            pred_label = logits.argmax(dim=-1).item()
        
        _sync_if_cuda()
        start = time.perf_counter()
        _ = gnn_explainer(x=data.x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None), batch=batch, target=torch.tensor([pred_label], device=DEVICE))
        _sync_if_cuda()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    times_ms = np.asarray(times_ms, dtype=float)
    mean_ms = float(times_ms.mean()) if times_ms.size else float('nan')
    std_ms = float(times_ms.std()) if times_ms.size else float('nan')
    median_ms = float(np.median(times_ms)) if times_ms.size else float('nan')
    p90_ms = float(np.percentile(times_ms, 90)) if times_ms.size else float('nan')
    graphs_per_s = float(1000.0 / mean_ms) if mean_ms > 0 else float('nan')
    total_s = float(times_ms.sum() / 1000.0) if times_ms.size else float('nan')

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "graphs_per_s": graphs_per_s,
        "total_s": total_s,
        "n_graphs": int(times_ms.size),
    }


HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 25

def run_multi_seed_gnnexplainer(seeds=(11, 22, 33, 44, 55)):
    results = []
    
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
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
        
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
                
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
                
        _sync_if_cuda()
        train_time_s = time.perf_counter() - train_start
        
        if best_state is not None:
            model.load_state_dict(best_state)
            
        print("Evaluating Explainer...")
        explainer = get_gnn_explainer(model, epochs=200, lr=0.01)
        
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
        
        # GNNExplainer is slow, limit to 1000 graphs or adjust as needed
        max_graphs_explainer = 1000
        node_jaccard = evaluate_custom_jaccard(explainer, model, test_dataset, max_graphs=max_graphs_explainer)
        node_auroc = evaluate_custom_auroc(explainer, model, test_dataset, max_graphs=max_graphs_explainer)
        timing = time_gnnexplainer_per_graph(explainer, model, test_dataset, warmup=2, max_graphs=max_graphs_explainer)
        
        print(f"Seed {seed} Results:")
        print(f"Test Acc: {test_acc:.4f} | Node Jaccard: {node_jaccard:.4f} | Node AUROC: {node_auroc:.4f}")
        print(f"Explainer Time: {timing['total_s']:.2f}s ({timing['mean_ms']:.2f} ms/graph)")
        
        results.append({
            "seed": seed,
            "test_acc": test_acc,
            "node_jaccard": node_jaccard,
            "node_auroc": node_auroc,
            "train_time_s": train_time_s,
            "explainer_total_s": timing["total_s"],
            "explainer_mean_ms": timing["mean_ms"]
        })
        
    return results

    # Smoke test
#smoke_results = run_multi_seed_gnnexplainer(seeds=[123])
#print("Smoke test complete.")
#print(smoke_results)

SEEDS = [11, 22, 33]
results = run_multi_seed_gnnexplainer(seeds=SEEDS)
import pandas as pd
df = pd.DataFrame(results)
print("\nFinal Results:")
print(df)

formatted_stats = df.apply(lambda col: f"{col.mean():.4f} ± {col.std():.4f}")

print(formatted_stats)