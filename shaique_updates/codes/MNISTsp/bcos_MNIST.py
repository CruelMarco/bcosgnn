import sys
import os
import copy
import random
from pathlib import Path

current_dir = Path.cwd()
script_dir = Path(__file__).resolve().parent if '__file__' in globals() else current_dir

def find_repo_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / 'pyproject.toml').exists() and (p / 'bcosgnn').is_dir():
            return p
    return None

project_root = find_repo_root(current_dir) or find_repo_root(script_dir)

if project_root is not None:
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    print(f"Repo root added: {project_root}")
else:
    print("Repo root not found in this execution context; using installed packages and local paths.")
    project_root = current_dir

default_data_root = project_root / 'data' / 'MNISTsp'
split_root_env = os.environ.get('MNISTSP_SPLIT_ROOT')
data_root = Path(os.environ.get('MNISTSP_DATA_ROOT', str(default_data_root))).expanduser()
print(f"Using data root: {data_root}")
if split_root_env:
    print(f"Using MNISTSP_SPLIT_ROOT: {split_root_env}")

import bcosgnn
import torch
import torch_geometric
import functools
import itertools
import operator
from typing import Any
import torch
from torch_geometric.data import Dataset, download_url
from torch.utils.data import random_split
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from bcos.modules import BcosLinear, BcosSequential
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils import add_self_loops, degree
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from bcosgnn.explain import explain
from tqdm import tqdm
from torch_geometric.datasets import MNISTSuperpixels
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import random_split
from torch_geometric.nn import SumAggregation
from sklearn.metrics import roc_auc_score
import time
import random
from copy import deepcopy
import time
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score




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


def load_and_split_data(batch_size=64, add_pos_features=True):
    print("Loading preprocessed sparsified .pt splits...")

    candidate_roots = []
    if split_root_env:
        candidate_roots.append(Path(split_root_env).expanduser())
    candidate_roots.extend([
        project_root / 'data' / 'MNIST' / 'sparsified_pt_splits',
        project_root / 'shaique_updates' / 'codes' / 'MNISTsp' / 'data' / 'MNIST' / 'sparsified_pt_splits',
    ])

    split_root = None
    for root in candidate_roots:
        if (root / 'train_sparsified.pt').exists() and (root / 'val_sparsified.pt').exists() and (root / 'test_sparsified.pt').exists():
            split_root = root
            break

    if split_root is None:
        raise FileNotFoundError(
            "Could not find saved sparsified split files. "
            "Expected train/val/test_sparsified.pt in one of the candidate directories."
        )

    train_path = split_root / 'train_sparsified.pt'
    val_path = split_root / 'val_sparsified.pt'
    test_path = split_root / 'test_sparsified.pt'

    train_dataset = torch.load(train_path, map_location='cpu', weights_only=False)
    val_dataset = torch.load(val_path, map_location='cpu', weights_only=False)
    test_dataset = torch.load(test_path, map_location='cpu', weights_only=False)

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
    avg_train_edges = float(np.mean([int(d.edge_index.size(1)) for d in train_dataset[:1000]]))

    dataset_info = {
        'num_node_features': num_node_features,
        'num_classes': num_classes,
        'num_edge_features': int(train_dataset[0].edge_attr.size(-1)) if train_dataset[0].edge_attr is not None else 0
    }

    print(f"Using split directory: {split_root}")
    print(
        f"Loaded splits. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    print(f"Avg train edges (first 1000 graphs): {avg_train_edges:.2f}")
    print(f"Node features: {num_node_features} (pos features added: {add_pos_features})")

    return train_loader, val_loader, test_loader, dataset_info

# Dataset Stats

train_loader, val_loader, test_loader, dataset_info = load_and_split_data(batch_size=64)

print(f"Dataset info: {dataset_info}")

from sklearn.metrics import roc_auc_score

class BcosGINConv(MessagePassing):
    def __init__(self, channels, b=2, max_out=1, eps=0.0, train_eps=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.transform = BcosSequential(
            *[
                BcosLinear(din, dout, b=b, max_out=max_out)
                for din, dout in zip(channels[:-1], channels[1:])
            ]
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor([eps], dtype=torch.float))
        else:
            self.register_buffer('eps', torch.tensor([eps], dtype=torch.float))

    def forward(self, x, edge_index):
        x_res = x
        out = self.propagate(edge_index, x=x)
        out = (1 + self.eps) * x_res + out
        return self.transform(out)

    def message(self, x_j):
        return x_j


class BCosGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, b=2, max_out=1):
        super().__init__()
        self.lin_node = BcosLinear(in_channels, hidden_channels, b=b, max_out=max_out)
        self.convs = torch.nn.ModuleList(
            [BcosGINConv([hidden_channels, hidden_channels], b=b, max_out=max_out, train_eps=True) for _ in range(num_layers)]
        )
        self.agg = SumAggregation()
        self.readout = BcosSequential(
            BcosLinear(hidden_channels, hidden_channels, b=b, max_out=max_out),
            BcosLinear(hidden_channels, out_channels, b=b, max_out=max_out),
        )

    def forward(self, x, edge_index, batch):
        x = self.lin_node(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.agg(x, batch)
        out = self.readout(x)
        return out

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            pred = torch.argmax(out, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average='weighted')

    return avg_loss, acc, f1


def train_bcos_gin(
    epochs=100,
    batch_size=64,
    lr=1e-3,
    hidden_channels=64,
    num_layers=4,
    use_scheduler=False,
    fixed_lr=5e-3,
    early_stopping=True,
    patience=15,
    min_delta=0.0,
    print_every = 10,
 ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader, dataset_info = load_and_split_data(batch_size=batch_size, add_pos_features=True)

    model = BCosGIN(
        in_channels=dataset_info['num_node_features'],
        hidden_channels=hidden_channels,
        out_channels=dataset_info['num_classes'],
        num_layers=num_layers,
        b=2.0,
        max_out=1,
    ).to(device)

    effective_lr = lr if use_scheduler else fixed_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        if use_scheduler
        else None
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    print('Starting BCos GIN training...')
    print(
        f'Config -> epochs={epochs}, batch_size={batch_size}, hidden={hidden_channels}, layers={num_layers}, '
        f'use_scheduler={use_scheduler}, lr={effective_lr}, early_stopping={early_stopping}, patience={patience}'
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        
        if epoch == 1 or epoch % print_every == 0:


            print(
                f'[BCOS] Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}'
            )



        if early_stopping and wait >= patience:
            print(f'[BCOS] Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs).')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'[BCOS] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

    return model, train_loader, val_loader, test_loader


def get_bcos_node_scores(model, data):
    model_device = next(model.parameters()).device
    data = data.to(model_device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=model_device)

    node_contrib = explain(model, data.x, data.edge_index, batch).detach()
    node_scores = node_contrib.abs().sum(dim=1).cpu().numpy()
    gt_mask = data.node_mask.detach().cpu().numpy().reshape(-1).astype(int)

    return node_scores, gt_mask


def evaluate_bcos_explanations(model, dataset, max_graphs=None):
    jaccards = []
    aurocs = []
    n = len(dataset) if max_graphs is None else min(len(dataset), max_graphs)

    for idx in range(n):
        data = dataset[idx].clone()
        if not hasattr(data, 'node_mask') or data.node_mask is None:
            continue

        scores, gt = get_bcos_node_scores(model, data)
        k = int(gt.sum())
        if k <= 0:
            continue

        top_k = np.argsort(scores)[-k:]
        pred = np.zeros_like(gt)
        pred[top_k] = 1

        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        jaccard = float(intersection / union) if union > 0 else 0.0
        jaccards.append(jaccard)

        if gt.min() != gt.max():
            aurocs.append(float(roc_auc_score(gt, scores)))

    mean_jacc = float(np.mean(jaccards)) if len(jaccards) else float('nan')
    mean_auroc = float(np.mean(aurocs)) if len(aurocs) else float('nan')

    print(f'[BCOS Explain] Graphs evaluated: {n}')
    print(f'[BCOS Explain] Mean Jaccard: {mean_jacc:.4f}')
    print(f'[BCOS Explain] Mean AUROC: {mean_auroc:.4f}')

    return {'jaccard': mean_jacc, 'auroc': mean_auroc, 'num_graphs': n}


    # Multi-seed BCOS experiment with timing + metrics

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_bcos_classification(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return acc, f1

def evaluate_bcos_explanations_with_timing(model, dataset):
    model.eval()
    jaccards = []
    aurocs = []
    per_graph_times = []
    for idx in range(len(dataset)):
        data = dataset[idx].clone()
        if not hasattr(data, 'node_mask') or data.node_mask is None:
            continue
        start = time.perf_counter()
        scores, gt = get_bcos_node_scores(model, data)
        elapsed = time.perf_counter() - start
        per_graph_times.append(elapsed)
        k = int(gt.sum())
        if k <= 0:
            continue
        top_k = np.argsort(scores)[-k:]
        pred = np.zeros_like(gt)
        pred[top_k] = 1
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        jaccard = float(intersection / union) if union > 0 else 0.0
        jaccards.append(jaccard)
        if gt.min() != gt.max():
            aurocs.append(float(roc_auc_score(gt, scores)))
    return {
        'jaccard': float(np.mean(jaccards)) if len(jaccards) else float('nan'),
        'auroc': float(np.mean(aurocs)) if len(aurocs) else float('nan'),
        'per_graph_times': per_graph_times,
        'num_graphs': len(per_graph_times),
    }

def run_bcos_multi_seed(seeds=(0, 1, 2)):
    results = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        set_all_seeds(seed)
        train_start = time.perf_counter()
        model, train_loader, val_loader, test_loader = train_bcos_gin(
            epochs=100,
            batch_size=64,
            lr=1e-3,
            hidden_channels=64,
            num_layers=4,
            use_scheduler=False,
            fixed_lr=5e-3,
            early_stopping=True,
            patience=15,
            min_delta=0.0,
            print_every=10,
        )
        train_time = time.perf_counter() - train_start
        device = next(model.parameters()).device
        test_acc, test_f1 = evaluate_bcos_classification(model, test_loader, device)
        expl_start = time.perf_counter()
        expl_metrics = evaluate_bcos_explanations_with_timing(model, test_loader.dataset)
        expl_time = time.perf_counter() - expl_start
        end_to_end = train_time + expl_time
        per_graph_times = np.array(expl_metrics['per_graph_times'], dtype=float)
        per_graph_ms = per_graph_times * 1000.0
        mean_ms = float(np.mean(per_graph_ms)) if len(per_graph_ms) else float('nan')
        median_ms = float(np.median(per_graph_ms)) if len(per_graph_ms) else float('nan')
        p90_ms = float(np.percentile(per_graph_ms, 90)) if len(per_graph_ms) else float('nan')
        throughput = float(expl_metrics['num_graphs'] / expl_time) if expl_time > 0 else float('nan')
        result = {
            'seed': seed,
            'train_time_sec': train_time,
            'explain_time_sec': expl_time,
            'end_to_end_time_sec': end_to_end,
            'mean_ms_per_graph': mean_ms,
            'median_ms_per_graph': median_ms,
            'p90_ms_per_graph': p90_ms,
            'throughput_graphs_per_s': throughput,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'jaccard': expl_metrics['jaccard'],
            'auroc': expl_metrics['auroc'],
            'num_graphs': expl_metrics['num_graphs'],
        }
        results.append(result)
        print(result)
    return results

def summarize_results(results):
    metrics = [
        'test_acc',
        'test_f1',
        'auroc',
        'jaccard',
        'train_time_sec',
        'explain_time_sec',
        'end_to_end_time_sec',
        'mean_ms_per_graph',
        'median_ms_per_graph',
        'p90_ms_per_graph',
        'throughput_graphs_per_s',
    ]
    summary = {}
    for key in metrics:
        values = np.array([r[key] for r in results], dtype=float)
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary

def print_summary(summary):
    print("\n=== Summary (mean ± std) ===")
    print(f"Test Accuracy : {summary['test_acc']['mean']:.4f} ± {summary['test_acc']['std']:.4f}")
    print(f"Test F1 Score : {summary['test_f1']['mean']:.4f} ± {summary['test_f1']['std']:.4f}")
    print(f"Explanation AUROC : {summary['auroc']['mean']:.4f} ± {summary['auroc']['std']:.4f}")
    print(f"Explanation Jaccard: {summary['jaccard']['mean']:.4f} ± {summary['jaccard']['std']:.4f}")
    print(f"Train time (s): {summary['train_time_sec']['mean']:.4f} ± {summary['train_time_sec']['std']:.4f}")
    print(f"Explain total time (s): {summary['explain_time_sec']['mean']:.4f} ± {summary['explain_time_sec']['std']:.4f}")
    print(f"End-to-end time (s): {summary['end_to_end_time_sec']['mean']:.4f} ± {summary['end_to_end_time_sec']['std']:.4f}")
    print(f"Mean ms/graph: {summary['mean_ms_per_graph']['mean']:.4f} ± {summary['mean_ms_per_graph']['std']:.4f}")
    print(f"Median ms/graph: {summary['median_ms_per_graph']['mean']:.4f} ± {summary['median_ms_per_graph']['std']:.4f}")
    print(f"P90 ms/graph: {summary['p90_ms_per_graph']['mean']:.4f} ± {summary['p90_ms_per_graph']['std']:.4f}")
    print(f"Throughput (graphs/s): {summary['throughput_graphs_per_s']['mean']:.4f} ± {summary['throughput_graphs_per_s']['std']:.4f}")

# Run the 3-seed experiment
seed_results = run_bcos_multi_seed(seeds=(0, 1, 2))
summary = summarize_results(seed_results)
print_summary(summary)