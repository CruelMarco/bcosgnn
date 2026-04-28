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
data_root = Path(os.environ.get('MNISTSP_DATA_ROOT', str(default_data_root))).expanduser()
print(f"Using data root: {data_root}")

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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_and_split_data(batch_size = 64, val_ratio = 0.1, seed: int = 42):

    print("Loading dataset...")

    full_train_dataset = MNISTSuperpixels(root=str(data_root), train=True)

    test_datset = MNISTSuperpixels(root = str(data_root), train=False)

    num_train = len(full_train_dataset)

    num_val = int(num_train * val_ratio)

    train_size =  num_train - num_val

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, num_val] , generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_datset)}")

    return train_loader, val_loader, test_loader, full_train_dataset

##### Bcos Model Defnitions #####

class Readout(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=1,
        b=2,
        max_out=1,
        agg: str = "sum",
    ):
        super().__init__()
        if hidden_channels is None:
            self.readout = BcosLinear(in_channels, out_channels, b=b, max_out=max_out)
        else:
            hidden_channels = (
                [hidden_channels]
                if isinstance(hidden_channels, int)
                else hidden_channels
            )
            channels = [in_channels] + hidden_channels + [out_channels]
            self.readout = BcosSequential(
                *[
                    BcosLinear(d_in, d_out, b=b, max_out=max_out)
                    for d_in, d_out in zip(channels[:-1], channels[1:])
                ]
            )
        match agg:
            case "sum":
                self.agg = SumAggregation()
            case _:
                raise ValueError(f"Aggregation '{agg}' not supported.")

    def forward(self, x, batch):
        raise NotImplementedError


class AggThenReadout(Readout):
    def forward(self, x, batch):
        z = self.agg(x, batch)
        out = self.readout(z)
        return out

class ReadoutThenAgg(Readout):
    def forward(self, x, batch):
        z = self.readout(x)
        out = self.agg(z, batch)
        return out

import torch
from torch_geometric.nn import MessagePassing

class BcosGINEConv(MessagePassing):
    def __init__(
        self,
        channels: list[int],
        edge_dim: int,
        b: float = 2.0,
        max_out: int = 1,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        
        # The MLP part of GIN, but using B-cos layers
        self.transform = BcosSequential(
            *[
                BcosLinear(din, dout, b=b, max_out=max_out)
                for din, dout in zip(channels[:-1], channels[1:])
            ]
        )
        
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr):

        
        # 1. Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 2. Combine step: (1 + eps) * center_node + aggregated_messages
        out = (1 + self.eps) * x + out
        
        # 3. Apply the B-cos MLP transformation
        return self.transform(out)

    def message(self, x_j, edge_attr):

        return torch.nn.functional.relu(x_j + edge_attr)
    
    ##############

class PureBcosGINE(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 9,
        b: float = 2.0,
        max_out: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.has_edge_attr = edge_dim > 0

        self.lin_node = BcosLinear(node_dim, hidden_dim, b=b, max_out=max_out)

        self.lin_edge = BcosLinear(edge_dim, hidden_dim, b=b, max_out=max_out) if self.has_edge_attr else None

        self.convs = nn.ModuleList([
            BcosGINEConv(
                channels=[hidden_dim, hidden_dim],
                edge_dim=hidden_dim,
                b=b,
                max_out=max_out
            ) for _ in range(num_layers)
        ])

        self.readout_mlp = BcosSequential(
            BcosLinear(hidden_dim, hidden_dim, b=b, max_out=max_out),
            BcosLinear(hidden_dim, num_classes, b=b, max_out=max_out)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self.agg = SumAggregation()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_node(x)

        if self.has_edge_attr and edge_attr is not None:
            e = self.lin_edge(edge_attr)
        else:
            e = x.new_zeros((edge_index.size(1), self.hidden_dim))
        
        for conv in self.convs:
            x = conv(x, edge_index, e)
        
        node_logits = self.readout_mlp(x)
        node_logits = self.dropout_layer(node_logits)
        graph_logits = self.agg(node_logits, batch)
        
        return graph_logits
    
    ###############

####### Training & Evaluation Loops #######

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_graphs

            pred = torch.argmax(out, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return avg_loss, acc, f1

def run_bcos_experiment(
    seeds=(42, 43, 44),
    batch_size=64,
    val_ratio=0.1,
    hidden_dim=64,
    num_layers=4,
    lr=1e-3,
    max_epochs=100,
    patience=30,
    min_delta=1e-4,
    b=2.0,
    max_out=1,
    dropout=0.5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    all_test_losses, all_test_accs, all_test_f1s = [], [], []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n===== B-Cos Run {run_idx}/{len(seeds)} | Seed: {seed} =====")
        set_seed(seed)

        train_loader, val_loader, test_loader, full_dataset = load_and_split_data(
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
        )

        model = PureBcosGINE(
            node_dim=full_dataset.num_node_features,
            edge_dim=full_dataset.num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=full_dataset.num_classes,
            b=b,
            max_out=max_out,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        print("Starting training...")
        for epoch in range(1, max_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:03d}/{max_epochs}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}, LR: {current_lr:.6f}"
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best Val Loss: {best_val_loss:.4f}"
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        print("Evaluating on test set...")
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        all_test_losses.append(test_loss)
        all_test_accs.append(test_acc)
        all_test_f1s.append(test_f1)

        print(
            f"Seed {seed} Test -> Loss: {test_loss:.4f}, "
            f"Acc: {test_acc:.4f}, F1: {test_f1:.4f}"
        )

    summary = {
        'test_loss_mean': float(np.mean(all_test_losses)),
        'test_loss_std': float(np.std(all_test_losses)),
        'test_acc_mean': float(np.mean(all_test_accs)),
        'test_acc_std': float(np.std(all_test_accs)),
        'test_f1_mean': float(np.mean(all_test_f1s)),
        'test_f1_std': float(np.std(all_test_f1s)),
        'all_test_losses': all_test_losses,
        'all_test_accs': all_test_accs,
        'all_test_f1s': all_test_f1s,
    }

    print("\n===== B-Cos Final Summary Across Seeds =====")
    print(f"Test Loss: {summary['test_loss_mean']:.4f} ± {summary['test_loss_std']:.4f}")
    print(f"Test Acc : {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"Test F1  : {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")

    return summary


bcos_results = run_bcos_experiment(
    seeds=(42, 43, 44),
    batch_size=64,
    val_ratio=0.1,
    hidden_dim=64,
    num_layers=4,
    lr=1e-3,
    max_epochs=100,
    patience=30,
    min_delta=1e-4,
    b=2.0,
    max_out=1,
    dropout=0.5,
)

bcos_results