import sys
import os
import copy
import random
from pathlib import Path

current_dir = Path.cwd()
script_dir = Path(__file__).resolve().parent

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
    print("Repo root not found in Condor scratch dir; using installed packages and local paths.")
    project_root = script_dir

default_data_root = project_root / 'data' / 'MNISTsp'
data_root = Path(os.environ.get('MNISTSP_DATA_ROOT', str(default_data_root))).expanduser()

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

    full_train_dataset = MNISTSuperpixels(root=str(project_root / 'data' / 'MNISTsp'), train=True)

    test_datset = MNISTSuperpixels(root = str(project_root / 'data' / 'MNISTsp'), train=False)

    num_train = len(full_train_dataset)

    num_val = int(num_train * val_ratio)

    train_size =  num_train - num_val

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, num_val] , generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_datset)}")

    return train_loader, val_loader, test_loader, full_train_dataset




## DEFINE GINE MODEL

class GINE(nn.Module):

    def __init__(self, in_channels, edge_channels, hidden_channels, out_channels , num_layers=3):
        
        super(GINE, self).__init__()

        self.node_emb = nn.Linear(in_channels, hidden_channels)

        self.use_edge_attr = edge_channels > 0

        if self.use_edge_attr:
            self.edge_emb = nn.Linear(edge_channels, hidden_channels)
        else:
            self.edge_emb = None

        self.convs = nn.ModuleList()

        for _ in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels)
            )

            self.convs.append(GINEConv(nn = mlp , train_eps = True))

        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels//2)

        self.fc2 = nn.Linear(hidden_channels//2, out_channels)

    
    def forward(self, x, edge_index, edge_attr, batch):

        x = self.node_emb(x)

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = x.new_zeros((edge_index.size(1), x.size(-1)))

        for conv in self.convs:

            x = conv(x, edge_index, edge_attr)

            x = F.relu(x)

        x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)

        return x



## Training Function

def train_epoch(model, loader, optimizer, criterion, device):
    
    model.train()

    total_loss = 0

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

    total_loss = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():
        for data in tqdm(loader, desc = "Evaluating" , leave = False):

            data = data.to(device)

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_graphs

            pred = torch.argmax(out, dim=1)

            all_preds.append(pred.cpu().numpy())

            all_labels.append(data.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))

    f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average='weighted')

    return avg_loss, acc, f1

## Main Training Loop

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    seeds = [42, 43, 44]
    max_epochs = 100
    early_stopping_patience = 30
    min_delta = 1e-4

    criterion = nn.CrossEntropyLoss()
    all_test_losses = []
    all_test_accs = []
    all_test_f1s = []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n===== Run {run_idx}/{len(seeds)} | Seed: {seed} =====")
        set_seed(seed)

        train_loader, val_loader, test_loader, full_dataset = load_and_split_data(
            batch_size=64,
            val_ratio=0.1,
            seed=seed,
        )

        model = GINE(
            in_channels=full_dataset.num_node_features,
            edge_channels=full_dataset.num_edge_features,
            hidden_channels=64,
            out_channels=full_dataset.num_classes,
            num_layers=4,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        best_val_loss = float("inf")
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
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
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

    print("\n===== Final Summary Across Seeds =====")
    print(
        f"Test Loss: {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}"
    )
    print(
        f"Test Acc : {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}"
    )
    print(
        f"Test F1  : {np.mean(all_test_f1s):.4f} ± {np.std(all_test_f1s):.4f}"
    )

if __name__ == "__main__":

    main()



