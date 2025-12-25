import os
import random
import math
from typing import Optional
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.nn import ModuleList, Sequential, Dropout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from sklearn.metrics import roc_auc_score

from bcos.modules import BcosLinear
from bcos.modules.norms import DetachableLayerNorm
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.aggr import MeanAggregation

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def prepare_dataset(root: str = "data/MOLHIV"):
    ds = MoleculeNet(root=root, name="HIV")
    print(f"Dataset has {ds.num_node_features} node features")
    print(f"Dataset has {ds.num_edge_features} edge features")
    
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample node features dtype: {sample.x.dtype if sample.x is not None else 'None'}")
        print(f"Sample edge features dtype: {sample.edge_attr.dtype if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'None'}")
    
    if ds.num_node_features == 0:
        from torch_geometric.utils import degree
        for d in ds:
            d.x = degree(d.edge_index[0], num_nodes=d.num_nodes).unsqueeze(1).float()
    else:
        for d in ds:
            if d.x is not None:
                d.x = d.x.float()
    
    for d in ds:
        if hasattr(d, 'edge_attr') and d.edge_attr is not None:
            d.edge_attr = d.edge_attr.float()
    
    return ds


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    torch.manual_seed(SEED)
    perms = torch.randperm(n).tolist()
    from torch.utils.data import Subset
    return Subset(dataset, perms[:n_train]), Subset(dataset, perms[n_train:n_train + n_val]), Subset(dataset, perms[n_train + n_val:])


def safe_masked_bce_loss(logits, y, loss_fn):
    y = y.view(-1)
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return None, mask
    return loss_fn(logits[mask], y[mask].float()), mask


def compute_roc_auc_all(y_true_list, y_score_list):
    import numpy as np
    if len(y_true_list) == 0:
        return float("nan")
    y_true = np.concatenate([t.cpu().numpy() for t in y_true_list])
    y_score = np.concatenate([s.cpu().numpy() for s in y_score_list])
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


class BcosGINEClassifier(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden=128, num_layers=3, b=2.0, max_out=1, dropout=0.5):
        super().__init__()
        #GNN Body 
        self.lin_node = BcosLinear(node_feat_dim, hidden, b=b, max_out=max_out)
        self.lin_edge = BcosLinear(edge_feat_dim, hidden, b=b, max_out=max_out)

        # Define GINE layers individually
        self.conv1 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm1 = DetachableLayerNorm(hidden)
        
        self.conv2 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm2 = DetachableLayerNorm(hidden)

        self.conv3 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm3 = DetachableLayerNorm(hidden)

        # Classification Head 
        self.agg = MeanAggregation()
        self.readout = Sequential(
            BcosLinear(hidden, hidden, b=b, max_out=max_out),
            Dropout(dropout),
            BcosLinear(hidden, 1, b=b, max_out=max_out),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 1. Initial Projection
        x = self.lin_node(x)
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)

        # 2. Message Passing Layers
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)

        # 3. Graph Pooling
        h_graph = self.agg(x, batch)

        # 4. Final Classification
        out = self.readout(h_graph)
        
        return out.view(-1)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for data in loader:
        data = data.to(device)
        # Ensure data types are correct
        if data.x is not None:
            data.x = data.x.float()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None), getattr(data, "batch", None))
        y = data.y.view(-1).to(device)
        loss_masked, mask = safe_masked_bce_loss(logits, y, loss_fn)
        if loss_masked is None:
            continue
        loss_masked.backward()
        optimizer.step()
        total_loss += float(loss_masked) * int(mask.sum().item())
        n_graphs += int(mask.sum().item())
    return total_loss / (n_graphs + 1e-12)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_list = []
    y_score_list = []
    for data in loader:
        data = data.to(device)
        # Ensure data types are correct
        if data.x is not None:
            data.x = data.x.float()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None), getattr(data, "batch", None))
        y = data.y.view(-1).to(device)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        y_true_list.append(y[mask].cpu())
        y_score_list.append(torch.sigmoid(logits[mask]).cpu())
    return compute_roc_auc_all(y_true_list, y_score_list)


def run(epochs=100, batch_size=64, hidden=128, num_layers=3, lr=1e-3, out_dir="outputs/bcosgine"):
    dataset = prepare_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    node_feat_dim = dataset.num_node_features or 0
    edge_feat_dim = dataset.num_edge_features or 0

    model = BcosGINEClassifier(node_feat_dim, edge_feat_dim, hidden=hidden, num_layers=num_layers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -math.inf
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)
        val_auc = evaluate(model, val_loader, DEVICE)
        test_auc = evaluate(model, test_loader, DEVICE)
        if val_auc > best_val:
            best_val = val_auc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_auc": val_auc, "test_auc": test_auc}, os.path.join(out_dir, "best.pt"))
        print(f"Epoch {epoch:02d} loss={loss:.4f} val_auc={val_auc:.4f} test_auc={test_auc:.4f}")
    print("best val auc:", best_val)


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(SEED)
    run()


