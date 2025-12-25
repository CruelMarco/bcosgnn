import os
import random
import math
import sys
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.nn import ModuleList, Sequential, Dropout
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


from bcos.modules import BcosLinear
from bcos.modules.norms import DetachableLayerNorm
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.aggr import MeanAggregation

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def prepare_dataset(root: str = "data/MOLHIV"):
    """Loads and preprocesses the MoleculeNet HIV dataset."""
    print("--> Loading and preparing dataset...")
    ds = MoleculeNet(root=root, name="HIV")
    print(f"Dataset has {ds.num_node_features} node features")
    print(f"Dataset has {ds.num_edge_features} edge features")

    if ds.num_node_features == 0:
        from torch_geometric.utils import degree
        print("Dataset has no node features. Using node degrees as features.")
        for d in ds:
            d.x = degree(d.edge_index[0], num_nodes=d.num_nodes).unsqueeze(1).float()
    else:
        for d in ds:
            if d.x is not None:
                d.x = d.x.float()

    for d in ds:
        if hasattr(d, 'edge_attr') and d.edge_attr is not None:
            d.edge_attr = d.edge_attr.float()

    print("--> Dataset preparation complete.")
    return ds

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):

    test_ratio = 1.0 - train_ratio - val_ratio
    n = len(dataset)
    
    # Extract labels for stratification
    # Squeeze to handle labels like tensor([[1.]])
    labels = [data.y.squeeze().long().item() for data in dataset]
    indices = list(range(n))
    
    # First split: Create training set and a temporary set for validation+test
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=SEED
    )
    
    # Second split: Split the temporary set into validation and test sets
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, y_temp,
        test_size=(test_ratio / (val_ratio + test_ratio)), # Proportion of test set within the temp set
        stratify=y_temp,
        random_state=SEED
    )

    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    print(f"Stratified dataset split: {len(train_subset)} train, {len(val_subset)} validation, {len(test_subset)} test.")
    
    def check_distribution(subset, name):
        subset_labels = [subset.dataset.get(i).y.squeeze().long().item() for i in subset.indices]
        pos_count = sum(subset_labels)
        total_count = len(subset_labels)
        print(f"  {name} set: {pos_count} positive samples ({pos_count/total_count:.2%})")
        
    check_distribution(train_subset, "Train")
    check_distribution(val_subset, "Validation")
    check_distribution(test_subset, "Test")
        
    return train_subset, val_subset, test_subset



class BcosGINEClassifier(nn.Module):

    def __init__(self, node_feat_dim, edge_feat_dim, hidden=128, b=2.0, max_out=1, dropout=0.5):
        super().__init__()

        self.lin_node = BcosLinear(node_feat_dim, hidden, b=b, max_out=max_out)
        self.lin_edge = BcosLinear(edge_feat_dim, hidden, b=b, max_out=max_out)

        self.conv1 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm1 = DetachableLayerNorm(hidden)

        self.conv2 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm2 = DetachableLayerNorm(hidden)

        self.conv3 = GINEConv(nn=BcosLinear(hidden, hidden, b=b, max_out=max_out))
        self.norm3 = DetachableLayerNorm(hidden)

        self.agg = MeanAggregation()
        self.readout = Sequential(
            #BcosLinear(hidden, hidden, b=b, max_out=max_out),
            #Dropout(dropout),
            BcosLinear(hidden, 1, b=b, max_out=max_out),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.lin_node(x)
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)

        h_graph = self.agg(x, batch)
        out = self.readout(h_graph)

        return out.view(-1)



def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for data in loader:
        data = data.to(device)

        if data.x is not None:
            data.x = data.x.float()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()

        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None), getattr(data, "batch", None))

        y = data.y.view(-1)
        mask = ~torch.isnan(y)

        if mask.sum() == 0:
            continue

        loss = loss_fn(logits[mask], y[mask].float())
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(mask.sum().item())
        n_graphs += int(mask.sum().item())

    return total_loss / (n_graphs + 1e-12)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:

    model.eval()
    y_true_list = []
    y_score_list = []
    for data in loader:
        data = data.to(device)

        if data.x is not None:
            data.x = data.x.float()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()

        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None), getattr(data, "batch", None))

        y = data.y.view(-1)
        mask = ~torch.isnan(y)

        if mask.sum() == 0:
            continue

        y_true_list.append(y[mask].cpu())
        y_score_list.append(torch.sigmoid(logits[mask]).cpu())

    if not y_true_list:
        return float("nan"), np.array([]), np.array([])

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)

    if len(set(y_true.tolist())) < 2:
        return float("nan"), y_true, y_score

    auc_score = roc_auc_score(y_true, y_score)
    return auc_score, y_true, y_score


def plot_roc_auc(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC AUC curve saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Computes, plots, and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()
    print(f" Confusion matrix saved to {save_path}")

# ___________________________
# MAIN EXECUTION SCRIPT
# ___________________________

def run(epochs=100, batch_size=128, hidden=128, lr=1e-3, out_dir="outputs/bcosgine"):
    os.makedirs(out_dir, exist_ok=True)

    #  Data Loading ---
    dataset = prepare_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model Initialization ---
    node_feat_dim = dataset.num_node_features
    edge_feat_dim = dataset.num_edge_features
    model = BcosGINEClassifier(node_feat_dim, edge_feat_dim, hidden=hidden).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Starting Model Training...")
    print(f"Device: {DEVICE}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # Training Loop 
    best_val_auc = -math.inf
    best_epoch = 0
    best_model_path = os.path.join(out_dir, "best_model.pt")
    best_val_preds = None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)
        val_auc, y_true_val, y_score_val = evaluate(model, val_loader, DEVICE)

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            best_val_preds = (y_true_val, y_score_val)
            print(f"  -> New best model saved with Val AUC: {best_val_auc:.4f}")

    print(f"Best model found at epoch {best_epoch} with Validation AUC: {best_val_auc:.4f}")

    # Plot Validation Set ROC Curve
    if best_val_preds:
        y_true_val_best, y_score_val_best = best_val_preds
        plot_roc_auc(y_true_val_best, y_score_val_best, save_path=os.path.join(out_dir, "val_roc_auc_curve.png"))

    # Final Evaluation on Test Set
    print(" Final Evaluation on Held-Out Test Set")

    model.load_state_dict(torch.load(best_model_path))

    test_auc, y_true_test, y_score_test = evaluate(model, test_loader, DEVICE)

    y_pred_test = (y_score_test > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='binary', zero_division=0)

    print(" Test Set Performance Metrics ")
    print(f"ROC AUC:  {test_auc:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    plot_roc_auc(y_true_test, y_score_test, save_path=os.path.join(out_dir, "test_roc_auc_curve.png"))
    plot_confusion_matrix(y_true_test, y_pred_test, save_path=os.path.join(out_dir, "test_confusion_matrix.png"))


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(SEED)
    run(epochs=100, batch_size=64, hidden=300 , lr=1e-3, out_dir="outputs/bcosgine")