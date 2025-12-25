import os
import tqdm
import random
import math
import sys
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader import DataLoader
from torch.nn import Module, ModuleList, Sequential, Dropout
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import LayerNorm
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn import global_add_pool


try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    print("Warning: '__file__' not defined. Assuming project root is two levels up from current working directory.")
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

try:
    from bcos.modules import BcosLinear
    from bcos.modules.norms import DetachableLayerNorm
except ImportError:
    print(" IMPORT ERRO")
    print("Exiting.")
    sys.exit(1)



from torch_geometric.nn.conv import GINConv


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataset(root: str = "data/BA2Motif"):
    print("--> Loading and preparing dataset...")

    transform = OneHotDegree(max_degree=10) 
    ds = BA2MotifDataset(root=root, pre_transform=transform)
    
    print(f"Dataset has {ds.num_node_features} node features (after OneHotDegree transform)")
    print("--> Dataset preparation complete.")
    return ds

def split_dataset(dataset):
    total_len = len(dataset)
    
    print("  Extracting all labels for stratification...")
    all_labels = [data.y.item() for data in dataset]
    
    train_len = int(total_len * 0.8)
    valid_len = int(total_len * 0.1)
    
    indices = list(range(total_len))
    
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_len,
        random_state=SEED,
        stratify=all_labels  
    )
    
    temp_labels = [all_labels[i] for i in temp_indices]
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=valid_len,
        random_state=SEED,
        stratify=temp_labels  
    )

    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    print(f"Dataset split: {len(train_subset)} train, {len(val_subset)} validation, {len(test_subset)} test.")
    
    def check_distribution(subset, name):
        subset_labels = [all_labels[i] for i in subset.indices]
        pos_count = sum(subset_labels)
        total_count = len(subset_labels)
        print(f"  {name} set: {pos_count} positive samples ({pos_count/total_count:.2%})")
        
    check_distribution(train_subset, "Train")
    check_distribution(val_subset, "Validation")
    check_distribution(test_subset, "Test")
        
    return train_subset, val_subset, test_subset




#  Model Definition 
class BcosGINClassifier(Module):
    def __init__(self, node_feat_dim, hidden=300, b=2.0, max_out=1, dropout=0.5):
        super().__init__()
        self.activation = nn.GELU()

        # Initial projection layer
        self.lin_node = BcosLinear(node_feat_dim, hidden, b=b, max_out=max_out)
        self.norm_initial = DetachableLayerNorm(hidden)

        # Layer 1
        self.norm1 = DetachableLayerNorm(hidden)
        nn1 = BcosLinear(hidden, hidden, b=b, max_out=max_out)
        self.conv1 = GINConv(nn=nn1, train_eps=True)

        # Layer 2
        self.norm2 = DetachableLayerNorm(hidden)
        nn2 = BcosLinear(hidden, hidden, b=b, max_out=max_out)
        self.conv2 = GINConv(nn=nn2, train_eps=True)
        
        self.norm_final = DetachableLayerNorm(hidden)

        
        # Readout
        self.dropout = Dropout(p=dropout)
        self.readout = BcosLinear(hidden, 1, b=b, max_out=max_out)

    def forward(self, x, edge_index, batch=None):
        h = self.lin_node(x)

        
        # Layer 1
        h_in1 = h
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h, edge_index)
        h = h + h_in1 # Residual connection

        # Layer 2
        h_in2 = h
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h, edge_index)
        h = h + h_in2 # Residual connection

        # Final Norm and Activation before Readout
        h = self.norm_final(h)
        h = self.activation(h)

        h_graph = global_add_pool(h, batch)
        h_graph = self.dropout(h_graph)
        out = self.readout(h_graph)

        return out.view(-1)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for data in tqdm.tqdm(loader, desc="Training"): 
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(logits, data.y.float())
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list = [], []
    total_loss = 0.0
    for data in tqdm.tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        
        logits = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(logits, data.y.float())
        
        total_loss += loss.item() * data.num_graphs
        
        preds = (logits > 0).long()
        y_true_list.append(data.y.cpu())
        y_pred_list.append(preds.cpu())

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, acc, y_true, y_pred


def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")

def plot_confusion_matrix_custom(y_true, y_pred, acc, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred House (0)', 'Pred Grid (1)'],
                yticklabels=['Actual House (0)', 'Actual Grid (1)'])
    plt.title(f'Confusion Matrix (Accuracy: {acc:.4f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# --- Main Execution ---

def run(epochs=100, batch_size=128, hidden=300, lr=1e-4, dropout=0.5, out_dir="outputs/bcos_gine_Ba2Motif"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Outputs will be saved to: {out_dir}")

    # Data
    dataset = prepare_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    # Model
    model = BcosGINClassifier(
        node_feat_dim=dataset.num_node_features,
        hidden=hidden,
        dropout=dropout
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr) 
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Hidden Dim: {hidden}")

    best_val_acc = -1
    best_epoch = 0
    best_model_path = os.path.join(out_dir, "best_model.pt")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, DEVICE)
        
        scheduler.step(val_loss) 
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with Val Acc: {best_val_acc:.4f}")

    print(f"Best model from epoch {best_epoch} with Validation Acc: {best_val_acc:.4f}")
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, save_path=os.path.join(out_dir, "loss_curves.png"))

    print("\n--- Final Evaluation on Held-Out Test Set ---")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, loss_fn, DEVICE)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='binary', zero_division=0)

    print("Test Set Performance Metrics:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Loss:      {test_loss:.4f}")

    plot_confusion_matrix_custom(y_true_test, y_pred_test, test_acc, save_path=os.path.join(out_dir, "test_confusion_matrix.png"))

if __name__ == "__main__":
    run()