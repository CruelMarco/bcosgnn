import os
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path | None:
    candidates = [start, *start.parents]

    for path in candidates:
        for candidate in (path, path / "bcosgnn"):
            if (candidate / "pyproject.toml").is_file() and (candidate / "bcosgnn").is_dir():
                return candidate

    return None


current_dir = Path.cwd()
script_dir = Path(__file__).resolve().parent

project_root = find_repo_root(current_dir) or find_repo_root(script_dir)

if project_root is None:
    raise RuntimeError("Could not locate repo root (pyproject.toml + bcosgnn/).")

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"Repo root added: {project_root}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.nn import MessagePassing
import math
import random
import numpy as np
from copy import deepcopy
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


data_dir = Path(os.environ.get("BCOSGNN_DATA_ROOT", project_root / "data")).expanduser()
dataset = GNNBenchmarkDataset(root=str(data_dir), name="PATTERN", split="train")



class MaskableGINConv(MessagePassing):
    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__(aggr="add")
        self.mlp = mlp
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float))
        else:
            self.register_buffer("eps", torch.tensor([eps], dtype=torch.float))

    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = (1 + self.eps) * x + out
        return self.mlp(out)

    def message(self, x_j, edge_weight=None):
        msg = x_j
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

class GINNodeBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.conv1 = MaskableGINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm1d(hidden_channels)

        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.conv2 = MaskableGINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm1d(hidden_channels)

        mlp3 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.conv3 = MaskableGINConv(mlp3, train_eps=True)
        self.bn3 = BatchNorm1d(hidden_channels)

        mlp4 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.conv4 = MaskableGINConv(mlp4, train_eps=True)
        self.bn4 = BatchNorm1d(hidden_channels)

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.bn4(self.conv4(x, edge_index, edge_weight=edge_weight)))
        return self.classifier(x)

class GSATNodeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, hidden_channels: int, temperature: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        for module in self.att_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def get_mask(self, x, edge_index, training: bool = True):
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_logits = self.att_mlp(edge_rep).view(-1)

        if training:
            u = torch.rand_like(edge_logits)
            noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            mask = torch.sigmoid((edge_logits + noise) / self.temperature)
        else:
            mask = torch.sigmoid(edge_logits)

        return mask, edge_logits

    def forward(self, data, training: bool = True):
        x, edge_index = data.x, data.edge_index
        mask, mask_logits = self.get_mask(x.float(), edge_index, training=training)
        logits = self.backbone(x, edge_index, edge_weight=mask)
        return logits, mask, mask_logits

backbone = GINNodeBackbone(
    in_channels=dataset.num_features,
    hidden_channels=64,
    out_channels=2,
 )
model = GSATNodeClassifier(
    backbone=backbone,
    in_channels=dataset.num_features,
    hidden_channels=64,
    temperature=1.0,
 )
print(model)

train_data = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='train')
val_data = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='val')
test_data = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='test')

print(f"Number of training graphs: {len(train_data)}")
print(f"Number of validation graphs: {len(val_data)}")
print(f"Number of test graphs: {len(test_data)}")




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gsat_loss(
    pred_logits,
    ground_truth_labels,
    mask_logits,
    r: float = 0.7,
    pred_loss_coef: float = 1.0,
    info_loss_coef: float = 1.0,
 ):
    pred_loss = nn.CrossEntropyLoss()(pred_logits, ground_truth_labels)
    mask_probs = torch.sigmoid(mask_logits)
    prior_target = torch.full_like(mask_probs, 1.0 - r)
    info_loss = F.binary_cross_entropy(mask_probs, prior_target, reduction="mean")
    loss = (pred_loss_coef * pred_loss) + (info_loss_coef * info_loss)
    return loss, pred_loss, info_loss

@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_info_loss = 0.0
    total_correct = 0
    total_nodes = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        logits, _, mask_logits = model(data, training=False)
        y = data.y.view(-1).long()

        loss, pred_loss, info_loss = gsat_loss(logits, y, mask_logits)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * y.numel()
        total_pred_loss += pred_loss.item() * y.numel()
        total_info_loss += info_loss.item() * y.numel()
        total_correct += int((preds == y).sum().item())
        total_nodes += int(y.numel())

        all_preds.append(preds.detach().cpu())
        all_labels.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = total_correct / max(total_nodes, 1)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return {
        "loss": total_loss / max(total_nodes, 1),
        "pred_loss": total_pred_loss / max(total_nodes, 1),
        "info_loss": total_info_loss / max(total_nodes, 1),
        "acc": acc,
        "f1": f1,
    }

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_info_loss = 0.0
    total_correct = 0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, _, mask_logits = model(data, training=True)
        y = data.y.view(-1).long()

        loss, pred_loss, info_loss = gsat_loss(logits, y, mask_logits)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.detach().item() * y.numel()
        total_pred_loss += pred_loss.detach().item() * y.numel()
        total_info_loss += info_loss.detach().item() * y.numel()
        total_correct += int((preds == y).sum().item())
        total_nodes += int(y.numel())

    return {
        "loss": total_loss / max(total_nodes, 1),
        "pred_loss": total_pred_loss / max(total_nodes, 1),
        "info_loss": total_info_loss / max(total_nodes, 1),
        "acc": total_correct / max(total_nodes, 1),
    }

# --- GSAT + GIN training on PATTERN node classification ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default GSAT hyperparameters
temperature = 1.0
r = 0.7
pred_loss_coef = 1.0
info_loss_coef = 1.0

# Training defaults
lr_init = 1e-3
lr_factor = 0.5
min_lr = 1e-6
num_epochs = 100
batch_size = 32
early_stop_patience = 20
min_delta = 0.0
seeds = [0, 1, 2, 3, 4]

train_dataset = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='train')
val_dataset = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='val')
test_dataset = GNNBenchmarkDataset(root=data_dir, name='PATTERN', split='test')

print(f"Using device: {device}")
print(f"Running seeds: {seeds}")
print(
    f"GSAT defaults -> temperature={temperature}, r={r}, "
    f"pred_loss_coef={pred_loss_coef}, info_loss_coef={info_loss_coef}"
)

results_acc = []
results_f1 = []
results_loss = []

for seed in seeds:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    backbone = GINNodeBackbone(
        in_channels=train_dataset.num_features,
        hidden_channels=64,
        out_channels=2,
    )
    model = GSATNodeClassifier(
        backbone=backbone,
        in_channels=train_dataset.num_features,
        hidden_channels=64,
        temperature=temperature,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=10, min_lr=min_lr
    )

    best_val_loss = math.inf
    best_state = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, _, mask_logits = model(data, training=True)
            y = data.y.view(-1).long()

            loss, pred_loss, info_loss = gsat_loss(
                logits,
                y,
                mask_logits,
                r=r,
                pred_loss_coef=pred_loss_coef,
                info_loss_coef=info_loss_coef,
            )

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss += loss.detach().item() * y.numel()
            total_correct += int((preds == y).sum().item())
            total_nodes += int(y.numel())

        tr_loss = total_loss / max(total_nodes, 1)
        tr_acc = total_correct / max(total_nodes, 1)

        model.eval()
        with torch.no_grad():
            va_total_loss = 0.0
            va_total_correct = 0
            va_total_nodes = 0
            va_preds, va_labels = [], []

            for data in val_loader:
                data = data.to(device)
                logits, _, mask_logits = model(data, training=False)
                y = data.y.view(-1).long()

                va_loss, _, _ = gsat_loss(
                    logits,
                    y,
                    mask_logits,
                    r=r,
                    pred_loss_coef=pred_loss_coef,
                    info_loss_coef=info_loss_coef,
                )

                preds = logits.argmax(dim=-1)
                va_total_loss += va_loss.item() * y.numel()
                va_total_correct += int((preds == y).sum().item())
                va_total_nodes += int(y.numel())
                va_preds.append(preds.detach().cpu())
                va_labels.append(y.detach().cpu())

            va_loss = va_total_loss / max(va_total_nodes, 1)
            va_acc = va_total_correct / max(va_total_nodes, 1)
            va_f1 = f1_score(
                torch.cat(va_labels).numpy(),
                torch.cat(va_preds).numpy(),
                average='binary',
            )

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        scheduler.step(va_loss)

        if va_loss < (best_val_loss - min_delta):
            best_val_loss = va_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Tr loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"Va loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_loader(model, test_loader, device)
    print(
        f"Seed {seed} -> Test acc: {test_metrics['acc']:.4f} | "
        f"Test f1: {test_metrics['f1']:.4f} | Test loss: {test_metrics['loss']:.4f}"
    )

    results_acc.append(test_metrics['acc'])
    results_f1.append(test_metrics['f1'])
    results_loss.append(test_metrics['loss'])

print("\n" + "=" * 55)
print("FINAL RESULTS (GIN + GSAT on PATTERN Node Classification)")
print("=" * 55)
print(f"Test Accuracy: {np.mean(results_acc):.4f} ± {np.std(results_acc):.4f}")
print(f"Test F1 Score: {np.mean(results_f1):.4f} ± {np.std(results_f1):.4f}")
print(f"Test Loss:     {np.mean(results_loss):.4f} ± {np.std(results_loss):.4f}")
print("=" * 55)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('GSAT total loss')
plt.title('Last-seed Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train acc')
plt.plot(val_accs, label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Last-seed Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()