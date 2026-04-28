import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# GSAT Explainer for MNIST (Vanilla GIN)"),
    nbf.v4.new_code_cell("""import os
import sys
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.metrics import roc_auc_score

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
print("Repo root added:", repo_root)"""),

    nbf.v4.new_code_cell("""def _augment_with_normalized_pos(dataset):
    augmented = []
    for data in dataset:
        d = data.clone()
        pos = d.pos.float()
        pos_min = pos.min(dim=0).values
        pos_max = pos.max(dim=0).values
        denom = (pos_max - pos_min).clamp(min=1e-8)
        pos_norm = (pos - pos_min) / denom
        d.x = torch.cat([d.x.float(), pos_norm], dim=-1)
        augmented.append(d)
    return augmented

def load_and_split_data(batch_size=64, add_pos_features=True):
    print("Loading preprocessed sparsified .pt splits...")
    candidate_roots = [
        repo_root / 'data' / 'MNIST' / 'sparsified_pt_splits',
        repo_root / 'shaique_updates' / 'codes' / 'MNISTsp' / 'data' / 'MNIST' / 'sparsified_pt_splits',
    ]
    split_root = None
    for root in candidate_roots:
        if (root / 'train_sparsified.pt').exists() and (root / 'val_sparsified.pt').exists() and (root / 'test_sparsified.pt').exists():
            split_root = root
            break
    if split_root is None:
        raise FileNotFoundError("Could not find saved sparsified split files.")

    train_dataset = torch.load(split_root / 'train_sparsified.pt', map_location='cpu', weights_only=False)
    val_dataset = torch.load(split_root / 'val_sparsified.pt', map_location='cpu', weights_only=False)
    test_dataset = torch.load(split_root / 'test_sparsified.pt', map_location='cpu', weights_only=False)

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
    }

    print(f"Using split directory: {split_root}")
    print(f"Loaded splits. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Avg train edges: {avg_train_edges:.2f}")
    print(f"Node features: {num_node_features} (pos features added: {add_pos_features})")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, dataset_info

BATCH_SIZE = 128
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, dataset_info = load_and_split_data(batch_size=BATCH_SIZE)"""),

    nbf.v4.new_code_cell("""class MaskableGINConv(MessagePassing):
    def __init__(self, nn_mlp, train_eps=False):
        super().__init__(aggr='add')
        self.nn = nn_mlp
        self.initial_eps = 0.0
        if train_eps:
            self.initial_eps = nn.Parameter(torch.Tensor([0.0]))
            
    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_r = x[1] if isinstance(x, tuple) else x
        out = out + (1 + self.initial_eps) * x_r
        return self.nn(out)

    def message(self, x_j, edge_weight):
        msg = x_j
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

class VanillaGINBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels)
            )
            self.convs.append(MaskableGINConv(nn_mlp=mlp, train_eps=True))
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.node_emb(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return x

class GSAT(nn.Module):
    def __init__(self, backbone, in_channels, hidden_channels, temperature=1.0):
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
        for m in self.att_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def get_mask(self, x, edge_index, training=True):
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

    def forward(self, data, training=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mask, mask_logits = self.get_mask(x, edge_index, training=training)
        pred_logits = self.backbone(x, edge_index, edge_weight=mask, batch=batch)
        return pred_logits, mask, mask_logits"""),

    nbf.v4.new_code_cell("""def gsat_loss(pred_logits, ground_truth_labels, mask_logits, r=0.7, pred_loss_coef=1.0, info_loss_coef=1.0):
    criterion = nn.CrossEntropyLoss()
    pred_loss = criterion(pred_logits, ground_truth_labels)
    mask_probs = torch.sigmoid(mask_logits)
    prior_target = torch.full_like(mask_probs, 1.0 - r)
    info_loss = F.binary_cross_entropy(mask_probs, prior_target, reduction="mean")
    loss = (pred_loss_coef * pred_loss) + (info_loss_coef * info_loss)
    return loss, pred_loss, info_loss"""),

    nbf.v4.new_code_cell("""HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 25
R_PRIOR = 0.7
INFO_LOSS_COEF = 3.0

backbone = VanillaGINBackbone(
    in_channels=dataset_info['num_node_features'],
    hidden_channels=HIDDEN_DIM,
    out_channels=dataset_info['num_classes'],
    num_layers=4
)

model = GSAT(
    backbone=backbone,
    in_channels=dataset_info['num_node_features'],
    hidden_channels=HIDDEN_DIM,
    temperature=1.0,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

print("Starting Training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    model.temperature = 1.0 - (epoch / EPOCHS) * (1.0 - 0.1)
    
    total_loss, total_pred, total_info = 0.0, 0.0, 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits, mask, mask_logits = model(batch, training=True)
        loss, pred_l, info_l = gsat_loss(
            logits, batch.y.view(-1).to(torch.long), mask_logits,
            r=R_PRIOR, pred_loss_coef=1.0, info_loss_coef=INFO_LOSS_COEF
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_pred += pred_l.item()
        total_info += info_l.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits, _, _ = model(batch, training=False)
            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(logits, batch.y.view(-1).to(torch.long)).item()
            
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Tau {model.temperature:.2f} | Loss {total_loss/len(train_loader):.4f} | Pred {total_pred/len(train_loader):.4f} | Info {total_info/len(train_loader):.4f} | Val {val_loss/len(val_loader):.4f}")
        
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

if best_state is not None:
    model.load_state_dict(best_state)
print("Training done.")"""),

    nbf.v4.new_code_cell("""def evaluate_results(loader, dataset):
    model.eval()
    node_aurocs = []
    node_jaccards = []
    
    all_gt_labels = []
    all_pred_scores = []
    
    correct_graphs = 0
    total_graphs = 0
    num_with_gt = 0
    num_with_gt_positive = 0
    
    # 1. Global Accuracy
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            logits, mask, _ = model(data, training=False)
            pred_classes = logits.argmax(dim=1)
            correct_graphs += (pred_classes == data.y).sum().item()
            total_graphs += data.num_graphs
            
            # 2. Global Edge AUC (Inspired by GSAT_di_halo_benzene.ipynb)
            if hasattr(data, "node_mask") or hasattr(data, "explanation_mask"):
                gt_mask = data.node_mask if hasattr(data, "node_mask") else data.explanation_mask
                row, col = data.edge_index
                edge_gt = gt_mask[row].bool() & gt_mask[col].bool()
                all_gt_labels.extend(edge_gt.cpu().numpy())
                all_pred_scores.extend(mask.cpu().numpy())

    # 3. Graph-level Node AUROC & Jaccard (Inspired by GSAT_BA2Motif.ipynb)
    for data in dataset:
        data = data.to(DEVICE)
        if hasattr(data, "node_mask"):
            gt_mask = data.node_mask.detach().cpu().numpy().astype(int)
        elif hasattr(data, "explanation_mask"):
            gt_mask = data.explanation_mask.detach().cpu().numpy().astype(int)
        else:
            continue
            
        num_with_gt += 1
        if gt_mask.sum() > 0:
            num_with_gt_positive += 1
            
        with torch.no_grad():
            _, mask, _ = model(data, training=False)
            
        # Convert edge mask to node scores
        row, col = data.edge_index
        mask_np = mask.cpu().numpy()
        node_scores = np.zeros(data.num_nodes)
        
        # Max of incident edge weights for each node
        for i in range(len(row)):
            u, v = row[i], col[i]
            node_scores[u] = max(node_scores[u], mask_np[i])
            node_scores[v] = max(node_scores[v], mask_np[i])
            
        if node_scores.max() > node_scores.min():
            node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min())
            
        if len(np.unique(gt_mask)) > 1:
            node_aurocs.append(roc_auc_score(gt_mask, node_scores))
            
        k = int(gt_mask.sum())
        if k > 0:
            top_k_indices = np.argsort(node_scores)[-k:]
            pred_binary = np.zeros_like(node_scores)
            pred_binary[top_k_indices] = 1
            intersect = (pred_binary * gt_mask).sum()
            union = (pred_binary + gt_mask).clip(0, 1).sum()
            node_jaccards.append(intersect / (union + 1e-8))
            
    acc = correct_graphs / max(total_graphs, 1)
    global_edge_auc = roc_auc_score(all_gt_labels, all_pred_scores) if len(np.unique(all_gt_labels)) > 1 else float('nan')
    node_auc = float(np.mean(node_aurocs)) if node_aurocs else float("nan")
    node_jaccard = float(np.mean(node_jaccards)) if node_jaccards else float("nan")
    
    return acc, global_edge_auc, node_auc, node_jaccard

test_acc, test_global_edge_auc, test_node_auc, test_jaccard = evaluate_results(test_loader, test_dataset)
print("=" * 60)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Global Edge AUC: {test_global_edge_auc:.4f}")
print(f"Node AUROC:   {test_node_auc:.4f}")
print(f"Node Jaccard: {test_jaccard:.4f}")
print("-" * 60)"""),

    nbf.v4.new_code_cell("""def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_gsat_explain_per_graph(model, dataset, warmup: int = 2, max_graphs=None):
    model.eval()
    times_ms = []
    graphs = dataset[:max_graphs] if max_graphs is not None else dataset
    
    with torch.no_grad():
        for data in graphs[:warmup]:
            data = data.to(DEVICE)
            _ = model(data, training=False)
            
        for data in graphs:
            data = data.to(DEVICE)
            _sync_if_cuda()
            start = time.perf_counter()
            _ = model(data, training=False)
            _sync_if_cuda()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)
        
    times_ms = np.asarray(times_ms, dtype=float)
    mean_ms = float(times_ms.mean()) if times_ms.size else float('nan')
    std_ms = float(times_ms.std()) if times_ms.size else float('nan')
    median_ms = float(np.median(times_ms)) if times_ms.size else float('nan')
    p90_ms = float(np.percentile(times_ms, 90)) if times_ms.size else float('nan')
    graphs_per_s = float(1000.0 / mean_ms) if mean_ms > 0 else float('nan')
    
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "graphs_per_s": graphs_per_s,
    }

timing_stats = time_gsat_explain_per_graph(model, test_dataset)
print(f"Timing (ms/graph): mean={timing_stats['mean_ms']:.2f}, median={timing_stats['median_ms']:.2f}, p90={timing_stats['p90_ms']:.2f} | throughput={timing_stats['graphs_per_s']:.2f} graphs/s")""")
]

with open('/Users/shaique/Desktop/BioInf_IMP/NMM_group/ICML/bcos_gnn/bcosgnn/shaique_updates/codes/post_hoc_analysis/GSAT_MNIST.ipynb', 'w') as f:
    nbf.write(nb, f)
