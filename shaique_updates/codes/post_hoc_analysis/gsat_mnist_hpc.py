import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.metrics import roc_auc_score


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "bcosgnn").is_dir():
            return path
    raise RuntimeError("Could not locate repo root (pyproject.toml + bcosgnn/).")


def augment_with_normalized_pos(dataset):
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


def load_saved_splits(repo_root: Path, batch_size: int, add_pos_features: bool = True):
    candidate_roots = [
        repo_root / "data" / "MNIST" / "sparsified_pt_splits",
        repo_root / "shaique_updates" / "codes" / "MNISTsp" / "data" / "MNIST" / "sparsified_pt_splits",
    ]

    split_root = None
    for root in candidate_roots:
        if (root / "train_sparsified.pt").exists() and (root / "val_sparsified.pt").exists() and (root / "test_sparsified.pt").exists():
            split_root = root
            break

    if split_root is None:
        raise FileNotFoundError("Could not find saved sparsified split files.")

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
        train_dataset = augment_with_normalized_pos(train_dataset)
        val_dataset = augment_with_normalized_pos(val_dataset)
        test_dataset = augment_with_normalized_pos(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_node_features = int(train_dataset[0].num_node_features)
    all_labels = torch.tensor([int(d.y.item()) for d in train_dataset + val_dataset + test_dataset])
    num_classes = int(all_labels.unique().numel())

    info = {
        "split_root": split_root,
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "num_test_graphs": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, test_dataset, info


class MaskableGINConv(MessagePassing):
    def __init__(self, nn_mlp, train_eps: bool = False):
        super().__init__(aggr="add")
        self.nn = nn_mlp
        self.initial_eps = 0.0
        if train_eps:
            self.initial_eps = nn.Parameter(torch.tensor([0.0]))

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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 4):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mask, mask_logits = self.get_mask(x, edge_index, training=training)
        pred_logits = self.backbone(x, edge_index, edge_weight=mask, batch=batch)
        return pred_logits, mask, mask_logits


def gsat_loss(pred_logits, labels, mask_logits, r: float = 0.7, pred_loss_coef: float = 1.0, info_loss_coef: float = 3.0):
    pred_loss = nn.CrossEntropyLoss()(pred_logits, labels)
    mask_probs = torch.sigmoid(mask_logits)
    prior_target = torch.full_like(mask_probs, 1.0 - r)
    info_loss = F.binary_cross_entropy(mask_probs, prior_target, reduction="mean")
    loss = (pred_loss_coef * pred_loss) + (info_loss_coef * info_loss)
    return loss


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_one_seed(
    seed: int,
    train_loader,
    val_loader,
    num_node_features: int,
    num_classes: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    early_stop_patience: int,
    r_prior: float,
    info_loss_coef: float,
    device: torch.device,
):
    set_all_seeds(seed)

    backbone = VanillaGINBackbone(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        num_layers=4,
    )
    model = GSAT(
        backbone=backbone,
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        temperature=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        model.temperature = 1.0 - (epoch / epochs) * (1.0 - 0.1)

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _, mask_logits = model(batch, training=True)
            loss = gsat_loss(
                logits,
                batch.y.view(-1).to(torch.long),
                mask_logits,
                r=r_prior,
                pred_loss_coef=1.0,
                info_loss_coef=info_loss_coef,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _, _ = model(batch, training=False)
                val_loss += criterion(logits, batch.y.view(-1).to(torch.long)).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_results_for_model(model, loader, dataset, device: torch.device):
    model.eval()
    node_aurocs = []
    node_jaccards = []
    all_gt_labels = []
    all_pred_scores = []
    correct_graphs = 0
    total_graphs = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, mask, _ = model(data, training=False)
            pred_classes = logits.argmax(dim=1)
            correct_graphs += (pred_classes == data.y).sum().item()
            total_graphs += data.num_graphs

            if hasattr(data, 'node_mask') or hasattr(data, 'explanation_mask'):
                gt_mask = data.node_mask if hasattr(data, 'node_mask') else data.explanation_mask
                row, col = data.edge_index
                edge_gt = gt_mask[row].bool() & gt_mask[col].bool()
                all_gt_labels.extend(edge_gt.cpu().numpy())
                all_pred_scores.extend(mask.cpu().numpy())

    for data in dataset:
        data = data.to(device)
        if hasattr(data, 'node_mask'):
            gt_mask = data.node_mask.detach().cpu().numpy().astype(int)
        elif hasattr(data, 'explanation_mask'):
            gt_mask = data.explanation_mask.detach().cpu().numpy().astype(int)
        else:
            continue

        with torch.no_grad():
            _, mask, _ = model(data, training=False)

        row, col = data.edge_index
        mask_np = mask.cpu().numpy()
        node_scores = np.zeros(data.num_nodes)

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
    node_auc = float(np.mean(node_aurocs)) if node_aurocs else float('nan')
    node_jaccard = float(np.mean(node_jaccards)) if node_jaccards else float('nan')

    return acc, global_edge_auc, node_auc, node_jaccard


def benchmark_full_test_timing(model, test_loader, test_dataset, device: torch.device):
    model.eval()
    num_graphs = len(test_dataset)

    sync_if_cuda()
    start_process = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _ = model(batch, training=False)
    sync_if_cuda()
    process_total_s = time.perf_counter() - start_process

    sync_if_cuda()
    start_explain = time.perf_counter()
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            _, edge_mask, _ = model(data, training=False)
            _ = edge_mask
    sync_if_cuda()
    explain_total_s = time.perf_counter() - start_explain

    return {
        "process_total_s": process_total_s,
        "explain_total_s": explain_total_s,
        "process_ms_per_graph": (process_total_s * 1000.0) / max(num_graphs, 1),
        "explain_ms_per_graph": (explain_total_s * 1000.0) / max(num_graphs, 1),
        "num_graphs": num_graphs,
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    repo_root = find_repo_root(Path.cwd())

    train_loader, val_loader, test_loader, test_dataset, info = load_saved_splits(
        repo_root=repo_root,
        batch_size=args.batch_size,
        add_pos_features=not args.disable_pos_features,
    )

    print(f"Device: {device}")
    print(f"Split dir: {info['split_root']}")
    print(f"Test graphs: {info['num_test_graphs']}")

    rows = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        model = train_one_seed(
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            num_node_features=info["num_node_features"],
            num_classes=info["num_classes"],
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            r_prior=args.r_prior,
            info_loss_coef=args.info_loss_coef,
            device=device,
        )

        t = benchmark_full_test_timing(model, test_loader, test_dataset, device=device)
        test_acc, test_global_edge_auc, test_node_auc, test_jaccard = evaluate_results_for_model(model, test_loader, test_dataset, device)
        row = {"seed": seed, "test_acc": test_acc, "global_edge_auc": test_global_edge_auc, "node_auroc": test_node_auc, "node_jaccard": test_jaccard, **t}
        rows.append(row)

        print(
            f"Acc: {test_acc:.4f} | Edge AUC: {test_global_edge_auc:.4f} | Node AUROC: {test_node_auc:.4f} | Node Jaccard: {test_jaccard:.4f}\n"
            f"Process total: {row['process_total_s']:.3f}s ({row['process_ms_per_graph']:.3f} ms/graph) | "
            f"Explain total: {row['explain_total_s']:.3f}s ({row['explain_ms_per_graph']:.3f} ms/graph)"
        )

    print("\n" + "=" * 78)
    print("FULL TEST-SET SUMMARY ACROSS SEEDS")
    print("=" * 78)

    metric_names = [
        "test_acc",
        "global_edge_auc",
        "node_auroc",
        "node_jaccard",
        "process_total_s",
        "explain_total_s",
        "process_ms_per_graph",
        "explain_ms_per_graph",
    ]

    for name in metric_names:
        vals = np.array([r[name] for r in rows], dtype=float)
        print(f"{name:22s}: {vals.mean():.4f} ± {vals.std():.4f}")

    total_process = float(np.sum([r["process_total_s"] for r in rows]))
    total_explain = float(np.sum([r["explain_total_s"] for r in rows]))
    total_graphs = int(np.sum([r["num_graphs"] for r in rows]))

    print("-" * 78)
    print(f"Total process time over all seeds : {total_process:.4f} s")
    print(f"Total explain time over all seeds : {total_explain:.4f} s")
    print(f"Total graphs across all seeds    : {total_graphs}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "seed",
                    "num_graphs",
                    "test_acc",
                    "global_edge_auc",
                    "node_auroc",
                    "node_jaccard",
                    "process_total_s",
                    "explain_total_s",
                    "process_ms_per_graph",
                    "explain_ms_per_graph",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-seed timing CSV: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="GSAT MNIST multi-seed timing benchmark for HPC.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r-prior", type=float, default=0.7)
    parser.add_argument("--info-loss-coef", type=float, default=3.0)
    parser.add_argument("--disable-pos-features", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--out-csv", type=str, default="", help="Optional path to save per-seed timing CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
