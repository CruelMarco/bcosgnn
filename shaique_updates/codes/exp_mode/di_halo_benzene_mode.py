import sys
import os
from pathlib import Path
import wandb

def find_repo_root(*starts: Path) -> Path:
    checked = set()
    for start in starts:
        if start is None:
            continue
        for p in [start.resolve(), *start.resolve().parents]:
            if p in checked:
                continue
            checked.add(p)
            if (p / "pyproject.toml").exists() and (p / "bcosgnn").is_dir():
                return p

    env_repo_root = os.environ.get("BCOSGNN_REPO_ROOT")
    if env_repo_root:
        env_repo_root = Path(env_repo_root).expanduser().resolve()
        if (env_repo_root / "pyproject.toml").exists() and (env_repo_root / "bcosgnn").is_dir():
            return env_repo_root

    raise RuntimeError(
        "Could not locate repo root. Checked CWD/script parents; set BCOSGNN_REPO_ROOT if needed."
    )

def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

repo_root = find_repo_root(Path.cwd(), _script_dir())
project_root = repo_root
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
print("Repo root added:", repo_root)

import random
import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score as sk_auroc

try:
    from IPython.display import display as _display
except Exception:
    def _display(obj):
        print(obj)



from bcosgnn.explain_edge_attr import explain as explain_edge_attr
from bcosgnn.evaluation import (
    evaluate_auroc_edge,
    evaluate_jaccard_edge,
    evaluate_gnnexplainer_jaccard_edge,
    evaluate_gnnexplainer_auroc_edge,
)
from bcosgnn.sanitized_models import BCosGINE, BcosGINEConv, ReadoutThenAgg

try:
    from torch_geometric.explain import CaptumExplainer
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("WARNING: CaptumExplainer not available. IG will be skipped.")


class ZincProcessedDataset(InMemoryDataset):
    """Thin wrapper around a pre-processed ZINC InMemoryDataset stored in data.pt."""

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        data_path = Path(self.processed_dir) / "data.pt"
        print(f"Loading data from: {data_path.resolve()}")
        try:
            self.data, self.slices = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):  return []
    @property
    def processed_file_names(self): return ['data.pt']
    def download(self): pass
    def process(self):  pass


# ── Path resolution ───────────────────────────────────────────────────────────
CANDIDATE_PATHS = [
    project_root / "shaique_updates/codes/multi_class_Zinc/zinc_di_halo_benzene_data"
]
DATASET_PATH = next((str(p) for p in CANDIDATE_PATHS if p.exists()), None)
if DATASET_PATH is None:
    raise FileNotFoundError(
        "Cannot find zinc_di_halo_benzene_data. Set DATASET_PATH manually."
    )

dataset = ZincProcessedDataset(root=DATASET_PATH)
print(f"\nDataset: {len(dataset)} graphs")
print(f"Node features : {dataset.num_node_features}")
print(f"Edge features : {dataset.num_edge_features}")
print(f"Classes       : {dataset.num_classes}")

# ── Quick stats ───────────────────────────────────────────────────────────────
n        = len(dataset)
avg_nodes = sum(dataset[i].num_nodes for i in range(n)) / n
avg_edges = sum(dataset[i].num_edges for i in range(n)) / n
y_counts  = {}
for i in range(n):
    y = int(dataset[i].y.item())
    y_counts[y] = y_counts.get(y, 0) + 1

print(f"\nAvg nodes/graph : {avg_nodes:.1f}")
print(f"Avg edges/graph : {avg_edges:.1f}")
print("Class distribution:")
CLASS_LABELS = [
    "di_chloro_ortho", "di_chloro_meta", "di_chloro_para",
    "di_fluoro_ortho", "di_fluoro_meta", "di_fluoro_para",
    "di_bromo_ortho",  "di_bromo_meta",  "di_bromo_para",
]
for c in sorted(y_counts):
    label = CLASS_LABELS[c] if c < len(CLASS_LABELS) else str(c)
    print(f"  Class {c} ({label}): {y_counts[c]} graphs")



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── Reproducibility helpers ───────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val_test(dataset, seed, test_size=0.1, val_size=0.1):
    """Stratified train / val / test split.  Returns index arrays.

    Defaults: test=10% (900 graphs = 100/class), val=10% (900 graphs),
    train=80% (7200 graphs) — matching bcos_gine_di_halo.ipynb test_size=900.
    """
    labels = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    tv_labels = labels[train_val_idx]
    val_ratio = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=seed, stratify=tv_labels
    )
    return train_idx, val_idx, test_idx


def make_loader(indices, dataset, batch_size=64, shuffle=False):
    subset = dataset.index_select(torch.tensor(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = total_graphs = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y.view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=-1)
        total_loss    += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.view(-1).long()).sum().item())
        total_graphs  += int(batch.num_graphs)
    return total_loss / total_graphs, total_correct / total_graphs


@torch.inference_mode()
def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total_graphs = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y.view(-1).long())
        pred   = logits.argmax(dim=-1)
        total_loss    += float(loss.item()) * batch.num_graphs
        total_correct += int((pred == batch.y.view(-1).long()).sum().item())
        total_graphs  += int(batch.num_graphs)
    return {"loss": total_loss / total_graphs, "acc": total_correct / total_graphs}


class EvalModelAdapter(torch.nn.Module):
    """Wraps BCosGINE so explain_edge_attr and PyG Explainer can call it uniformly.

    Converts the strict positional signature ``forward(x, edge_index, edge_attr, batch)``
    into ``forward(x, edge_index, edge_attr=None, batch=None)`` with auto-batch for
    single graphs.  No logic is changed — edge_attr is always passed through.
    """
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.base_model(x, edge_index, edge_attr, batch)


# ── Hyperparameters ───────────────────────────────────────────────────────────
node_size   = dataset.num_node_features
edge_size   = dataset.num_edge_features
num_classes = dataset.num_classes

SEEDS              = [0, 1, 2]
EPOCHS             = 200
BATCH_SIZE         = 64
LR                 = 1e-3
hidden_dim         = 128
b                  = 2.0
EARLY_STOP_PATIENCE = 25   # stop if val loss doesn't improve by MIN_DELTA for this many epochs
MIN_DELTA           = 1e-4  # minimum improvement to count as "better"

seed_results      = []
best_result       = None
best_result_score = -float("inf")

# ── Multi-seed training loop ──────────────────────────────────────────────────
for seed in SEEDS:
    print("=" * 80)
    print(f"Seed {seed}")
    print("=" * 80)
    set_seed(seed)

    train_idx, val_idx, test_idx = split_train_val_test(dataset, seed=seed)
    train_loader = make_loader(train_idx, dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_idx,   dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Build list for per-graph explanation evaluation (avoid DataLoader overhead)
    test_data = [dataset[int(i)] for i in test_idx]

    print(f"  Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    readout = ReadoutThenAgg(
        in_channels=hidden_dim,
        hidden_channels=[hidden_dim],   # hidden_dim → hidden_dim → num_classes
        out_channels=num_classes,
        b=b,
        max_out=1,
        agg="sum",
    )

    model = BCosGINE(
        node_size=node_size,
        edge_size=edge_size,
        hidden_channels=[hidden_dim, hidden_dim],   # conv MLP: hidden_dim → hidden_dim
        num_convs=4,
        readout=readout,
        b=b,
        max_out=1,
    ).to(DEVICE)

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_state        = None
    best_val_loss     = float("inf")
    best_epoch        = -1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics     = evaluate_loader(model, val_loader,   criterion)
        scheduler.step(val_metrics["loss"])

        if (best_val_loss - val_metrics["loss"]) > MIN_DELTA:
            best_val_loss     = val_metrics["loss"]
            best_epoch        = epoch
            best_state        = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    eval_model = EvalModelAdapter(model).to(DEVICE)

    test_loader     = make_loader(test_idx, dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_metrics    = evaluate_loader(model, test_loader, criterion)
    test_jaccard    = float(evaluate_jaccard_edge(eval_model, test_data))
    test_expl_auroc = float(evaluate_auroc_edge(eval_model,   test_data))

    result = {
        "seed":            seed,
        "best_epoch":      best_epoch,
        "best_val_loss":   float(best_val_loss),
        "test_loss":       float(test_metrics["loss"]),
        "test_acc":        float(test_metrics["acc"]),
        "test_expl_auroc": test_expl_auroc,
        "test_jaccard":    test_jaccard,
        "model":           model,
        "eval_model":      eval_model,
        "test_dataset":    test_data,
    }
    seed_results.append(result)
    print(
        f"Seed {seed} | best_epoch={best_epoch} | test_acc={result['test_acc']:.4f} "
        f"| test_expl_auroc={result['test_expl_auroc']:.4f} "
        f"| test_jaccard={result['test_jaccard']:.4f}"
    )

    rank_score = (
        np.nan_to_num(test_expl_auroc, nan=-1e9)
        + np.nan_to_num(test_jaccard,  nan=-1e9)
    )
    if best_result is None or rank_score > best_result_score:
        best_result_score = rank_score
        best_result = result

# ── Summary ───────────────────────────────────────────────────────────────────
metrics_keys = ["test_loss", "test_acc", "test_expl_auroc", "test_jaccard"]
summary = {}
for key in metrics_keys:
    vals = np.array([r[key] for r in seed_results], dtype=float)
    summary[f"{key}_mean"] = float(np.nanmean(vals))
    summary[f"{key}_std"]  = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

df_seed_results = pl.DataFrame([
    {k: v for k, v in r.items() if k in ["seed", "best_epoch", "best_val_loss", *metrics_keys]}
    for r in seed_results
])
df_summary = pl.DataFrame([summary])

print("\nPer-seed test metrics:")
_display(df_seed_results)
print("\nMean ± std across seeds:")
_display(df_summary)

if best_result is None:
    raise RuntimeError("No valid seed result produced. Check training logs.")

# Expose best-seed model for downstream cells
model        = best_result["model"]
eval_model   = best_result["eval_model"]
test_dataset = best_result["test_dataset"]
print("\nBest seed:", best_result["seed"])



######### Post Hoc Explainer Comparision ###########


# ── Post-hoc explainer setup ──────────────────────────────────────────────────




def make_gnn_explainer(model, gnn_epochs: int = 200, gnn_lr: float = 0.01):
    """GNNExplainer for a multi-class GINE.

    ``edge_mask_type='object'`` is set because our model uses edge_attr —
    PyG will also produce an edge mask alongside the node mask.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=gnn_epochs, lr=gnn_lr),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',        # ← edge_attr is used by GINE
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )


def make_ig_explainer(model):
    """IG explainer for a multi-class GINE.

    Uses ``node_mask_type='attributes'`` so the node mask has shape
    ``[num_nodes, num_features]`` — required by Captum's path-integral.
    The B-COS model outputs ``[1, num_classes]``  directly (no wrapper needed).
    """
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("Install captum (pip install captum) to use IG.")
    return Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs',
        ),
    )


def evaluate_ig_on_gine(ig_explainer, model, dataset, transform=None,
                        gt_attr: str = "explanation_mask"):
    """Jaccard@|GT| and Node AUROC for IG on the multi-class GINE.

    Node scores = abs(node_mask).sum(dim=1) — same approach as IG_BA2Motif.ipynb.
    """
    jaccards, aurocs = [], []
    device = next(model.parameters()).device

    for data in tqdm(dataset, desc="Evaluating IG (post-hoc)", leave=True):
        data_t  = transform(data.clone()) if transform else data.clone()
        data_t  = data_t.to(device)
        gt      = getattr(data_t, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        k       = int(gt_mask.sum())
        batch   = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            logits    = model(data_t.x, data_t.edge_index, data_t.edge_attr, batch)
            pred_cls  = int(logits.argmax(dim=-1).item())

        try:
            expl   = ig_explainer(
                x=data_t.x, edge_index=data_t.edge_index,
                edge_attr=data_t.edge_attr, batch=batch, target=pred_cls
            )
            scores = expl.node_mask.abs().sum(dim=1).detach().cpu().numpy()

            if gt_mask.min() != gt_mask.max():
                aurocs.append(sk_auroc(gt_mask, scores))

            if k > 0:
                top_k    = set(np.argsort(scores)[-k:].tolist())
                gt_set   = set(np.where(gt_mask)[0].tolist())
                inter    = len(gt_set & top_k)
                union    = len(gt_set | top_k)
                jaccards.append(inter / union if union > 0 else 0.0)
        except Exception:
            pass

    return (
        float(np.mean(jaccards)) if jaccards else float('nan'),
        float(np.mean(aurocs))   if aurocs   else float('nan'),
    )


print("Post-hoc GINE explainer utilities ready.")
print(f"  Captum available: {CAPTUM_AVAILABLE}")



# ── Per-seed post-hoc evaluation ─────────────────────────────────────────────
posthoc_results = []

for result in seed_results:
    seed_id = result["seed"]
    ev_m    = result["eval_model"].to(DEVICE)
    t_d     = result["test_dataset"]

    print(f"\n{'='*60}")
    print(f"Seed {seed_id}: post-hoc explainers on B-COS GINE")
    print(f"{'='*60}")

    ev_m.eval()

    # ── GNNExplainer ──────────────────────────────────────────────────────
    gnn_exp   = make_gnn_explainer(ev_m, gnn_epochs=200, gnn_lr=0.01)
    gnn_jacc  = evaluate_gnnexplainer_jaccard_edge(gnn_exp, ev_m, t_d)
    gnn_auroc = evaluate_gnnexplainer_auroc_edge(gnn_exp,   ev_m, t_d)
    print(f"  GNNExplainer  →  Jaccard: {gnn_jacc:.4f}  |  AUROC: {gnn_auroc:.4f}")

    # ── Integrated Gradients ──────────────────────────────────────────────
    if CAPTUM_AVAILABLE:
        ig_exp = make_ig_explainer(ev_m)
        ig_jacc, ig_auroc = evaluate_ig_on_gine(ig_exp, ev_m, t_d)
        print(f"  IG (Captum)   →  Jaccard: {ig_jacc:.4f}  |  AUROC: {ig_auroc:.4f}")
    else:
        ig_jacc = ig_auroc = float('nan')
        print("  IG (Captum)   →  SKIPPED (captum not installed)")

    posthoc_results.append({
        "seed":        seed_id,
        "gnn_jaccard": float(gnn_jacc),
        "gnn_auroc":   float(gnn_auroc),
        "ig_jaccard":  float(ig_jacc),
        "ig_auroc":    float(ig_auroc),
    })

print("\nPost-hoc evaluation complete.")




# ── Comparison summary table ──────────────────────────────────────────────────
def _ms(vals):
    """Mean ± std string, ignoring NaN values."""
    vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return "N/A"
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return f"{vals.mean():.4f} ± {std:.4f}"


bcos_jacc  = [r["test_jaccard"]    for r in seed_results]
bcos_auroc = [r["test_expl_auroc"] for r in seed_results]
gnn_jacc   = [r["gnn_jaccard"]     for r in posthoc_results]
gnn_auroc  = [r["gnn_auroc"]       for r in posthoc_results]
ig_jacc    = [r["ig_jaccard"]      for r in posthoc_results]
ig_auroc   = [r["ig_auroc"]        for r in posthoc_results]

comparison_df = pl.DataFrame([
    {
        "Method":       "B-COS Intrinsic",
        "Model":        "B-COS GINE",
        "Explainer":    "Linear decomposition (explain_edge_attr.py)",
        "Jaccard@|GT|": _ms(bcos_jacc),
        "Node AUROC":   _ms(bcos_auroc),
    },
    {
        "Method":       "GNNExplainer (post-hoc)",
        "Model":        "B-COS GINE",
        "Explainer":    "PyG GNNExplainer (200 epochs)",
        "Jaccard@|GT|": _ms(gnn_jacc),
        "Node AUROC":   _ms(gnn_auroc),
    },
    {
        "Method":       "IG (post-hoc)",
        "Model":        "B-COS GINE",
        "Explainer":    "Captum IntegratedGradients",
        "Jaccard@|GT|": _ms(ig_jacc),
        "Node AUROC":   _ms(ig_auroc),
    },
])

print("=" * 75)
print("Explanation Quality Comparison — Di-Halo Benzene (B-COS GINE Model)")
print(f"Across {len(seed_results)} seeds, mean ± std")
print("=" * 75)
_display(comparison_df)
