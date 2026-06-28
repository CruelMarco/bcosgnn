import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from bcosgnn.explain import explain

try:
    from torch_geometric.explain import Explainer as PyGExplainer
except Exception:  # pragma: no cover
    PyGExplainer = None


def _reduce_node_mask_to_scores(node_mask: np.ndarray) -> np.ndarray:
    """Normalize PyG Explainer node masks into a flat per-node score vector."""
    # node_mask can be [num_nodes], [num_nodes, 1], or [num_nodes, F]
    if node_mask.ndim == 2:
        node_mask = node_mask.mean(axis=1)
    return node_mask.reshape(-1)


@torch.no_grad()
def _predict_class(model, data, threshold: float = 0.0) -> int:
    model.eval()
    batch = getattr(data, 'batch', None)
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    out = model(data.x, data.edge_index, data.edge_attr, batch)
    out = out.view(out.shape[0], -1) if out.dim() == 1 else out

    # Binary (single logit) vs multi-class logits
    if out.shape[-1] == 1:
        return 1 if float(out.view(-1)[0].item()) > threshold else 0
    return int(out.argmax(dim=-1).view(-1)[0].item())


def get_gnnexplainer_scores(
    explainer,
    model,
    data,
    *,
    target: int | None = None,
):
    """Run a (PyG) GNNExplainer-style explainer and return per-node scores.

    Parameters
    ----------
    explainer:
        A `torch_geometric.explain.Explainer` instance.
    model:
        The trained GNN.
    data:
        A `torch_geometric.data.Data` graph.
    target:
        Target class index to explain. If None, uses the model prediction.
    """
    if PyGExplainer is None:
        raise RuntimeError(
            "torch_geometric.explain is unavailable; please install a recent PyG version.",
        )
    if not isinstance(explainer, PyGExplainer):
        raise TypeError(
            "Expected explainer to be torch_geometric.explain.Explainer instance.",
        )

    data = data.clone()
    device = next(model.parameters()).device
    data = data.to(device)

    batch = getattr(data, 'batch', None)
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    if target is None:
        target = _predict_class(model, data)

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=getattr(data, 'edge_attr', None),
        batch=batch,
        target=target,
    )

    if explanation.node_mask is None:
        raise RuntimeError("Explainer returned no node_mask.")

    node_scores = _reduce_node_mask_to_scores(
        explanation.node_mask.detach().cpu().numpy(),
    )
    return node_scores, int(target)

def get_attribution_scores(model, data, method='explain', batch=None, threshold=0.0):

    if method != 'explain':
        raise ValueError(
            f"Only method='explain' is supported for B-COS attribution, got: {method}",
        )

    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, batch)
    pred_class = 1 if out.item() > threshold else 0

    node_contrib = explain(model, data.x, data.edge_index, data.edge_attr, batch).detach()
    scores = node_contrib.sum(1)

    # Apply sign flipping for Class 0
    if pred_class == 0:
        scores = -scores
        
    return scores.cpu().numpy(), pred_class

def evaluate_jaccard(model, dataset, method='explain', transform=None):

    jaccard_scores = []
    
    for data in tqdm(dataset, desc="Evaluating Jaccard (explain)"):
        if transform:
            data_transformed = transform(data.clone())
        else:
            data_transformed = data.clone()
            
        # Ground Truth
        gt_nodes = set(torch.where(data_transformed.node_mask.squeeze() == 1)[0].numpy())
        k = len(gt_nodes)
        
        if k == 0:
            continue
            
        try:
            scores, _ = get_attribution_scores(model, data_transformed, method=method)
            
            # Top-k
            # scores is numpy array
            # We want indices of top-k scores. 
            # np.argsort returns indices that sort the array. 
            # We want descending order.
            top_k_indices = np.argsort(scores)[-k:]
            pred_nodes = set(top_k_indices)
            
            intersection = len(gt_nodes.intersection(pred_nodes))
            union = len(gt_nodes.union(pred_nodes))
            
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
            
        except Exception as e:
            # print(f"Error processing graph: {e}")
            jaccard_scores.append(0.0)
            
    return np.mean(jaccard_scores)

def evaluate_auroc(model, dataset, method='explain', transform=None):

    auroc_scores = []
    
    for data in tqdm(dataset, desc="Evaluating AUROC (explain)"):
        if transform:
            data_transformed = transform(data.clone())
        else:
            data_transformed = data.clone()
            
        gt_mask = data_transformed.node_mask.squeeze().numpy()
        
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue
            
        try:
            scores, _ = get_attribution_scores(model, data_transformed, method=method)
            
            auroc = roc_auc_score(gt_mask, scores)
            auroc_scores.append(auroc)
        except Exception as e:
            # print(f"Error processing graph: {e}")
            pass
            
    return np.mean(auroc_scores)


def evaluate_gnnexplainer_jaccard(
    explainer,
    model,
    dataset,
    *,
    transform=None,
):
    jaccard_scores = []

    for data in tqdm(dataset, desc="Evaluating Jaccard (gnnexplainer)"):
        if transform:
            data_transformed = transform(data.clone())
        else:
            data_transformed = data.clone()

        if not hasattr(data_transformed, 'node_mask'):
            continue

        gt_nodes = set(torch.where(data_transformed.node_mask.squeeze() == 1)[0].cpu().numpy())
        k = len(gt_nodes)
        if k == 0:
            continue

        try:
            scores, _ = get_gnnexplainer_scores(explainer, model, data_transformed)
            top_k_indices = np.argsort(scores)[-k:]
            pred_nodes = set(top_k_indices)

            intersection = len(gt_nodes.intersection(pred_nodes))
            union = len(gt_nodes.union(pred_nodes))
            jaccard_scores.append(intersection / union if union > 0 else 0.0)
        except Exception:
            jaccard_scores.append(0.0)

    return float(np.mean(jaccard_scores)) if len(jaccard_scores) else float('nan')


def evaluate_gnnexplainer_auroc(
    explainer,
    model,
    dataset,
    *,
    transform=None,
):
    auroc_scores = []

    for data in tqdm(dataset, desc="Evaluating AUROC (gnnexplainer)"):
        if transform:
            data_transformed = transform(data.clone())
        else:
            data_transformed = data.clone()

        if not hasattr(data_transformed, 'node_mask'):
            continue

        gt_mask = data_transformed.node_mask.squeeze().detach().cpu().numpy()
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue

        try:
            scores, _ = get_gnnexplainer_scores(explainer, model, data_transformed)
            auroc_scores.append(roc_auc_score(gt_mask, scores))
        except Exception:
            pass

    return float(np.mean(auroc_scores)) if len(auroc_scores) else float('nan')


# ── Edge-attr / multi-class evaluation (BCosGINE) ────────────────────────────
#
# Key differences from the binary GIN evaluators above:
#   • Ground-truth attribute is ``explanation_mask`` (not ``node_mask``).
#   • Model outputs multi-class logits → target class is passed to
#     ``explain_edge_attr`` explicitly; **no sign flip** is needed.
#   • Node scores = abs(contrib['x']).sum(dim=-1)  (unsigned, per-feature sum).
# ─────────────────────────────────────────────────────────────────────────────

def get_attribution_scores_edge(model, data, batch=None):
    """B-COS attribution for edge-attr models (multi-class GINE).

    Uses ``explain_edge_attr`` with the predicted class as the target.
    Node scores are the L1 norm of the node feature contribution map,
    which is always non-negative and does not require sign flipping.

    Parameters
    ----------
    model:
        An ``EvalModelAdapter`` wrapping a ``BCosGINE`` (or any model with
        signature ``forward(x, edge_index, edge_attr=None, batch=None)``).
    data:
        A ``torch_geometric.data.Data`` object already on the target device.
    batch:
        Optional batch index tensor.  Created automatically if ``None``.

    Returns
    -------
    scores : np.ndarray, shape [num_nodes]
    pred_class : int
    """
    from bcosgnn.explain_edge_attr import explain as _explain_edge

    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, getattr(data, "edge_attr", None), batch)
    pred_class = int(out.view(1, -1).argmax(dim=-1).item())

    contrib = _explain_edge(
        model,
        data.x,
        data.edge_index,
        getattr(data, "edge_attr", None),
        batch,
        target=pred_class,
    )
    # Sum absolute attributions across feature dimension → scalar per node
    scores = contrib["x"].abs().sum(dim=-1).detach().cpu().numpy()
    return scores, pred_class


def evaluate_jaccard_edge(
    model,
    dataset,
    transform=None,
    gt_attr: str = "explanation_mask",
):
    """Jaccard@|GT| for edge-attr B-COS models (multi-class).

    ``k`` is set dynamically to the number of ground-truth motif nodes per graph.
    Ground truth is read from ``data.<gt_attr>`` (default: ``explanation_mask``).
    """
    jaccard_scores = []
    device = next(model.parameters()).device

    for data in tqdm(dataset, desc="Evaluating Jaccard (bcos_edge)"):
        data_t = transform(data.clone()) if transform else data.clone()

        gt = getattr(data_t, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        k = int(gt_mask.sum())
        if k == 0:
            continue

        try:
            data_t = data_t.to(device)
            batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)
            scores, _ = get_attribution_scores_edge(model, data_t, batch)
            top_k  = set(np.argsort(scores)[-k:].tolist())
            gt_set = set(np.where(gt_mask)[0].tolist())
            inter  = len(gt_set & top_k)
            union  = len(gt_set | top_k)
            jaccard_scores.append(inter / union if union > 0 else 0.0)
        except Exception:
            jaccard_scores.append(0.0)

    return float(np.mean(jaccard_scores)) if jaccard_scores else float("nan")


def evaluate_auroc_edge(
    model,
    dataset,
    transform=None,
    gt_attr: str = "explanation_mask",
):
    """Node AUROC for edge-attr B-COS models (multi-class).

    Skips graphs where the ground-truth mask is all-0 or all-1 (degenerate AUROC).
    """
    auroc_scores = []
    device = next(model.parameters()).device

    for data in tqdm(dataset, desc="Evaluating AUROC (bcos_edge)"):
        data_t = transform(data.clone()) if transform else data.clone()

        gt = getattr(data_t, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue

        try:
            data_t = data_t.to(device)
            batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)
            scores, _ = get_attribution_scores_edge(model, data_t, batch)
            auroc_scores.append(roc_auc_score(gt_mask, scores))
        except Exception:
            pass

    return float(np.mean(auroc_scores)) if auroc_scores else float("nan")


def evaluate_gnnexplainer_jaccard_edge(
    explainer,
    model,
    dataset,
    *,
    transform=None,
    gt_attr: str = "explanation_mask",
):
    """Jaccard@|GT| for GNNExplainer on edge-attr models (multi-class).

    Reuses ``get_gnnexplainer_scores`` (which already forwards ``edge_attr``
    and handles batch creation) but reads ground truth from ``gt_attr``.
    """
    jaccard_scores = []

    for data in tqdm(dataset, desc="Evaluating Jaccard (gnnexplainer_edge)"):
        data_t = transform(data.clone()) if transform else data.clone()

        gt = getattr(data_t, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        k = int(gt_mask.sum())
        if k == 0:
            continue

        try:
            scores, _ = get_gnnexplainer_scores(explainer, model, data_t)
            top_k  = set(np.argsort(scores)[-k:].tolist())
            gt_set = set(np.where(gt_mask)[0].tolist())
            inter  = len(gt_set & top_k)
            union  = len(gt_set | top_k)
            jaccard_scores.append(inter / union if union > 0 else 0.0)
        except Exception:
            jaccard_scores.append(0.0)

    return float(np.mean(jaccard_scores)) if jaccard_scores else float("nan")


def evaluate_gnnexplainer_auroc_edge(
    explainer,
    model,
    dataset,
    *,
    transform=None,
    gt_attr: str = "explanation_mask",
):
    """Node AUROC for GNNExplainer on edge-attr models (multi-class)."""
    auroc_scores = []

    for data in tqdm(dataset, desc="Evaluating AUROC (gnnexplainer_edge)"):
        data_t = transform(data.clone()) if transform else data.clone()

        gt = getattr(data_t, gt_attr, None)
        if gt is None:
            continue
        gt_mask = gt.squeeze().detach().cpu().numpy().astype(int)
        if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
            continue

        try:
            scores, _ = get_gnnexplainer_scores(explainer, model, data_t)
            auroc_scores.append(roc_auc_score(gt_mask, scores))
        except Exception:
            pass

    return float(np.mean(auroc_scores)) if auroc_scores else float("nan")
