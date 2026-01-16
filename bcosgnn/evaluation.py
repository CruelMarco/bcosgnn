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

    out = model(data.x, data.edge_index, batch)
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

def get_attribution_scores(model, data, method='gradient', batch=None, threshold=0.0):

    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        
    x = data.x.clone().requires_grad_()
    model.eval()
    
    # Forward pass to get prediction and gradients
    # We need gradients for 'gradient' and 'input_x_gradient' methods
    # For 'explain', the explain function handles its own forward/backward
    
    if method == 'explain':
        # Method A: Use the bcosgnn.explain.explain function written by Joschka
        # We still need the prediction to know if we should flip signs, this is handled in the notebook
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch)
        pred_class = 1 if out.item() > threshold else 0
        
        node_contrib = explain(model, data.x, data.edge_index, batch).detach()
        scores = node_contrib.sum(1)
        
    elif method in ['gradient', 'input_x_gradient']:
        with torch.enable_grad():
            out = model(x, data.edge_index, batch)
            out.backward()
            
        pred_class = 1 if out.item() > threshold else 0
        grad = x.grad.detach()
        
        if method == 'gradient':
            # Method B: Sensitivity / Raw Gradients
            scores = grad.sum(1)
        elif method == 'input_x_gradient':
            # Method C: Input * Gradient
            scores = (x.detach() * grad).sum(1)
            
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply sign flipping for Class 0
    if pred_class == 0:
        scores = -scores
        
    return scores.cpu().numpy(), pred_class

def evaluate_jaccard(model, dataset, method='gradient', transform=None):

    jaccard_scores = []
    
    for data in tqdm(dataset, desc=f"Evaluating Jaccard ({method})"):
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

def evaluate_auroc(model, dataset, method='gradient', transform=None):

    auroc_scores = []
    
    for data in tqdm(dataset, desc=f"Evaluating AUROC ({method})"):
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
