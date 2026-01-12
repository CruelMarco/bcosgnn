import os
import sys

_RINGS_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "rings"))
if _RINGS_PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _RINGS_PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from sklearn.metrics import (
	average_precision_score,
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)

from rings.complementarity import ComplementarityFunctor, FrobeniusMatrixNormComparator
from rings.perturbations import CompleteGraph, RandomFeatures, RandomGraph


def remove_edges_keep_nodes(data):
	"""Structure ablation for node tasks: keep nodes/labels/features, remove all edges."""
	data = data.clone()
	# Preserve nodes; Explanation objects sometimes confuse PyG's num_nodes inference.
	data.num_nodes = _get_num_nodes(data)
	data.edge_index = torch.empty((2, 0), dtype=torch.long)
	return data


def _get_num_nodes(data) -> int:
	n = getattr(data, "num_nodes", None)
	if n is not None:
		return int(n)
	y = getattr(data, "y", None)
	if y is None:
		raise RuntimeError("Cannot infer num_nodes: data has no num_nodes and no y.")
	return int(y.view(-1).numel())


def add_degree_one_hot_features(data, max_degree: int):
	data = data.clone()
	n = _get_num_nodes(data)
	if getattr(data, "edge_index", None) is None:
		data.edge_index = torch.empty((2, 0), dtype=torch.long)
	deg = degree(data.edge_index[0], num_nodes=n).to(torch.long)
	deg = deg.clamp(max=max_degree)
	data.x = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
	return data


def add_constant_features(data, dim: int = 1):
	data = data.clone()
	n = _get_num_nodes(data)
	data.x = torch.ones((n, dim), dtype=torch.float)
	return data


def compute_max_degree(dataset_list, limit: int | None = None) -> int:
	n = len(dataset_list) if limit is None else min(len(dataset_list), limit)
	max_deg = 0
	for i in range(n):
		d = dataset_list[i]
		if getattr(d, "edge_index", None) is None or d.edge_index.numel() == 0:
			continue
		deg = degree(d.edge_index[0], num_nodes=_get_num_nodes(d))
		if deg.numel() > 0:
			max_deg = max(max_deg, int(deg.max().item()))
	return max_deg


class GNNNodeClassifier(torch.nn.Module):
	def __init__(self, in_channels: int):
		super().__init__()
		self.conv1 = GCNConv(in_channels, 64)
		self.conv2 = GCNConv(64, 64)
		self.lin = torch.nn.Linear(64, 1)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = x.float()
		x = F.relu(self.conv1(x, edge_index))
		x = F.relu(self.conv2(x, edge_index))
		return self.lin(x).view(-1)  # [num_nodes]


def _binary_targets(batch) -> torch.Tensor:
	y = getattr(batch, "y", None)
	if y is None:
		raise RuntimeError("Batch has no labels (batch.y is None).")
	return (y.view(-1) > 0).float()


def empty_features_keep_dim(data):
	data = data.clone()
	dim = 1 if getattr(data, "x", None) is None else int(data.x.size(1))
	data.x = torch.zeros((_get_num_nodes(data), dim), dtype=torch.float)
	return data


def apply_perturbation(dataset_list, perturbation):
	if perturbation is None:
		return [d.clone() for d in dataset_list]
	out = []
	for d in dataset_list:
		d2 = perturbation(d.clone()) if callable(perturbation) else perturbation(d.clone())
		# These masks refer to the original motif edges/nodes; after structure perturbations
		# they can be inconsistent with edge_index and aren't needed for P1/P2.
		if hasattr(d2, "edge_mask"):
			d2.edge_mask = None
		if hasattr(d2, "node_mask"):
			d2.node_mask = None
		out.append(d2)
	return out


def _is_structure_perturbation(p) -> bool:
	return p is remove_edges_keep_nodes or isinstance(p, (RandomGraph, CompleteGraph))


def _is_feature_perturbation(p) -> bool:
	return isinstance(p, RandomFeatures) or p in (empty_features_keep_dim,)


def build_condition_dataset(
	base_graphs,
	perturbation,
	*,
	feature_mode: str,
	max_degree: int,
):
	"""Build a dataset for one condition without leaking original topology into features.

	feature_mode:
	  - 'degree': recompute degree one-hot from the *current* edge_index
	  - 'constant': all-ones features
	"""
	perturbed = apply_perturbation(base_graphs, perturbation)

	# If this is a structure perturbation, node features must be recomputed after it.
	out = []
	for d in perturbed:
		if feature_mode == "degree":
			d = add_degree_one_hot_features(d, max_degree)
		elif feature_mode == "constant":
			d = add_constant_features(d, dim=1)
		else:
			raise ValueError(f"Unknown feature_mode: {feature_mode}")
		out.append(d)

	# If this is a feature perturbation, apply it after features are built.
	if _is_feature_perturbation(perturbation):
		out = [perturbation(d.clone()) for d in out]

	return out


def _bce_pos_weight_from_dataset(dataset_list) -> torch.Tensor:
	ys = []
	for d in dataset_list:
		y = getattr(d, "y", None)
		if y is None:
			continue
		ys.append((y.view(-1) > 0).to(torch.float))
	if not ys:
		return torch.tensor(1.0)
	y_all = torch.cat(ys, dim=0)
	pos = float(y_all.sum().item())
	neg = float(y_all.numel() - y_all.sum().item())
	if pos <= 0:
		return torch.tensor(1.0)
	return torch.tensor(neg / pos)


def _topk_jaccard_from_batch(logits: torch.Tensor, y: torch.Tensor, batch_vec: torch.Tensor):
	# For each graph, select top-k nodes where k = #positive nodes (GT), compute Jaccard.
	probs = torch.sigmoid(logits.detach()).cpu()
	y = y.detach().cpu().to(torch.int64)
	batch_vec = batch_vec.detach().cpu().to(torch.int64)

	jaccs = []
	for g in batch_vec.unique():
		idx = (batch_vec == g).nonzero(as_tuple=False).view(-1)
		y_g = y[idx]
		k = int(y_g.sum().item())
		if k <= 0:
			continue
		scores = probs[idx]
		topk = torch.topk(scores, k=k, largest=True).indices
		pred = torch.zeros_like(y_g)
		pred[topk] = 1
		inter = int(((pred == 1) & (y_g == 1)).sum().item())
		union = int(((pred == 1) | (y_g == 1)).sum().item())
		if union > 0:
			jaccs.append(inter / union)
	if not jaccs:
		return None
	return float(sum(jaccs) / len(jaccs))


def run_p1_test(name, train_loader, test_loader, in_channels: int, *, pos_weight: torch.Tensor, epochs: int = 10):
	model = GNNNodeClassifier(in_channels)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	print(f"Training P1 - {name}...")
	for _epoch in range(epochs):
		model.train()
		for batch in train_loader:
			optimizer.zero_grad()
			logits = model(batch)
			y = _binary_targets(batch)
			loss = criterion(logits, y)
			loss.backward()
			optimizer.step()

	model.eval()
	total_loss = 0.0
	logits_all = []
	y_all = []
	batch_all = []
	with torch.no_grad():
		for batch in test_loader:
			logits = model(batch)
			y = _binary_targets(batch)
			total_loss += float(criterion(logits, y).item()) * int(y.numel())
			logits_all.append(logits.detach().cpu())
			y_all.append(y.detach().cpu())
			batch_all.append(batch.batch.detach().cpu())

	logits_all = torch.cat(logits_all, dim=0)
	y_all = torch.cat(y_all, dim=0).to(torch.int64)
	batch_all = torch.cat(batch_all, dim=0)

	probs = torch.sigmoid(logits_all).numpy()
	y_np = y_all.numpy()
	pred = (probs >= 0.5).astype(int)

	# Accuracy is included for reference, but is not reliable under heavy imbalance.
	acc = float((pred == y_np).mean()) if y_np.size else 0.0
	bal_acc = float(balanced_accuracy_score(y_np, pred)) if len(set(y_np.tolist())) > 1 else None
	f1 = float(f1_score(y_np, pred, zero_division=0))
	prec = float(precision_score(y_np, pred, zero_division=0))
	rec = float(recall_score(y_np, pred, zero_division=0))

	auroc = None
	auprc = None
	if len(set(y_np.tolist())) > 1:
		auroc = float(roc_auc_score(y_np, probs))
		auprc = float(average_precision_score(y_np, probs))

	topk_j = _topk_jaccard_from_batch(logits_all, y_all.to(torch.float), batch_all)

	avg_loss = total_loss / max(int(y_all.numel()), 1)
	return {
		"acc": acc,
		"bal_acc": bal_acc,
		"f1": f1,
		"precision": prec,
		"recall": rec,
		"auroc": auroc,
		"auprc": auprc,
		"topk_jaccard": topk_j,
		"bce": float(avg_loss),
	}


def run_p2_test(dataset_list):
	print("Computing P2 - Mode Complementarity...")
	comp_functor = ComplementarityFunctor(
		graph_metric="diffusion_distance",
		feature_metric="euclidean",
		comparator=FrobeniusMatrixNormComparator,
		num_steps=1,
	)

	subset = dataset_list[:100]

	subset_ef = [empty_features_keep_dim(d) for d in subset]
	results_ef = comp_functor(subset_ef, as_dataframe=False)
	gamma_ef = float(results_ef["complementarity"].mean().item())
	delta_s = 1 - abs(1 - 2 * gamma_ef)

	subset_eg = [remove_edges_keep_nodes(d) for d in subset]
	results_eg = comp_functor(subset_eg, as_dataframe=False)
	gamma_eg = float(results_eg["complementarity"].mean().item())
	delta_f = 1 - abs(1 - 2 * gamma_eg)

	return float(delta_s), float(delta_f)


def main():
	torch.manual_seed(0)

	dataset_path = os.environ.get(
		"RINGS_DATASET_PATH",
		os.path.join(
			os.path.dirname(__file__),
			"..",
			"shaique_updates",
			"data",
			"Custom_BA2Motif",
			"custom_ba2motif_dataset.pt",
		),
	)
	dataset_path = os.path.abspath(dataset_path)

	# NOTE: PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`.
	# This dataset is a pickled list of PyG `Explanation` objects, so we must
	# set weights_only=False. Only do this for trusted local files.
	raw_list = torch.load(dataset_path, weights_only=False)

	if not isinstance(raw_list, list) or len(raw_list) == 0:
		raise RuntimeError(f"Expected a non-empty list from {dataset_path!r}, got: {type(raw_list)}")

	quick = os.environ.get("RINGS_QUICK", "0") == "1"
	max_graphs = int(os.environ.get("RINGS_MAX_GRAPHS", "200" if quick else "1000"))
	raw_list = raw_list[: min(len(raw_list), max_graphs)]

	# Base graphs: keep only structure + node labels; features will be generated per condition
	base_graphs = []
	for d in raw_list:
		d = d.clone()
		if getattr(d, "edge_index", None) is None:
			d.edge_index = torch.empty((2, 0), dtype=torch.long)
		# Explicitly set num_nodes to avoid PyG inference warnings.
		d.num_nodes = _get_num_nodes(d)
		d.x = None
		base_graphs.append(d)

	max_deg = compute_max_degree(base_graphs)
	feature_mode = os.environ.get("RINGS_FEATURE_MODE", "degree")  # 'degree' or 'constant'
	in_channels = (max_deg + 1) if feature_mode == "degree" else 1

	train_idx = int(len(base_graphs) * 0.8)

	def make_loaders(dataset_list):
		return (
			DataLoader(dataset_list[:train_idx], batch_size=64, shuffle=True),
			DataLoader(dataset_list[train_idx:], batch_size=64),
		)

	print("\n--- RINGS DIAGNOSTIC RESULTS (Custom_BA2Motif) ---")
	print(f"Dataset: {dataset_path}")
	print(f"Graphs: {len(base_graphs)} | max_degree: {max_deg} | feature_mode: {feature_mode} | in_channels: {in_channels}")

	p1_epochs = int(os.environ.get("RINGS_P1_EPOCHS", "2" if quick else "5"))

	p1_conditions = [
		("Original", None),
		("NoEdges (edge ablation)", remove_edges_keep_nodes),
		("RandomGraph (same |E|)", RandomGraph()),
		("RandomGraph (shuffle edges)", RandomGraph(shuffle=True)),
		("CompleteGraph", CompleteGraph()),
		("RandomFeatures (gaussian)", RandomFeatures(shuffle=False)),
		("RandomFeatures (shuffle)", RandomFeatures(shuffle=True)),
		("EmptyFeatures (zeros)", empty_features_keep_dim),
	]

	p1_results = []
	for condition_name, perturb in p1_conditions:
		cond_dataset = build_condition_dataset(
			base_graphs,
			perturb,
			feature_mode=feature_mode,
			max_degree=max_deg,
		)
		train_loader, test_loader = make_loaders(cond_dataset)
		pos_weight = _bce_pos_weight_from_dataset(cond_dataset[:train_idx])
		metrics = run_p1_test(
			condition_name,
			train_loader,
			test_loader,
			in_channels,
			pos_weight=pos_weight,
			epochs=p1_epochs,
		)
		p1_results.append((condition_name, metrics))

	orig = p1_results[0][1]
	empty = p1_results[1][1]

	# For P2, use the unperturbed dataset with generated features.
	full_for_p2 = build_condition_dataset(
		base_graphs,
		None,
		feature_mode=feature_mode,
		max_degree=max_deg,
	)
	ds, df = run_p2_test(full_for_p2)

	print("\nRESULTS SUMMARY:")
	print(
		"P1 - Original: "
		f"Acc={orig['acc']:.4f} | BalAcc={orig['bal_acc'] if orig['bal_acc'] is not None else 'N/A'} | "
		f"F1={orig['f1']:.4f} | AUROC={orig['auroc'] if orig['auroc'] is not None else 'N/A'} | "
		f"AUPRC={orig['auprc'] if orig['auprc'] is not None else 'N/A'} | "
		f"TopKJ={orig['topk_jaccard'] if orig['topk_jaccard'] is not None else 'N/A'} | "
		f"BCE={orig['bce']:.4f}"
	)
	print(
		"P1 - EmptyGraph: "
		f"Acc={empty['acc']:.4f} | BalAcc={empty['bal_acc'] if empty['bal_acc'] is not None else 'N/A'} | "
		f"F1={empty['f1']:.4f} | AUROC={empty['auroc'] if empty['auroc'] is not None else 'N/A'} | "
		f"AUPRC={empty['auprc'] if empty['auprc'] is not None else 'N/A'} | "
		f"TopKJ={empty['topk_jaccard'] if empty['topk_jaccard'] is not None else 'N/A'} | "
		f"BCE={empty['bce']:.4f}"
	)
	print(f"P2 - Structural Diversity (ΔS): {ds:.4f}")
	print(f"P2 - Feature Diversity (ΔF):    {df:.4f}")

	print("\nP1 - METRICS ACROSS PERTURBATIONS:")
	print(f"{'Condition':30s} {'Acc':>6s} {'BalAcc':>8s} {'F1':>6s} {'AUROC':>7s} {'AUPRC':>7s} {'TopKJ':>7s} {'BCE':>8s}")
	for condition_name, m in p1_results:
		bal = f"{m['bal_acc']:.4f}" if m["bal_acc"] is not None else "  N/A"
		auroc = f"{m['auroc']:.4f}" if m["auroc"] is not None else "  N/A"
		auprc = f"{m['auprc']:.4f}" if m["auprc"] is not None else "  N/A"
		topkj = f"{m['topk_jaccard']:.4f}" if m["topk_jaccard"] is not None else "  N/A"
		print(
			f"{condition_name[:30]:30s} {m['acc']:6.4f} {bal:>8s} {m['f1']:6.4f} {auroc:>7s} {auprc:>7s} {topkj:>7s} {m['bce']:8.4f}"
		)

	print("\n--- INTERPRETATION (RINGS) ---")
	print("1. Performance Separability (P1):")
	print("   - Prefer AUROC/AUPRC/TopKJ over raw accuracy for this imbalanced node task.")
	print("   - Tests whether graph structure is task-relevant by removing edges.")
	# Use AUROC if available; otherwise TopKJ; otherwise balanced accuracy.
	score_name = None
	orig_score = None
	empty_score = None
	for key, label in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("topk_jaccard", "TopKJ"), ("bal_acc", "BalAcc")]:
		if orig.get(key) is not None and empty.get(key) is not None:
			score_name = label
			orig_score = float(orig[key])
			empty_score = float(empty[key])
			break
	if score_name is None:
		print("   - Outcome: N/A (not enough label variation to score)")
	else:
		print(f"   - {score_name}: Original={orig_score:.4f} vs EmptyGraph={empty_score:.4f}")
		print(
			f"   - Outcome: {'Separable (structure matters)' if orig_score > empty_score + 0.05 else 'Not separable (weak evidence structure matters)'}"
		)

	print("\n2. Mode Complementarity (P2):")
	print("   - Measures non-redundancy between structure and node features (degree here).")
	print(f"   - Structural Diversity (ΔS): {ds:.2f} ({'High' if ds > 0.4 else 'Low'})")
	print(f"   - Feature Diversity (ΔF):     {df:.2f} ({'High' if df > 0.4 else 'Low'})")


if __name__ == "__main__":
	main()
