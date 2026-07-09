import functools
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import degree


def find_repo_root(start: Path) -> Path:
	for path in [start, *start.parents]:
		if (path / "pyproject.toml").exists() and (path / "bcosgnn").is_dir():
			return path
	raise RuntimeError("Could not locate repo root (pyproject.toml + bcosgnn/).")


PROJECT_ROOT = find_repo_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from bcosgnn.evaluation import evaluate_auroc, evaluate_jaccard
from bcosgnn.sanitized_models import (
	AggThenReadout,
	BCosGNN,
	BcosGINConv,
	ReadoutThenAgg,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_one_hot_degree_clamped(data, max_degree: int = 4):
	_, col = data.edge_index
	deg = degree(col, data.num_nodes, dtype=torch.long)
	deg = deg.clamp(max=max_degree)
	one_hot_deg = torch.nn.functional.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
	data.x = one_hot_deg.float()
	return data


def split_train_val_test(dataset, seed: int, test_size: float = 0.2, val_size: float = 0.1):
	labels = np.array([int(d.y.item()) for d in dataset])
	indices = np.arange(len(dataset))

	train_val_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		random_state=seed,
		stratify=labels,
	)

	train_val_labels = labels[train_val_idx]
	val_ratio_on_trainval = val_size / (1.0 - test_size)
	train_idx, val_idx = train_test_split(
		train_val_idx,
		test_size=val_ratio_on_trainval,
		random_state=seed,
		stratify=train_val_labels,
	)

	train_data = [dataset[i] for i in train_idx]
	val_data = [dataset[i] for i in val_idx]
	test_data = [dataset[i] for i in test_idx]
	return train_data, val_data, test_data


def make_loader(data_list, transform, batch_size: int = 64, shuffle: bool = False):
	data_t = [transform(d.clone()) for d in data_list]
	return DataLoader(data_t, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, criterion, optimizer):
	model.train()
	total_loss = 0.0
	total_correct = 0
	total_graphs = 0

	for batch in loader:
		batch = batch.to(DEVICE)
		logits = model(batch.x, batch.edge_index, batch.batch).flatten()
		loss = criterion(logits, batch.y.float())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pred = (logits > 0).long()
		total_loss += float(loss.item()) * batch.num_graphs
		total_correct += int((pred == batch.y.long()).sum().item())
		total_graphs += int(batch.num_graphs)

	return total_loss / total_graphs, total_correct / total_graphs


@torch.inference_mode()
def evaluate_loader(model, loader, criterion):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_graphs = 0
	all_logits = []
	all_labels = []

	for batch in loader:
		batch = batch.to(DEVICE)
		logits = model(batch.x, batch.edge_index, batch.batch).flatten()
		loss = criterion(logits, batch.y.float())
		pred = (logits > 0).long()

		total_loss += float(loss.item()) * batch.num_graphs
		total_correct += int((pred == batch.y.long()).sum().item())
		total_graphs += int(batch.num_graphs)

		all_logits.append(logits.detach().cpu())
		all_labels.append(batch.y.detach().cpu().long())

	logits_all = torch.cat(all_logits).numpy()
	labels_all = torch.cat(all_labels).numpy()
	probs = 1.0 / (1.0 + np.exp(-logits_all))
	try:
		clf_auroc = float(roc_auc_score(labels_all, probs))
	except ValueError:
		clf_auroc = float("nan")

	return {
		"loss": total_loss / total_graphs,
		"acc": total_correct / total_graphs,
		"clf_auroc": clf_auroc,
	}


class EvalModelAdapter(torch.nn.Module):
	def __init__(self, base_model: torch.nn.Module):
		super().__init__()
		self.base_model = base_model

	def forward(self, x, edge_index, edge_attr=None, batch=None):
		if batch is None:
			batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
		return self.base_model(x, edge_index, batch)


class VanillaGIN(torch.nn.Module):
	def __init__(self, node_size: int, hidden_dim: int = 64, num_convs: int = 3):
		super().__init__()
		self.node_encoder = nn.Linear(node_size, hidden_dim)

		convs = []
		for _ in range(num_convs):
			mlp = nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim),
			)
			convs.append(GINConv(mlp, train_eps=False))
		self.convs = nn.ModuleList(convs)

		self.readout = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, x, edge_index, batch):
		x = self.node_encoder(x)
		for conv in self.convs:
			x = conv(x, edge_index)
		graph_emb = global_mean_pool(x, batch)
		return self.readout(graph_emb)


def build_vanilla_model(node_size: int):
	return VanillaGIN(node_size=node_size, hidden_dim=64, num_convs=3).to(DEVICE)


def build_bcos_model(node_size: int, readout_mode: str, b: float):
	if readout_mode == "readout_then_agg":
		readout = ReadoutThenAgg(
			in_channels=64,
			hidden_channels=[64, 64],
			out_channels=1,
			b=b,
			max_out=1,
			agg="mean",
		)
	elif readout_mode == "agg_then_readout":
		readout = AggThenReadout(
			in_channels=64,
			hidden_channels=[64, 64],
			out_channels=1,
			b=b,
			max_out=1,
			agg="mean",
		)
	else:
		raise ValueError(f"Unsupported readout_mode: {readout_mode}")

	return BCosGNN(
		node_size=node_size,
		hidden_channels=[64, 64],
		conv_layer=BcosGINConv,
		num_convs=3,
		readout=readout,
		b=b,
		max_out=1,
		conv_kwargs={"aggr": "mean"},
	).to(DEVICE)


def run_experiment(
	experiment_name: str,
	dataset_list,
	transform,
	seeds,
	epochs: int,
	batch_size: int,
	lr: float,
	b: float,
	early_stop_patience: int,
	min_delta: float,
):
	criterion = BCEWithLogitsLoss()
	seed_results = []
	best_result = None
	best_result_score = -float("inf")

	for seed in seeds:
		print("=" * 100)
		print(f"{experiment_name} | seed={seed}")
		print("=" * 100)
		set_seed(seed)

		train_data, val_data, test_data = split_train_val_test(
			dataset_list,
			seed=seed,
			test_size=0.2,
			val_size=0.1,
		)
		train_loader = make_loader(train_data, transform, batch_size=batch_size, shuffle=True)
		val_loader = make_loader(val_data, transform, batch_size=batch_size, shuffle=False)
		test_loader = make_loader(test_data, transform, batch_size=batch_size, shuffle=False)

		node_size = train_loader.dataset[0].x.shape[1]
		if experiment_name == "vanilla_gin_mean":
			model = build_vanilla_model(node_size=node_size)
			do_expl_metrics = False
		elif experiment_name == "bcos_gin_mean_readout_then_agg":
			model = build_bcos_model(node_size=node_size, readout_mode="readout_then_agg", b=b)
			do_expl_metrics = True
		elif experiment_name == "bcos_gin_mean_agg_then_readout":
			model = build_bcos_model(node_size=node_size, readout_mode="agg_then_readout", b=b)
			do_expl_metrics = True
		else:
			raise ValueError(f"Unknown experiment: {experiment_name}")

		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=0.5,
			patience=4,
			min_lr=1e-6,
		)

		best_state = None
		best_val_loss = float("inf")
		best_epoch = -1
		epochs_without_improvement = 0
		SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
		checkpoint_path = SAVED_MODELS_DIR / f"best_{experiment_name}_seed_{seed}.pt"

		for epoch in range(1, epochs + 1):
			tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
			val_metrics = evaluate_loader(model, val_loader, criterion)
			scheduler.step(val_metrics["loss"])

			if (best_val_loss - val_metrics["loss"]) > min_delta:
				best_val_loss = val_metrics["loss"]
				best_epoch = epoch
				best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
				epochs_without_improvement = 0
				torch.save(
					{
						"experiment": experiment_name,
						"seed": seed,
						"best_epoch": best_epoch,
						"best_val_loss": float(best_val_loss),
						"model_state_dict": best_state,
					},
					checkpoint_path,
				)
			else:
				epochs_without_improvement += 1

			if epoch % 5 == 0 or epoch == 1:
				print(
					f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
					f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
					f"| lr={optimizer.param_groups[0]['lr']:.2e}",
				)

			if epochs_without_improvement >= early_stop_patience:
				print(
					f"Early stopping at epoch {epoch} "
					f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})",
				)
				break

		if checkpoint_path.exists():
			checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			model.load_state_dict(best_state)
		eval_model = EvalModelAdapter(model).to(DEVICE)

		test_metrics = evaluate_loader(model, test_loader, criterion)
		if do_expl_metrics:
			test_jaccard = float(
				evaluate_jaccard(eval_model, test_data, method="explain", transform=transform),
			)
			test_expl_auroc = float(
				evaluate_auroc(eval_model, test_data, method="explain", transform=transform),
			)
		else:
			test_jaccard = float("nan")
			test_expl_auroc = float("nan")

		result = {
			"experiment": experiment_name,
			"seed": seed,
			"best_epoch": best_epoch,
			"best_val_loss": float(best_val_loss),
			"best_model_path": str(checkpoint_path),
			"test_loss": float(test_metrics["loss"]),
			"test_acc": float(test_metrics["acc"]),
			"test_clf_auroc": float(test_metrics["clf_auroc"]),
			"test_expl_auroc": test_expl_auroc,
			"test_jaccard": test_jaccard,
			"model": model,
			"eval_model": eval_model,
			"test_dataset": test_data,
		}
		seed_results.append(result)

		print(
			f"{experiment_name} | seed={seed} | best_epoch={best_epoch} "
			f"| ckpt={checkpoint_path.name} "
			f"| test_acc={result['test_acc']:.4f} "
			f"| test_clf_auroc={result['test_clf_auroc']:.4f} "
			f"| test_expl_auroc={result['test_expl_auroc']:.4f} "
			f"| test_jaccard={result['test_jaccard']:.4f}",
		)

		rank_score = float(
			np.nan_to_num(result["test_expl_auroc"], nan=-1e9)
			+ np.nan_to_num(result["test_jaccard"], nan=-1e9)
			+ np.nan_to_num(result["test_acc"], nan=-1e9)
		)
		if (best_result is None) or (rank_score > best_result_score):
			best_result_score = rank_score
			best_result = result

	metrics_keys = [
		"test_loss",
		"test_acc",
		"test_clf_auroc",
		"test_expl_auroc",
		"test_jaccard",
	]
	summary = {}
	for key in metrics_keys:
		values = np.array([r[key] for r in seed_results], dtype=float)
		summary[f"{key}_mean"] = float(np.nanmean(values))
		summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
	summary["experiment"] = experiment_name

	return seed_results, best_result, summary


def main():
	print(f"Device: {DEVICE}")

	data_path = PROJECT_ROOT / "shaique_updates" / "data" / "Custom_BA2Motif" / "custom_ba2motif_dataset.pt"
	print(f"Loading data from: {data_path}")
	if not data_path.exists():
		raise FileNotFoundError(f"Dataset file not found at: {data_path}")

	dataset_list = torch.load(data_path, weights_only=False)
	print(f"Successfully loaded {len(dataset_list)} graphs.")

	for i, data in enumerate(dataset_list):
		data.y = torch.tensor([0]) if i < 500 else torch.tensor([1])
		data.x = torch.ones((data.num_nodes, 1))

	max_degree = 4
	transform = functools.partial(get_one_hot_degree_clamped, max_degree=max_degree)

	SEEDS = [0, 1, 2]
	EPOCHS = 200
	BATCH_SIZE = 64
	LR = 3e-4
	b = 2.0
	EARLY_STOP_PATIENCE = 25
	MIN_DELTA = 1e-4

	experiments = [
		"vanilla_gin_mean",
		"bcos_gin_mean_readout_then_agg",
		"bcos_gin_mean_agg_then_readout",
	]

	all_seed_rows = []
	summary_rows = []
	best_by_experiment = {}

	for experiment_name in experiments:
		seed_results, best_result, summary = run_experiment(
			experiment_name=experiment_name,
			dataset_list=dataset_list,
			transform=transform,
			seeds=SEEDS,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			lr=LR,
			b=b,
			early_stop_patience=EARLY_STOP_PATIENCE,
			min_delta=MIN_DELTA,
		)

		rows = [
			{
				key: value
				for key, value in result.items()
				if key
				in {
					"experiment",
					"seed",
					"best_epoch",
					"best_val_loss",
					"test_loss",
					"test_acc",
					"test_clf_auroc",
					"test_expl_auroc",
					"test_jaccard",
				}
			}
			for result in seed_results
		]
		all_seed_rows.extend(rows)
		summary_rows.append(summary)
		best_by_experiment[experiment_name] = best_result

	df_seed_results = pl.DataFrame(all_seed_rows)
	df_summary = pl.DataFrame(summary_rows)

	print("\nPer-seed test metrics:")
	print(df_seed_results)
	print("\nMean ± std across seeds:")
	print(df_summary)

	print("\nBest run picked per experiment (by explanation+accuracy rank score):")
	for experiment_name, best_result in best_by_experiment.items():
		print(
			f"- {experiment_name}: seed={best_result['seed']} "
			f"| test_acc={best_result['test_acc']:.4f} "
			f"| test_clf_auroc={best_result['test_clf_auroc']:.4f} "
			f"| test_expl_auroc={best_result['test_expl_auroc']:.4f} "
			f"| test_jaccard={best_result['test_jaccard']:.4f}",
		)


if __name__ == "__main__":
	main()
