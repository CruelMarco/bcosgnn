from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from bcosgnn.evaluation import evaluate_auroc, evaluate_jaccard
from bcosgnn.explain import explain


@dataclass(frozen=True)
class ExportConfig:
    """Saving-only config.

    This module intentionally contains no training code.

    """

    dataset_name: str
    test_size: float = 0.2
    test_split_seed: int = 42
    export_dir: str = "outputs/explanations"


def default_test_indices_path(cfg: ExportConfig) -> Path:
    export_dir = Path(cfg.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir / f"{cfg.dataset_name}__testseed{cfg.test_split_seed}__test_indices.npy"


def load_or_make_fixed_test_split(
    dataset_list: list[Data],
    *,
    cfg: ExportConfig,
    indices_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trainval_idx, test_idx) and persist test_idx to `.npy`.

    If the `.npy` exists, it is loaded unless overwrite=True.

    """

    if indices_path is None:
        indices_path = default_test_indices_path(cfg)

    idx = np.arange(len(dataset_list))

    if indices_path.exists() and not overwrite:
        test_idx = np.load(indices_path).astype(int)
    else:
        y = np.array([int(d.y.item()) for d in dataset_list])
        _, test_idx = train_test_split(
            idx,
            test_size=cfg.test_size,
            random_state=cfg.test_split_seed,
            stratify=y,
        )
        np.save(indices_path, np.asarray(test_idx, dtype=int))

    mask = np.ones(len(dataset_list), dtype=bool)
    mask[test_idx] = False
    trainval_idx = idx[mask]
    return trainval_idx.astype(int), np.asarray(test_idx, dtype=int)


@torch.inference_mode()
def evaluate_accuracy(
    model: torch.nn.Module,
    dataset: list[Data],
    *,
    transform: Callable[[Data], Data] | None,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> float:
    device = torch.device(device)
    model.eval()
    loader = DataLoader(
        [transform(d.clone()) if transform else d.clone() for d in dataset],
        batch_size=batch_size,
        shuffle=False,
    )
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch).flatten()
        pred = (logits > 0).long()
        y = batch.y.long().flatten()
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def export_test_explanations_for_model(
    model: torch.nn.Module,
    dataset_list: list[Data],
    *,
    raw_data_path: str,
    transform: Callable[[Data], Data] | None,
    cfg: ExportConfig,
    device: torch.device | str = "cpu",
    output_name: str | None = None,
    overwrite: bool = False,
    indices_path: Path | None = None,
    transform_metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, float]]:
    """Export explanations for the fixed held-out test set for an *already-trained* model.

    Saves a `.pt` payload that is PyG-friendly:
      - `explained_test` is a list of `torch_geometric.data.Data` objects with extra fields.

    Notes:
      - `evaluate_jaccard/evaluate_auroc` currently expect CPU (they call `.numpy()` internally).

    """

    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError(
            "Export currently requires CPU because evaluation uses `.numpy()` internally. "
            "Use device='cpu' or patch evaluation to call `.cpu().numpy()`."
        )

    export_dir = Path(cfg.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    trainval_idx, test_idx = load_or_make_fixed_test_split(
        dataset_list,
        cfg=cfg,
        indices_path=indices_path,
        overwrite=False,
    )

    test_data = [dataset_list[i] for i in test_idx]

    # Metrics on held-out test
    test_acc = evaluate_accuracy(model, test_data, transform=transform, device=device)
    test_jacc = evaluate_jaccard(model, test_data, method="explain", transform=transform)
    test_auroc = evaluate_auroc(model, test_data, method="explain", transform=transform)

    # Explanations per test graph
    explained_test: list[Data] = []
    model.eval()

    for dataset_index in tqdm(test_idx.tolist(), desc="Exporting test explanations"):
        data = dataset_list[dataset_index].clone()
        data = transform(data) if transform else data

        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # `explain()` is gradient-based; ensure x requires grad and autograd is enabled.
        with torch.enable_grad():
            x = data.x.to(device).detach().requires_grad_(True)
            contrib = explain(model, x, data.edge_index.to(device), batch).cpu()

        with torch.no_grad():
            logit = model(data.x.to(device), data.edge_index.to(device), batch).flatten()[0].cpu()
        pred_class = int(logit.item() > 0.0)

        score = contrib.sum(1)

        data.dataset_index = int(dataset_index)
        data.dataset_name = cfg.dataset_name
        data.pred_logit = logit
        data.pred_class = torch.tensor([pred_class], dtype=torch.long)
        data.explain_contrib = contrib
        data.explain_score = score

        explained_test.append(data.cpu())

    if output_name is None:
        output_name = f"{cfg.dataset_name}__testseed{cfg.test_split_seed}__export.pt"

    out_path = export_dir / output_name
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    if indices_path is None:
        indices_path = default_test_indices_path(cfg)

    payload = {
        "dataset_name": cfg.dataset_name,
        "raw_data_path": raw_data_path,
        "test_size": cfg.test_size,
        "test_split_seed": cfg.test_split_seed,
        "trainval_indices": trainval_idx,
        "test_indices": test_idx,
        "test_indices_path": str(indices_path),
        "model_state_dict": model.state_dict(),
        "transform": transform_metadata,
        "explained_test": explained_test,
        "metrics_test": {
            "accuracy": float(test_acc),
            "jaccard_explain": float(test_jacc),
            "auroc_explain": float(test_auroc),
        },
    }

    torch.save(payload, out_path)
    return out_path, payload["metrics_test"]
