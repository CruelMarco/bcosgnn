from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from bcosgnn.explain import explain
from bcosgnn.explain_edge_attr import explain as explain_edge_attr


def evaluate_bcos_node_completeness(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logits: list[float] = []
    contrib_sums: list[float] = []

    for data in dataset:
        data_t = transform(data.clone()) if transform else data.clone()
        data_t = data_t.to(device)

        edge_attr = getattr(data_t, "edge_attr", None)
        batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(data_t.x, data_t.edge_index, edge_attr, batch)
            logit = float(out.reshape(-1)[0].item())

        node_contrib = explain(
            model,
            data_t.x,
            data_t.edge_index,
            edge_attr,
            batch,
        ).detach()

        logits.append(logit)
        contrib_sums.append(float(node_contrib.sum().item()))

    logits_np = np.asarray(logits, dtype=float)
    contrib_np = np.asarray(contrib_sums, dtype=float)

    if logits_np.size == 0:
        mae = float("nan")
    else:
        mae = float(np.mean(np.abs(logits_np - contrib_np)))

    return {
        "num_graphs": int(logits_np.size),
        "mae": mae,
        "logits": logits_np,
        "contrib_sums": contrib_np,
    }


def save_bcos_completeness_scatter(
    logits: np.ndarray,
    contrib_sums: np.ndarray,
    *,
    title: str,
    output_path: str | Path,
    mae: float | None = None,
    x_label: str = "Model Output",
    y_label: str = "Row-Sum of Contribution Map",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.figsize": (6, 6),
            "axes.linewidth": 2.0,
            "font.family": "sans-serif",
        },
    )

    fig, ax = plt.subplots()

    if mae is None:
        if logits.size == 0:
            mae = float("nan")
        else:
            mae = float(np.mean(np.abs(logits - contrib_sums)))

    ax.scatter(
        logits,
        contrib_sums,
        alpha=1.0,
        s=60,
        label=f"B-cos (MAE: {mae:.2e})",
        color="navy",
        edgecolors="black",
        linewidths=0.5,
    )

    if logits.size > 0:
        min_val = float(min(logits.min(), contrib_sums.min()))
        max_val = float(max(logits.max(), contrib_sums.max()))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="#d62728",
            linestyle="--",
            linewidth=2.5,
            label="Perfect Completeness (y=x)",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    legend = ax.legend(loc="upper left", frameon=True, edgecolor="black", framealpha=1.0)
    for text in legend.get_texts():
        text.set_weight("bold")

    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", which="major", width=2.0, length=6)

    plt.tight_layout()
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path


def run_bcos_node_completeness_check(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
    title: str = "B-cos Completeness Check",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_bcos_node_completeness(
        model,
        dataset,
        transform=transform,
        device=device,
    )

    plot_path = None
    if output_path is not None:
        plot_path = save_bcos_completeness_scatter(
            metrics["logits"],
            metrics["contrib_sums"],
            title=title,
            output_path=output_path,
            mae=metrics["mae"],
        )

    return {
        "num_graphs": metrics["num_graphs"],
        "mae": metrics["mae"],
        "plot_path": str(plot_path) if plot_path is not None else None,
    }


def evaluate_bcos_gine_completeness(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logits: list[float] = []
    contrib_sums: list[float] = []

    for data in dataset:
        data_t = transform(data.clone()) if transform else data.clone()
        data_t = data_t.to(device)

        batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)
        edge_attr = getattr(data_t, "edge_attr", None)

        with torch.no_grad():
            out = model(data_t.x, data_t.edge_index, edge_attr, batch)
            out_2d = out.view(1, -1) if out.dim() == 1 else out
            pred_class = int(out_2d.argmax(dim=-1).item())
            target_logit = float(out_2d[0, pred_class].item())

        contrib = explain_edge_attr(
            model,
            data_t.x,
            data_t.edge_index,
            edge_attr,
            batch,
            target=pred_class,
        )
        node_sum = float(contrib["x"].sum().item())
        edge_sum = (
            float(contrib["edge_attr"].sum().item())
            if contrib.get("edge_attr", None) is not None
            else 0.0
        )

        logits.append(target_logit)
        contrib_sums.append(node_sum + edge_sum)

    logits_np = np.asarray(logits, dtype=float)
    contrib_np = np.asarray(contrib_sums, dtype=float)

    if logits_np.size == 0:
        mae = float("nan")
    else:
        mae = float(np.mean(np.abs(logits_np - contrib_np)))

    return {
        "num_graphs": int(logits_np.size),
        "mae": mae,
        "logits": logits_np,
        "contrib_sums": contrib_np,
    }


def run_bcos_gine_completeness_check(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
    title: str = "B-cos GINE Completeness Check",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_bcos_gine_completeness(
        model,
        dataset,
        transform=transform,
        device=device,
    )

    plot_path = None
    if output_path is not None:
        plot_path = save_bcos_completeness_scatter(
            metrics["logits"],
            metrics["contrib_sums"],
            title=title,
            output_path=output_path,
            mae=metrics["mae"],
            x_label="Logit for Predicted Class",
            y_label="Sum of Node+Edge Contribution Map",
        )

    return {
        "num_graphs": metrics["num_graphs"],
        "mae": metrics["mae"],
        "plot_path": str(plot_path) if plot_path is not None else None,
    }


def evaluate_bcos_multiclass_node_completeness(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logits: list[float] = []
    contrib_sums: list[float] = []

    for data in dataset:
        data_t = transform(data.clone()) if transform else data.clone()
        data_t = data_t.to(device)

        edge_attr = getattr(data_t, "edge_attr", None)
        batch = torch.zeros(data_t.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(data_t.x, data_t.edge_index, edge_attr, batch)
            out_2d = out.view(1, -1) if out.dim() == 1 else out
            pred_class = int(out_2d.argmax(dim=-1).item())
            target_logit = float(out_2d[0, pred_class].item())

        node_contrib = explain(
            model,
            data_t.x,
            data_t.edge_index,
            edge_attr,
            batch,
        ).detach()

        logits.append(target_logit)
        contrib_sums.append(float(node_contrib.sum().item()))

    logits_np = np.asarray(logits, dtype=float)
    contrib_np = np.asarray(contrib_sums, dtype=float)
    mae = float(np.mean(np.abs(logits_np - contrib_np))) if logits_np.size > 0 else float("nan")

    return {
        "num_graphs": int(logits_np.size),
        "mae": mae,
        "logits": logits_np,
        "contrib_sums": contrib_np,
    }


def run_bcos_multiclass_node_completeness_check(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    transform=None,
    device: torch.device | str | None = None,
    title: str = "B-cos Multiclass Node Completeness Check",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_bcos_multiclass_node_completeness(
        model,
        dataset,
        transform=transform,
        device=device,
    )

    plot_path = None
    if output_path is not None:
        plot_path = save_bcos_completeness_scatter(
            metrics["logits"],
            metrics["contrib_sums"],
            title=title,
            output_path=output_path,
            mae=metrics["mae"],
            x_label="Logit for Predicted Class",
            y_label="Sum of Node Contribution Map",
        )

    return {
        "num_graphs": metrics["num_graphs"],
        "mae": metrics["mae"],
        "plot_path": str(plot_path) if plot_path is not None else None,
    }
