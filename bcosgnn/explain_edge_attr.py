import torch
from torch import Tensor, enable_grad
from bcos import explanation_mode


def explain(
    model,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
    *other_inputs: Tensor,
    target: int | Tensor | None = None,
    contrast: tuple[int, int] | None = None,
) -> dict[str, Tensor | None]:
    """
    This is the edge-aware analogue of `bcosgnn.explain.explain`.

    - If `edge_attr` is provided, we attribute w.r.t. both `x` and `edge_attr`.
    - If `edge_attr` is None, behavior matches the node-only explainer.

    Returns a dict with keys:
      - `x`: contribution map shaped like `x`
      - `edge_attr`: contribution map shaped like `edge_attr` (or None)

    Contribution map is computed as: input * d(target)/d(input) under `explanation_mode`.
    """

    input_x = x.clone().requires_grad_()

    if edge_attr is not None:
        input_edge_attr = edge_attr.clone().requires_grad_()
        forward_args = (input_x, edge_index, input_edge_attr, *other_inputs)
        backward_inputs = [input_x, input_edge_attr]
    else:
        input_edge_attr = None
        forward_args = (input_x, edge_index, *other_inputs)
        backward_inputs = [input_x]

    if contrast is not None and target is not None:
        raise ValueError("Pass only one of `target` or `contrast`.")

    with enable_grad(), explanation_mode(model):
        out = model(*forward_args)

        # Select a scalar/vector target to backprop.
        # - target=None: match original behavior (max logit for multi-logit outputs).
        # - target=int: explain that class logit for all items.
        # - target=Tensor: per-item class indices (shape [B]) for batched outputs.
        if contrast is not None:
            if out.dim() == 0 or out.shape[-1] <= 1:
                raise ValueError("`contrast` requires a multi-logit model output.")
            c_pos, c_neg = contrast
            prediction_logit = out[..., c_pos] - out[..., c_neg]
        elif target is None:
            if out.dim() > 1 and out.shape[-1] > 1:
                prediction_logit = out.max(dim=-1).values
            else:
                prediction_logit = out
        else:
            if out.dim() == 0:
                prediction_logit = out
            elif out.shape[-1] == 1:
                prediction_logit = out.squeeze(-1)
            else:
                if isinstance(target, int):
                    prediction_logit = out[..., target]
                else:
                    target_idx = target.to(out.device).long()
                    if out.dim() == 1:
                        prediction_logit = out[target_idx]
                    else:
                        prediction_logit = out.gather(-1, target_idx.view(*target_idx.shape, 1)).squeeze(-1)

        prediction_logit.backward(
            gradient=torch.ones_like(prediction_logit),
            inputs=backward_inputs,
        )

    x_grad = input_x.grad
    if x_grad is None:
        x_grad = torch.zeros_like(input_x)
    x_contrib = (x_grad * input_x).detach()

    if input_edge_attr is None:
        edge_contrib = None
    else:
        edge_grad = input_edge_attr.grad
        if edge_grad is None:
            edge_grad = torch.zeros_like(input_edge_attr)
        edge_contrib = (edge_grad * input_edge_attr).detach()

    return {"x": x_contrib, "edge_attr": edge_contrib}
