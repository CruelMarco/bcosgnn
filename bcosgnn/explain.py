import torch
from torch import Tensor, enable_grad
from bcos import explanation_mode


# NOTE this does not support edge attributes *yet*
def explain(
    model,
    x: Tensor,
    edge_index: Tensor,
    *other_inputs: Tensor,
):
    x = x.clone().requires_grad_()
    with enable_grad(), explanation_mode(model):
        out = model(x, edge_index, *other_inputs)
        
        if out.dim() > 1 and out.shape[-1] > 1:
            prediction_logit = out.max(dim=-1).values
        else:
            prediction_logit = out
            
        prediction_logit.backward(gradient=torch.ones_like(prediction_logit), inputs=[x])

    dynamic_linear_weights = x.grad
    contributions = dynamic_linear_weights * x
    return contributions.detach()