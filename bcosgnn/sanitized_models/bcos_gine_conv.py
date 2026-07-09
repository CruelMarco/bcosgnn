import torch
from bcos.modules import BcosLinear, BcosSequential
from torch_geometric.nn import MessagePassing


class BcosGINEConv(MessagePassing):
    """B-COS GINEConv: edge-aware message passing with pure B-COS linear layers.

    Edge attributes **must be pre-projected** (by ``BCosGINE.lin_edge``) to the
    same channel dimension as node features before being passed here.

    The message is defined as::

        message(x_j, edge_attr) = x_j + edge_attr   # no activation

    This additive form is the simplest way to incorporate edge information while
    preserving the dynamic linearity required for B-COS explanations.  Any
    standard non-linearity (ReLU, etc.) in the message would break the
    ``explanation_mode`` linearity invariant.
    """

    def __init__(
        self,
        channels: list[int],
        b: float = 2.0,
        max_out: int = 1,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.transform = BcosSequential(
            *[
                BcosLinear(din, dout, b=b, max_out=max_out)
                for din, dout in zip(channels[:-1], channels[1:])
            ]
        )
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = (1 + self.eps) * x + out
        return self.transform(out)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Pure additive message — no activation, preserves B-COS linearity."""
        return x_j + edge_attr
