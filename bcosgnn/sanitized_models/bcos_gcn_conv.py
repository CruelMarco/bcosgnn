import torch
from bcos.modules import BcosLinear, BcosSequential
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class BcosGCNConv(MessagePassing):
    def __init__(
        self,
        channels: list[int],
        b: float = 2.0,
        max_out: int = 1,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
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
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def forward(self, x, edge_index):
        edge_weight = None
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(0),
                self.improved,
                self.add_self_loops,
                flow=self.flow,
                dtype=x.dtype,
            )

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return self.transform(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j
