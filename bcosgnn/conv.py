from typing import Any, Dict, List
from bcos.modules import BcosLinear
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing


class BCC(MessagePassing):
    def __init__(
        self,
        channels: int,
        b: int = 1,
        max_out: int = 1,
        aggr: str | List[str] | Aggregation | None = "sum",
        aggr_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self.channels = channels
        self.b = b
        self.max_out = max_out
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.lin_message = BcosLinear(channels, channels, b=b, max_out=max_out)
        self.lin_update = BcosLinear(channels, channels, b=b, max_out=max_out)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
    ) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor = None) -> Tensor:
        if(edge_attr is not None):
            return self.lin_message(x_j + x_i + edge_attr)
        else:
            return self.lin_message(x_j + x_i)

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        return self.lin_update(x) + inputs
    
