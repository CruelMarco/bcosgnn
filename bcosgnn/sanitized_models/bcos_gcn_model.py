from typing import Any

import torch
from bcos.modules import BcosLinear

from .bcos_gcn_conv import BcosGCNConv
from .readout import Readout


class BCosGCN(torch.nn.Module):
    def __init__(
        self,
        node_size: int,
        hidden_channels: int | list[int],
        num_convs: int,
        readout: Readout,
        b: float = 2.0,
        max_out: int = 1,
        conv_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        conv_kwargs = conv_kwargs or {}

        self.lin_node = BcosLinear(node_size, hidden_channels[0])
        self.convs = torch.nn.ModuleList(
            [
                BcosGCNConv(
                    channels=hidden_channels,
                    b=b,
                    max_out=max_out,
                    **conv_kwargs,
                )
                for _ in range(num_convs)
            ]
        )
        self.readout = readout

    def forward(self, x, edge_index, batch):
        x = self.lin_node(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        return self.readout(x, batch)
