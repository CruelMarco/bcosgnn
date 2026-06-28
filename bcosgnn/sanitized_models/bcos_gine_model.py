from typing import Any

import torch
from bcos.modules import BcosLinear

from .bcos_gine_conv import BcosGINEConv
from .readout import Readout


class BCosGINE(torch.nn.Module):
    """B-COS Graph Isomorphism Network with Edge features (GINE).

    Designed for multi-class molecular property prediction with full intrinsic
    B-COS explanation support via ``bcosgnn.explain_edge_attr``.

    Architecture
    ------------
    1. ``lin_node``: BcosLinear(node_size → hidden_channels[0])
    2. ``lin_edge``: BcosLinear(edge_size → hidden_channels[0])  ← pre-projects edges
    3. ``num_convs`` × BcosGINEConv(channels=hidden_channels)
    4. ``readout``: typically ReadoutThenAgg(hidden_channels[-1] → num_classes)

    All operations are B-COS linear layers.  **No dropout or standard
    non-linearities** are used so that ``explanation_mode`` produces exact
    (complete) linear decompositions for both node and edge features.

    Parameters
    ----------
    node_size:
        Dimensionality of raw node features.
    edge_size:
        Dimensionality of raw edge features.
    hidden_channels:
        Channel sequence for the per-conv MLP.  E.g. ``[128, 128]`` gives a
        single B-COS linear layer 128 → 128 inside each conv.
    num_convs:
        Number of BcosGINEConv layers.
    readout:
        A ``Readout`` instance (e.g. ``ReadoutThenAgg``) that maps node
        embeddings to a graph-level output of shape ``[num_classes]``.
    b:
        B-COS exponent (default 2.0).
    max_out:
        B-COS max-out parameter (default 1).
    conv_kwargs:
        Extra keyword arguments forwarded to each BcosGINEConv constructor.
    """

    def __init__(
        self,
        node_size: int,
        edge_size: int,
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

        # Initial projections: both map to the same hidden dimension so that
        # additive edge–node combination in the message function is dimension-safe.
        self.lin_node = BcosLinear(node_size, hidden_channels[0])
        self.lin_edge = BcosLinear(edge_size, hidden_channels[0])

        self.convs = torch.nn.ModuleList(
            [
                BcosGINEConv(
                    channels=hidden_channels,
                    b=b,
                    max_out=max_out,
                    **conv_kwargs,
                )
                for _ in range(num_convs)
            ]
        )
        self.readout = readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.lin_node(x)
        e = self.lin_edge(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, e)
        return self.readout(x, batch)
