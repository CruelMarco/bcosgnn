from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, Type
from torch.nn import Module, ModuleList, Sequential, Dropout
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.aggr import MeanAggregation, SumAggregation

from bcos.modules import BcosLinear
from bcos.modules.norms import DetachableLayerNorm

from bcosgnn.conv import BCC


class AddedResidualConnection(Module):
    def __init__(self, base: Module):
        self.base = base

    def forward(self, *args, **kwargs):
        inp = args[0]
        out = self.base(*args, **kwargs)
        return inp + out


class AddedLayerNorm(Module):
    def __init__(self, base: Module, layer_norm: Module):
        super(AddedLayerNorm, self).__init__()
        self.base = base
        self.layer_norm = layer_norm

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)
        return self.layer_norm(out)


class BcosGNN(Module, ABC):
    @abstractmethod
    def make_conv(self):
        raise NotImplementedError

    def make_linear(self, in_channels: int, out_channels: int):
        return BcosLinear(
            in_channels,
            out_channels,
            b=self.b,
            max_out=self.max_out,
        )

    def wrap_conv_module(
        self,
        conv_module: Module,
        residual_connection: bool = False,
        layer_norm: bool = False,
    ) -> Module:
        if residual_connection:
            conv_module = AddedResidualConnection(conv_module)
        if layer_norm:
            conv_module = AddedLayerNorm(
                conv_module, DetachableLayerNorm(self.hidden_channels)
            )
        return conv_module

    def __init__(
        self,
        node_size: int,
        edge_size: int,
        hidden_channels: int,
        num_layers: int,
        b: float = 2,
        max_out: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_channels = hidden_channels
        self.b = b
        self.max_out = max_out

        self.lin_node = self.make_linear(node_size, hidden_channels)
        self.lin_edge = self.make_linear(edge_size, hidden_channels)
        self.convs = ModuleList(
            [
                self.wrap_conv_module(self.make_conv(), layer_norm=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.lin_node(x)
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x


class BcosGINE(BcosGNN):
    def make_conv(self):
        return GINEConv(
            self.make_linear(self.hidden_channels, self.hidden_channels),
            train_eps=False,
        )


class BcosMPNN(BcosGNN):
    def make_conv(self):
        return BCC(self.hidden_channels, b=self.b, max_out=self.max_out)


class BcosGCN(BcosGNN):
    # TODO Shaique
    ...


class GNNCls(Enum):
    GINE = "gine"
    BCOS_GINE = "bcos_gine"
    BCOS_MPNN = "bcos_mpnn"


TGNNKey = Literal[GNNCls.GINE, GNNCls.BCOS_GINE, GNNCls.BCOS_MPNN]


def resolve_gnn_cls(gnn_cls: TGNNKey) -> Type[BcosGNN]:
    if gnn_cls == GNNCls.GINE:
        return BcosGINE
    if gnn_cls == GNNCls.BCOS_GINE:
        return BcosGINE
    if gnn_cls == GNNCls.BCOS_MPNN:
        return BcosMPNN
    raise ValueError(f"Unknown GNN class: {gnn_cls}")


class BcosReadout(Module):
    def make_linear(self, in_channels: int, out_channels: int):
        return BcosLinear(in_channels, out_channels, b=self.b, max_out=self.max_out)

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int = 1,
        agg: str | None = None,
        b: float = 2,
        max_out: int = 1,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.agg = agg
        self.b = b
        self.max_out = max_out
        self.dropout = dropout

        if agg is None:
            self.fn_agg = None
        if agg == "sum":
            self.fn_agg = SumAggregation()
        if agg == "mean":
            self.fn_agg = MeanAggregation()
        self.readout = Sequential(
            self.make_linear(hidden_channels, hidden_channels),
            Dropout(dropout),
            self.make_linear(hidden_channels, self.out_channels),
        )

    def forward(self, x, batch):
        if self.fn_agg is not None:
            x = self.fn_agg(x, batch)
        return self.readout(x)
