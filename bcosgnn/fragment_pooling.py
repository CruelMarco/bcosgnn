from typing import Literal
from torch.nn import Module, Linear, Sequential, ReLU, Sigmoid
from torch import Tensor
from torch_geometric.nn.pool import global_add_pool, global_add_pool
from torch_geometric.utils import scatter, segment

from bcos.modules import BcosLinear


class FragmentPooling(Module):

    def __init__(
        self,
        input_channels,
        hidden_channels=None,
        fragment_reduce: Literal["sum", "mean", "max"] = "mean",
        pre_decode_reduce: Literal["sum", "mean", "max"] = "sum",
        transform: Literal["linear", "bcos"] = "bcos",
        max_out: int = 1,
        b: int = 2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else input_channels * 2
        )
        self.fragment_reduce = fragment_reduce
        self.pre_decode_reduce = pre_decode_reduce
        if transform == "linear":
            self.encoder = Sequential(
                *(Linear(self.input_channels, self.hidden_channels), Sigmoid())
            )
            self.decoder = Sequential(
                *(Linear(self.hidden_channels, self.input_channels), Sigmoid())
            )
        elif transform == "bcos":
            self.encoder = BcosLinear(
                self.input_channels, self.hidden_channels, max_out=max_out, b=b
            )
            self.decoder = BcosLinear(
                self.hidden_channels, self.input_channels, max_out=max_out, b=b
            )

    def forward(
        self,
        x: Tensor,
        fragment_index: Tensor,
        fragment_ptr: Tensor,
    ):
        # (Nx, d) -> (Nf, d)
        x_fragment = scatter(x, fragment_index, reduce=self.fragment_reduce)
        x_fragment = self.encoder(x_fragment)
        # (Nf, d) -> (Nf, h) -> (1, h)
        x_molecule = segment(x_fragment, fragment_ptr, reduce=self.pre_decode_reduce)
        return self.decoder(x_molecule)
