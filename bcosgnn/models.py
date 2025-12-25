from collections import defaultdict
from typing import Any, Dict, Literal, Mapping

import torch
import bcos
import torch.nn.functional as F
from bcos.common import BcosUtilMixin
from bcos.modules import BcosLinear
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.nn import Dropout, Linear, ReLU, Sequential
from torchmetrics import AUROC, Accuracy, Precision, Recall
from torch_geometric.nn.aggr import SumAggregation

from bcosgnn.bcos_gnn import BcosGNN, BcosReadout, resolve_gnn_cls, TGNNKey, GNNCls
from bcosgnn.fragment_pooling import FragmentPooling


class MLP(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
        base_transform: Literal["linear", "bcos"] = "linear",
        **kwargs,
    ):
        super().__init__()
        if base_transform == "linear":
            new_layer = lambda in_dim, out_dim: [Linear(in_dim, out_dim), ReLU()]
        elif base_transform == "bcos":
            new_layer = lambda in_dim, out_dim: [
                BcosLinear(in_dim, out_dim, max_out=kwargs["max_out"], b=kwargs["b"])
            ]
        layers = new_layer(input_dim, hidden_dim)
        for i in range(num_hidden_layers):
            layers = layers + new_layer(hidden_dim, hidden_dim)
            if dropout > 0:
                layers = layers + [Dropout(dropout)]
        layers = layers + new_layer(hidden_dim, output_dim)
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLPClassifier(LightningModule, BcosUtilMixin):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = MLP(input_dim, hidden_dim, base_transform="bcos", b=2, max_out=1)
        self.recall = Recall("binary")
        self.precision = Precision("binary")
        self.auroc = AUROC("binary")
        self.train_acc = Accuracy("binary")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        self.train_acc.update(y_hat, y.long())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        self.recall.update(y_hat, y)
        self.precision.update(y_hat, y)
        self.auroc.update(y_hat, y)

    def on_test_epoch_end(self) -> None:
        self.log("test/recall", self.recall.compute(), prog_bar=True)
        self.log("test/precision", self.precision.compute(), prog_bar=True)
        self.log("test/auroc", self.auroc.compute(), prog_bar=True)


class BaseDiscriminativeGNN(BcosUtilMixin, LightningModule):
    def __init__(
        self,
        node_size: int,
        edge_size: int,
        hidden_dim: int,
        num_layers: int,
        gnn_cls: TGNNKey = GNNCls.BCOS_MPNN,
        fragment_pooling: bool = True,
        b: float = 2,
        max_out: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        out_channels: int = 1,
        dropout: float = 0.5,
        node_classification: bool = False,
    ):
        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fragment_pooling = fragment_pooling
        self.lr = lr
        self.weight_decay = weight_decay
        self.out_channels = out_channels
        super().__init__()
        self.save_hyperparameters()
        gnn_cls = resolve_gnn_cls(gnn_cls)
        self.gnn_encoder = gnn_cls(
            node_size, edge_size, hidden_dim, num_layers, b=b, max_out=max_out
        )
        self.pooling_layer = None
        if self.fragment_pooling:
            self.pooling_layer = FragmentPooling(
                hidden_dim, b=b, max_out=max_out, transform="bcos"
            )

        if(node_classification):
            self.readout = BcosReadout(
                hidden_dim,
                out_channels=out_channels,
                agg=None,
                b=b,
                max_out=max_out,
                dropout=dropout
            )
        else:
            self.readout = BcosReadout(
                hidden_dim,
                out_channels=out_channels,
                agg=None if self.fragment_pooling else "sum",
                b=b,
                max_out=max_out,
                dropout=dropout
            )

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(
        self,
        x: Tensor,
        edge_index,
        edge_attr=None,
        fragment_index=None,
        fragment_ptr=None,
        batch=None,
    ) -> Tensor:
        x = self.gnn_encoder(x, edge_index, edge_attr, batch)
        if self.fragment_pooling:
            x = self.pooling_layer(x, fragment_index, fragment_ptr)
        return self.readout(x, batch)

    def compute_fragment_utils(self, data):
        if self.pooling_layer is None:
            return None, None
        zero = torch.zeros(1, dtype=torch.long, device=data.num_fragments.device)
        fragment_ptr = torch.cat((zero, data.num_fragments))
        fragment_ptr = torch.cumsum(fragment_ptr, dim=0)
        return data.fragment_index, fragment_ptr

    def criterion(self, data, y_hat, y):
        raise NotImplementedError

    def step(self, data, batch_idx):
        fragment_index, fragment_ptr = self.compute_fragment_utils(data)
        y_hat = self(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch=data.batch,
            fragment_index=fragment_index,
            fragment_ptr=fragment_ptr,
        )

        y_hat = y_hat.squeeze()
        y = data.y.squeeze()
        loss = self.criterion(data, y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def select_prediction_logit(self, data, out, idx):
        raise NotImplementedError

    def explain(self, data, idx=0) -> Dict[str, Any]:
        if(data.edge_attr is not None):
            with torch.enable_grad(), self.explanation_mode():
                input_x = data.x.clone().requires_grad_()
                input_edge_attr = data.edge_attr.clone().requires_grad_()
                fragment_index, fragment_ptr = self.compute_fragment_utils(data)
                out = self(
                    input_x,
                    data.edge_index,
                    input_edge_attr,
                    batch=data.batch,
                    fragment_index=fragment_index,
                    fragment_ptr=fragment_ptr,
                )

                if(self.out_channels > 1):
                    out = out.max(1)[0]

                prediction_logit, prediction_info = self.select_prediction_logit(data, out, idx)
                prediction_logit.backward(inputs=[input_x, input_edge_attr])

            result = defaultdict(dict)
            for inp, key in zip([input_x, input_edge_attr], ["x", "edge_attr"]):
                result[key]["dynamic_linear_weights"] = inp.grad
                result[key]["contribution_map"] = inp * inp.grad
            result["predicted_logit"] = prediction_logit
            result["prediction_info"] = prediction_info
            result["data"] = data
        else:
            with torch.enable_grad(), self.explanation_mode():
                input_x = data.x.clone().requires_grad_()
                fragment_index, fragment_ptr = self.compute_fragment_utils(data)
                out = self(
                    input_x,
                    data.edge_index,
                    batch=data.batch,
                    fragment_index=fragment_index,
                    fragment_ptr=fragment_ptr,
                )
                
                if(self.out_channels > 1):
                    out = out.max(1)[0]

                prediction_logit, prediction_info = self.select_prediction_logit(data, out, idx)
                prediction_logit.backward(inputs=[input_x])
            result = defaultdict(dict)
            for inp, key in zip([input_x], ["x"]):
                result[key]["dynamic_linear_weights"] = inp.grad
                result[key]["contribution_map"] = inp * inp.grad
            result["predicted_logit"] = prediction_logit
            result["prediction_info"] = prediction_info
            result["data"] = data
            
        return result


class BinaryClassifierGNN(BaseDiscriminativeGNN):
    def __init__(
        self,
        node_size,
        edge_size,
        hidden_dim,
        num_layers,
        gnn_cls=GNNCls.BCOS_MPNN,
        fragment_pooling=True,
        b=2,
        max_out=1,
        lr=3e-4,
        weight_decay=1e-4,
        out_channels: int = 1,
        dropout = 0.5,
        node_classification=False,
    ):
        # assert out_channels == 1
        super().__init__(
            node_size,
            edge_size,
            hidden_dim,
            num_layers,
            gnn_cls,
            fragment_pooling,
            b,
            max_out,
            lr,
            weight_decay,
            out_channels,
            dropout,
            node_classification
        )
        self.recall = Recall("binary")
        self.precision = Precision("binary")
        self.auroc = AUROC("binary")
        self.accuracy = Accuracy("binary")
        self.train_acc = Accuracy("binary")

    def criterion(self, data, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

    def select_prediction_logit(self, data, out, idx=0):
        return out[idx], {"prediction_type": "binary"}

    def training_step(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Mapping[str, Any] | None:
        info = self.step(*args, **kwargs)
        self.train_acc.update(info["y_hat"], info["y"])
        self.log("train/loss", info["loss"], prog_bar=True)
        return info["loss"]

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)

    def test_step(self, data, batch_idx):
        info = self.step(data, batch_idx)
        self.recall.update(info["y_hat"], info["y"])
        self.precision.update(info["y_hat"], info["y"])
        self.auroc.update(info["y_hat"], info["y"])
        self.accuracy.update(info["y_hat"], info["y"])

    def on_test_epoch_end(self) -> None:
        self.log("test/recall", self.recall.compute(), prog_bar=True)
        self.log("test/precision", self.precision.compute(), prog_bar=True)
        self.log("test/auroc", self.auroc.compute(), prog_bar=True)
        self.log("test/acc", self.accuracy.compute(), prog_bar=True)
