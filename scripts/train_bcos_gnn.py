from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer

from bcosgnn.models import BinaryClassifierGNN
from bcosgnn.bcos_gnn import GNNCls
from bcosgnn.data.splits import split_random
from bcosgnn.data import load_dataset, NamedDataset


if __name__ == "__main__":
    dataset = load_dataset(NamedDataset.DEBUG, root="data")
    train_idx, val_idx, test_idx = split_random(dataset)
    model = BinaryClassifierGNN(
        dataset.num_node_features,
        dataset.num_edge_features,
        hidden_dim=32,
        num_layers=3,
        b=2,
        max_out=2,
        gnn_cls=GNNCls.BCOS_MPNN,
        fragment_pooling=False,
    )
    trainer = Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(
        model,
        DataLoader(
            dataset[train_idx],
            batch_size=32,
            shuffle=True,
        ),
    )
    test_loader = DataLoader(dataset[test_idx], batch_size=32, shuffle=False)
    trainer.test(model, test_loader)
    trainer.save_checkpoint("model.ckpt")
