import sys
import os
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from bcosgnn.models import BinaryClassifierGNN
from bcosgnn.bcos_gnn import GNNCls
from bcosgnn.data.splits import split_random
from bcosgnn.data import load_dataset, NamedDataset


def train_and_evaluate(model, train_loader, val_loader, test_loader):
    """Trains a model and evaluates it on the test set."""
    trainer = Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader)
    return model, test_results[0]


def get_predictions(model, loader):
    """Get model predictions for a given data loader."""
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(output.sigmoid().cpu())
            labels.append(batch.y.cpu())
    return torch.cat(predictions), torch.cat(labels)


if __name__ == "__main__":
    # 1. Load dataset and create splits
    dataset = load_dataset(NamedDataset.MUTAG, root="data")
    train_idx, val_idx, test_idx = split_random(dataset)

    train_loader = DataLoader(dataset[train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=32)
    test_loader = DataLoader(dataset[test_idx], batch_size=32)

    # 2. Train and evaluate BcosGINE
    print("\nTraining BcosGINE...")
    bcos_gine = BinaryClassifierGNN(
        dataset.num_node_features,
        dataset.num_edge_features,
        hidden_dim=32,
        num_layers=3,
        b=2,
        max_out=2,
        gnn_cls=GNNCls.BCOS_GINE,
    )
    bcos_gine, bcos_results = train_and_evaluate(
        bcos_gine, train_loader, val_loader, test_loader
    )
    print("BcosGINE Test Results:", bcos_results)

    # 3. Get predictions and save them
    bcos_preds, labels = get_predictions(bcos_gine, test_loader)
    
    results = {
        'predictions': bcos_preds,
        'labels': labels
    }
    torch.save(results, "bcos_gine_results.pt")
    print("\nSaved BcosGINE predictions and labels to bcos_gine_results.pt")

    # 4. Generate and save BcosGINE explanations
    print("\nGenerating BcosGINE explanations...")
    explanations = {}
    for i in test_idx:
        data = dataset[i]
        explanation = bcos_gine.explain(data.to(bcos_gine.device))
        smiles = data.smiles if hasattr(data, "smiles") else str(i)
        explanations[smiles] = explanation

    torch.save(explanations, "bcos_explanations.pt")
    print("Saved BcosGINE explanations to bcos_explanations.pt")
