import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

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

    # 2. Train and evaluate Vanilla GINE
    print("Training Vanilla GINE...")
    vanilla_gine = BinaryClassifierGNN(
        dataset.num_node_features,
        dataset.num_edge_features,
        hidden_dim=32,
        num_layers=3,
        gnn_cls=GNNCls.GINE,
    )
    vanilla_gine, vanilla_results = train_and_evaluate(
        vanilla_gine, train_loader, val_loader, test_loader
    )
    print("Vanilla GINE Test Results:", vanilla_results)

    # 3. Train and evaluate BcosGINE
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

    # 4. Get predictions for ROC curve
    vanilla_preds, labels = get_predictions(vanilla_gine, test_loader)
    bcos_preds, _ = get_predictions(bcos_gine, test_loader)

    # 5. Calculate AUC-ROC
    vanilla_auc = roc_auc_score(labels, vanilla_preds)
    bcos_auc = roc_auc_score(labels, bcos_preds)
    print(f"\nVanilla GINE AUC: {vanilla_auc:.4f}")
    print(f"BcosGINE AUC: {bcos_auc:.4f}")

    # 6. Plot ROC curves
    fpr_vanilla, tpr_vanilla, _ = roc_curve(labels, vanilla_preds)
    fpr_bcos, tpr_bcos, _ = roc_curve(labels, bcos_preds)

    plt.figure()
    plt.plot(
        fpr_vanilla,
        tpr_vanilla,
        label=f"Vanilla GINE (AUC = {vanilla_auc:.2f})",
    )
    plt.plot(fpr_bcos, tpr_bcos, label=f"BcosGINE (AUC = {bcos_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.savefig("roc_comparison.png")
    print("\nSaved ROC curve comparison plot to roc_comparison.png")

    # 7. Generate and save BcosGINE explanations
    print("\nGenerating BcosGINE explanations...")
    explanations = {}
    for i in test_idx:
        data = dataset[i]
        explanation = bcos_gine.explain(data.to(bcos_gine.device))
        smiles = data.smiles if hasattr(data, "smiles") else str(i)
        explanations[smiles] = explanation

    torch.save(explanations, "bcos_explanations.pt")
    print("Saved BcosGINE explanations to bcos_explanations.pt")
