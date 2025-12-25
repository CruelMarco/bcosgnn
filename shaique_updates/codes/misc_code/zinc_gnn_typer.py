import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, L1Loss, BatchNorm1d
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os
import csv
import yaml
import subprocess
from datetime import datetime
import typer
from typing_extensions import Annotated

# Import the shared argument definitions
from shared_args import (
    experiment_name_arg,
    prod_mode_arg,
    data_dir_arg,
    hidden_channels_arg,
    learning_rate_arg,
    epochs_arg,
    batch_size_arg,
    seed_arg,
)

app = typer.Typer()


def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = 'N/A'
    return commit

class VanillaGNN(torch.nn.Module):

    def __init__(self, hidden_channels):

        super(VanillaGNN, self).__init__()

        self.node_emb = Embedding(28, hidden_channels)

        self.bn1 = BatchNorm1d(hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)

        self.bn2 = BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.bn2 = BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.bn3 = BatchNorm1d(hidden_channels)

        self.conv4 = GCNConv(hidden_channels, hidden_channels)

        self.bn4 = BatchNorm1d(hidden_channels)

        self.mlp = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x.squeeze())
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()
        x = self.conv4(x , edge_index)
        x = self.bn4(x).relu()
        graph_x = global_add_pool(x, batch)
        return self.mlp(graph_x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        error = criterion(out.squeeze(), data.y)
        total_error += error.item() * data.num_graphs
    return total_error / len(loader.dataset)


@app.command()
def main(

    experiment_name: experiment_name_arg = "vanilla_gnn_regression",
    prod_mode: prod_mode_arg = True,
    data_dir: data_dir_arg = "data",
    hidden_channels: hidden_channels_arg = 128,
    learning_rate: learning_rate_arg = 0.0001, # Using a slightly higher LR
    epochs: epochs_arg = 101,
    batch_size: batch_size_arg = 128,
    seed: seed_arg = 42,
):
  

    if prod_mode:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_id = f"{experiment_name}_{timestamp}"
    else:
        experiment_id = experiment_name
    parent_dir = 'experiments'
    experiment_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    model_save_path = os.path.join(experiment_dir, "best_model.pth")
    config_save_path = os.path.join(experiment_dir, "hparams.yaml")
    plot_save_path = os.path.join(experiment_dir, "train_val_mae_plot.png")
    results_csv_path = os.path.join(experiment_dir, 'results.csv')
    print(f"\nExperiment files will be saved in: {experiment_dir}")
    hparams = { 'experiment_id': experiment_id, 'model_name': 'VanillaGNN', 'dataset': 'ZINC_SMALL', 'seed': seed, 'learning_rate': learning_rate, 'optimizer': 'Adam', 'hidden_channels': hidden_channels, 'batch_size': batch_size, 'epochs': epochs, 'production_mode': prod_mode }
    with open(config_save_path, 'w') as f:
        yaml.dump(hparams, f, indent=4)

    dataset_path = os.path.join(data_dir, 'ZINC')
    train_dataset = ZINC(root=dataset_path, subset=True, split='train')
    val_dataset = ZINC(root=dataset_path, subset=True, split='val')
    test_dataset = ZINC(root=dataset_path, subset=True, split='test')
    
    mean = train_dataset.data.y.mean()
    std = train_dataset.data.y.std()
    print(f"\nNormalizing targets with mean={mean:.4f} and std={std:.4f} from the training set.")
    train_dataset.data.y = (train_dataset.data.y - mean) / std
    val_dataset.data.y = (val_dataset.data.y - mean) / std
    test_dataset.data.y = (test_dataset.data.y - mean) / std
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'])

    #  Model, Optimizer, and Loss (Your setup with L1Loss is correct)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Using seed: {hparams['seed']} for reproducibility")
    torch.manual_seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hparams['seed'])

    model = VanillaGNN(hidden_channels=hparams['hidden_channels']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    criterion = L1Loss()

    # (CSV Logging Setup is correct)
    # ...

    #  Training and Validation (Your logic is correct)
    best_val_loss = float('inf')
    best_epoch = -1
    train_loss_history, val_loss_history = [], []

    for epoch in range(1, hparams['epochs']):
        train(model, train_loader, optimizer, criterion, device)
        train_loss = evaluate(model, train_loader, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        # This is the L1 Loss on the NORMALIZED targets
        print(f'Epoch: {epoch:03d}, Train L1 Loss: {train_loss:.4f}, Val L1 Loss: {val_loss:.4f}')
        


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)

    # (Plotting is correct, but I'll adjust the label for clarity)
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train L1 Loss')
    plt.plot(val_loss_history, label='Validation L1 Loss')
    plt.title(f'Train vs. Validation Loss\n({experiment_id})')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (on normalized targets)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")


    print(f"\nLoading best model from {model_save_path} (saved at epoch {best_epoch})...")
    model.load_state_dict(torch.load(model_save_path))

    y_true_orig, y_pred_orig = [], []
    for data in test_loader:
        data = data.to(device)
        predictions_norm = model(data.x, data.edge_index, data.batch)
        
        y_pred_unnormalized = predictions_norm.squeeze().detach().cpu() * std + mean
        y_true_unnormalized = data.y.cpu() * std + mean
        
        y_true_orig.extend(y_true_unnormalized.numpy())
        y_pred_orig.extend(y_pred_unnormalized.numpy())

    final_test_mae = np.mean(np.abs(np.array(y_true_orig) - np.array(y_pred_orig)))
    final_test_r2 = r2_score(y_true_orig, y_pred_orig)
    
    print(f"Final Test MAE (un-normalized): {final_test_mae:.4f}")
    print(f"Final Test R2 Score: {final_test_r2:.4f}")


if __name__ == "__main__":
    app()