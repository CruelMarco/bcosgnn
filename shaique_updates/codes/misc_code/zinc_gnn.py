import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, L1Loss
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import csv
import yaml
import subprocess
from datetime import datetime


def get_git_commit():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "N/A"
    return commit


#### Data set Directory ####

data_dir = "data"
dataset_path = os.path.join(data_dir, "ZINC")
print(f"Dataset will be downloaded to/loaded from: {dataset_path}")
train_dataset = ZINC(root=dataset_path, subset=True, split="train")
val_dataset = ZINC(root=dataset_path, subset=True, split="val")
test_dataset = ZINC(root=dataset_path, subset=True, split="test")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

##### Define the GNN Model #####


class VanillaGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(VanillaGNN, self).__init__()
        self.node_emb = Embedding(28, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x.squeeze())
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        graph_x = global_add_pool(x, batch)
        return self.mlp(graph_x)


# --- 3. Define the Training and Evaluation Loops ---
def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        error = criterion(out.squeeze(), data.y)
        total_error = total_error + error.item() * data.num_graphs
    return total_error / len(loader.dataset)


#### Experiment name and directory definition ####
experiment_name = input("Enter the experiment name (e.g., vanilla_gnn_regression): ")
if not experiment_name:
    experiment_name = "gnn_experiment"

#### Check if we are in EXPERIMENT or PROD mode ####
use_timestamp = input(
    f"Create a unique timestamped folder for '{experiment_name}'? (y/n) [default: y]: "
)

if use_timestamp.lower() == "n":  # EXPERIMENT MODE: use the experiment name directly
    experiment_id = experiment_name
    print(
        "EXPERIMENT MODE: Using fixed folder. This run will overwrite previous prototype runs."
    )
else:  # RROD Mode: add a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    print("PROD MODE: Creating unique timestamped folder for this run.")


#### Defining the specific directory for this experiment files ####
parent_dir = "experiments"
experiment_dir = os.path.join(parent_dir, experiment_id)
os.makedirs(experiment_dir, exist_ok=True)

#### Defining all file paths to be inside the experiment directory
model_save_path = os.path.join(experiment_dir, "best_model.pth")
config_save_path = os.path.join(experiment_dir, "hparams.yaml")
plot_save_path = os.path.join(experiment_dir, "train_val_mae_plot.png")
results_csv_path = os.path.join(experiment_dir, "results.csv")

print(f"\nThe files pertaining to this experiment will be saved in: {experiment_dir}")

#### Define hyperparameters and save them to a YAML file ####
hparams = {
    "experiment_id": experiment_id,
    "model_name": "VanillaGNN",
    "dataset": "ZINC_SMALL",
    "seed": 42,
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "hidden_channels": 64,
    "batch_size": 256,
    "epochs": 101,
}

with open(config_save_path, "w") as f:
    yaml.dump(hparams, f, indent=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(hparams["seed"])
np.random.seed(hparams["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams["seed"])

print(f"\nUsing device: {device}")

# TODO add control flow
# add an argument/option to choose the model type
model = VanillaGNN(hidden_channels=hparams["hidden_channels"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
criterion = L1Loss()

#### Result CSV File Setup ####
csv_header = [
    "experiment_id",
    "dataset",
    "split",
    "model_name",
    "commit",
    "metric_name",
    "metric_value",
    "seed",
    "B",
    "epoch",
]

write_header = not os.path.exists(results_csv_path)
with open(results_csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(csv_header)

git_commit = get_git_commit()

####Training Loop####
best_val_mae = np.inf
best_epoch = -1
train_mae_history, val_mae_history = [], []

for epoch in range(1, hparams["epochs"] + 1):
    train(model, train_loader, optimizer, criterion)
    train_mae = evaluate(model, train_loader, criterion)
    val_mae = evaluate(model, val_loader, criterion)
    train_mae_history.append(train_mae)
    val_mae_history.append(val_mae)
    print(f"Epoch: {epoch:03d}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    #### Saving resilts to results.csv ####
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                experiment_id,
                hparams["dataset"],
                "train",
                hparams["model_name"],
                git_commit,
                "mae",
                f"{train_mae:.4f}",
                hparams["seed"],
                "NaN",
                epoch,
            ]
        )
        writer.writerow(
            [
                experiment_id,
                hparams["dataset"],
                "val",
                hparams["model_name"],
                git_commit,
                "mae",
                f"{val_mae:.4f}",
                hparams["seed"],
                "NaN",
                epoch,
            ]
        )

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch = epoch
        torch.save(model.state_dict(), model_save_path)


#### Plots pertaining to training and validation MAE ####
plt.figure(figsize=(10, 6))
plt.plot(train_mae_history, label="Train MAE")
plt.plot(val_mae_history, label="Validation MAE")
plt.title(f"Train vs. Validation MAE\n({experiment_id})")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend()
plt.grid(True)
plt.savefig(plot_save_path)
print(f"Plot saved to {plot_save_path}")
plt.show()

#### Evaliating the best model on the test set ####
print(f"\nLoading best model from {model_save_path} (saved at epoch {best_epoch})...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        predictions = model(data.x, data.edge_index, data.batch)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(predictions.squeeze().cpu().numpy())

final_test_mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
final_test_r2 = r2_score(y_true, y_pred)
print(f"Final Test MAE: {final_test_mae:.4f}")
print(f"Final Test R2 Score: {final_test_r2:.4f}")

#### Save final results to results.csv ####
with open(results_csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            experiment_id,
            hparams["dataset"],
            "test",
            hparams["model_name"],
            git_commit,
            "mae",
            f"{final_test_mae:.4f}",
            hparams["seed"],
            "NaN",
            best_epoch,
        ]
    )
    writer.writerow(
        [
            experiment_id,
            hparams["dataset"],
            "test",
            hparams["model_name"],
            git_commit,
            "r2",
            f"{final_test_r2:.4f}",
            hparams["seed"],
            "NaN",
            best_epoch,
        ]
    )

print(f"\nResults appended to {results_csv_path}")


# TODO
# as typer command
def run_experiment():
    #
    ...


if __name__ == "__main__":
    ...
