from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear, Embedding, BatchNorm1d
from sklearn.metrics import roc_auc_score
import os
import torch
import typer
# Add this import
import torch.serialization
import torch_geometric.data.data 

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

class VanillaGNN(torch.nn.Module):
    # ... (Your class definition remains unchanged)
    def __init__(self, hidden_channels):
        super(VanillaGNN, self).__init__()
        self.node_emb = Embedding(100, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x.squeeze())
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()
        x = self.conv4(x, edge_index)
        x = self.bn4(x).relu()
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.view(-1)


_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 3. Replace the original torch.load with our new patched version
torch.load = _patched_torch_load

# Now, when PygGraphPropPredDataset calls torch.load, it will use our version
dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="data")

# 4. (Optional but good practice) Restore the original function
torch.load = _original_torch_load

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

