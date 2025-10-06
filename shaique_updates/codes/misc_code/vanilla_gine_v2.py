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
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Module, Embedding
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm

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

class VanillaGINE(torch.nn.Module):
    
    ## define the Graph Isomorphism Network with Edge Features (GINE) model here
    ## Args: num_tasks (int): Number of output tasks (e.g., 1 for binary classification).
    ##       emb_dim (int): Dimensionality of hidden embeddings.
    ##       drop_ratio (float): Dropout probability.
    def __init__(self, num_tasks, emb_dim = 300 , drop_ratio = 0.5):
        super(VanillaGINE, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.atom_encoder = Embedding(119, emb_dim)
        self.bond_encoder = Embedding(5, emb_dim)


        # Layer 1
        nn1 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))

        self.conv1 = GINEConv(nn=nn1, train_eps=True)
        
        self.bn1 = BatchNorm1d(emb_dim)


        # Layer 2
        nn2 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))

        self.conv2 = GINEConv(nn=nn2, train_eps=True)

        self.bn2 = BatchNorm1d(emb_dim)

        # Layer 3
        nn3 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))

        self.conv3 = GINEConv(nn=nn3, train_eps=True)

        self.bn3 = BatchNorm1d(emb_dim)

        # Layer 4
        nn4 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))

        self.conv4 = GINEConv(nn=nn4, train_eps=True)

        self.bn4 = BatchNorm1d(emb_dim)

        # Layer 5
        nn5 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))

        self.conv5 = GINEConv(nn=nn5, train_eps=True)

        self.bn5 = BatchNorm1d(emb_dim)

        # Output layer
        self.graph_pred_linear = Linear(emb_dim, num_tasks)
    
    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        # Encode node and edge features
        h = self.atom_encoder(x.squeeze())
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        
        # --- Apply layers sequentially ---
        h = F.relu(self.bn1(self.conv1(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn3(self.conv3(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn4(self.conv4(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn5(self.conv5(h, edge_index, edge_attr=edge_embedding)))

        # Graph-level pooling
        h_graph = global_add_pool(h, batch)
        
        # Dropout and final prediction
        h_graph = F.dropout(h_graph, p=self.drop_ratio, training=self.training)
        output = self.graph_pred_linear(h_graph)

        return output
    
def train(mode, device , model, loader, optimizer , criterion):
    model.train()
    total

     