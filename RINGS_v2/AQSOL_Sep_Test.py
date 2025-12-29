import torch
import torch.nn.functional as F
from torch_geometric.datasets import AQSOL
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
from rings.perturbations import EmptyGraph, RandomGraph, CompleteGraph, RandomFeatures
from rings.complementarity import ComplementarityFunctor, FrobeniusMatrixNormComparator
import numpy as np

# 1. LOAD DATASET
dataset = AQSOL(root='./data/AQSOL')
# Split: 80% train, 20% test
train_idx = int(len(dataset) * 0.8)
base_train = dataset[:train_idx]
base_test = dataset[train_idx:]


def _edge_attr_dim_and_dtype(sample):
    ea = getattr(sample, "edge_attr", None)
    if ea is None:
        return 1, torch.float
    if ea.dim() == 1:
        return 1, torch.float
    return int(ea.size(-1)), torch.float


EDGE_ATTR_DIM, EDGE_ATTR_DTYPE = _edge_attr_dim_and_dtype(dataset[0])


def make_loaders(train_list, test_list):
    return (
        DataLoader(train_list, batch_size=64, shuffle=True),
        DataLoader(test_list, batch_size=64),
    )


def apply_perturbation(dataset_list, perturbation):
    if perturbation is None:
        return dataset_list
    out = []
    for d in dataset_list:
        d = d.clone()
        if hasattr(d, "x") and d.x is not None:
            if d.x.dim() == 1:
                d.x = d.x.view(-1, 1)
            d.x = d.x.float()

        d2 = perturbation(d) if callable(perturbation) else perturbation(d)

        # Structure perturbations can change edge_index without updating edge_attr.
        # Keep edge_attr consistent so edge-aware models work.
        if hasattr(d2, "edge_index") and d2.edge_index is not None:
            num_edges = int(d2.edge_index.size(1))
            ea = getattr(d2, "edge_attr", None)
            if ea is None:
                d2.edge_attr = torch.zeros((num_edges, EDGE_ATTR_DIM), dtype=EDGE_ATTR_DTYPE)
            else:
                if ea.dim() == 1:
                    ea = ea.view(-1, 1)
                else:
                    ea = ea.view(ea.size(0), -1)
                if int(ea.size(0)) != num_edges or int(ea.size(1)) != EDGE_ATTR_DIM:
                    ea = torch.zeros((num_edges, EDGE_ATTR_DIM), dtype=EDGE_ATTR_DTYPE)
                d2.edge_attr = ea.to(dtype=EDGE_ATTR_DTYPE)

        out.append(d2)
    return out


def empty_features_keep_dim(data):
    data = data.clone()
    if hasattr(data, "x") and data.x is not None and data.x.dim() == 1:
        data.x = data.x.view(-1, 1)
    dim = 1 if data.x is None else int(data.x.size(1))
    data.x = torch.zeros((data.num_nodes, dim), dtype=torch.float)
    return data

class GNNRegressor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden_dim = 64
        self.node_encoder = Linear(in_channels, hidden_dim)
        self.edge_encoder = Linear(EDGE_ATTR_DIM, hidden_dim)

        nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        
        # FIX: Ensure x is 2D [num_nodes, num_features]
        if x.dim() == 1:
            x = x.view(-1, 1)
        # Handle cases where x might be integer indices (common in molecules)
        x = x.float()

        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), EDGE_ATTR_DIM), dtype=torch.float, device=edge_index.device)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = edge_attr.to(dtype=torch.float)

        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        x = F.relu(self.conv1(x, edge_index, e))
        x = F.relu(self.conv2(x, edge_index, e))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# 3. P1: PERFORMANCE SEPARABILITY TEST
def run_p1_test(name, train_list, test_list, perturbation=None, epochs: int = 10):
    # AQSOL num_features is often 1 (atomic number) or 9 
    # Use actual shape to be safe
    sample_x = dataset[0].x
    in_channels = sample_x.shape[1] if sample_x.dim() > 1 else 1
    
    model = GNNRegressor(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = make_loaders(train_list, test_list)
    
    print(f"Training P1 - {name}...")
    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    total_mae = 0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch)
            total_mae += F.l1_loss(pred.view(-1), batch.y, reduction='sum').item()
            n += int(batch.y.numel())
    
    mae = total_mae / max(n, 1)
    return mae

# 4. P2: MODE COMPLEMENTARITY TEST
def run_p2_test():
    print("Computing P2 - Mode Complementarity...")
    comp_functor = ComplementarityFunctor(
        graph_metric='diffusion_distance', 
        feature_metric='euclidean',
        comparator=FrobeniusMatrixNormComparator,
        num_steps=1 
    )
    # Test on subset for efficiency
    subset = dataset[:100]
    
    # Fix feature dimensions if necessary
    for i, d in enumerate(subset):
        if d.x.dim() == 1:
            d.x = d.x.view(-1, 1)
        d.x = d.x.float()
        if d.x.dim() != 2:
            print(f"Warning: Graph {i} has x shape {d.x.shape}")
    
    # Mode Diversity (Delta) measures internal richness [cite: 165]
    # Delta_S: Structural diversity (remove feature information but keep feature dim)
    subset_ef = [empty_features_keep_dim(d) for d in subset]
    results_ef = comp_functor(subset_ef, as_dataframe=False)
    gamma_ef = float(results_ef['complementarity'].mean().item())
    delta_s = 1 - abs(1 - 2 * gamma_ef)
    
    # Delta_F: Feature diversity (using EmptyGraph to isolate features)
    subset_eg = [EmptyGraph()(d.clone()) for d in subset]
    results_eg = comp_functor(subset_eg, as_dataframe=False)
    gamma_eg = float(results_eg['complementarity'].mean().item())
    delta_f = 1 - abs(1 - 2 * gamma_eg)

    return float(delta_s), float(delta_f)

# EXECUTION
print("\n--- RINGS DIAGNOSTIC RESULTS ---")

P1_EPOCHS = 5

p1_conditions = [
    ("Original", None),
    ("EmptyGraph (no bonds)", EmptyGraph()),
    ("RandomGraph (same |E|)", RandomGraph()),
    ("RandomGraph (shuffle edges)", RandomGraph(shuffle=True)),
    ("CompleteGraph", CompleteGraph()),
    ("RandomFeatures (gaussian)", RandomFeatures(shuffle=False)),
    ("RandomFeatures (shuffle)", RandomFeatures(shuffle=True)),
    ("EmptyFeatures (zeros)", empty_features_keep_dim),
]

p1_results = []
for condition_name, perturb in p1_conditions:
    train_list = apply_perturbation(base_train, perturb)
    test_list = apply_perturbation(base_test, perturb)
    mae = run_p1_test(condition_name, train_list, test_list, epochs=P1_EPOCHS)
    p1_results.append((condition_name, mae))

orig_mae = p1_results[0][1]
empty_mae = p1_results[1][1]
ds, df = run_p2_test()

print("\nRESULTS SUMMARY:")
print(f"P1 - Original MAE:      {orig_mae:.4f}")
print(f"P1 - Empty Graph MAE:   {empty_mae:.4f}")
print(f"P2 - Structural Diversity (ΔS): {ds:.4f}")
print(f"P2 - Feature Diversity (ΔF):    {df:.4f}")

print("\nP1 - MAE ACROSS PERTURBATIONS:")
print(f"{'Condition':30s} {'MAE':>10s}")
for condition_name, mae in p1_results:
    print(f"{condition_name[:30]:30s} {mae:10.4f}")

# PRINT TEST MEANINGS
print("\n--- TEST MEANINGS & INTERPRETATION ---")
print("1. Performance Separability (P1):")
print("   - Measures if a mode (structure or features) is task-relevant")
print(f"   - Outcome: {'Separable (Good)' if orig_mae < empty_mae * 0.9 else 'Not Separable (Redundant)'}")
print("   - Meaning: If MAE doesn't increase when bonds are removed, GNN is likely just using features.")

print("\n2. Mode Complementarity (P2):")
print("   - Measures if structure and features provide non-redundant information.")
print(f"   - Structural Diversity (ΔS): {ds:.2f} ({'High' if ds > 0.4 else 'Low'})")
print("   - Meaning: High diversity means the graph shapes vary significantly across the dataset.")