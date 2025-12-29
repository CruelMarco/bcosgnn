import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU

from rings.perturbations import EmptyGraph, RandomGraph, CompleteGraph, RandomFeatures
from rings.complementarity import ComplementarityFunctor, FrobeniusMatrixNormComparator


def make_loaders(train_list, test_list, batch_size: int = 64):
    return (
        DataLoader(train_list, batch_size=batch_size, shuffle=True),
        DataLoader(test_list, batch_size=batch_size),
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
        # Some structure perturbations change edge_index without updating edge_attr.
        # For edge-aware models, keep edge_attr consistent by resetting it when needed.
        if hasattr(d2, "edge_index") and d2.edge_index is not None:
            num_edges = int(d2.edge_index.size(1))
            if not hasattr(d2, "edge_attr") or d2.edge_attr is None:
                d2.edge_attr = torch.zeros((num_edges,), dtype=torch.long)
            else:
                if d2.edge_attr.dim() > 1:
                    d2.edge_attr = d2.edge_attr.view(-1)
                if int(d2.edge_attr.numel()) != num_edges:
                    d2.edge_attr = torch.zeros((num_edges,), dtype=torch.long)
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
    def __init__(self, in_channels: int, hidden_dim: int = 64, num_bond_types: int = 4):
        super().__init__()
        self.node_encoder = Linear(in_channels, hidden_dim)
        self.edge_encoder = torch.nn.Embedding(num_embeddings=num_bond_types, embedding_dim=hidden_dim)

        nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        if x.dim() == 1:
            x = x.view(-1, 1)
        x = x.float()

        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1),), dtype=torch.long, device=edge_index.device)
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.view(-1)
        edge_attr = edge_attr.long()

        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        x = F.relu(self.conv1(x, edge_index, e))
        x = F.relu(self.conv2(x, edge_index, e))
        x = global_mean_pool(x, batch)
        return self.lin(x).view(-1)


def run_p1_test(name, train_list, test_list, in_channels: int, epochs: int = 5):
    model = GNNRegressor(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = make_loaders(train_list, test_list)

    print(f"Training P1 - {name}...")
    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch)
            y = batch.y.view(-1)
            total_mae += float(F.l1_loss(pred, y, reduction="sum").item())
            n += int(y.numel())

    return total_mae / max(n, 1)


def run_p2_test(dataset_list, subset_size: int = 200):
    print("Computing P2 - Mode Complementarity...")
    comp_functor = ComplementarityFunctor(
        graph_metric="diffusion_distance",
        feature_metric="euclidean",
        comparator=FrobeniusMatrixNormComparator,
        num_steps=1,
    )

    subset = dataset_list[: min(len(dataset_list), subset_size)]

    # Ensure x is 2D + float for robust feature distance computation
    subset = [d.clone() for d in subset]
    for d in subset:
        if d.x is not None and d.x.dim() == 1:
            d.x = d.x.view(-1, 1)
        if d.x is not None:
            d.x = d.x.float()

    subset_ef = [empty_features_keep_dim(d) for d in subset]
    results_ef = comp_functor(subset_ef, as_dataframe=False)
    gamma_ef = float(results_ef["complementarity"].mean().item())
    delta_s = 1 - abs(1 - 2 * gamma_ef)

    subset_eg = [EmptyGraph()(d.clone()) for d in subset]
    results_eg = comp_functor(subset_eg, as_dataframe=False)
    gamma_eg = float(results_eg["complementarity"].mean().item())
    delta_f = 1 - abs(1 - 2 * gamma_eg)

    return float(delta_s), float(delta_f)


# 1) LOAD DATASET (ZINC)
train_ds = ZINC(root="./data/ZINC", split="train")
test_ds = ZINC(root="./data/ZINC", split="test")

# Determine input feature dim robustly
sample_x = train_ds[0].x
in_channels = sample_x.shape[1] if sample_x.dim() > 1 else 1

# 2) P1: PERFORMANCE SEPARABILITY across perturbations
TRAIN_SUBSET = 5000
P1_EPOCHS = 10

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

print("\n--- RINGS DIAGNOSTIC RESULTS (ZINC) ---")
print(f"Train graphs: {len(train_ds)} | Test graphs: {len(test_ds)} | in_channels: {in_channels}")

p1_results = []
for condition_name, perturb in p1_conditions:
    train_list = apply_perturbation(train_ds[:TRAIN_SUBSET], perturb)
    test_list = apply_perturbation(test_ds, perturb)
    mae = run_p1_test(condition_name, train_list, test_list, in_channels, epochs=P1_EPOCHS)
    p1_results.append((condition_name, mae))

orig_mae = p1_results[0][1]
empty_mae = p1_results[1][1]

# 3) P2: MODE COMPLEMENTARITY (on subset of test)
ds, df = run_p2_test(list(test_ds), subset_size=200)

print("\nRESULTS SUMMARY:")
print(f"P1 - Original MAE:      {orig_mae:.4f}")
print(f"P1 - Empty Graph MAE:   {empty_mae:.4f}")
print(f"P2 - Structural Diversity (ΔS): {ds:.4f}")
print(f"P2 - Feature Diversity (ΔF):    {df:.4f}")

print("\nP1 - MAE ACROSS PERTURBATIONS:")
print(f"{'Condition':30s} {'MAE':>10s}")
for condition_name, mae in p1_results:
    print(f"{condition_name[:30]:30s} {mae:10.4f}")


print("\n--- RESULT ANALYSIS (ZINC) ---")
mae_by_name = {k: v for k, v in p1_results}
mae_empty_features = mae_by_name.get("EmptyFeatures (zeros)")

structure_reliance = (empty_mae - orig_mae) / max(orig_mae, 1e-12)
feature_reliance = None
if mae_empty_features is not None:
    feature_reliance = (mae_empty_features - orig_mae) / max(orig_mae, 1e-12)

print(f"Structure reliance score (EmptyGraph vs Original): {structure_reliance:+.2%}")
if feature_reliance is not None:
    print(f"Feature reliance score (EmptyFeatures vs Original): {feature_reliance:+.2%}")

print("\nRelative MAE change vs Original:")
print(f"{'Condition':30s} {'ΔMAE':>10s} {'Δ%':>8s}")
for condition_name, mae in p1_results:
    delta = mae - orig_mae
    delta_pct = delta / max(orig_mae, 1e-12)
    print(f"{condition_name[:30]:30s} {delta:10.4f} {delta_pct:8.2%}")

print("\n--- TEST MEANINGS & INTERPRETATION (RINGS) ---")
print("1. Performance Separability (P1):")
print("   - Tests whether graph structure is task-relevant by perturbing edges/features and measuring MAE.")
print(
    f"   - Outcome (structure): {'Separable (Good)' if orig_mae < empty_mae * 0.9 else 'Not Separable (Redundant)'}"
)
print("\n2. Mode Complementarity (P2):")
print("   - Measures non-redundancy between structure and node features.")
print(f"   - Structural Diversity (ΔS): {ds:.2f} ({'High' if ds > 0.4 else 'Low'})")
print(f"   - Feature Diversity (ΔF):    {df:.2f} ({'High' if df > 0.4 else 'Low'})")
