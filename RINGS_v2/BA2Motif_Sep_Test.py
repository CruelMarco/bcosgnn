import torch
import torch.nn.functional as F
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from rings.perturbations import EmptyGraph, RandomGraph, CompleteGraph, RandomFeatures
from rings.complementarity import ComplementarityFunctor, FrobeniusMatrixNormComparator


def add_degree_one_hot_features(data, max_degree: int):
    data = data.clone()
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)
    deg = deg.clamp(max=max_degree)
    data.x = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
    return data


def compute_max_degree(dataset, limit: int | None = None) -> int:
    n = len(dataset) if limit is None else min(len(dataset), limit)
    max_deg = 0
    for i in range(n):
        d = dataset[i]
        if d.edge_index.numel() == 0:
            continue
        deg = degree(d.edge_index[0], num_nodes=d.num_nodes)
        if deg.numel() > 0:
            max_deg = max(max_deg, int(deg.max().item()))
    return max_deg


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).view(-1)  # node-level logits


def _binary_targets(batch) -> torch.Tensor:
    y = batch.y
    if y is None:
        raise RuntimeError("Batch has no labels (batch.y is None).")
    # ExplainerDataset generates node labels: 0=background, >0=motif node
    return (y.view(-1) > 0).float()


def run_p1_test(name, train_loader, test_loader, in_channels: int, perturbation=None, epochs: int = 10):
    model = GNNClassifier(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Training P1 - {name}...")
    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if perturbation is not None:
                batch = perturbation(batch)
            optimizer.zero_grad()
            logits = model(batch)
            y = _binary_targets(batch)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            if perturbation is not None:
                batch = perturbation(batch)
            logits = model(batch)
            y = _binary_targets(batch)
            total_loss += float(criterion(logits, y).item()) * y.numel()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == y).sum().item())
            total += int(y.numel())

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return acc, avg_loss


def apply_perturbation(dataset_list, perturbation):
    if perturbation is None:
        return dataset_list
    return [perturbation(d.clone()) for d in dataset_list]


def _empty_features_keep_dim(data):
    data = data.clone()
    dim = 1 if data.x is None else int(data.x.size(1))
    data.x = torch.zeros((data.num_nodes, dim), dtype=torch.float)
    return data


def run_p2_test(dataset_list):
    print("Computing P2 - Mode Complementarity...")
    comp_functor = ComplementarityFunctor(
        graph_metric="diffusion_distance",
        feature_metric="euclidean",
        comparator=FrobeniusMatrixNormComparator,
        num_steps=1,
    )

    subset = dataset_list[:100]

    # ΔS: structural diversity (remove features)
    subset_ef = [_empty_features_keep_dim(d) for d in subset]
    results_ef = comp_functor(subset_ef, as_dataframe=False)
    gamma_ef = float(results_ef["complementarity"].mean().item())
    delta_s = 1 - abs(1 - 2 * gamma_ef)

    # ΔF: feature diversity (remove edges)
    subset_eg = [EmptyGraph()(d.clone()) for d in subset]
    results_eg = comp_functor(subset_eg, as_dataframe=False)
    gamma_eg = float(results_eg["complementarity"].mean().item())
    delta_f = 1 - abs(1 - 2 * gamma_eg)

    return float(delta_s), float(delta_f)


# 1. LOAD/GENERATE DATASET (BA2Motif)
# In PyG 2.6, BA2Motif is generated via `ExplainerDataset`.
torch.manual_seed(0)
raw_dataset = ExplainerDataset(
    graph_generator="ba",
    motif_generator="house",
    num_motifs=2,
    num_graphs=2000,
    graph_generator_kwargs={"num_nodes": 25, "num_edges": 2},
)
max_deg = compute_max_degree(raw_dataset)

# Build a list dataset with degree one-hot node features
full_dataset = [add_degree_one_hot_features(raw_dataset[i], max_deg) for i in range(len(raw_dataset))]

# Split: 80% train, 20% test
train_idx = int(len(full_dataset) * 0.8)

def make_loaders(dataset_list):
    return (
        DataLoader(dataset_list[:train_idx], batch_size=64, shuffle=True),
        DataLoader(dataset_list[train_idx:], batch_size=64),
    )

in_channels = max_deg + 1

# EXECUTION
print("\n--- RINGS DIAGNOSTIC RESULTS (BA2Motif) ---")
print(f"Graphs: {len(full_dataset)} | max_degree: {max_deg} | in_channels: {in_channels}")

P1_EPOCHS = 5

p1_conditions = [
    ("Original", None),
    ("EmptyGraph (no edges)", EmptyGraph()),
    ("RandomGraph (same |E|)", RandomGraph()),
    ("RandomGraph (shuffle edges)", RandomGraph(shuffle=True)),
    ("CompleteGraph", CompleteGraph()),
    ("RandomFeatures (gaussian)", RandomFeatures(shuffle=False)),
    ("RandomFeatures (shuffle)", RandomFeatures(shuffle=True)),
    ("EmptyFeatures (zeros)", _empty_features_keep_dim),
]

p1_results = []
for condition_name, perturb in p1_conditions:
    perturbed_dataset = apply_perturbation(full_dataset, perturb)
    train_loader, test_loader = make_loaders(perturbed_dataset)
    acc, bce = run_p1_test(condition_name, train_loader, test_loader, in_channels, epochs=P1_EPOCHS)
    p1_results.append((condition_name, acc, bce))

orig_acc, orig_bce = p1_results[0][1], p1_results[0][2]
empty_acc, empty_bce = p1_results[1][1], p1_results[1][2]
ds, df = run_p2_test(full_dataset)

print("\nRESULTS SUMMARY:")
print(f"P1 - Original Acc:           {orig_acc:.4f} (BCE {orig_bce:.4f})")
print(f"P1 - Empty Graph Acc:        {empty_acc:.4f} (BCE {empty_bce:.4f})")
print(f"P2 - Structural Diversity (ΔS): {ds:.4f}")
print(f"P2 - Feature Diversity (ΔF):    {df:.4f}")

print("\nP1 - ACC/BCE ACROSS PERTURBATIONS:")
print(f"{'Condition':30s} {'Acc':>8s} {'BCE':>10s}")
for condition_name, acc, bce in p1_results:
    print(f"{condition_name[:30]:30s} {acc:8.4f} {bce:10.4f}")

print("\n--- TEST MEANINGS & INTERPRETATION (RINGS) ---")
print("1. Performance Separability (P1):")
print("   - Tests whether graph structure is task-relevant by removing edges.")
print("   - If performance drops under EmptyGraph, structure is separable/task-relevant.")
print(
    f"   - Outcome: {'Separable (Good)' if orig_acc > empty_acc + 0.05 else 'Not Separable (Redundant)'}"
)

print("\n2. Mode Complementarity (P2):")
print("   - Measures non-redundancy between structure and node features.")
print(f"   - Structural Diversity (ΔS): {ds:.2f} ({'High' if ds > 0.4 else 'Low'})")
print(f"   - Feature Diversity (ΔF):    {df:.2f} ({'High' if df > 0.4 else 'Low'})")
