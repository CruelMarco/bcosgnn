import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np

# 1. Define a minimal generic class to load the saved .pt files directly
class SimpleZincDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".") # Dummy root
        # Load the (data, slices) tuple saved previously
        self.data, self.slices = torch.load(pt_path, weights_only=False)

# 2. The reviewer directly loads the pre-split subsets! 
train_dataset = SimpleZincDataset("exported_zinc_di_halo_dataset/train.pt")
val_dataset   = SimpleZincDataset("exported_zinc_di_halo_dataset/val.pt")
test_dataset  = SimpleZincDataset("exported_zinc_di_halo_dataset/test.pt")

full_dataset = SimpleZincDataset("exported_zinc_di_halo_dataset/full_dataset.pt")
train_idx = np.loadtxt("exported_zinc_di_halo_dataset/train_ids.txt", dtype=int)
train_dataset = full_dataset[train_idx]

# 3. Pass the datasets directly into PyG DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train graphs: {len(train_dataset)}")
print(f"Val graphs:   {len(val_dataset)}")
print(f"Test graphs:  {len(test_dataset)}")