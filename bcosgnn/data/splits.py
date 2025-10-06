import torch
from torch_geometric.data import InMemoryDataset


def split_random(dataset, seed=42, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    torch.manual_seed(seed)
    assert frac_train + frac_val + frac_test == 1
    n = len(dataset)
    indices = torch.randperm(n)
    train_indices = indices[: int(n * frac_train)]
    val_indices = indices[int(n * frac_train) : int(n * (frac_train + frac_val))]
    test_indices = indices[int(n * (frac_train + frac_val)) :]
    return train_indices, val_indices, test_indices
