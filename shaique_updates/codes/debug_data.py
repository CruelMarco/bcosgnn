import torch
from torch_geometric.datasets import MoleculeNet

def debug_dataset():
    ds = MoleculeNet(root="data/MOLHIV", name="HIV")
    print(f"Dataset has {ds.num_node_features} node features")
    print(f"Dataset has {ds.num_edge_features} edge features")
    
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample has {sample.num_nodes} nodes")
        print(f"Sample node features shape: {sample.x.shape if sample.x is not None else 'None'}")
        print(f"Sample node features dtype: {sample.x.dtype if sample.x is not None else 'None'}")
        print(f"Sample edge features dtype: {sample.edge_attr.dtype if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'None'}")
        print(f"Sample edge features shape: {sample.edge_attr.shape if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'None'}")

if __name__ == "__main__":
    debug_dataset()