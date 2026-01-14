import torch
import os
import traceback
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif
from torch_geometric.utils import to_networkx, from_networkx, subgraph

def keep_motif_component(data):
    """
    Finds the connected component containing the motif and removes all other nodes.
    """
    # 1. Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # 2. Get all connected components (list of sets of node indices)
    components = list(nx.connected_components(G))
    
    # 3. Find the component that contains the motif nodes
    # (Motif nodes are where node_mask == 1)
    motif_indices = data.node_mask.nonzero(as_tuple=True)[0].tolist()
    
    target_component = None
    for comp in components:
        # Check if any motif node is in this component
        # We assume the motif itself is connected, so finding one node is enough
        if motif_indices[0] in comp:
            target_component = list(comp)
            break
            
    if target_component is None:
        return None # Should not happen if motif exists

    # 4. Create a Tensor of the nodes we want to keep
    subset = torch.tensor(target_component, dtype=torch.long)
    
    # 5. Use PyG's subgraph function
    # This automatically filters edge_index and re-indexes nodes from 0 to N
    new_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
    
    # 6. Filter the masks and features manually
    new_node_mask = data.node_mask[subset]
    new_x = data.x[subset]
    
    # Edge mask is tricky because 'subgraph' doesn't return the edge mask.
    # We must reconstruct it. 
    # Logic: If an edge exists in the new graph, we check if it was a motif edge.
    # A simple way: Retain the concept that "Motif Edges" connect "Motif Nodes".
    # However, to be precise, we should rely on the original mask.
    # (Simplification: We re-calculate edge mask based on node mask for the specific Cycle/House shapes
    # OR we just map the old edges. The easiest robust way is below:)
    
    # Create a mapping from old_index -> new_index
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(target_component)}
    
    # Filter the original edge_mask
    # We iterate original edges, check if both nodes are in the subset, then keep the mask value
    mask_list = []
    # We need to iterate the *new* edge_index to ensure alignment
    # But since subgraph reorders, alignment is hard.
    
    # ALTERNATIVE STRATEGY for Edge Mask:
    # Re-calculate it. If both source and target are in the motif (node_mask=1), it's a motif edge.
    # Note: This assumes no "internal" non-motif edges exist between motif nodes (true for Cycle/House).
    new_edge_mask = torch.zeros(new_edge_index.shape[1], dtype=torch.long)
    
    for i in range(new_edge_index.shape[1]):
        u, v = new_edge_index[:, i]
        if new_node_mask[u] == 1 and new_node_mask[v] == 1:
            new_edge_mask[i] = 1
            
    # 7. Create new Data object
    new_data = Data(x=new_x, 
                    edge_index=new_edge_index, 
                    y=data.y,
                    node_mask=new_node_mask,
                    edge_mask=new_edge_mask)
                    
    new_data.num_nodes = new_x.shape[0]
    
    return new_data

def generate_custom_dataset(num_graphs, graph_generator, motif_generator, label):
    dataset = []
    
    # We might need to generate MORE than num_graphs because we might skip some
    # if they turn out too small (trivial).
    graphs_generated = 0
    
    while graphs_generated < num_graphs:
        # 1. Generate Base (Likely Disjoint)
        base_data = graph_generator()
        motif_data = motif_generator()
        
        # 2. Naive Merge (Just like your original code, creating disjoint parts)
        base_node_mask = torch.zeros(base_data.num_nodes, dtype=torch.long)
        motif_node_mask = torch.ones(motif_data.num_nodes, dtype=torch.long)
        
        shift = base_data.num_nodes
        motif_edge_index = motif_data.edge_index + shift
        
        # Connect Base to Motif (One random edge)
        # We STILL need to connect it to *something*, otherwise the motif is ALWAYS its own component.
        # We attach it to a random base node. Even if that base node is in a small disjoint island,
        # we will keep that island and discard the rest.
        source = torch.randint(0, base_data.num_nodes, (1,)).item()
        target = torch.randint(base_data.num_nodes, base_data.num_nodes + motif_data.num_nodes, (1,)).item()
        
        row = torch.tensor([source, target], dtype=torch.long)
        col = torch.tensor([target, source], dtype=torch.long)
        connect_edge = torch.stack([row, col], dim=0)
        
        final_edge_index = torch.cat([base_data.edge_index, motif_edge_index, connect_edge], dim=1)
        final_node_mask = torch.cat([base_node_mask, motif_node_mask], dim=0)
        
        # Features
        total_nodes = base_data.num_nodes + motif_data.num_nodes
        x = torch.ones((total_nodes, 10), dtype=torch.float)
        
        temp_data = Data(x=x, edge_index=final_edge_index, y=torch.tensor([label]), node_mask=final_node_mask)
        temp_data.num_nodes = total_nodes

        # 3. APPLY THE FILTER: Delete disjoint parts
        clean_data = keep_motif_component(temp_data)
        
        # 4. (Optional) Filter out trivial graphs
        # If the graph has < 10 nodes, it's basically just the motif. Let's skip it to keep data hard.
        if clean_data.num_nodes < 10:
            continue

        dataset.append(clean_data)
        graphs_generated += 1
        
    return dataset

def visualize_graph(data, label, index, output_dir="graph_visualizations"):
    try:
        G = to_networkx(data, to_undirected=True)
        node_colors = ['red' if data.node_mask[i] == 1 else '#A0CBE2' for i in range(data.num_nodes)]
        
        motif_edge_indices = []
        if hasattr(data, 'edge_mask'):
            motif_edge_indices = data.edge_index[:, data.edge_mask.bool()].T.tolist()

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=80, width=0.8, edge_color='gray', alpha=0.6)
        if motif_edge_indices:
            nx.draw_networkx_edges(G, pos, edgelist=motif_edge_indices, edge_color='red', width=3.0)
                               
        plt.title(f"Graph #{index} - Nodes: {data.num_nodes}", fontsize=16)
        plt.savefig(os.path.join(output_dir, f"graph_{index}_class_{label}.png"))
        plt.close()
    except Exception as e:
        traceback.print_exc()

def motif_masks_dataset():
    print("Generating Cleaned BA2Motif (Deleting Disjoint Parts)...")
    save_dir = os.path.join(os.path.dirname(__file__), ".." ,"data", "Custom_BA2Motif")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("graph_visualizations", exist_ok=True)
    
    # Class 0
    d0 = generate_custom_dataset(500, BAGraph(num_nodes=25, num_edges=1), CycleMotif(5), 0)
    # Class 1
    d1 = generate_custom_dataset(500, BAGraph(num_nodes=25, num_edges=1), HouseMotif(), 1)
    
    full_dataset = d0 + d1
    torch.save(full_dataset, os.path.join(save_dir, "custom_ba2motif_dataset.pt"))
    
    # Visualize
    for i, data in enumerate(full_dataset):
        if i % 100 == 0: # Plot every 100th graph
            visualize_graph(data, data.y.item(), i)
            print(f"Plotting graph {i} (Nodes: {data.num_nodes})")

if __name__ == "__main__":
    motif_masks_dataset()