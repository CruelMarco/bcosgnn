import torch
import networkx as nx
import random
import os
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

def build_custom_ba2motif(dataset_size=1000, num_base_nodes=20):
    """
    Generates the dataset with GUARANTEED connections between Base and Motif.
    """
    print(f"Generating {dataset_size} graphs...")
    data_list = []
    
    for i in range(dataset_size):
        # 1. Base Graph (Tree-like structure, m=1)
        base_graph = nx.barabasi_albert_graph(n=num_base_nodes, m=1, seed=i)
        
        # 2. Motif Selection
        if i < dataset_size // 2:
            label = 0
            motif = nx.house_graph() 
        else:
            label = 1
            motif = nx.cycle_graph(5)
            
        # 3. Relabel Motif Nodes (to append to end of base)
        motif_relabeled = nx.convert_node_labels_to_integers(
            motif, first_label=num_base_nodes
        )
        
        # 4. Compose
        full_graph = nx.compose(base_graph, motif_relabeled)
        
        # 5. FORCE CONNECTION (Critical Step)
        # Connect random Base node to random Motif node
        u = random.randint(0, num_base_nodes - 1)
        v = random.randint(num_base_nodes, num_base_nodes + 4)
        full_graph.add_edge(u, v)
        
        # 6. Convert to PyG
        data = from_networkx(full_graph)
        
        # 7. Create Masks
        num_total_nodes = full_graph.number_of_nodes()
        
        # Node Mask (Motif = True)
        node_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
        node_mask[num_base_nodes:] = True
        data.node_mask = node_mask
        
        # Edge Mask (Edges inside Motif = True)
        edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
        row, col = data.edge_index
        for edge_idx in range(data.num_edges):
            n1 = row[edge_idx].item()
            n2 = col[edge_idx].item()
            # Only edges strictly INSIDE the motif are marked True
            # The connecting edge is NOT marked True (standard practice)
            if n1 >= num_base_nodes and n2 >= num_base_nodes:
                edge_mask[edge_idx] = True
        data.edge_mask = edge_mask
        
        # 8. Features & Labels
        data.x = torch.ones((num_total_nodes, 1), dtype=torch.float)
        data.y = torch.tensor([label], dtype=torch.long)
        data.id = i
        
        data_list.append(data)

    return data_list

def save_and_visualize():
    # --- 1. Setup Paths ---
    # Define directories relative to where the script is run
    base_dir = os.path.dirname(os.getcwd()) # Go up one level usually
    if base_dir == '': base_dir = '.' # Handle edge case
        
    data_dir = os.path.join(base_dir, 'data', 'Custom_BA2Motif')
    vis_dir = os.path.join(os.getcwd(), 'custom_dataset_vis') # Save images in current folder
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'custom_ba2motif_dataset.pt')
    
    # --- 2. Generate Data ---
    dataset = build_custom_ba2motif()
    
    # --- 3. Save Data ---
    torch.save(dataset, dataset_path)
    print(f"Dataset saved to: {dataset_path}")
    
    # --- 4. Visualize and Save Images ---
    print(f"Saving visualizations to: {vis_dir}")
    
    # We will save the first 5 examples of Class 0 and first 5 of Class 1
    indices_to_plot = [0, 1, 2, 3, 4, 500, 501, 502, 503, 504]
    
    for idx in indices_to_plot:
        data = dataset[idx]
        G = nx.Graph()
        G.add_edges_from(data.edge_index.t().tolist())
        
        # Define Colors
        node_colors = []
        for n in G.nodes():
            if n >= 20: 
                node_colors.append('red')    # Motif
            else: 
                node_colors.append('#A0CBE2') # Base
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42) # Consistent layout
        
        nx.draw(G, pos, 
                node_color=node_colors, 
                with_labels=True,
                node_size=300,
                edge_color='gray')
        
        label_name = "House" if data.y.item() == 0 else "Cycle"
        plt.title(f"Graph {idx} - {label_name} (Connected)")
        
        # Save to file
        file_name = os.path.join(vis_dir, f"graph_{idx}_{label_name}.png")
        plt.savefig(file_name)
        plt.close() # Close memory
        print(f"  - Saved {file_name}")

if __name__ == "__main__":
    save_and_visualize()