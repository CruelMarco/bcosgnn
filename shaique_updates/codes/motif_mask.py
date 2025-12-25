import torch
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif
from torch.utils.data import ConcatDataset
import traceback  # Import the traceback module

import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
# --- End New Imports ---

def visualize_graph(data, label, index, output_dir="graph_visualizations"):
    """
    Visualizes a torch_geometric graph with its ground-truth motif highlighted
    and saves it to a file.
    """
    try:
        # Convert the torch_geometric Data object to a networkx Graph
        # to_undirected=True makes plotting simpler
        G = to_networkx(data, to_undirected=True)
        
        # Determine node colors: red if in motif, blue otherwise
        node_colors = []
        for i in range(data.num_nodes):
            if data.node_mask[i] == 1:
                node_colors.append('red')
            else:
                node_colors.append('#A0CBE2') # A light blue
                
        # Get the list of edges that are part of the motif
        # data.edge_mask is a boolean tensor for edges in data.edge_index
        # We find the indices where the mask is true and get those edges
        motif_edge_indices = data.edge_index[:, data.edge_mask.bool()].T.tolist()
        
        # Create a new figure
        plt.figure(figsize=(10, 10))
        
        # Use a spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # 1. Draw all nodes and "background" edges (not in motif)
        nx.draw(G, pos,
                with_labels=False,
                node_color=node_colors,
                node_size=50,
                width=0.5,        # Thin lines for non-motif edges
                edge_color='gray',
                alpha=0.6)        # Faintly
                
        # 2. Draw the motif edges on top, in red and thicker
        nx.draw_networkx_edges(G, pos,
                               edgelist=motif_edge_indices,
                               edge_color='red',
                               width=2.0)
                               
        motif_name = "Cycle" if label == 0 else "House"
        plt.title(f"Graph #{index} - Class {label} ({motif_name} Motif)", fontsize=20)
        
        # Save the figure
        filename = os.path.join(output_dir, f"graph_{index}_class_{label}.png")
        plt.savefig(filename)
        plt.close() # Close the figure to save memory
        
        print(f"    - Saved plot to {filename}")

    except Exception as e:
        print(f"\nError during visualization of graph {index}: {e}")
        traceback.print_exc()


def motif_masks_dataset():
    """
    Loads/Generates a BA2Motif-like dataset, prints the ground-truth
    node and edge masks, and visualizes the graphs.
    """
    print("Generating BA2Motif-style dataset using ExplainerDataset...")
    print("This may take a moment...")

    save_dir = os.path.join(os.path.dirname(__file__), ".." ,"data", "Custom_BA2Motif")

    os.makedirs(save_dir, exist_ok=True)

    dataset_filename = "custom_ba2motif_dataset.pt"

    save_path = os.path.join(save_dir, dataset_filename)

    print(f"Dataset will be saved to: {os.path.abspath(save_path)}")
    

    # --- Create output directory for plots ---
    output_dir = "graph_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving graph images to: {os.path.abspath(output_dir)}")
    # ---
    
    try:
        # Class 0: CycleMotif
        dataset_class_0 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=CycleMotif(5),
            num_graphs=500,
            num_motifs=1,
            graph_generator_kwargs={'seed': 123},
            motif_generator_kwargs={'seed': 123},
        )
        
        # Class 1: HouseMotif
        dataset_class_1 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=HouseMotif(),
            num_graphs=500,
            num_motifs=1,
            graph_generator_kwargs={'seed': 456},
            motif_generator_kwargs={'seed': 456},
        )

        print("Converting ConcatDataset to a simple Python list...")

        dataset_as_list = [graph for graph in ConcatDataset([dataset_class_0, dataset_class_1])]

        # --- Save the entire dataset to a file ---
        print(f"\nSaving the generated dataset of {len(dataset_as_list)} graphs...")
        torch.save(dataset_as_list, save_path) # Save the list, not the ConcatDataset
        print(f"Successfully saved dataset to {save_path}")
        # ---
        
    except Exception as e:
        print("\n" + "!"*60)
        print(f"Error generating dataset. The script cannot continue.")
        print("\nFull Error Traceback:")
        traceback.print_exc()
        print("!"*60 + "\n")
        return

    print(f"Dataset generated. Total graphs: {len(dataset_as_list)}")
    
    if len(dataset_as_list) == 0:
        print("Error: The dataset loaded but contains 0 graphs. Cannot proceed.")
        return
        
    # --- DEBUGGING STEP ---
    try:
        first_data = dataset_as_list[0]
        print("\n" + "*"*60)
        print("DEBUG: Inspecting keys of the first graph object...")
        print(f"Available keys: {first_data.keys()}")
        print(f"First data object: {first_data}")
        print("*"*60 + "\n")
    except Exception as e:
        print(f"Could not inspect first data object: {e}")
    # --- END DEBUGGING STEP ---
        
    print("Class 0: CycleMotif (Index 0-499), Class 1: HouseMotif (Index 500-999)")
    
    class_counts = {0: 0, 1: 0}
    max_examples = 10
    graphs_processed = 0

    for i, data in enumerate(dataset_as_list):
        graphs_processed += 1
        
        if i < 500:
            label = 0
        else:
            label = 1
        
        if label not in class_counts:
            print(f"Warning: Found unexpected label {label}. Skipping.")
            continue
            
        if class_counts[label] < max_examples:
            class_counts[label] += 1
            
            print("\n" + "="*50)
            print(f"--- Example {class_counts[label]}/{max_examples} for CLASS {label} (Graph #{i}) ---")
            print(f"  Graph properties:")
            print(f"    - Nodes: {data.num_nodes}")
            print(f"    - Edges: {data.num_edges}")
            
            if not hasattr(data, 'node_mask'):
                 print("\nCRITICAL ERROR: 'data.node_mask' not found.")
                 print(f"Available keys in this data object: {data.keys()}")
                 print("Stopping loop.")
                 break 
            
            nodes_in_motif = data.node_mask.sum().item()
            print(f"\n  Ground-Truth Node Mask ({int(nodes_in_motif)} nodes in motif):")
            print(f"    {data.node_mask.T}")
            
            if not hasattr(data, 'edge_mask'):
                 print("\nCRITICAL ERROR: 'data.edge_mask' not found.")
                 print(f"Available keys in this data object: {data.keys()}")
                 print("Stopping loop.")
                 break 

            edges_in_motif = data.edge_mask.sum().item()
            print(f"\n  Ground-Truth Edge Mask ({int(edges_in_motif)} edges in motif):")
            print(f"    {data.edge_mask}")
            
            # --- Call the new visualize function ---
            print("\n  Visualizing graph...")
            visualize_graph(data, label, i, output_dir=output_dir)
            # ---
            
            print("="*50)

        if class_counts[0] >= max_examples and class_counts[1] >= max_examples:
            print(f"\nSuccessfully printed and plotted {max_examples} examples from each class.")
            break

    if graphs_processed == 0:
         print("\nWarning: The loop finished without processing any graphs.")
    elif (class_counts[0] < max_examples or class_counts[1] < max_examples) and graphs_processed > 0:
        print(f"\nWarning: Loop stopped early (likely due to error) after processing {graphs_processed} graphs.")
        print(f"Found {class_counts[0]} of Class 0 and {class_counts[1]} of Class 1.")
    elif graphs_processed > 0:
         print(f"\nWarning: Finished iterating, but did not find {max_examples} of each class.")
         print(f"Found {class_counts[0]} of Class 0 and {class_counts[1]} of Class 1.")


if __name__ == "__main__":
    print("Script starting...")
    motif_masks_dataset()