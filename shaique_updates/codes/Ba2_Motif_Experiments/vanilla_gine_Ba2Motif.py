import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Module
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import BA2MotifDataset
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class GINE(Module):
    """
    Graph Isomorphism Network (GIN) model adapted for BA_2motifs dataset.

    This model uses GINConv layers. It first projects the node features to a
    common embedding dimension, then applies a series of GINConv layers,
    followed by global add pooling and a final linear layer for
    graph-level classification.
    """

    def __init__(self, num_node_features, num_tasks=1, emb_dim=300, drop_ratio=0.5):
        super(GINE, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.node_proj = Linear(num_node_features, emb_dim)

        # Layer 1
        nn1 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv1 = GINConv(nn=nn1, train_eps=True)
        self.bn1 = BatchNorm1d(emb_dim)

        # Layer 2
        nn2 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv2 = GINConv(nn=nn2, train_eps=True)
        self.bn2 = BatchNorm1d(emb_dim)

        # Layer 3
        nn3 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv3 = GINConv(nn=nn3, train_eps=True)
        self.bn3 = BatchNorm1d(emb_dim)

        # Output layer
        self.graph_pred_linear = Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )
        
        h = self.node_proj(x)

        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        h_graph = global_add_pool(h, batch)
        
        h_graph = F.dropout(h_graph, p=self.drop_ratio, training=self.training)
        output = self.graph_pred_linear(h_graph)

        return output

def train(model, device, loader, optimizer, criterion):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        
        pred = model(batch).squeeze()
        optimizer.zero_grad()
        
        loss = criterion(pred.to(torch.float32), batch.y.to(torch.float32))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)

def evaluate(model, device, loader, criterion):
    """
    Evaluates the model on a given dataset.
    
    Returns:
        float: The accuracy score.
        np.array: The confusion matrix.
        float: The average loss.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            continue
        
        with torch.no_grad():
            pred_logits = model(batch).squeeze()
            pred_labels = (pred_logits > 0).long()
            
            # Calculate loss
            loss = criterion(pred_logits.to(torch.float32), batch.y.to(torch.float32))
            total_loss += loss.item() * batch.num_graphs
        
        y_true_list.append(batch.y.long().detach().cpu())
        y_pred_list.append(pred_labels.detach().cpu())

    all_y_true = torch.cat(y_true_list, dim=0).numpy()
    all_y_pred = torch.cat(y_pred_list, dim=0).numpy()

    # Calculate metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)
    avg_loss = total_loss / len(loader.dataset)
    
    return acc, cm, avg_loss

def main():
    """Main function to run the graph classification task on BA_2motifs."""
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 128
    epochs = 100
    learning_rate = 0.00001
    emb_dim = 300
    drop_ratio = 0.5
    
    # --- Output Directory Setup ---
    output_dir = "/home/moso00002/Desktop/gnn/bcosgnn-bcos_gnn_shaique/shaique_updates/codes/outputs/vanilla_gine_BA2Motif"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    best_model_path = os.path.join(output_dir, 'gine_ba2motif_best_model.pt')
    loss_plot_path = os.path.join(output_dir, 'train_val_loss.png')
    cm_plot_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    
    # --- Dataset and Dataloaders ---
    dataset = BA2MotifDataset(root="data/BA_2motifs")
    
    num_node_features = dataset.num_node_features
    num_tasks = 1 # Binary classification
    
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    valid_len = int(total_len * 0.1)
    test_len = total_len - train_len - valid_len
    
    print(f"Total graphs: {total_len}, Train: {train_len}, Valid: {valid_len}, Test: {test_len}")

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_len, valid_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = GINE(
        num_node_features=num_node_features,
        num_tasks=num_tasks,
        emb_dim=emb_dim,
        drop_ratio=drop_ratio,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # --- Training Loop ---
    best_val_acc = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = train(model, device, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        print("Evaluating on validation set...")
        val_acc, val_cm, val_loss = evaluate(model, device, valid_loader, criterion)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

    print("\n--- Training Finished ---")
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')

    # --- Plotting Training and Validation Loss ---
    print(f"\nPlotting and saving loss curve to {loss_plot_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close() 
    print("Loss plot saved.")

    # --- Final Test Set Evaluation ---
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on the test set
    final_test_acc, final_test_cm, final_test_loss = evaluate(model, device, test_loader, criterion)
    
    print(f'\n--- Final Test Results ---')
    print(f'Final Test Accuracy: {final_test_acc:.4f}')
    print(f'Final Test Loss: {final_test_loss:.4f}')
    print(f'Final Test Confusion Matrix:\n{final_test_cm}')

    # --- Plotting and Saving Test Confusion Matrix ---
    print(f"\nPlotting and saving confusion matrix to {cm_plot_path}...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(final_test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred House (0)', 'Pred Grid (1)'],
                yticklabels=['Actual House (0)', 'Actual Grid (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Final Test Confusion Matrix (Accuracy: {final_test_acc:.4f})')
    plt.savefig(cm_plot_path)
    plt.close() 
    print("Confusion matrix plot saved.")

if __name__ == "__main__":
    main()
