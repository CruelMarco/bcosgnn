import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Module, Embedding
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm

class GINE(Module):

    def __init__(self, num_tasks, emb_dim=300, drop_ratio=0.5):

        super(GINE, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        # Atom and bond encoders
        self.atom_encoder = Embedding(119, emb_dim)
        self.bond_encoder = Embedding(5, emb_dim)


        # Layer 1
        nn1 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv1 = GINEConv(nn=nn1, train_eps=True)
        self.bn1 = BatchNorm1d(emb_dim)

        # Layer 2
        nn2 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv2 = GINEConv(nn=nn2, train_eps=True)
        self.bn2 = BatchNorm1d(emb_dim)

        # Layer 3
        nn3 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        self.conv3 = GINEConv(nn=nn3, train_eps=True)
        self.bn3 = BatchNorm1d(emb_dim)

        # # Layer 4
        # nn4 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        # self.conv4 = GINEConv(nn=nn4, train_eps=True)
        # self.bn4 = BatchNorm1d(emb_dim)

        # # Layer 5
        # nn5 = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(), Linear(2 * emb_dim, emb_dim))
        # self.conv5 = GINEConv(nn=nn5, train_eps=True)
        # self.bn5 = BatchNorm1d(emb_dim)

        # Output layer
        self.graph_pred_linear = Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        """
        Forward pass of the GINE model.

        Args:
            batched_data (Data): A batch of graph data from PyTorch Geometric.

        Returns:
            torch.Tensor: The model's prediction for each graph in the batch.
        """
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        # Encode node and edge features
        h = self.atom_encoder(x[:, 0])
        edge_embedding = self.bond_encoder(edge_attr[:, 0])
        
        # --- Apply layers sequentially ---
        h = F.relu(self.bn1(self.conv1(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr=edge_embedding)))
        h = F.relu(self.bn3(self.conv3(h, edge_index, edge_attr=edge_embedding)))
        # h = F.relu(self.bn4(self.conv4(h, edge_index, edge_attr=edge_embedding)))
        # h = F.relu(self.bn5(self.conv5(h, edge_index, edge_attr=edge_embedding)))

        # Graph-level pooling
        h_graph = global_add_pool(h, batch)
        
        # Dropout and final prediction
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
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            
            is_labeled = batch.y == batch.y  # Filter out unlabeled data
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)

def evaluate(model, device, loader, evaluator):
    """Evaluates the model on a given dataset."""
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def main():
    """Main function to run the graph classification task."""
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    epochs = 100
    learning_rate = 0.001
    emb_dim = 300
    drop_ratio = 0.5
    
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    # 3. Replace the original torch.load with our new patched version
    torch.load = _patched_torch_load

    # Now, when PygGraphPropPredDataset calls torch.load, it will use our version
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="data")

    # 4. (Optional but good practice) Restore the original function
    torch.load = _original_torch_load

    split_idx = dataset.get_idx_split()    

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=0)
    
    # --- Model, Optimizer, and Loss ---
    model = GINE(
        num_tasks=dataset.num_tasks,
        emb_dim=emb_dim,
        drop_ratio=drop_ratio,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = Evaluator(name='ogbg-molhiv')
    
    # --- Training Loop ---
    best_val_auc = 0
    test_auc_at_best_val = 0
    
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss = train(model, device, train_loader, optimizer, criterion)
        
        print("Evaluating on validation and test sets...")
        val_result = evaluate(model, device, valid_loader, evaluator)
        test_result = evaluate(model, device, test_loader, evaluator)
        
        val_auc = val_result['rocauc']
        test_auc = test_result['rocauc']
        
        print(f'Train Loss: {train_loss:.4f}, Val ROC-AUC: {val_auc:.4f}, Test ROC-AUC: {test_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc_at_best_val = test_auc
            print(f"New best validation ROC-AUC: {best_val_auc:.4f}. Saving model.")
            torch.save(model.state_dict(), 'gine_molhiv_best_model.pt')

    print("\n--- Final Results ---")
    print(f'Best Validation ROC-AUC: {best_val_auc:.4f}')
    print(f'Test ROC-AUC at Best Validation: {test_auc_at_best_val:.4f}')

if __name__ == "__main__":
    main()

