import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

torch.manual_seed(42)
np.random.seed(42)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

class MLP(nn.Module):
    ## A simple MLP
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16,16)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  # Output layer for binary
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

### Data preparation

# 1. Create dataset
X , y  = make_moons(n_samples= 1000 , noise = 0.25 , random_state=42)

# 2. Split data into train, validation, and test sets (as NumPy arrays)
X_train , X_temp , y_train , y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val , X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- START OF CORRECTIONS ---

# 3. Scale the features
# Instantiate the scaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
# and transform the training data
X_train = scaler.fit_transform(X_train)

# Use the SAME scaler to transform validation and test data
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Convert NumPy arrays to PyTorch tensors and send to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# --- END OF CORRECTIONS ---


def train_validate(model , model_name):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_logits = model(X_train)
        loss = loss_fn(y_logits.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_logits = model(X_val)
            val_loss = loss_fn(y_val_logits.squeeze(), y_val)

            val_preds = torch.sigmoid(y_val_logits).squeeze()
            val_preds = (val_preds > 0.5).float()
            val_accuracy = (val_preds == y_val).float().sum() / len(y_val)
        
        # Printing only every 10 epochs to reduce output clutter
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

def test_model(model):
    model.eval()
    with torch.no_grad():
        y_test_logits = model(X_test)
        test_loss = nn.BCEWithLogitsLoss()(y_test_logits.squeeze(), y_test)

        test_preds = torch.sigmoid(y_test_logits).squeeze()
        test_preds = (test_preds > 0.5).float()
        test_accuracy = (test_preds == y_test).float().sum() / len(y_test)

    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")


# --- Visualization Function ---
def plot_decision_boundary(model, model_name):
    # Create a meshgrid using the original (unscaled) data range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # The grid data must be scaled just like the training data before being fed to the model
    # It also needs to be on the same device as the model
    grid_tensor = torch.tensor(scaler.transform(np.c_[xx.ravel(), yy.ravel()]), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        Z_logits = model(grid_tensor)
        Z = (torch.sigmoid(Z_logits) > 0.5).float()
        # Reshape predictions to match the grid shape and move to CPU for plotting
        Z = Z.cpu().reshape(xx.shape)
        
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    
    # For plotting, use the original unscaled test data and move labels to CPU
    unscaled_X_test = scaler.inverse_transform(X_test.cpu().numpy())
    plt.scatter(unscaled_X_test[:, 0], unscaled_X_test[:, 1], c=y_test.cpu().squeeze(), cmap=plt.cm.RdYlBu, edgecolors='k')
    
    plt.title(f'Decision Boundary for {model_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # --- ADD THIS LINE ---
    # Save the figure to a file before trying to show it.
    plt.savefig('decision_boundary.png')
    # --- END OF ADDITION ---
    
    plt.show()
    
    # Add a print statement to confirm saving
    print("Plot has been saved to decision_boundary.png")

model = MLP().to(device)
train_validate(model, "MLP_Moons")
test_model(model)
plot_decision_boundary(model, "MLP_Moons")
print("Training and evaluation completed.")
print("Decision boundary plot displayed.")