import numpy as np
import torch
import torch.nn as nn

class VRPLSTMPyTorch(nn.Module):
    """
    A simple PyTorch-based LSTM module to predict VRP from sequences.
    Each sequence step has 'input_dim' features, 
    and we produce a single VRP prediction at the end.
    """

    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
        super(VRPLSTMPyTorch, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # out shape: (batch_size, seq_length, hidden_dim)
        # We only want the last time step
        last_out = out[:, -1, :]   # (batch_size, hidden_dim)
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_training_data_torch(prices, implied_vols, seq_length=10):
    """
    Minimal feature engineering:
    Feature dims => [daily return, next day implied vol].
    Then VRP = implied - realizedVol(returns).
    We produce (X, y) for supervised learning:
    X: [batch_size, seq_length, input_dim]
    y: [batch_size, 1]
    """
    rets = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i-1]) / prices[i-1]
        rets.append(r)
    # Align implied
    implied_vals = implied_vols[1:]  # shift to match rets

    data = np.column_stack([rets, implied_vals])  # shape: (N-1, 2)

    sequences = []
    labels = []
    for i in range(seq_length, data.shape[0]):
        seq_x = data[i-seq_length:i, :]   # shape: (seq_length, 2)
        # Next day VRP => implied[i] - std(returns[i-seq..i])
        local_returns = data[i-seq_length:i, 0]
        rvol_est = np.std(local_returns) * 100.0
        iv_ = data[i, 1]
        vrp_ = iv_ - rvol_est
        sequences.append(seq_x)
        labels.append(vrp_)
    X = np.array(sequences)
    y = np.array(labels).reshape(-1, 1)
    return X, y

class VRPLSTMPyTorchTrainer:
    """
    A small wrapper for training the VRPLSTMPyTorch model.
    """
    def __init__(self, input_dim=2, hidden_dim=32, seq_length=10, lr=1e-3, device='cpu'):
        self.seq_length = seq_length
        self.device = device
        self.model = VRPLSTMPyTorch(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, epochs=5, batch_size=16):
        """
        X_train: numpy array, shape: (num_samples, seq_len, input_dim)
        y_train: numpy array, shape: (num_samples, 1)
        """
        dataset_size = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(dataset_size)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            batch_losses = []
            for i in range(0, dataset_size, batch_size):
                xb_np = X_train_shuffled[i:i+batch_size]
                yb_np = y_train_shuffled[i:i+batch_size]

                xb_t = torch.tensor(xb_np, dtype=torch.float32, device=self.device)
                yb_t = torch.tensor(yb_np, dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()
                preds = self.model(xb_t)
                loss = self.criterion(preds, yb_t)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)
            print(f"Epoch {epoch+1}/{epochs}, Loss={epoch_loss:.4f}")

    def predict(self, X):
        """
        X: shape (num_samples, seq_length, input_dim) in numpy
        Return shape => (num_samples,) in numpy
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(X_t)
        return preds.cpu().numpy().flatten()