import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

class VRPLSTMModel:
    """
    A simple LSTM-based forecasting model: 
    Takes processed features (market data, macro signals, etc.) 
    and predicts the next day's VRP.
    """

    def __init__(self, input_dim=5, sequence_length=10):
        """
        input_dim: number of features at each time step
        sequence_length: how many time steps the LSTM sees
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.sequence_length, self.input_dim)))
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))  # Predict a single VRP value
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=5, batch_size=16):
        """
        X_train: shape (num_samples, sequence_length, input_dim)
        y_train: shape (num_samples,)
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X).flatten()

def create_training_data(prices, implied_vols, seq_length=10):
    """
    Example function to create sequences from daily price & implied vol arrays.
    We might add more features, e.g. realized vol, VIX term structure, macro data, etc.
    Output X: array of shape (num_samples, seq_length, input_dim)
    Output y: array of shape (num_samples,)
    """
    # Minimal feature set: [price_return, implied_vol]
    returns = []
    for i in range(1, len(prices)):
        retr = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(retr)
    # Align implied_vols with same indexing
    implied_vols = implied_vols[1:]  # shift to match returns length

    data = np.column_stack([returns, implied_vols])  # shape: (N-1, 2)
    X, y = [], []
    for i in range(seq_length, data.shape[0]):
        seq_x = data[i-seq_length:i, :]   # seq_length x input_dim
        # Next day VRP = implied_vol - realized_vol
        # realized_vol ~ std of returns in seq, for example
        local_returns = data[i-seq_length:i, 0]
        rvol_est = np.std(local_returns) * 100.0
        iv_ = data[i, 1]  # next day implied vol
        vrp_ = iv_ - rvol_est
        X.append(seq_x)
        y.append(vrp_)
    return np.array(X), np.array(y)