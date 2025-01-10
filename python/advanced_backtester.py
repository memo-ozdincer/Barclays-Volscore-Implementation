import numpy as np
import pandas as pd
from advanced_vrp import AdvancedVRP
# We remove references to "ml_lstm" and use our new PyTorch version:
from ml_lstm_torch import create_training_data_torch, VRPLSTMPyTorchTrainer

class AdvancedBacktester:
    """
    A more involved backtester that uses:
      1) Historical data for prices & implied vol
      2) A PyTorch LSTM to forecast VRP for the next day
      3) A position sizing logic based on predicted VRP
    """

    def __init__(self, eq_threshold=0.0, lookback=10):
        self.eq_threshold = eq_threshold   # If predicted VRP > eq_threshold => 'short vol'
        self.lookback = lookback
        self.adv_vrp = AdvancedVRP()
        # We'll create the PyTorch LSTM trainer
        self.ml_trainer = VRPLSTMPyTorchTrainer(input_dim=2, hidden_dim=32, seq_length=lookback, lr=1e-3)

    def prepare_training(self, price_list, implied_vol_list):
        """
        Convert raw data into sequences for LSTM training
        """
        X, y = create_training_data_torch(price_list, implied_vol_list, seq_length=self.lookback)
        return X, y

    def train_model(self, X, y, epochs=5, batch_size=16):
        self.ml_trainer.train(X, y, epochs=epochs, batch_size=batch_size)

    def run_backtest(self, price_list, implied_vol_list):
        """
        Similar idea: for every day, predict VRP with the LSTM.
        Then short vol if VRP > eq_threshold.
        """
        X, _ = create_training_data_torch(price_list, implied_vol_list, seq_length=self.lookback)
        y_pred = self.ml_trainer.predict(X)

        daily_pnl = []
        for i in range(len(y_pred)):
            vrp_pred = y_pred[i]
            # If VRP is above threshold => short vol
            position = 1 if vrp_pred > self.eq_threshold else 0
            # Simple naive PnL
            daily_pnl.append((vrp_pred / 100.0) * position)

        results_df = pd.DataFrame({
            'day': range(len(daily_pnl)),
            'predicted_vrp': y_pred,
            'Position': [1 if y_pred[i] > self.eq_threshold else 0 for i in range(len(y_pred))],
            'daily_pnl': daily_pnl
        })
        results_df['cumulative_pnl'] = results_df['daily_pnl'].cumsum()
        return results_df

if __name__ == "__main__":
    # Example usage
    mock_prices = np.linspace(100, 130, 100) + np.random.normal(0,1,100).cumsum()
    mock_iv = 20 + np.random.normal(0,2,100).cumsum()

    bt = AdvancedBacktester(eq_threshold=1.5, lookback=10)
    X_train, y_train = bt.prepare_training(mock_prices, mock_iv)
    bt.train_model(X_train, y_train, epochs=3, batch_size=8)
    results = bt.run_backtest(mock_prices, mock_iv)
    print(results.head(12))
    print("Final Cumulative PnL:", results['cumulative_pnl'].iloc[-1])