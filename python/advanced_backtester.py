import numpy as np
import pandas as pd
from advanced_vrp import AdvancedVRP
from ml_lstm import VRPLSTMModel, create_training_data

class AdvancedBacktester:
    """
    A more involved backtester that uses:
      1) Historical data for prices & implied vol
      2) An LSTM to forecast VRP for the next day
      3) A position sizing logic based on predicted VRP
    """

    def __init__(self, eq_threshold=0.0, lookback=10):
        self.eq_threshold = eq_threshold   # If predicted VRP > threshold => 'short vol'
        self.lookback = lookback
        self.adv_vrp = AdvancedVRP()
        self.ml_model = VRPLSTMModel(input_dim=2, sequence_length=lookback)

    def prepare_training(self, price_list, implied_vol_list):
        """
        Convert raw data into sequences for LSTM training
        """
        X, y = create_training_data(price_list, implied_vol_list, seq_length=self.lookback)
        return X, y

    def train_model(self, X, y, epochs=5, batch_size=16):
        self.ml_model.train(X, y, epochs=epochs, batch_size=batch_size)

    def run_backtest(self, price_list, implied_vol_list):
        """
        For every day, use the LSTM to predict VRP. 
        If VRP > eq_threshold => short vol. 
        Gains = predicted_vrp / 100.0 * notional
        (very naive logic - purely illustrative)
        """
        # We'll reuse the same sequences:
        X, y_truth = create_training_data(price_list, implied_vol_list, seq_length=self.lookback)
        # Predict
        y_pred = self.ml_model.predict(X)

        daily_pnl = []
        # Align indexing properly
        # For i in [lookback..len(price_list)-1], backtest day:
        for i in range(len(y_pred)):
            # predicted VRP:
            vrp_pred = y_pred[i]
            # Decide position
            position = 1 if vrp_pred > self.eq_threshold else 0
            # Naive PnL = VRP * fraction
            # e.g., if VRP=3 => 3 vol points => 3% ann? This is simplistic.
            daily_pnl.append((vrp_pred / 100.0) * position)

        # Build a DataFrame
        results_df = pd.DataFrame({
            'day': range(len(daily_pnl)),
            'predicted_vrp': y_pred,
            'Position': [1 if y_pred[i] > self.eq_threshold else 0 for i in range(len(y_pred))],
            'daily_pnl': daily_pnl
        })
        results_df['cumulative_pnl'] = results_df['daily_pnl'].cumsum()
        return results_df

if __name__ == "__main__":
    # Example usage:
    # (1) Mock data
    mock_prices = np.linspace(100, 130, 100) + np.random.normal(0,1,100).cumsum()
    mock_iv = 20 + np.random.normal(0,2,100).cumsum()  # some random walk around 20

    bt = AdvancedBacktester(eq_threshold=1.0, lookback=10)
    # Prepare data
    X_train, y_train = bt.prepare_training(mock_prices, mock_iv)
    # Simple model training
    bt.train_model(X_train, y_train, epochs=2, batch_size=8)
    # Backtest
    results = bt.run_backtest(mock_prices, mock_iv)
    print(results.head(15))
    print("Final Cumulative PnL:", results['cumulative_pnl'].iloc[-1])