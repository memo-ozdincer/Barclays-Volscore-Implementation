import pandas as pd
import numpy as np

from data_pipeline import DataPipeline
from vrp_computation import VRPComputer
from adjust_vrp_ml import adjust_vrp_with_ml

class SimpleBacktester:
    def __init__(
        self, 
        start="2020-01-01", 
        end="2020-06-01", 
        vol_threshold=0.0,
        use_ml=False
    ):
        """
        vol_threshold: If computed VRP > vol_threshold, we go short vol; else do nothing.
        use_ml: Whether to adjust VRP with the fake ML model from Phase 2.
        """
        self.start = start
        self.end = end
        self.vol_threshold = vol_threshold
        self.use_ml = use_ml

        # Setup pipeline and VRP
        self.dp = DataPipeline(start=self.start, end=self.end)
        self.vrp_comp = VRPComputer()

        # Will hold PnL results day by day
        self.pnl_series = []

    def run_backtest(self):
        """
        We'll fetch daily data for some range, compute VRP each day (or for the next day),
        and track a trivial PnL if we choose to short or not.
        """
        main_data, vix_data = self.dp.fetch_market_data()
        
        # Ensure both dataframes align on dates
        combined = pd.concat(
            [main_data['Close'], vix_data['Close']],
            axis=1,
            join='inner'
        )
        combined.columns = ['main_close', 'vix_close']
        
        # Sort by ascending date to simulate forward progression
        combined.sort_index(inplace=True)

        # We'll track the position (short vol or flat)
        position = 0  # 1 => short vol, 0 => flat

        for i in range(1, len(combined)):
            # Current day’s data
            current_day = combined.iloc[:i]  # all data up to day i
            main_prices_list = current_day['main_close'].tolist()
            vix_prices_list  = current_day['vix_close'].tolist()

            # Compute today's VRP
            vrp_value = self.vrp_comp.compute_vrp(main_prices_list, vix_prices_list)
            if self.use_ml:
                vrp_value = adjust_vrp_with_ml(vrp_value)  # ML-minimal stub

            # Decide position for the next day
            if vrp_value > self.vol_threshold:
                position = 1  # short vol
            else:
                position = 0  # flat

            # Compute daily PnL (fake logic):
            # If short vol => assume daily PnL ~ VRP_value / 100, else 0.
            # This is obviously not realistic, but enough to illustrate a backtest.
            daily_pnl = (vrp_value / 100.0) * position
            self.pnl_series.append(daily_pnl)

        # Convert to a DataFrame for easy analysis
        return pd.DataFrame({
            'day': range(len(self.pnl_series)),
            'daily_pnl': self.pnl_series
        })

if __name__ == "__main__":
    bt = SimpleBacktester(start="2020-01-01", end="2020-06-01", vol_threshold=0.0, use_ml=True)
    results = bt.run_backtest()
    results['cumulative_pnl'] = results['daily_pnl'].cumsum()
    print(results.head(10))
    print("Final Cumulative PnL =", results['cumulative_pnl'].iloc[-1])
