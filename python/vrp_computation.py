import pandas as pd
import numpy as np
import volscore_wrapper as vw
from data_pipeline import DataPipeline

class VRPComputer:
    def __init__(self):
        self.vol_engine = vw.VolScore()

    def compute_vrp(self, main_prices, vix_prices):
        """
        A trivial VRP calculation:
          VRP = VolScore_of_main - daily_average_of_vix
        """
        # 1) Use C++ to compute realized vol
        realized_vol_score = self.vol_engine.computeVolScore(main_prices)  # some numeric
        # 2) Implied vol ~ average of VIX from vix_prices
        implied_vol = np.mean(vix_prices) if len(vix_prices) > 0 else 0.0

        # This is our naive “spread”
        vrp = realized_vol_score - implied_vol
        return vrp

if __name__ == "__main__":
    dp = DataPipeline()
    main_data, vix_data = dp.fetch_market_data()
    main_prices_list = dp.get_prices_as_list(main_data)
    vix_prices_list  = dp.get_prices_as_list(vix_data)

    vrp_comp = VRPComputer()
    result = vrp_comp.compute_vrp(main_prices_list, vix_prices_list)
    print("Naive VRP result =", result)
    from adjust_vrp_ml import adjust_vrp_with_ml

    # near the end of the __main__ section
    adjusted_vrp = adjust_vrp_with_ml(result)
    print("ML-adjusted VRP =", adjusted_vrp)  
