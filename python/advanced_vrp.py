import numpy as np
import volscore_wrapper as vw

class AdvancedVRP:
    """
    An advanced VRP calculator that uses the C++ 'VolScore' 
    for realized vol, and merges with a user-supplied implied vol input.
    It also calculates realized skew/kurt for reference.
    """
    def __init__(self):
        self.vol_engine = vw.VolScore()

    def compute_vrp(self, price_list, implied_vol):
        """
        VRP = (Implied Vol) - (Realized Vol)
        'implied_vol' expected in volatility points (e.g. 20 => 20%).
        """
        if not price_list:
            return 0.0
        rvol = self.vol_engine.computeRealizedVol(price_list) * 100.0  # Convert fractional to points
        return implied_vol - rvol

    def detail_stats(self, price_list):
        """
        Return a dict with realized vol, skew, kurtosis from the VolScore engine.
        """
        rv = self.vol_engine.computeRealizedVol(price_list)  # fractional
        skew = self.vol_engine.computeRealizedSkew(price_list)
        kurt = self.vol_engine.computeRealizedKurt(price_list)
        return {
            'realized_vol_%': rv*100.0,
            'realized_skew': skew,
            'realized_excess_kurt': kurt
        }