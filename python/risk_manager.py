import pandas as pd
import numpy as np

class RiskManager:
    """
    Conduct advanced drawdown analysis, VaR, and CVaR on the 
    daily PnL from the advanced backtester.
    """

    def __init__(self):
        pass

    def compute_drawdowns(self, pnl_series):
        """
        Return a DataFrame with drawdown metrics for the given PnL.
        """
        cum_pnl = pnl_series.cumsum()
        running_max = cum_pnl.cummax()
        dd = (cum_pnl - running_max)
        dd_percent = dd / running_max.replace(0, 1e-12)
        max_dd = dd.min()
        return {
            'max_drawdown_abs': max_dd,
            'max_drawdown_pct': dd_percent.min()
        }

    def compute_var_cvar(self, pnl_series, alpha=0.95):
        """
        Compute historical VaR and CVaR at level alpha.
        For daily changes in PnL or returns.
        """
        sorted_pnl = np.sort(pnl_series)
        index = int((1 - alpha)*len(sorted_pnl))
        var = sorted_pnl[index]
        cvar = sorted_pnl[:index].mean() if index > 0 else var
        return {
            'var': var,
            'cvar': cvar
        }

    def analyse(self, df_results):
        """
        Takes the output from advanced_backtester and 
        provides risk metrics like drawdown, VaR, CVaR.
        """
        daily_pnl = df_results['daily_pnl']
        dd_info = self.compute_drawdowns(daily_pnl)
        var_info = self.compute_var_cvar(daily_pnl, alpha=0.95)
        summary = {}
        summary.update(dd_info)
        summary.update(var_info)
        return summary

if __name__ == "__main__":
    # Quick test with random PnL
    random_pnl = np.random.normal(0, 1, 250)
    rm = RiskManager()
    dd_info = rm.compute_drawdowns(pd.Series(random_pnl))
    var_info = rm.compute_var_cvar(random_pnl, alpha=0.95)
    print("Drawdown:", dd_info)
    print("VaR / CVaR:", var_info)