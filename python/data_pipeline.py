import yfinance as yf
import pandas as pd
import numpy as np

class DataPipeline:
    """
    Advanced data pipeline for fetching and caching data. 
    Supports dynamic lookback windows, optional data resampling, 
    and multi-asset retrieval.
    """
    def __init__(self, tickers=None, start="2020-01-01", end="2021-01-01", freq='1D'):
        self.tickers = tickers if tickers else ["SPY", "^VIX"]
        self.start = start
        self.end = end
        self.freq = freq

    def fetch_market_data(self):
        """
        Fetch data for self.tickers using yfinance.
        Resample to self.freq (daily by default).
        """
        data_dict = {}
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start, end=self.end, progress=False)
            if not df.empty:
                # Basic cleaning
                df.dropna(inplace=True)
                # Optional resampling
                df = df.resample(self.freq).last().dropna()
                data_dict[ticker] = df
        return data_dict

    def get_close_prices(self, df):
        """
        Return a list of close prices from the given df.
        """
        if 'Close' not in df.columns:
            return []
        return df['Close'].tolist()

if __name__ == "__main__":
    dp = DataPipeline(tickers=["SPY", "^VIX"], start="2022-01-01", end="2022-12-31")
    data = dp.fetch_market_data()
    for k, v in data.items():
        print(f"{k} data sample:\n", v.head())