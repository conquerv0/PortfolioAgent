import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_community.llms import OpenAI
import sys
import logging 
import numpy as np
from sklearn.covariance import LedoitWolf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import *
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DataCollector:
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Abstract method to collect and return a combined DataFrame of features.
        """
        raise NotImplementedError("Subclasses must implement collect_data.")
    
    def download_etf_full_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        """
        Downloads full price data for a list of ETF tickers between the given dates.
        It extracts Open, High, Low, Close, Volume and computes VWAP for each ticker.
        """
        logger.info(f"Downloading full price data for {len(tickers)} ETFs from {start_date} to {end_date}")
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            if data.empty:
                logger.error("No data downloaded")
                return pd.DataFrame()
            # For multiple tickers, data comes as a MultiIndex DataFrame; extract required columns.
            price_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            # Calculate VWAP for each ticker: (High + Low + Close) / 3
            for ticker in tickers:
                if (ticker, 'High') in price_data.columns:
                    price_data[(ticker, 'VWAP')] = (
                        price_data[(ticker, 'High')] +
                        price_data[(ticker, 'Low')] +
                        price_data[(ticker, 'Close')]
                    ) / 3
            return price_data
        except Exception as e:
            logger.error(f"Error downloading ETF data: {str(e)}")
            return pd.DataFrame()
    
    def estimate_covariance_matrix(self, price_data: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
        """
        Estimates the covariance matrix of ETF returns using downloaded price data.
        
        This function first extracts the "Close" prices (if the DataFrame has a MultiIndex),
        calculates daily percentage returns, and then computes the covariance matrix using one of:
            - "sample": the standard sample covariance estimator.
            - "ledoit_wolf": a robust covariance estimator using the Ledoit-Wolf shrinkage method.
        
        Parameters:
            price_data (pd.DataFrame): Historical price data with dates as the index.
                For MultiIndex columns, the "Close" price is expected.
            method (str): Estimation method ("sample" or "ledoit_wolf").
        
        Returns:
            pd.DataFrame: Estimated covariance matrix of asset returns.
        """
        # If price_data has MultiIndex columns, extract the 'Close' prices.
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Close' in price_data.columns.levels[1]:
                close_prices = price_data.xs('Close', axis=1, level=1)
            else:
                raise ValueError("MultiIndex price data must include 'Close' prices.")
        else:
            close_prices = price_data
        
        # Calculate daily returns
        returns = close_prices.pct_change().dropna()
        
        if method == "sample":
            cov_matrix = returns.cov()
        elif method == "ledoit_wolf":
            lw = LedoitWolf().fit(returns)
            cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'sample', 'ledoit_wolf'")
        
        return cov_matrix
    

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickers = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    collector = DataCollector()
    # Here, one would subclass DataCollector to implement collect_data; however, for covariance estimation we use download_etf_full_data.
    price_data = collector.download_etf_full_data(tickers, start_date, end_date)
    cov_matrix = collector.estimate_covariance_matrix(price_data, method="ledoit_wolf")
    
    print("Estimated Covariance Matrix:")
    print(cov_matrix)