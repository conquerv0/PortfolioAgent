import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from openai import OpenAI
import sys
import logging 
import numpy as np
from sklearn.covariance import LedoitWolf
import wrds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import *
from dotenv import load_dotenv

load_dotenv()

class DataCollector:
    def __init__(self, full_start_date: str, target_start_date: str, end_date: str):
        self.FRED_API_KEY = os.getenv("FRED_API_KEY")
        self.WRDS_USERNAME = os.getenv("WRDS_USERNAME")
        self.full_start_date = full_start_date
        self.target_start_date = target_start_date
        self.end_date = end_date

    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Abstract method to collect and return a combined DataFrame of features.
        """
        raise NotImplementedError("Subclasses must implement collect_data.")
    
    def get_etf_data(self, tickers, start_date, end_date):
        logger.info(f"Downloading data for {len(tickers)} ETFs...")
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            logger.error("Error: No ETF data available")
            return pd.DataFrame()
        etf_data = data['Close'].copy()
        missing_data = etf_data.isnull().sum()
        if missing_data.any():
            logger.warning("Missing data points per ETF:")
            for ticker, count in missing_data.items():
                if count > 0:
                    logger.warning(f"{ticker}: {count} missing points")
        return etf_data
    
    def get_etf_return(self,tickers, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads daily price data for the treasury ETFs from CRSP by joining dsf with dsenames.
        
        Parameters:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: DataFrame containing date, ticker, price, and daily return.
        """
        tickers_str = ",".join(f"'{ticker}'" for ticker in tickers)
        
        db = wrds.Connection(USER_NAME=self.WRDS_USERNAME)
        
        # Join dsf (daily file) with dsenames to get ticker information.
        # The join condition uses the date to ensure we use the correct ticker for each period.
        query = f"""
        SELECT a.date, b.ticker, a.prc as price, a.ret as daily_return
        FROM crsp.dsf as a
        JOIN crsp.dsenames as b
        ON a.permno = b.permno
        WHERE b.ticker IN ({tickers_str})
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
        AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '{end_date}')
        ORDER BY a.date, b.ticker
        """
        df_prices = db.raw_sql(query)
        db.close()
        
        # Pivot the data so that the date becomes the index and each ticker is a column.
        df_returns = df_prices.pivot(index='date', columns='ticker', values='daily_return')
        df_returns.fillna(0, inplace=True)
        
        return df_returns

    
    def get_etf_adj_close(self, tickers, start_date, end_date) -> pd.DataFrame:
        """
        Downloads adjusted close price data for a list of ETF tickers between start_date and end_date.
        Returns a panel DataFrame with dates as index and tickers as columns.
        
        Parameters:
            tickers (list): List of ETF ticker strings.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
        
        Returns:
            pd.DataFrame: DataFrame of adjusted close prices.
        """
        logger.info(f"Downloading adjusted close data for {len(tickers)} ETFs from {start_date} to {end_date}")
        try:
            # Download raw data with auto_adjust set to False so that 'Adj Close' is available.
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if data.empty:
                logger.error("No data downloaded")
                return pd.DataFrame()
            # Extract only the 'Adj Close' column; this creates a DataFrame with dates as index and tickers as columns.
            adj_close = data['Adj Close']
            return adj_close
        except Exception as e:
            logger.error(f"Error downloading ETF data: {str(e)}")
            return pd.DataFrame()

    def estimate_covariance_matrix(self, price_data: pd.DataFrame, method: str = "ledoit_wolf") -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Given panel data of adjusted close prices (dates as index, tickers as columns),
        computes daily returns and estimates the covariance matrix.
        """
        # If the data has columns labeled "Adj Close", use that; otherwise, assume price_data is already panel.
        if "Adj Close" in price_data.columns:
            close_prices = price_data["Adj Close"]
        else:
            close_prices = price_data
        close_prices.index = pd.to_datetime(close_prices.index)
        returns = close_prices.pct_change().dropna()
        
        if method == "sample":
            cov_matrix = returns.cov()
        elif method == "ledoit_wolf":
            lw = LedoitWolf().fit(returns)
            cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        else:
            raise ValueError(f"Unknown method: {method}")
        return cov_matrix, close_prices, returns
    
def extract_etf_tickers(portfolio: dict, key: str = "treasuries") -> list:
    """
    Extracts ETF tickers from a portfolio dictionary.

    Parameters:
        portfolio (dict): Portfolio dictionary structured with a key (e.g. "treasuries") 
                          mapping to a list of asset dictionaries.
                          Example:
                          {
                              "treasuries": [
                                  {"name": "Short-Term Treasury", "etf": "SHV", "maturity": "0-1yr", "weight": 0.0},
                                  {"name": "1-3 Year Treasury", "etf": "SHY", "maturity": "1-3yr", "weight": 0.0},
                                  ...
                              ]
                          }
        key (str): The key in the dictionary where the asset list is stored (default "treasuries").

    Returns:
        list: A list of ETF tickers extracted from the portfolio.
    """
    tickers = []
    assets = portfolio.get(key, [])
    for asset in assets:
        ticker = asset.get("etf")
        if ticker:
            tickers.append(ticker)
    return tickers


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickers = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    collector = DataCollector(start_date, start_date, end_date)
    # Here, one would subclass DataCollector to implement collect_data; however, for covariance estimation we use download_etf_full_data.
    price_data = collector.get_etf_adj(tickers, start_date, end_date)
    cov_matrix = collector.estimate_covariance_matrix(price_data, method="ledoit_wolf")
    
    print("Estimated Covariance Matrix:")
    print(cov_matrix)