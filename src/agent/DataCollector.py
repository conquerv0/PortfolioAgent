import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_community.llms import OpenAI
import sys
import logging 

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
    
    def download_etf_full_data(self, tickers, start_date, end_date):
        logger.info(f"Downloading full price data for {len(tickers)} ETFs from {start_date} to {end_date}")
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            if data.empty:
                logger.error("No data downloaded")
                return pd.DataFrame()
            # For multiple tickers, we have a MultiIndex DataFrame; extract required columns.
            price_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            # Calculate VWAP for each ticker: (High+Low+Close)/3
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
