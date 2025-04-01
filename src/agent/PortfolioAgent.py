import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from openai import OpenAI
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import *
from dotenv import load_dotenv
from agent.DataCollector import DataCollector

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Base Classes for the Common Pipeline
# ----------------------------

class PortfolioAgent:
    def __init__(self, name: str, data_collector: DataCollector, llm_client: OpenAI):
        self.name = name
        self.data_collector = data_collector
        self.llm_client = llm_client
    
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.data_collector.collect_data(start_date, end_date)
    
    def prepare_prompt(self, row: pd.Series) -> str:
        """
        Abstract method for creating a prompt given a row of data.
        """
        raise NotImplementedError("Subclasses must implement prepare_prompt.")
    
    def estimate_returns(self, data: dict) -> dict:
        """
        Empirical return estimation for each asset in the portfolio.
        For simplicity, we use the mean daily return computed from historical prices.
        """
        raise NotImplementedError("Subclasses must implement collect_data.")
        # returns = {}
        # for asset, df in data.items():
        #     # Compute daily percent change and average (simple empirical return)
        #     daily_returns = df.pct_change().dropna()
        #     returns[asset] = daily_returns.mean()
        # return returns
    
    def get_llm_analysis(self, prompt: str) -> dict:
        """
        Abstract method for performing LLM analysis given a prompt.
        """
        raise NotImplementedError("Subclasses must implement get_llm_analysis.")
    
    def run_pipeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Common pipeline that:
          1. Collects combined feature data.
          2. Iterates over each day (row) to:
             - Prepare a detailed prompt.
             - Perform LLM analysis.
          3. Returns a DataFrame of LLM responses indexed by date.
        """
        raise NotImplementedError("Subclasses must implement run_pipeline.")
    
