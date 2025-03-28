import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_community.llms import OpenAI
import sys

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