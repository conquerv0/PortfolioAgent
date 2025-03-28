import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_community.llms import OpenAI
from agent.portfolioAgent import DataCollector, PortfolioAgent
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import *
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Equity-Specific Implementations
# ----------------------------

class EquityDataCollector(DataCollector):
    """
    Equity data collector that retrieves historical price data for each ETF
    in the equity portfolio using yfinance.
    """
    def __init__(self, etf_portfolio: dict):
        self.etf_portfolio = etf_portfolio  # Expecting a dict with a "sectors" key
    
    def collect_data(self, start_date: str, end_date: str) -> dict:
        data = {}
        for sector in self.etf_portfolio.get("sectors", []):
            etf = sector["etf"]
            try:
                df = yf.download(etf, start=start_date, end=end_date)["Close"]
                data[etf] = df
            except Exception as e:
                print(f"Error downloading data for {etf}: {e}")
        return data

class EquityAgent(PortfolioAgent):
    """
    Equity agent that uses an EquityDataCollector and a customized prompt template.
    The prompt instructs the LLM to provide a forecast view (expected return and confidence)
    for each equity ETF.
    """
    def __init__(self, etf_portfolio: dict, llm: OpenAI):
        # Equity-specific prompt template.
        # Note: In production you may want to use a structured prompt and LLMChain.
        prompt_template = (
            "You are an expert in equity portfolio management. Based on the following estimated daily returns for each ETF:\n"
            "{estimated_returns}\n"
            "Provide a forecast view with an expected return and a confidence score for each ETF. "
            "Explain your reasoning concisely."
        )
        data_collector = EquityDataCollector(etf_portfolio)
        super().__init__(name="EquityAgent", data_collector=data_collector, prompt_template=prompt_template, llm=llm)


# ----------------------------
# Example Execution
# ----------------------------
if __name__ == "__main__":
    # Define a time window for data collection
    start_date = "2022-11-31"
    end_date = "2022-12-01"
    # Define an example equity portfolio (as provided)
    equity_portfolio = PORTFOLIOS["equity"]

    llm = OpenAI(temperature=0)
    
    # Run the pipeline for the EquityAgent
    equity_agent.ruy_pipeline(start_date, end_date)
    
    # Output the result (structured view containing expected return, confidence, etc.)
    print("Equity Agent Forecast View:")
    print(result)