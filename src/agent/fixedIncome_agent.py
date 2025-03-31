import os
import sys
import time
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred
from openai import OpenAI
from PortfolioAgent import *
from DataCollector import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from config.settings import *

# ----------------------------
# API Configurations and Schema
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# JSON schema for structured output (we reuse the same schema for simplicity)
PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_return": {
            "type": "number",
            "description": "Predicted return for the ETF tomorrow (as a decimal)"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level in the prediction"
        },
        "rationale": {
            "type": "string",
            "description": "Brief reasoning behind the prediction"
        }
    },
    "required": ["predicted_return", "confidence", "rationale"]
}

# ----------------------------
# Fixed Income Data Download Functions
# ----------------------------
def download_daily_fred_series(api_key, start_date="2020-01-01", end_date=None):
    fred_local = Fred(api_key=api_key)
    series_dict = {
        "EFFR": "EFFR",          # Effective Federal Funds Rate
        "Headline_PCE": "PCE",   # Headline PCE Price Index
        "Core_PCE": "PCEPILFE",  # Core PCE Price Index
        "3M_Yield": "DGS3MO",    # 3-Month Treasury Constant Maturity Rate
        "6M_Yield": "DGS6MO",    # 6-Month Treasury Constant Maturity Rate
        "1Y_Yield": "DGS1",      # 1-Year Treasury Constant Maturity Rate
        "2Y_Yield": "DGS2",      # 2-Year Treasury Constant Maturity Rate
        "5Y_Yield": "DGS5",      # 5-Year Treasury Constant Maturity Rate
        "10Y_Yield": "DGS10",    # 10-Year Treasury Constant Maturity Rate
        'IRLTLT01EZM156N': 'EUR_T10Y',  # Euro Area 10-Year Rate
        'IRLTLT01JPM156N': 'JPY_T10Y',  # Japan 10-Year Rate
        'IRLTLT01GBM156N': 'GBP_T10Y',  # UK 10-Year Rate
    }
    
    data_frames = {}
    for label, series_id in series_dict.items():
        try:
            series_data = fred_local.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df_series = series_data.to_frame(name=label)
            data_frames[label] = df_series
        except Exception as e:
            logger.error(f"Error downloading {series_id}: {e}")
    
    df = pd.concat(data_frames.values(), axis=1)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_index)
    df = df.fillna(method='ffill')
    return df
# ----------------------------
# Fixed Income Specific Implementations
# ----------------------------
class FixedIncomeDataCollector(DataCollector):
    """
    Downloads macro data from FRED and full price/volume data for a fixed income portfolio.
    Combines the macro indicators with ETF price data into a single DataFrame.
    """
    def __init__(self, portfolio: dict, macro_start_date: str = "2020-01-01", target_start_date: str = "2023-11-01", end_date: str = "2025-02-28"):
        self.portfolio = portfolio
        self.macro_start_date = macro_start_date
        self.target_start_date = target_start_date
        self.end_date = end_date

    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        # Extract ETF tickers from the portfolio
        tickers = [entry["etf"] for entry in self.portfolio.get("treasuries", [])]
        # Download ETF data (price, volume, VWAP)
        etf_data = self.download_etf_full_data(tickers, self.macro_start_date, self.end_date)
        # For simplicity, select the 'Close' price from the MultiIndex columns and flatten the columns.
        if not etf_data.empty:
            close_prices = etf_data.xs('Close', axis=1, level=1)
            close_prices.columns = [col for col in close_prices.columns]
        else:
            close_prices = pd.DataFrame()

        # Download macro data
        macro_data = download_daily_fred_series(api_key=FRED_API_KEY, start_date=self.macro_start_date, end_date=self.end_date)
        # Merge the two datasets on the date index
        combined = pd.merge(close_prices, macro_data, left_index=True, right_index=True, how="inner")
        # Filter for the target period
        combined = combined[self.target_start_date:self.end_date]
        combined = combined.dropna()
        return combined

class FixedIncomeAgent(PortfolioAgent):
    """
    FixedIncomeAgent processes a fixed income ETF portfolio.
    It uses a FixedIncomeDataCollector to gather both ETF price data and macro indicators,
    then builds a detailed prompt for each day to forecast tomorrow's returns for each treasury ETF.
    """
    def __init__(self, data_collector: FixedIncomeDataCollector, llm_client: OpenAI):
        super().__init__(name="FixedIncomeAgent", data_collector=data_collector, llm_client=llm_client)
        # You might extend the prediction schema or prompt details if needed.
    
    def prepare_prompt(self, row: pd.Series) -> str:
        # Build a prompt that includes the fixed income portfolio ETF data and macro indicators.
        prompt_lines = []
        prompt_lines.append("Based on the following market data, predict tomorrow's return for each treasury ETF in the fixed income portfolio and provide a brief rationale with a confidence level.\n")
        prompt_lines.append("ETF Price Data (Close Prices):")
        # Assume the ETF tickers are the columns coming from the ETF data portion.
        for etf in row.index:
            # Skip macro columns by checking if etf is not one of the known macro indicators.
            if etf in ["EFFR", "Headline_PCE", "Core_PCE", "3M_Yield", "6M_Yield", "1Y_Yield", "2Y_Yield", "5Y_Yield", "10Y_Yield", "EUR_T10Y", "JPY_T10Y", "GBP_T10Y"]:
                continue
            prompt_lines.append(f"- {etf} Close: {row[etf]:.2f}")
        prompt_lines.append("\nMacro Indicators:")
        macro_keys = ["EFFR", "Headline_PCE", "Core_PCE", "3M_Yield", "6M_Yield", "1Y_Yield", "2Y_Yield", "5Y_Yield", "10Y_Yield", "EUR_T10Y", "JPY_T10Y", "GBP_T10Y"]
        for key in macro_keys:
            value = row.get(key, None)
            if value is not None:
                prompt_lines.append(f"- {key}: {value:.2f}")
        prompt = "\n".join(prompt_lines)
        return prompt

    def get_llm_analysis(self, prompt: str) -> dict:
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial market expert specialized in fixed income analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object", "schema": PREDICTION_SCHEMA, "strict": True},
                temperature=0.7,
                max_tokens=300
            )
            prediction_str = response.choices[0].message.content.strip()
            prediction = json.loads(prediction_str)
            return prediction
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return {"predicted_return": None, "confidence": None, "rationale": f"Error: {str(e)}"}

# ----------------------------
# Example Execution of the Fixed Income Agent Pipeline
# ----------------------------
if __name__ == "__main__":
    # Define the fixed income portfolio
    fixed_income_portfolio = PORTFOLIOS['bond']['treasuries']
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Instantiate the data collector and agent
    fixed_income_data_collector = FixedIncomeDataCollector(
        portfolio=fixed_income_portfolio,
        macro_start_date="2020-01-01",
        target_start_date="2023-11-01",
        end_date="2025-02-28"
    )
    fixed_income_agent = FixedIncomeAgent(data_collector=fixed_income_data_collector, llm_client=client)
    
    # Define the time window for data collection/prediction
    start_date = "2023-11-01"
    end_date = "2025-02-28"
    
    # Run the pipeline and print the resulting predictions
    result_df = fixed_income_agent.run_pipeline(start_date, end_date)
    print("Fixed Income Agent Predictions:")
    print(result_df)
    result_df.to_csv('data/fixed_income_predictions.csv')
