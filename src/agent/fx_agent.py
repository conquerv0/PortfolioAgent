import os
import sys
import time
import json
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from fredapi import Fred
from openai import OpenAI
from agent.portfolioAgent import DataCollector, PortfolioAgent

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRED_API_KEY, OPENAI_API_KEY

# ----------------------------
# Configure APIs and Schema
# ----------------------------
fred = Fred(api_key=FRED_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

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
# FX Data Collector Functions 
# ----------------------------

def get_currency_etf_data(start_date='2020-01-01', end_date='2025-02-28'):
    tickers = ['UUP']
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

def get_exchange_rates(start_date='2020-01-01', end_date='2025-02-28'):
    pairs = ['EURUSD=X', 'JPYUSD=X', 'GBPUSD=X']
    data = yf.download(pairs, start=start_date, end=end_date)
    return data['Close']

def get_momentum_factors(df):
    for ticker in df.columns:
        df[f'{ticker}_mom_1m'] = df[ticker].pct_change(periods=21)
        df[f'{ticker}_mom_3m'] = df[ticker].pct_change(periods=63)
        df[f'{ticker}_mom_12m'] = df[ticker].pct_change(periods=252)
    return df

def get_risk_sentiment_data(start_date='2020-01-01', end_date='2025-02-28'):
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    risk_data = pd.DataFrame()
    risk_data['VIX'] = vix
    spy = yf.download('SPY', start=start_date, end=end_date)['Close']
    risk_data['SPY_returns'] = spy.pct_change()
    return risk_data

def get_interest_rates(start_date='2020-01-01', end_date='2025-02-28'):
    series = {
        'DGS10': 'US_T10Y',
        'IRLTLT01EZM156N': 'EUR_T10Y',
        'IRLTLT01JPM156N': 'JPY_T10Y',
        'IRLTLT01GBM156N': 'GBP_T10Y',
    }
    rates_data = pd.DataFrame()
    for code, name in series.items():
        try:
            data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            rates_data[name] = data
        except Exception as e:
            print(f"Error fetching {code}: {e}")
            continue
    rates_data.index = pd.to_datetime(rates_data.index)
    for col in rates_data.columns:
        rates_data[f'{col}_change'] = rates_data[col].diff()
    rates_data = rates_data.ffill()
    return rates_data

def get_macro_data(start_date='2020-01-01', end_date='2025-02-28'):
    series = {
        'EFFR': 'FedRate',
        'T10Y2Y': 'YieldSpread'
    }
    macro_data = pd.DataFrame()
    for code, name in series.items():
        try:
            data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            macro_data[name] = data
        except Exception as e:
            print(f"Error fetching {code}: {e}")
            continue
    macro_data.index = pd.to_datetime(macro_data.index)
    for col in macro_data.columns:
        macro_data[f'{col}_change'] = macro_data[col].diff()
    macro_data = macro_data.ffill()
    return macro_data

# ----------------------------
# FX-Specific Implementations
# ----------------------------
class FXDataCollector(DataCollector):
    """
    Aggregates FX and macro data:
      - Currency ETF data (with momentum factors)
      - Exchange rates
      - Risk sentiment data
      - Interest rates and macro data
    Combines these into a single DataFrame for the target period.
    """
    def __init__(self, full_start_date: str = '2020-01-01', target_start_date: str = '2023-11-01', end_date: str = '2025-02-28'):
        self.full_start_date = full_start_date
        self.target_start_date = target_start_date
        self.end_date = end_date
    
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        etf_data = get_currency_etf_data(self.full_start_date, self.end_date)
        etf_data = get_momentum_factors(etf_data)
        exchange_rates = get_exchange_rates(self.full_start_date, self.end_date)
        risk_data = get_risk_sentiment_data(self.full_start_date, self.end_date)
        rates_data = get_interest_rates(self.full_start_date, self.end_date)
        macro_data = get_macro_data(self.full_start_date, self.end_date)
        
        combined_data = pd.concat([etf_data, exchange_rates, risk_data, rates_data, macro_data], axis=1)
        combined_data = combined_data[self.target_start_date:self.end_date]
        combined_data = combined_data.dropna(subset=['UUP'])
        return combined_data

class FXAgent(PortfolioAgent):
    """
    FXAgent uses FXDataCollector to retrieve detailed features and then
    performs deep LLM analysis (via GPT‑4o) to forecast UUP’s next-day return.
    """
    def prepare_prompt(self, row: pd.Series) -> str:
        # Build a detailed prompt that includes all the necessary market data.
        prompt = (
            "Based on the following financial market data, predict the next day's return for the US Dollar Index (UUP ETF).\n\n"
            "Today's market data:\n"
            "1. Currency ETF:\n"
            f"- UUP price: {row.get('UUP', 'N/A')}\n"
            f"- UUP 1-month momentum: {row.get('UUP_mom_1m', 0):.4f}\n"
            f"- UUP 3-month momentum: {row.get('UUP_mom_3m', 0):.4f}\n"
            f"- UUP 12-month momentum: {row.get('UUP_mom_12m', 0):.4f}\n\n"
            "2. Exchange Rates:\n"
            f"- EUR/USD: {row.get('EURUSD=X', 0):.4f}\n"
            f"- JPY/USD: {row.get('JPYUSD=X', 0):.4f}\n"
            f"- GBP/USD: {row.get('GBPUSD=X', 0):.4f}\n\n"
            "3. Interest Rates:\n"
            f"- US 10Y Treasury: {row.get('US_T10Y', 0):.2f}%\n"
            f"- Eurozone 10Y: {row.get('EUR_T10Y', 0):.2f}%\n"
            f"- Japan 10Y: {row.get('JPY_T10Y', 0):.2f}%\n"
            f"- UK 10Y: {row.get('GBP_T10Y', 0):.2f}%\n\n"
            "4. Rate Changes:\n"
            f"- US 10Y Change: {row.get('US_T10Y_change', 0):.3f}%\n"
            f"- EUR 10Y Change: {row.get('EUR_T10Y_change', 0):.3f}%\n"
            f"- JPY 10Y Change: {row.get('JPY_T10Y_change', 0):.3f}%\n"
            f"- UK 10Y Change: {row.get('GBP_T10Y_change', 0):.3f}%\n\n"
            "5. Risk Metrics:\n"
            f"- VIX Index: {row.get('VIX', 0):.2f}\n"
            f"- S&P 500 Return: {row.get('SPY_returns', 0):.4f}\n\n"
            "6. Macro Indicators:\n"
            f"- Fed Funds Rate: {row.get('FedRate', 0):.2f}%\n"
            f"- 10Y-2Y Spread: {row.get('YieldSpread', 0):.2f}%\n"
            f"- Fed Rate Change: {row.get('FedRate_change', 0):.3f}%\n"
            f"- Yield Spread Change: {row.get('YieldSpread_change', 0):.3f}%\n\n"
            "Predict tomorrow's return for UUP and provide a brief rationale."
        )
        return prompt

    def get_llm_analysis(self, prompt: str) -> dict:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial market expert providing accurate predictions for currency movements."},
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
            print(f"Error getting LLM analysis: {e}")
            return {"predicted_return": None, "confidence": None, "rationale": f"Error: {str(e)}"}
    
    def run_pipeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        combined_data = self.collect_data(start_date, end_date)
        predictions = []
        dates = []
        for date, row in combined_data.iterrows():
            try:
                prompt = self.prepare_prompt(row)
                analysis = self.get_llm_analysis(prompt)
            except Exception as e:
                analysis = {"error": str(e)}
            predictions.append(analysis)
            dates.append(date)
            time.sleep(1) 
        return pd.DataFrame(predictions, index=dates)

# ----------------------------
# Example Execution of the FX Agent Pipeline
# ----------------------------
if __name__ == "__main__":
    fx_data_collector = FXDataCollector(full_start_date="2020-01-01", target_start_date="2023-11-01", end_date="2025-02-28")
    fx_agent = FXAgent(data_collector=fx_data_collector, llm_client=client)
    
    start_date = "2023-11-01"
    end_date = "2025-02-28"
    
    result_df = fx_agent.run_pipeline(start_date, end_date)
    print("FX Agent Predictions:")
    print(result_df)
    result_df.to_csv('data/fx_llm_predictions_refactored.csv')
