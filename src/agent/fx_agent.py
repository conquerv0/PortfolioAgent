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
from PortfolioAgent import *
from DataCollector import *
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from config.settings import *

# ----------------------------
# Configure APIs and Schema
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Define the JSON schema for structured output
PREDICTION_SCHEMA = {
    "name": "fx_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "instruments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "instrument": {
                            "type": "string",
                            "enum": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD"],
                            "description": "The currency pair"
                        },
                        "predicted_return": {
                            "type": "number",
                            "description": "Predicted return for the currency pair (as a decimal)"
                        },
                        "predicted_volatility": {
                            "type": "number",
                            "description": "Predicted volatility for the currency pair (as a decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the prediction (0-1)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reasoning for this specific currency pair"
                        }
                    },
                    "required": ["instrument", "predicted_return", "predicted_volatility", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Overall market analysis and cross-currency factors"
            }
        },
        "required": ["instruments", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}

# ----------------------------
# FX Data Collector Functions 
# ----------------------------
def get_momentum_factors(df):
    """Calculate momentum factors"""
    momentum_data = df.copy()
    
    # Calculate various momentum indicators
    for ticker in df.columns:
        # Fill NaN values before calculating momentum
        series = df[ticker].ffill()
        # 1-month momentum
        momentum_data[f'{ticker}_mom_1m'] = series.pct_change(periods=21, fill_method=None)
        # 3-month momentum
        momentum_data[f'{ticker}_mom_3m'] = series.pct_change(periods=63, fill_method=None)
        # 12-month momentum
        momentum_data[f'{ticker}_mom_12m'] = series.pct_change(periods=252, fill_method=None)
    
    return momentum_data


def get_ewma_factor(df, span=21):
    """
    Calculate Exponentially Weighted Moving Average of returns for each currency
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data for currencies
    span : int
        Span parameter for EWMA (default: 21 days, roughly a trading month)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with EWMA values for each currency
    """
    ewma_data = pd.DataFrame(index=df.index)
    
    for ticker in df.columns:
        # Calculate daily returns first
        returns = df[ticker].pct_change()
        # Calculate EWMA of returns
        ewma_data[f'{ticker}_ewma_1m'] = returns.ewm(span=span).mean()
        
    return ewma_data


def get_risk_sentiment_data(start_date='2020-01-01', end_date='2025-03-31'):
    """Get risk sentiment data"""
    # Download market indices
    indices = {
        '^VIX': 'VIX',           # Volatility Index
        '^MOVE': 'MOVE'          # ICE BofA MOVE Index
    }
    
    risk_data = pd.DataFrame()
    
    for ticker, name in indices.items():
        try:
            print(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if data.empty:
                print(f"Warning: No data received for {ticker}")
                continue
            risk_data[name] = data
            print(f"Successfully downloaded {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    if risk_data.empty:
        print("Warning: No risk sentiment data was collected")
    else:
        print(f"Collected risk sentiment data with columns: {risk_data.columns.tolist()}")
    
    return risk_data

def get_interest_rates(start_date='2020-01-01', end_date='2025-03-31'):
    """Get 10-year interest rates for major economies from local CSV files"""
    try:
        # Define file paths and column mappings for major countries
        rate_files = {
            'US_T10Y': 'data/Government Bond Yield Data/United States 10-Year Bond Yield Historical Data.csv',
            'GBP_T10Y': 'data/Government Bond Yield Data/United Kingdom 10-Year Bond Yield Historical Data.csv',
            'JPY_T10Y': 'data/Government Bond Yield Data/Japan 10-Year Bond Yield Historical Data.csv',
            'EUR_T10Y': 'data/Government Bond Yield Data/Germany 10-Year Bond Yield Historical Data.csv',
            'CHF_T10Y': 'data/Government Bond Yield Data/Switzerland 10-Year Bond Yield Historical Data.csv',
            'CAD_T10Y': 'data/Government Bond Yield Data/Canada 10-Year Bond Yield Historical Data.csv'
        }
        
        rates_data = pd.DataFrame()
        
        # Read each file and combine the data
        for name, file_path in rate_files.items():
            try:
                # Read CSV file
                data = pd.read_csv(file_path)
                
                # Convert Date column to datetime
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Set Date as index and sort
                data.set_index('Date', inplace=True)
                data.sort_index(inplace=True)
                
                # Get the Price column (which contains the yield)
                rates_data[name] = data['Price']
                
                print(f"Successfully read {name}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if not rates_data.empty:
            # Ensure index is sorted
            rates_data.sort_index(inplace=True)
            
            # Filter for the desired date range
            mask = (rates_data.index >= pd.to_datetime(start_date)) & (rates_data.index <= pd.to_datetime(end_date))
            rates_data = rates_data[mask]
            
            # Forward fill missing values (some rates might have gaps)
            rates_data = rates_data.ffill()
            
            print(f"Successfully processed interest rates data with shape: {rates_data.shape}")
        else:
            print("Warning: No interest rates data was collected")
        
        return rates_data
    
    except Exception as e:
        print(f"Error processing interest rates data: {e}")
        return pd.DataFrame()


def calculate_historical_volatility(df, windows=[21, 63, 126]):
    """
    Calculate historical volatility using rolling standard deviation of returns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data for currencies
    windows : list
        List of rolling window sizes (default: [21, 63, 126] for 1m, 3m, 6m)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with volatility values for each currency and window
    """
    vol_data = pd.DataFrame(index=df.index)
    
    for ticker in df.columns:
        # Calculate daily returns
        returns = df[ticker].pct_change().dropna()
        
        # Calculate rolling volatility for each window
        window_to_month = {21: "1m", 63: "3m", 126: "6m"}
        for window in windows:
            # Annualize the volatility (sqrt(252) is the annualization factor for daily data)
            vol_data[f'{ticker}_vol_{window_to_month[window]}'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    return vol_data

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
    def __init__(self, portfolio: dict, full_start_date: str = '2020-01-01', target_start_date: str = '2023-11-01', end_date: str = '2025-02-28'):
        self.portfolio = portfolio
        self.full_start_date = full_start_date
        self.target_start_date = target_start_date
        self.end_date = end_date
    
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        tickers = [entry["etf"] for entry in self.portfolio]
        logger.info("Starting fixed income data collection...")
        etf_data = self.get_etf_data(tickers, self.full_start_date, self.end_date)
        print(f"Collected ETF data with shape: {etf_data.shape}")
            
        # Calculate momentum factors
        print("\nCalculating momentum factors...")
        data_with_momentum = get_momentum_factors(etf_data)
        print(f"Added momentum factors. Shape: {data_with_momentum.shape}")
        
        # Calculate EWMA for ETF returns
        print("\nCalculating EWMA of returns...")
        ewma_data = get_ewma_factor(etf_data)
        print(f"Added EWMA data. Shape: {ewma_data.shape}")
        
        # Calculate historical volatility
        print("\nCalculating historical volatility...")
        vol_data = calculate_historical_volatility(etf_data)
        print(f"Added volatility data. Shape: {vol_data.shape}")
        
        # Get risk sentiment data
        print("\nCollecting risk sentiment data...")
        risk_data = get_risk_sentiment_data(self.full_start_date, end_date)
        if risk_data.empty:
            print("Warning: No risk sentiment data collected")
        else:
            print(f"Collected risk sentiment data. Shape: {risk_data.shape}")
        
        # Get interest rates data
        print("\nCollecting interest rates data...")
        rates_data = get_interest_rates(self.full_start_date, end_date)
        if not rates_data.empty:
            print(f"Collected interest rates data. Shape: {rates_data.shape}")
        
        # Combine all data
        print("\nCombining all data...")
        combined_data = pd.concat([
            data_with_momentum,
            ewma_data,
            vol_data,
            risk_data,
            rates_data
        ], axis=1)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Filter for target period
        combined_data = combined_data[self.target_start_date:end_date]
        print(f"Data filtered to target period. Shape: {combined_data.shape}")
        
        # Drop rows with missing ETF data
        initial_rows = len(combined_data)
        combined_data = combined_data.dropna(subset=etf_data.columns)
        dropped_rows = initial_rows - len(combined_data)
        print(f"Dropped {dropped_rows} rows with missing ETF data. Remaining rows: {len(combined_data)}")
        
        if len(combined_data) == 0:
            print("Error: No data remaining after filtering")
            return
        
        # Save daily data to CSV
        print("\nSaving daily data to CSV...")
        output_file_daily = 'data/fx_combined_features_daily.csv'
        combined_data.to_csv(output_file_daily)
        print(f"Daily data saved to '{output_file_daily}'")
        
        # Create weekly data
        print("\nCreating weekly data...")
        # Resample to weekly (end of week)
        weekly_data = combined_data.resample('W-FRI').last()
        
        # Calculate weekly changes for 10-year interest rates
        print("Calculating weekly changes for 10-year interest rates...")
        for rate in rates_data.columns:
            weekly_data[f'{rate}_weekly_change'] = weekly_data[rate].pct_change()

        # Calculate weekly changes for risk sentiment data
        print("Calculating weekly changes for risk sentiment data...")
        for sentiment in risk_data.columns:
            weekly_data[f'{sentiment}_weekly_change'] = weekly_data[sentiment].pct_change()
        
        # Save weekly data to CSV
        print("\nSaving weekly data to CSV...")
        output_file_weekly = 'data/fx_combined_features_weekly.csv'
        weekly_data.to_csv(output_file_weekly)
        print(f"Weekly data collection completed. Final weekly dataset shape: {weekly_data.shape}")
        print(f"Weekly data saved to '{output_file_weekly}'")
        return weekly_data

class FXAgent(PortfolioAgent):
    """
    FXAgent uses FXDataCollector to retrieve detailed features and then
    performs deep LLM analysis (via GPT‑4o) to forecast UUP's next-day return.
    """
    def __init__(self, data_collector: FXDataCollector, llm_client: OpenAI):
        super().__init__(name="FXAgent", data_collector=data_collector, llm_client=llm_client)
        # You might extend the prediction schema or prompt details if needed.

    def prepare_prompt(self, row: pd.Series) -> str:
        # Build a detailed prompt that includes all the necessary market data.
        prompt = """
        Based on the following weekly financial market data, predict the next week's returns and volatility for the major currency pairs.
        This week's market data:
        1. Currency ETFs:
        - Euro (FXE): {FXE:.4f} (mom_1m: {FXE_mom_1m:.4f}, mom_3m: {FXE_mom_3m:.4f}, mom_12m: {FXE_mom_12m:.4f}, ewma_1m: {FXE_ewma_1m:.4f})
          Historical Vol: 1m: {FXE_vol_1m:.4f}, 3m: {FXE_vol_3m:.4f}, 6m: {FXE_vol_6m:.4f}
        - British Pound (FXB): {FXB:.4f} (mom_1m: {FXB_mom_1m:.4f}, mom_3m: {FXB_mom_3m:.4f}, mom_12m: {FXB_mom_12m:.4f}, ewma_1m: {FXB_ewma_1m:.4f})
          Historical Vol: 1m: {FXB_vol_1m:.4f}, 3m: {FXB_vol_3m:.4f}, 6m: {FXB_vol_6m:.4f}
        - Japanese Yen (FXY): {FXY:.4f} (mom_1m: {FXY_mom_1m:.4f}, mom_3m: {FXY_mom_3m:.4f}, mom_12m: {FXY_mom_12m:.4f}, ewma_1m: {FXY_ewma_1m:.4f})
          Historical Vol: 1m: {FXY_vol_1m:.4f}, 3m: {FXY_vol_3m:.4f}, 6m: {FXY_vol_6m:.4f}
        - Swiss Franc (FXF): {FXF:.4f} (mom_1m: {FXF_mom_1m:.4f}, mom_3m: {FXF_mom_3m:.4f}, mom_12m: {FXF_mom_12m:.4f}, ewma_1m: {FXF_ewma_1m:.4f})
          Historical Vol: 1m: {FXF_vol_1m:.4f}, 3m: {FXF_vol_3m:.4f}, 6m: {FXF_vol_6m:.4f}
        - Canadian Dollar (FXC): {FXC:.4f} (mom_1m: {FXC_mom_1m:.4f}, mom_3m: {FXC_mom_3m:.4f}, mom_12m: {FXC_mom_12m:.4f}, ewma_1m: {FXC_ewma_1m:.4f})
          Historical Vol: 1m: {FXC_vol_1m:.4f}, 3m: {FXC_vol_3m:.4f}, 6m: {FXC_vol_6m:.4f}

        2. Interest Rates and Changes:
        - US 10Y: {US_T10Y:.2f}% (Weekly Δ: {US_T10Y_weekly_change:.3f}%)
        - EUR 10Y: {EUR_T10Y:.2f}% (Weekly Δ: {EUR_T10Y_weekly_change:.3f}%)
        - GBP 10Y: {GBP_T10Y:.2f}% (Weekly Δ: {GBP_T10Y_weekly_change:.3f}%)
        - JPY 10Y: {JPY_T10Y:.2f}% (Weekly Δ: {JPY_T10Y_weekly_change:.3f}%)
        - CHF 10Y: {CHF_T10Y:.2f}% (Weekly Δ: {CHF_T10Y_weekly_change:.3f}%)
        - CAD 10Y: {CAD_T10Y:.2f}% (Weekly Δ: {CAD_T10Y_weekly_change:.3f}%)

        3. Risk Sentiment:
        - VIX Index: {VIX:.2f} (Weekly Δ: {VIX_weekly_change:.3f})
        - MOVE Index: {MOVE:.2f} (Weekly Δ: {MOVE_weekly_change:.3f})

        For each of the major currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD):
        1. Predict next week's return as a decimal (e.g., 0.0025 for a 0.25% increase)
        2. Provide a confidence score between 0 and 1 (where 0 is no confidence and 1 is complete certainty)
        3. Give a brief rationale specific to each currency pair
        4. Provide an overall market analysis

        Your response must be structured in the required JSON format."""

        # Handle potential missing values
        formatted_prompt = ""
        try:
            formatted_prompt = prompt.format(**row.to_dict())
        except KeyError as e:
            print(f"Warning: Missing data for key {e}. Using default values.")
            # Create a copy with missing values filled
            row_copy = row.copy()
            for col in ['FXE_mom_1m', 'FXE_mom_3m', 'FXE_mom_12m', 'FXE_ewma_1m',
                        'FXE_vol_1m', 'FXE_vol_3m', 'FXE_vol_6m',
                        'FXB_mom_1m', 'FXB_mom_3m', 'FXB_mom_12m', 'FXB_ewma_1m',
                        'FXB_vol_1m', 'FXB_vol_3m', 'FXB_vol_6m',
                        'FXY_mom_1m', 'FXY_mom_3m', 'FXY_mom_12m', 'FXY_ewma_1m',
                        'FXY_vol_1m', 'FXY_vol_3m', 'FXY_vol_6m',
                        'FXF_mom_1m', 'FXF_mom_3m', 'FXF_mom_12m', 'FXF_ewma_1m',
                        'FXF_vol_1m', 'FXF_vol_3m', 'FXF_vol_6m',
                        'FXC_mom_1m', 'FXC_mom_3m', 'FXC_mom_12m', 'FXC_ewma_1m',
                        'FXC_vol_1m', 'FXC_vol_3m', 'FXC_vol_6m',
                        'US_T10Y_weekly_change', 'EUR_T10Y_weekly_change',
                        'GBP_T10Y_weekly_change', 'JPY_T10Y_weekly_change', 'CHF_T10Y_weekly_change', 'CAD_T10Y_weekly_change',
                        'VIX_weekly_change', 'MOVE_weekly_change']:
                if col not in row_copy or pd.isna(row_copy[col]):
                    row_copy[col] = 0.0
            formatted_prompt = prompt.format(**row_copy.to_dict())
        
        return formatted_prompt

    def get_gpt4o_prediction(self, prompt: str) -> dict:
        """
        Get prediction from GPT-4o API with structured output
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o mini
                messages=[
                    {"role": "system", "content": "You are a financial market expert providing accurate predictions for major currency movements. Your analysis should be based on fundamental and technical factors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
                temperature=0
            )
            
            # Extract the JSON response
            prediction_str = response.choices[0].message.content.strip()
            prediction = json.loads(prediction_str)
            
            return prediction
        except Exception as e:
            print(f"Error getting prediction: {e}")
            # Return a structured error response
            return {
                "instruments": [
                    {"instrument": "EUR/USD", "predicted_return": None, "predicted_volatility": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "GBP/USD", "predicted_return": None, "predicted_volatility": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "USD/JPY", "predicted_return": None, "predicted_volatility": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "USD/CHF", "predicted_return": None, "predicted_volatility": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "USD/CAD", "predicted_return": None, "predicted_volatility": None, "confidence": 0, "rationale": f"Error: {str(e)}"}
                ],
                "overall_analysis": f"Failed to generate predictions due to error: {str(e)}"
            }
    
    def run_pipeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        print("Loading weekly data...")
        if os.path.exists('data/fx_combined_features_weekly.csv'):
            data = pd.read_csv('data/fx_combined_features_weekly.csv', index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            data = self.data_collector.collect_data(start_date, end_date)
        
        # Initialize list to store predictions
        predictions = []
        dates = []
        
        # Get predictions for each week
        total_weeks = len(data)
        for i, (date, row) in enumerate(data.iterrows(), 1):
            print(f"Processing week ending {date} ({i}/{total_weeks})...")
            
            # Prepare prompt with current week's data
            prompt = self.prepare_prompt(row)
            
            # Get prediction from GPT-4o
            prediction = self.get_gpt4o_prediction(prompt)
            
            # Store results
            predictions.append(prediction)
            dates.append(date)
            
            # Sleep to respect API rate limits
            time.sleep(1)
            
            # Save progress every 5 predictions
            if i % 5 == 0 or i == total_weeks:
                # Process and save the predictions in a more analysis-friendly format
                processed_data = {
                    'date': [],
                    'etf': [],  
                    'instrument': [],
                    'predicted_return': [],
                    'predicted_volatility': [],
                    'confidence': [],
                    'rationale': [],
                    'overall_analysis': []
                }
                
                # Map currency pairs to their corresponding ETFs
                pair_to_etf = {
                    "EUR/USD": "FXE",
                    "GBP/USD": "FXB",
                    "USD/JPY": "FXY",
                    "USD/CHF": "FXF",
                    "USD/CAD": "FXC"
                }
                
                for j, pred in enumerate(predictions):
                    pred_date = dates[j]
                    overall = pred.get('overall_analysis', '')
                    
                    for inst_data in pred.get('instruments', []):
                        instrument = inst_data.get('instrument', '')
                        processed_data['date'].append(pred_date)
                        processed_data['etf'].append(pair_to_etf.get(instrument, ""))  
                        processed_data['instrument'].append(instrument)
                        processed_data['predicted_return'].append(inst_data.get('predicted_return'))
                        processed_data['predicted_volatility'].append(inst_data.get('predicted_volatility'))
                        processed_data['confidence'].append(inst_data.get('confidence'))
                        processed_data['rationale'].append(inst_data.get('rationale', ''))
                        processed_data['overall_analysis'].append(overall)
                
                # Create DataFrame and save
                pred_df = pd.DataFrame(processed_data)
                pred_df.to_csv(f'data/temp/fx_weekly_predictions_temp_{i}.csv', index=False)
                print(f"Progress saved: {i}/{total_weeks} predictions")
        
        # Process all predictions for final output
        processed_data = {
            'date': [],
            'etf': [],  
            'instrument': [],
            'predicted_return': [],
            'predicted_volatility': [],
            'confidence': [],
            'rationale': [],
            'overall_analysis': []
        }
        
        # Map currency pairs to their corresponding ETFs
        pair_to_etf = {
            "EUR/USD": "FXE",
            "GBP/USD": "FXB",
            "USD/JPY": "FXY",
            "USD/CHF": "FXF",
            "USD/CAD": "FXC"
        }
        
        for j, pred in enumerate(predictions):
            pred_date = dates[j]
            overall = pred.get('overall_analysis', '')
            
            for inst_data in pred.get('instruments', []):
                instrument = inst_data.get('instrument', '')
                processed_data['date'].append(pred_date)
                processed_data['etf'].append(pair_to_etf.get(instrument, ""))  
                processed_data['instrument'].append(instrument)
                processed_data['predicted_return'].append(inst_data.get('predicted_return'))
                processed_data['predicted_volatility'].append(inst_data.get('predicted_volatility'))
                processed_data['confidence'].append(inst_data.get('confidence'))
                processed_data['rationale'].append(inst_data.get('rationale', ''))
                processed_data['overall_analysis'].append(overall)
        
        # Create final DataFrame
        final_df = pd.DataFrame(processed_data)
        
        # Save predictions
        final_df.to_csv('data/fx_weekly_predictions.csv', index=False)
        print("Weekly predictions completed and saved to 'data/fx_weekly_predictions.csv'")

        # Also save the raw predictions
        raw_df = pd.DataFrame({'date': dates, 'predictions': predictions})
        raw_df.to_json('data/fx_weekly_predictions_raw.json', orient='records')
        print("Raw predictions saved to 'data/fx_weekly_predictions_raw.json'")
        return final_df


if __name__ == "__main__":
    fx_portfolio = PORTFOLIOS['fx']['currencies']
    client = OpenAI(api_key=OPENAI_API_KEY)

    fx_data_collector = FXDataCollector(
        portfolio=fx_portfolio,
        full_start_date="2020-01-01", 
        target_start_date="2023-11-01", 
        end_date="2025-03-31")
    
    fx_agent = FXAgent(data_collector=fx_data_collector, llm_client=client)
    
    start_date = "2023-11-01"
    end_date = "2025-03-31"
    
    result_df = fx_agent.run_pipeline(start_date, end_date)
    print("FX Agent Predictions:")
    print(result_df.tail())
