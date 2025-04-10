import os
import sys
import time
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred
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
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

PREDICTION_SCHEMA = {
    "name": "fi_prediction",
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
                            "enum": [
                                "Short-Term Treasury",
                                "1-3 Year Treasury",
                                "3-7 Year Treasury",
                                "7-10 Year Treasury",
                                "10-20 Year Treasury",
                                "20+ Year Treasury"
                            ],
                            "description": "The treasury ETF instrument"
                        },
                        "predicted_return": {
                            "type": "number",
                            "description": "Predicted return for the instrument (as a decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the prediction (0-1)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reasoning for this specific instrument"
                        }
                    },
                    "required": ["instrument", "predicted_return", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Overall fixed income market analysis and related macro insights"
            }
        },
        "required": ["instruments", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}

# ----------------------------
# Fixed Income Data Download Functions
# ----------------------------
def get_fred_series(api_key, start_date, end_date, freq='D'):
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
        "10Y_Yield": "DGS10"   # 10-Year Treasury Constant Maturity Rate
        # 'IRLTLT01EZM156N': 'EUR_T10Y',  # Euro Area 10-Year Rate
        # 'IRLTLT01JPM156N': 'JPY_T10Y',  # Japan 10-Year Rate
        # 'IRLTLT01GBM156N': 'GBP_T10Y'   # UK 10-Year Rate
    }
    
    data_frames = {}
    for label, series_id in series_dict.items():
        try:
            series_data = fred_local.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df_series = series_data.to_frame(name=label)
            data_frames[label] = df_series
            logger.info(f"Downloaded {label} from FRED.")
        except Exception as e:
            logger.error(f"Error downloading {series_id} ({label}): {e}")
    
    df = pd.concat(data_frames.values(), axis=1)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df = df.fillna(method='ffill')
    return df

def get_yield_momentum(df):
    momentum_data = df.copy()
    yield_cols = ["3M_Yield", "6M_Yield", "1Y_Yield", "2Y_Yield", "5Y_Yield", "10Y_Yield"]
    for col in yield_cols:
        if col in df.columns:
            series = df[col].ffill()
            momentum_data[f'{col}_mom_1m'] = series.pct_change(periods=21)
            momentum_data[f'{col}_mom_3m'] = series.pct_change(periods=63)
            momentum_data[f'{col}_mom_12m'] = series.pct_change(periods=252)
            logger.info(f"Computed momentum for {col}.")
    return momentum_data

def get_risk_sentiment_data(start_date, end_date):
    indices = {
        '^VIX': 'VIX',
        '^MOVE': 'MOVE'
    }
    risk_data = pd.DataFrame()
    for ticker, name in indices.items():
        try:
            logger.info(f"Downloading {ticker} data...")
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if not data.empty:
                risk_data[name] = data
                logger.info(f"Successfully downloaded {ticker}.")
            else:
                logger.warning(f"No data received for {ticker}.")
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
    return risk_data

def load_bond_etf_returns(filepath: str) -> pd.DataFrame:
    """
    Load bond ETF data from a CSV file and compute the full daily return for each ETF.
    
    The CSV file is expected to have at least the following columns:
        - date: Trading date
        - ticker: ETF ticker
        - prc: Price (which may be negative for adjustments)
        - ret: Daily price return (as a decimal)
        - divamt: Dividend (or coupon) cash amount
        - vwretd: Value-weighted total return (includes distributions)
        
    The function computes:
        dividend_yield = divamt / abs(prc)   if prc is nonzero, else 0.
        full_return = vwretd (if available) else ret + dividend_yield.
    
    The final DataFrame is pivoted so that the row index is the date and the columns are the ticker names.
    
    Parameters:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Daily full return DataFrame with date as index and ticker as columns.
    """
    # Load the CSV file; ensure the date column is parsed as datetime.
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Ensure the required columns are present.
    required_columns = ['date', 'ticker', 'prc', 'ret', 'divamt', 'vwretd']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    
    # Compute dividend yield when price is nonzero.
    df['dividend_yield'] = df.apply(lambda row: row['divamt'] / abs(row['prc']) if row['prc'] != 0 else 0, axis=1)
    
    # Compute full return:
    # If vwretd is available (non-null), use it; otherwise, add the dividend yield to the price return.
    df['computed_return'] = df.apply(
        lambda row: row['vwretd'] if pd.notnull(row['vwretd']) else (row['ret'] + row['dividend_yield']),
        axis=1
    )
    
    # Pivot the DataFrame so that 'date' becomes the index and each ticker is a column.
    full_return_df = df.pivot(index='date', columns='ticker', values='computed_return')
    full_return_df.fillna(0, inplace=True)
    
    return full_return_df

# ----------------------------
# Fixed Income Specific Implementations
# ----------------------------
class FixedIncomeDataCollector(DataCollector):
    """
    Downloads macro data from FRED and full price/volume data for a fixed income portfolio.
    Combines the macro indicators with ETF price data into a single DataFrame.
    """
    def __init__(self, portfolio: dict, full_start_date: str, target_start_date: str, end_date: str):
        super().__init__(full_start_date, target_start_date, end_date)
        self.portfolio = portfolio

    def get_bond_etf_full_return(self, tickers, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads daily price data for bond ETFs from the CRSP Q library (DSF table) by joining DSF with DSENAMES,
        computes the full daily return (including coupon/dividend yield) for each ETF, and returns a DataFrame with 
        date as the row index and ticker as the column.
        
        The full return is computed as:
        full_return = ret + (divamt/abs(prc))   if prc is nonzero and dividend data is available;
        otherwise, if vwretd is available (and non-null), that is used.
        
        Parameters:
        tickers (list): List of ETF tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns containing the full daily return.
        """
        tickers_str = ",".join(f"'{ticker}'" for ticker in tickers)
        
        # Connect to WRDS (using CRSP Q library)
        db = wrds.Connection(USER_NAME=self.WRDS_USERNAME)
        
        query = f"""
        SELECT a.date, b.ticker, a.prc, a.ret, a.vwretd
        FROM crspq.dsf as a
        JOIN crspq.dsenames as b
        ON a.permno = b.permno
        WHERE b.ticker IN ({tickers_str})
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
        AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '{end_date}')
        ORDER BY a.date, b.ticker
        """
        
        df = db.raw_sql(query)
        db.close()
        
        # Compute dividend yield (if price is nonzero)
        # df['dividend_yield'] = df.apply(lambda row: row['divamt'] / abs(row['prc']) if row['prc'] != 0 else 0, axis=1)
        
        # Compute full return:
        # If vwretd is available (and non-null), use it; otherwise, add the dividend yield to the price return.
        df['full_return'] = df.apply(
            lambda row: row['vwretd'] if pd.notnull(row['vwretd']) else row['ret'],
            axis=1
        )
        
        # Pivot the data so that the index is date and the columns are ticker names.
        df_full_return = df.pivot(index='date', columns='ticker', values='full_return')
        df_full_return.index = pd.to_datetime(df_full_return.index)
        df_full_return.fillna(0, inplace=True)
        
        return df_full_return


    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        # Extract ETF tickers from the portfolio
        tickers = [entry["etf"] for entry in self.portfolio.get("treasuries", [])]

        logger.info("Starting fixed income data collection...")
        
        # Download macro and yield data from FRED
        logger.info("Downloading FRED series...")
        fred_data = get_fred_series(FRED_API_KEY, start_date=self.full_start_date, end_date=self.end_date)
        logger.info(f"Downloaded FRED data with shape: {fred_data.shape}")
        print(fred_data.tail())
        
        # Compute momentum for yield series
        logger.info("Computing yield momentum factors...")
        fred_with_momentum = get_yield_momentum(fred_data)
        logger.info(f"Data with momentum factors shape: {fred_with_momentum.shape}")
        
        # Get risk sentiment data (VIX and MOVE)
        logger.info("Downloading risk sentiment data...")
        risk_data = get_risk_sentiment_data(start_date=self.full_start_date, end_date=self.end_date)
        if risk_data.empty:
            logger.warning("No risk sentiment data collected.")
        else:
            logger.info(f"Risk sentiment data shape: {risk_data.shape}")
        
        # Get Treasury ETF data for the fixed income portfolio
        logger.info("Downloading Treasury ETF data...")
        treasury_etf_data = load_bond_etf_returns('./data/bond_etf.csv')
        if treasury_etf_data.empty:
            logger.warning("No Treasury ETF data collected.")
        else:
            logger.info(f"Treasury ETF data shape: {treasury_etf_data.shape}")
        
        # Combine all data: macro, momentum, risk sentiment, and Treasury ETF prices
        if not isinstance(fred_with_momentum.index, pd.DatetimeIndex):
            fred_with_momentum.index = pd.to_datetime(fred_with_momentum.index)
        if not isinstance(risk_data.index, pd.DatetimeIndex):
            risk_data.index = pd.to_datetime(risk_data.index)
        if not isinstance(treasury_etf_data.index, pd.DatetimeIndex):
            treasury_etf_data.index = pd.to_datetime(treasury_etf_data.index)
            
        # Now concat with axis=1 to preserve row alignment by datetime
        combined_data = pd.concat([fred_with_momentum, risk_data, treasury_etf_data], axis=1)
        
        # Filter for target period
        print(combined_data.tail())
        combined_data = combined_data[self.target_start_date:self.end_date]
        logger.info(f"Data filtered to target period: {combined_data.shape}")
        
        # Drop rows with missing essential yield data
        required_yields = ["3M_Yield", "6M_Yield", "1Y_Yield", "2Y_Yield", "5Y_Yield", "10Y_Yield"]
        initial_rows = len(combined_data)
        combined_data = combined_data.dropna(subset=required_yields)
        logger.info(f"Dropped {initial_rows - len(combined_data)} rows due to missing yield data. Remaining rows: {len(combined_data)}")
        
        # Save daily data to CSV
        daily_file = 'data/fi_combined_features_daily.csv'
        os.makedirs(os.path.dirname(daily_file), exist_ok=True)
        combined_data.to_csv(daily_file)
        logger.info(f"Daily fixed income data saved to '{daily_file}'")
        
        # Create weekly data by resampling (using Friday's data)
        logger.info("Resampling daily data to weekly frequency...")
        weekly_data = combined_data.resample('W-FRI').last()
        
        # Calculate weekly changes for yield series and risk sentiment
        for col in required_yields:
            weekly_data[f'{col}_weekly_change'] = weekly_data[col].pct_change()
        for sentiment in ["VIX", "MOVE"]:
            if sentiment in weekly_data.columns:
                weekly_data[f'{sentiment}_weekly_change'] = weekly_data[sentiment].pct_change()
        
        weekly_file = 'data/fi_combined_features_weekly.csv'
        weekly_data.to_csv(weekly_file)
        logger.info(f"Weekly fixed income data saved to '{weekly_file}'")

        return weekly_data

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
        """
        Prepare a prompt for GPT-4 based on the weekly fixed income market data.
        """
        prompt = """Based on the following weekly fixed income market data, predict the next week's yield changes for the fixed income instruments.

        Macro Indicators:
        - Effective Federal Funds Rate (EFFR): {EFFR:.4f}
        - Headline PCE: {Headline_PCE:.4f}
        - Core PCE: {Core_PCE:.4f}

        US Treasury Yields and Momentum:
        - 3-Month Yield: {3M_Yield:.2f}% (Momentum: 1M: {3M_Yield_mom_1m:.4f}, 3M: {3M_Yield_mom_3m:.4f}, 12M: {3M_Yield_mom_12m:.4f})
        - 6-Month Yield: {6M_Yield:.2f}% (Momentum: 1M: {6M_Yield_mom_1m:.4f}, 3M: {6M_Yield_mom_3m:.4f}, 12M: {6M_Yield_mom_12m:.4f})
        - 1-Year Yield: {1Y_Yield:.2f}% (Momentum: 1M: {1Y_Yield_mom_1m:.4f}, 3M: {1Y_Yield_mom_3m:.4f}, 12M: {1Y_Yield_mom_12m:.4f})
        - 2-Year Yield: {2Y_Yield:.2f}% (Momentum: 1M: {2Y_Yield_mom_1m:.4f}, 3M: {2Y_Yield_mom_3m:.4f}, 12M: {2Y_Yield_mom_12m:.4f})
        - 5-Year Yield: {5Y_Yield:.2f}% (Momentum: 1M: {5Y_Yield_mom_1m:.4f}, 3M: {5Y_Yield_mom_3m:.4f}, 12M: {5Y_Yield_mom_12m:.4f})
        - 10-Year Yield: {10Y_Yield:.2f}% (Momentum: 1M: {10Y_Yield_mom_1m:.4f}, 3M: {10Y_Yield_mom_3m:.4f}, 12M: {10Y_Yield_mom_12m:.4f})

        Risk Sentiment:
        - VIX Index: {VIX:.2f} (Weekly change: {VIX_weekly_change:.3f})
        - MOVE Index: {MOVE:.2f} (Weekly change: {MOVE_weekly_change:.3f})

        Treasury ETF Return:
        - Short-Term Treasury (SHV): {SHV:.2f}
        - 1-3 Year Treasury (SHY): {SHY:.2f}
        - 3-7 Year Treasury (IEI): {IEI:.2f}
        - 7-10 Year Treasury (IEF): {IEF:.2f}
        - 10-20 Year Treasury (TLH): {TLH:.2f}
        - 20+ Year Treasury (TLT): {TLT:.2f}

        Based on the above data, please predict next week's return for the following treasury ETF instruments:
        - Short-Term Treasury (SHV)
        - 1-3 Year Treasury (SHY)
        - 3-7 Year Treasury (IEI)
        - 7-10 Year Treasury (IEF)
        - 10-20 Year Treasury (TLH)
        - 20+ Year Treasury (TLT)

        For each instrument, provide:
        1. The predicted expected return for next week as a decimal,
        2. A confidence score between 0 and 1,
        3. A brief rationale for the prediction.
        4. Provide an overall fixed income market analysis that incorporates these macro indicators, yield levels, momentum signals, and risk sentiment.

        Your response must be structured in the required JSON format.
        """
        try:
            formatted_prompt = prompt.format(**row.to_dict())
        except KeyError as e:
            print(f"Warning: Missing data for key {e}. Filling missing values with 0.0.")
            row_copy = row.copy()
            required_cols = [
                'EFFR', 'Headline_PCE', 'Core_PCE',
                '3M_Yield', '6M_Yield', '1Y_Yield', '2Y_Yield', '5Y_Yield', '10Y_Yield',
                'EUR_T10Y', 'JPY_T10Y', 'GBP_T10Y',
                '3M_Yield_mom_1m', '3M_Yield_mom_3m', '3M_Yield_mom_12m',
                '6M_Yield_mom_1m', '6M_Yield_mom_3m', '6M_Yield_mom_12m',
                '1Y_Yield_mom_1m', '1Y_Yield_mom_3m', '1Y_Yield_mom_12m',
                '2Y_Yield_mom_1m', '2Y_Yield_mom_3m', '2Y_Yield_mom_12m',
                '5Y_Yield_mom_1m', '5Y_Yield_mom_3m', '5Y_Yield_mom_12m',
                '10Y_Yield_mom_1m', '10Y_Yield_mom_3m', '10Y_Yield_mom_12m',
                'VIX', 'VIX_weekly_change', 'MOVE', 'MOVE_weekly_change',
                'SHV', 'SHY', 'IEI', 'IEF', 'TLH', 'TLT'
            ]
            for col in required_cols:
                if col not in row_copy or pd.isna(row_copy[col]):
                    row_copy[col] = 0.0
            formatted_prompt = prompt.format(**row_copy.to_dict())
        
        return formatted_prompt
    
    def get_gpt4o_prediction(self, prompt: str) -> dict:
        """
        Get prediction from GPT-4 API with structured output for fixed income.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o mini
                messages=[
                    {"role": "system", "content": "You are a fixed income market expert. Provide yield change predictions based on macroeconomic, treasury, and risk sentiment data."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
                temperature=0
            )
            prediction_str = response.choices[0].message.content.strip()
            prediction = json.loads(prediction_str)
            return prediction
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return {
                "instruments": [
                    {"instrument": "Short-Term Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "1-3 Year Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "3-7 Year Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "7-10 Year Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "10-20 Year Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                    {"instrument": "20+ Year Treasury", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"}
                ],
                "overall_analysis": f"Failed to generate predictions due to error: {str(e)}"
            }
    

    # def get_llm_analysis(self, prompt: str) -> dict:
    #     try:
    #         response = self.llm_client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=[
    #                 {"role": "system", "content": "You are a financial market expert specialized in fixed income analysis."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             response_format={"type": "json_object", "schema": PREDICTION_SCHEMA, "strict": True},
    #             temperature=0.7,
    #             max_tokens=300
    #         )
    #         prediction_str = response.choices[0].message.content.strip()
    #         prediction = json.loads(prediction_str)
    #         return prediction
    #     except Exception as e:
    #         logger.error(f"Error getting LLM analysis: {e}")
    #         return {"predicted_return": None, "confidence": None, "rationale": f"Error: {str(e)}"}
    def run_pipeline(self, start_date, end_date):
        print("Loading weekly fixed income data...")
        if os.path.exists('data/fi_combined_features_weekly.csv'):
            data = pd.read_csv('data/fi_combined_features_weekly.csv', index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            data = self.data_collector.collect_data(start_date, end_date)
        predictions = []
        dates = []
        total_weeks = len(data)
        
        # Mapping from instrument names to ETF tickers
        instrument_to_etf = {
            "Short-Term Treasury": "SHV",
            "1-3 Year Treasury": "SHY",
            "3-7 Year Treasury": "IEI",
            "7-10 Year Treasury": "IEF",
            "10-20 Year Treasury": "TLH",
            "20+ Year Treasury": "TLT"
        }

        for i, (date, row) in enumerate(data.iterrows(), 1):
            print(f"Processing week ending {date} ({i}/{total_weeks})...")
            prompt = self.prepare_prompt(row)
            prediction = self.get_gpt4o_prediction(prompt)
            predictions.append(prediction)
            dates.append(date)
            time.sleep(1)  # Respect API rate limits
            
            if i % 5 == 0 or i == total_weeks:
                # Process and save the predictions in a more analysis-friendly format
                processed_data = {
                    'date': [],
                    'etf': [],  # New ETF column based on instrument mapping
                    'instrument': [],
                    'predicted_return': [],
                    'confidence': [],
                    'rationale': [],
                    'overall_analysis': []
                }
                for j, pred in enumerate(predictions):
                    pred_date = dates[j]
                    overall = pred.get('overall_analysis', '')
                    for inst in pred.get('instruments', []):
                        instrument = inst.get('instrument', '')
                        processed_data['date'].append(pred_date)
                        # Map instrument to ETF ticker
                        processed_data['etf'].append(instrument_to_etf.get(instrument, ""))
                        processed_data['instrument'].append(instrument)
                        processed_data['predicted_return'].append(inst.get('predicted_return'))
                        processed_data['confidence'].append(inst.get('confidence'))
                        processed_data['rationale'].append(inst.get('rationale', ''))
                        processed_data['overall_analysis'].append(overall)
                
                temp_filename = f'data/temp/fi_weekly_predictions_temp_{i}.csv'
                os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
                temp_df = pd.DataFrame(processed_data)
                temp_df.to_csv(temp_filename, index=False)
                print(f"Progress saved: {i}/{total_weeks} predictions")

        # Final processing of predictions
        processed_data = {
            'date': [],
            'etf': [],
            'instrument': [],
            'predicted_return': [],
            'confidence': [],
            'rationale': [],
            'overall_analysis': []
        }
        for j, pred in enumerate(predictions):
            pred_date = dates[j]
            overall = pred.get('overall_analysis', '')
            for inst in pred.get('instruments', []):
                instrument = inst.get('instrument', '')
                processed_data['date'].append(pred_date)
                processed_data['etf'].append(instrument_to_etf.get(instrument, ""))
                processed_data['instrument'].append(instrument)
                processed_data['predicted_return'].append(inst.get('predicted_return'))
                processed_data['confidence'].append(inst.get('confidence'))
                processed_data['rationale'].append(inst.get('rationale', ''))
                processed_data['overall_analysis'].append(overall)
        
        final_df = pd.DataFrame(processed_data)
        final_csv = 'data/fi_weekly_predictions.csv'
        final_df.to_csv(final_csv, index=False)
        print(f"Weekly fixed income predictions completed and saved to '{final_csv}'")
        
        raw_df = pd.DataFrame({'date': dates, 'predictions': predictions})
        raw_json = 'data/fi_weekly_predictions_raw.json'
        raw_df.to_json(raw_json, orient='records')
        print(f"Raw predictions saved to '{raw_json}'")

        return final_df
        
# ----------------------------
# Example Execution of the Fixed Income Agent Pipeline
# ----------------------------
if __name__ == "__main__":
    # Define the fixed income portfolio
    fixed_income_portfolio = PORTFOLIOS['bond']
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Instantiate the data collector and agent
    fixed_income_data_collector = FixedIncomeDataCollector(
        portfolio=fixed_income_portfolio,
        full_start_date="2023-01-01",
        target_start_date="2023-11-01",
        end_date="2025-12-31"
    )
    fixed_income_agent = FixedIncomeAgent(data_collector=fixed_income_data_collector, llm_client=client)
    
    # Define the time window for data collection/prediction
    start_date = "2023-11-01"
    end_date = "2025-12-31"
    
    # Run the pipeline and print the resulting predictions
    result_df = fixed_income_agent.run_pipeline(start_date, end_date)
    print("Fixed Income Agent Predictions:")
    print(result_df.tail())
