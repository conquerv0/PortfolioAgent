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
import textwrap
from statsmodels.tsa.arima.model import ARIMA
from string import Formatter

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
                        "variance_view": {
                            "type": "number",
                            "description": "Alpha adjustment v.s. baseline return"
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
                    "required": ["instrument", "variance_view", "confidence", "rationale"],
                    "additionalProperties": False,
                    "example":    {
                    "predictions": 
                        {
                        "instrument":"1-3 Year Treasury",
                        "variance_view":0.0050,
                        "confidence":0.85,
                        "rationale":"EFFR_weekly_change -0.015 & VIX_weekly_change −0.1 signal market expects near term easing, short term treasury outperforming."
                        },
                    },
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Cross-sector and macro narrative"
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
    def get_yield_momentum(self, df):
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

    def get_fred_data(self, start_date: str, end_date: str, freq: str = 'D') -> pd.DataFrame:
        fred_local = Fred(api_key=self.FRED_API_KEY)
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
        fred_data = self.get_fred_data(start_date=self.full_start_date, end_date=self.end_date)
        logger.info(f"Downloaded FRED data with shape: {fred_data.shape}")
        print(fred_data.tail())
        
        # Compute momentum for yield series
        logger.info("Computing yield momentum factors...")
        fred_with_momentum = self.get_yield_momentum(fred_data)
        logger.info(f"Data with momentum factors shape: {fred_with_momentum.shape}")
        
        # Get risk sentiment data (VIX and MOVE)
        logger.info("Downloading risk sentiment data...")
        risk_data = self.get_risk_sentiment_data(start_date=self.full_start_date, end_date=self.end_date)
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
        daily_file = 'data/features/fi_combined_features_daily.csv'
        os.makedirs(os.path.dirname(daily_file), exist_ok=True)
        combined_data.to_csv(daily_file)
        logger.info(f"Daily fixed income data saved to '{daily_file}'")
        
        # Create weekly data by resampling (using Friday's data)
        logger.info("Resampling daily data to weekly frequency...")
        weekly_data = combined_data.resample('W-FRI').last()
        
        # Calculate weekly changes for yield series and risk sentiment
        macro_cols = ['EFFR', 'Headline_PCE', 'Core_PCE', '3M_Yield', '6M_Yield', '1Y_Yield', '2Y_Yield', '5Y_Yield', '10Y_Yield']
        for col in macro_cols + ["VIX", "MOVE"] :
            if col in weekly_data.columns:
                weekly_data[f"{col}_weekly_change"] = weekly_data[col].pct_change().fillna(0)

        weekly_file = 'data/features/fi_combined_features_weekly.csv'
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
    
    
    def estimate_returns(self,
                         daily_ret_df: pd.DataFrame,
                         tickers: list,
                         span_days: int = 21
                        ) -> pd.DataFrame:
        """
        Compute an EWMA‐based baseline of 1‐week expected returns for each ETF.
        - daily_ret_df: DataFrame of daily total returns (decimal) for each ticker.
        - tickers: list of ETF tickers in daily_ret_df.
        - span_days: lookback span for the EWMA (default ~1 trading month).
        """
        # 1) EWMA of daily returns
        ewma_daily = daily_ret_df[tickers].ewm(span=span_days, adjust=False).mean()
        # 2) take last EWMA on each Friday as the "expected" weekly return
        weekly_baseline = ewma_daily.resample('W-FRI').last()
        # 3) rename columns
        weekly_baseline.columns = [f"{tck}_baseline_ret" for tck in tickers]
        return weekly_baseline

    def estimate_returns_ARIMA(self,
                               daily_ret_df: pd.DataFrame,
                               tickers: list
                              ) -> pd.DataFrame:
        """
        Fit an ARIMA(1,0,0) on the **weekly** returns series, then one‐step forecast.
        - First aggregates daily returns into weekly returns: (1+rt).prod()−1.
        - Then for each ETF, fits ARIMA on that weekly series and forecasts next week.
        Returns a DataFrame indexed by the same Fridays, with the forecast in 
        columns ['SHV_baseline_ret', …].
        """
        # aggregate to weekly total returns
        weekly_ret = daily_ret_df[tickers] \
                         .resample('W-FRI') \
                         .apply(lambda x: (1 + x).prod() - 1) \
                         .dropna()

        # placeholder for forecasts
        forecast_df = pd.DataFrame(index=weekly_ret.index,
                                   columns=[f"{t}_baseline_ret" for t in tickers],
                                   dtype=float)

        for t in tickers:
            series = weekly_ret[t].dropna()
            try:
                model     = ARIMA(series, order=(1, 0, 0))
                fit       = model.fit()
                # one-step forecast → a length‐1 array
                fc        = fit.forecast(steps=1)
                # align the forecast to the next Friday index
                forecast_df[f"{t}_baseline_ret"] = fc.values
            except Exception as e:
                logger.error(f"ARIMA baseline failed for {t}: {e}")
                forecast_df[f"{t}_baseline_ret"] = None

        return forecast_df

    def prepare_prompt(self, row: pd.Series, default: float = 0.0) -> str:
        """
        Prepare a prompt for GPT-4 based on the weekly fixed income market data.
        """
        PROMPT_TEMPLATE = textwrap.dedent("""Based only on the given data for the week that just passed:
        Some U.S. macro indicator changes, risk-sentiment proxy, yield of treasury bond etf of different maturities and baseline return estimated by a quantitative mode
        and some technical features features of the etf returns of given period: 

        ▌Macro indicator changes:
        • Effective Fed Funds Rate (EFFR): Δ {EFFR_weekly_change:.4f}%  
        • Headline PCE: Δ {Headline_PCE_weekly_change:.4f}%  
        • Core PCE: Δ {Core_PCE_weekly_change:.4f}% 

        ▌US Treasury Yields and Momentum:
        - 3-Month Yield: {3M_Yield:.2f}% (Momentum: 1M: {3M_Yield_mom_1m:.4f}, 3M: {3M_Yield_mom_3m:.4f}, 12M: {3M_Yield_mom_12m:.4f})
        - 6-Month Yield: {6M_Yield:.2f}% (Momentum: 1M: {6M_Yield_mom_1m:.4f}, 3M: {6M_Yield_mom_3m:.4f}, 12M: {6M_Yield_mom_12m:.4f})
        - 1-Year Yield: {1Y_Yield:.2f}% (Momentum: 1M: {1Y_Yield_mom_1m:.4f}, 3M: {1Y_Yield_mom_3m:.4f}, 12M: {1Y_Yield_mom_12m:.4f})
        - 2-Year Yield: {2Y_Yield:.2f}% (Momentum: 1M: {2Y_Yield_mom_1m:.4f}, 3M: {2Y_Yield_mom_3m:.4f}, 12M: {2Y_Yield_mom_12m:.4f})
        - 5-Year Yield: {5Y_Yield:.2f}% (Momentum: 1M: {5Y_Yield_mom_1m:.4f}, 3M: {5Y_Yield_mom_3m:.4f}, 12M: {5Y_Yield_mom_12m:.4f})
        - 10-Year Yield: {10Y_Yield:.2f}% (Momentum: 1M: {10Y_Yield_mom_1m:.4f}, 3M: {10Y_Yield_mom_3m:.4f}, 12M: {10Y_Yield_mom_12m:.4f})

        ▌Risk sentiment
        - VIX Index: {VIX:.2f} (Weekly change: {VIX_weekly_change:.3f})
        - MOVE Index: {MOVE:.2f} (Weekly change: {MOVE_weekly_change:.3f})

        ▌Treasury ETF Return:
        - Short-Term Treasury (SHV): {SHV:.2f}
        - 1-3 Year Treasury (SHY): {SHY:.2f}
        - 3-7 Year Treasury (IEI): {IEI:.2f}
        - 7-10 Year Treasury (IEF): {IEF:.2f}
        - 10-20 Year Treasury (TLH): {TLH:.2f}
        - 20+ Year Treasury (TLT): {TLT:.2f}
                                          
        | Instrument (ETF)            | past_weekly_full_return | baseline_ret |
        |-----------------------------|---------------|--------------|
        | Short-Term Treasury (SHV)   | {SHV:.4f}       | {SHV_baseline_ret:.4f}   |
        | 1–3 Year Treasury (SHY)     | {SHY:.4f}       | {SHY_baseline_ret:.4f}   |
        | 3–7 Year Treasury (IEI)     | {IEI:.4f}       | {IEI_baseline_ret:.4f}   |
        | 7–10 Year Treasury (IEF)    | {IEF:.4f}       | {IEF_baseline_ret:.4f}   |
        | 10–20 Year Treasury (TLH)   | {TLH:.4f}       | {TLH_baseline_ret:.4f}   |
        | 20+ Year Treasury (TLT)     | {TLT:.4f}       | {TLT_baseline_ret:.4f}   |

        Based on the above data, please predict next week's total return (price + dividend) for the following treasury ETF instruments:
        - Short-Term Treasury (SHV)
        - 1-3 Year Treasury (SHY)
        - 3-7 Year Treasury (IEI)
        - 7-10 Year Treasury (IEF)
        - 10-20 Year Treasury (TLH)
        - 20+ Year Treasury (TLT)

        For every ETF provided above, please provide:
        1. Your**variance_view** , it should be a high conviction (can be aggressive, avoid neutral, conservative) view on the alpha you expect above/below baseline weekly return based on the analysis and reflect the relative outperformance or underperformance of the different ETF in the portfolio for next week as a decimal. 
        2. A confidence score between 0 and 1,
        3. A brief rationale for the prediction. reference the data we supplied. (ex. "INDPRO +0.6 % and VIX −12 % indicate risk-on rotation.")
        4. Provide an include a top-level field **overall_analysis** summarizing how the the provided macro indicators, yield levels, momentum signals, and risk sentiment informed your decision.
        
        Each baseline_ret is the **expected 1-week total return** (decimal).
        Return variance_view in the *same* units (weekly total return).
                                          
        Do **NOT** adjust the baseline yourself – we'll add it afterwards.
        Include a top-level field **overall_analysis**.
                                          
        Respond **only** with valid JSON conforming to the provided schema.
        """)  
        # 1️⃣ extract every {placeholder} name in the template
        field_names = {name.split(':')[0]                    # strip any format spec
                    for _, name, _, _ in Formatter().parse(PROMPT_TEMPLATE)
                    if name}

        # 2️⃣ build a mapping, filling missing/NaN with the chosen default
        mapping = {f: float(row.get(f, default) or default) for f in field_names}

        # 3️⃣ format
        return PROMPT_TEMPLATE.format(**mapping)
    
    def get_gpt4o_prediction(self, prompt: str) -> dict:
        """
        Get prediction from GPT-4o API with structured output for equity sectors.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",                                   # GPT-4o
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an fixed income portfolio strategist. Provide 1-week alpha return (price gain + dividend) forecast on top of the baseline return we provide, for "
                            "the portfolio consists of short, medium and long term maturity treasury ETF based on macro, risk-sentiment, price data and some technical signal."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
                temperature=0,
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
    
    def run_pipeline(self, start_date, end_date):
        data_d = pd.read_csv("data/features/fi_combined_features_daily.csv", index_col=0, parse_dates=True)
        # load weekly data
        print("Loading weekly equity data…")
        weekly_path = "data/features/fi_combined_features_weekly.csv"
        if os.path.exists(weekly_path):
            data_w = pd.read_csv(weekly_path, index_col=0, parse_dates=True)
        else:
            data_w = self.data_collector.collect_data(start_date, end_date)

        # ── 1.  add baseline_ret columns  ─────────────────────────────────
        tickers     = [p["etf"] for p in PORTFOLIOS["bond"]["treasuries"]]
        baseline_w  = self.estimate_returns(data_d[tickers], tickers, span_days=260)             # <- ewma_1m
        # baseline_w  = self.estimate_returns_ARIMA(data_d[tickers], tickers)       # <- arima_1w
        baseline_w   = baseline_w.reindex(data_w.index)
        data_w      = pd.concat([data_w, baseline_w], axis=1)

        # ── 2.  main loop  ───────────────────────────────────────────────
        predictions, dates = [], []
        total_weeks        = len(data_w)

        name_to_etf = {p["name"]: p["etf"] for p in PORTFOLIOS["bond"]["treasuries"]}

        for i, (date, row) in enumerate(data_w.iterrows(), 1):
            print(f"Processing week ending {date.date()}  ({i}/{total_weeks})")
            prompt      = self.prepare_prompt(row)
            prediction  = self.get_gpt4o_prediction(prompt)   # returns variance_view
            predictions.append(prediction)
            dates.append(date)
            time.sleep(1)                                     # respect API limits

            if i % 5 == 0 or i == total_weeks:                # checkpoint
                self._save_temp(predictions, dates, name_to_etf, data_w, i)

        # ── 3.  final post-processing  ───────────────────────────────────
        final_df = self._build_output(predictions, dates, name_to_etf, data_w)

        final_path = "data/predictions/fi_weekly_predictions.csv"
        final_df.to_csv(final_path, index=False)
        print(f"Weekly equity predictions saved to '{final_path}'")

        # raw JSON (optional – keeps whole LLM response)
        pd.DataFrame({"date": dates, "predictions": predictions}).to_json("data/predictions/fi_weekly_predictions_raw.json", orient="records")

        return final_df

    # ---------------------------------------------------------------------
    #  helper: save a temp checkpoint every 5 predictions
    # ---------------------------------------------------------------------
    def _save_temp(self, preds, dates, name_to_etf, data_w, idx):
        out = self._records_from_preds(preds, dates, name_to_etf, data_w)
        temp = f"data/temp/fi_weekly_predictions_temp_{idx}.csv"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        pd.DataFrame(out).to_csv(temp, index=False)
        print(f"  ↳ progress saved ({idx} weeks)")


    # ---------------------------------------------------------------------
    #  helper: turn (prediction, date) pairs into a list of dicts
    # ---------------------------------------------------------------------
    def _records_from_preds(self, preds, dates, name_to_etf, data_w):
        rows = []
        for d, p in zip(dates, preds):
            overall = p.get("overall_analysis", "")
            for inst in p.get("instruments", []):
                etf       = name_to_etf[inst["instrument"]]
                baseline = data_w.at[d, f"{etf}_baseline_ret"]
                if pd.isna(baseline):
                    # fallback = last available EWMA-1w  (or 0.0 if not present)
                    baseline = data_w.at[d, f"{etf}_ewma_1w"] if f"{etf}_ewma_1w" in data_w.columns else 0.0

                variance  = inst["variance_view"]
                pred_ret  = baseline + variance
                rows.append({
                    "date":             d,
                    "etf":              etf,
                    "instrument":       inst["instrument"],
                    "baseline_return":  baseline,
                    "variance_view":    variance,
                    "predicted_return": pred_ret,
                    "confidence":       inst.get("confidence"),
                    "rationale":        inst.get("rationale", ""),
                    "overall_analysis": overall
                })
        return rows


    # ---------------------------------------------------------------------
    #  helper: build the final DataFrame
    # ---------------------------------------------------------------------
    def _build_output(self, preds, dates, name_to_etf, data_w):
        records = self._records_from_preds(preds, dates, name_to_etf, data_w)
        return pd.DataFrame(records)
        
# ----------------------------
# Example Execution of the Fixed Income Agent Pipeline
# ----------------------------
if __name__ == "__main__":
    # Define the fixed income portfolio
    fixed_income_portfolio = PORTFOLIOS['bond']
    client = OpenAI(api_key=OPENAI_API_KEY)
    start_date = "2023-11-01"
    end_date = "2025-04-28"
    full_start_date="2020-01-01"

    # Instantiate the data collector and agent
    fixed_income_data_collector = FixedIncomeDataCollector(
        portfolio=fixed_income_portfolio,
        full_start_date=full_start_date,
        target_start_date=start_date,
        end_date=end_date
    )
    fixed_income_data_collector.collect_data(start_date, end_date)
    fixed_income_agent = FixedIncomeAgent(data_collector=fixed_income_data_collector, llm_client=client)
    
    
    # Run the pipeline and print the resulting predictions
    result_df = fixed_income_agent.run_pipeline(start_date, end_date)
    print("Fixed Income Agent Predictions:")
    print(result_df.tail())
