import os
import sys
import time
import json
import logging
import textwrap
import re
from typing import List, Dict
from datetime import datetime
from string import Formatter

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from openai import OpenAI
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA

from DataCollector import DataCollector
from PortfolioAgent import PortfolioAgent

# -----------------------------------------------------------------------------
# Environment & configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PORTFOLIOS  # noqa: E402

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# -----------------------------------------------------------------------------
# Portfolio helpers
# -----------------------------------------------------------------------------
SECTOR_DEFS: List[Dict] = PORTFOLIOS["commodity"]["sectors"]
SECTOR_TO_ETF: Dict[str, str] = {s["name"]: s["etf"] for s in SECTOR_DEFS}  # primary ETF per sector
TICKERS: List[str] = list(SECTOR_TO_ETF.values())
SECTOR_NAMES: List[str] = list(SECTOR_TO_ETF.keys())

# -----------------------------------------------------------------------------
# Prediction JSON schema 
# -----------------------------------------------------------------------------
PREDICTION_SCHEMA = {
    "name": "commodity_prediction",
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
                            "enum": SECTOR_NAMES,
                            "description": "Commodity sector name"
                        },
                        "variance_view": {
                            "type": "number",
                            "description": "Alpha adjustment vs. baseline return (decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence 0-1"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief rationale"
                        }
                    },
                    "required": ["instrument", "variance_view", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Cross-sector commodity narrative"
            }
        },
        "required": ["instruments", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}

# -----------------------------------------------------------------------------
# Commodity Data Collector (daily + weekly feature sets)
# -----------------------------------------------------------------------------
class CommodityDataCollector(DataCollector):
    def __init__(self, full_start_date: str, target_start_date: str, end_date: str):
        super().__init__(full_start_date, target_start_date, end_date)
        self.tickers = TICKERS

    # ---------------- Macro & sentiment helpers ----------------
    def _get_fred_series(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve economic data series from FRED."""
        fred = Fred(api_key=self.FRED_API_KEY)
        
        # Map of FRED series IDs to descriptive labels
        series_map = {
            "DTWEXBGS": "USD_Index",         # Trade‑weighted dollar index
            "DCOILWTICO": "WTI_Oil",         # WTI spot $/bbl
            "GOLDAMGBD228NLBM": "Gold_Spot", # Gold London fix $/oz
            "PCOPPUSDM": "Copper_Spot",      # Copper LME $/mt
            "NAPMNOI": "ISM_NewOrders",      # ISM New Orders diffusion index
        }
        
        # Try to retrieve each series
        frames = []
        for sid, label in series_map.items():
            try:
                data = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
                frames.append(data.to_frame(name=label))
            except Exception as e:
                logger.warning(f"FRED {sid} failed: {e}")
                
        if not frames:
            return pd.DataFrame()
            
        # Combine all series and fill forward missing values
        df = pd.concat(frames, axis=1)
        return df.asfreq("D").ffill()

    # ---------------- Main collect_data ----------------
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect and process commodity data features."""
        logger.info("Downloading commodity ETF adj close...")
        adj_close = self.get_etf_adj_close(self.tickers, self.full_start_date, self.end_date)
        
        if adj_close.empty:
            logger.error("No ETF data - abort")
            return pd.DataFrame()

        # Calculate technical features
        returns = adj_close.pct_change(fill_method=None).dropna()
        
        # Momentum features (1m, 3m, 12m)
        mom = pd.DataFrame(index=adj_close.index)
        for t in self.tickers:
            mom[f"{t}_mom_1m"] = adj_close[t].pct_change(21, fill_method=None)
            mom[f"{t}_mom_3m"] = adj_close[t].pct_change(63, fill_method=None)
            mom[f"{t}_mom_12m"] = adj_close[t].pct_change(252, fill_method=None)
            
        # EWMA features (1m & 1w)
        ewma_1m = self.get_ewma_factor(adj_close, span=21)
        ewma_1w = self.get_ewma_factor(adj_close, span=5)
        
        # Standardize column names
        ewma_1m.columns = [
            c.replace("_ewma_1m", "") + "_ewma_1m" if "ewma_1m" not in c else c 
            for c in ewma_1m.columns
        ]
        ewma_1w.columns = [c.replace("_ewma_1m", "") for c in ewma_1w.columns]  # ensure suffix
        ewma_1w.columns = [col.replace("_ewma", "_ewma_1w") for col in ewma_1w.columns]
        
        # Volatility features (1m, 3m)
        vol = self.calculate_historical_volatility(adj_close, windows=[21, 63])

        # External data
        macro = self._get_fred_series(self.full_start_date, self.end_date)
        risk = self.get_risk_sentiment_data(self.full_start_date, self.end_date)

        # Combine all features
        daily = pd.concat([adj_close, mom, ewma_1m, ewma_1w, vol, macro, risk], axis=1)
        daily = daily[self.full_start_date:self.end_date]
        
        # Save daily features
        os.makedirs("data/features", exist_ok=True)
        daily.to_csv("data/features/commodity_combined_features_daily.csv")
        logger.info("Saved daily commodity features")

        # Resample to weekly and calculate additional features
        weekly = daily.resample("W-FRI").last()
        for col in ["USD_Index", "WTI_Oil", "Gold_Spot", "ISM_NewOrders", "VIX", "MOVE"]:
            if col in weekly.columns:
                weekly[f"{col}_weekly_change"] = weekly[col].pct_change(fill_method=None)
                
        # Filter by target date range and save
        weekly = weekly[self.target_start_date:self.end_date]
        weekly.to_csv("data/features/commodity_combined_features_weekly.csv")
        logger.info("Saved weekly commodity features")
        
        return weekly

# -----------------------------------------------------------------------------
# CommodityAgent (mirrors latest EquityAgent structure)
# -----------------------------------------------------------------------------
class CommodityAgent(PortfolioAgent):
    def __init__(self, data_collector: CommodityDataCollector, llm_client: OpenAI):
        super().__init__("CommodityAgent", data_collector, llm_client)

    # ---------------- Baseline estimation (EWMA 1M) ----------------
    def estimate_returns(self, daily_px: pd.DataFrame, tickers: List[str], span_days: int = 21):
        """Calculate baseline returns using EWMA of daily returns."""
        # Ensure all tickers are in the dataframe
        available_tickers = [t for t in tickers if t in daily_px.columns]
        if len(available_tickers) != len(tickers):
            missing = set(tickers) - set(available_tickers)
            logger.warning(f"Missing tickers in data: {missing}")
        
        # Calculate EWMA of daily returns
        daily_ret = daily_px[available_tickers].pct_change(fill_method=None)
        ewma = daily_ret.ewm(span=span_days).mean()
        
        # Resample to weekly and rename columns
        alpha_w = ewma.resample("W-FRI").last()
        alpha_w.columns = [f"{c}_baseline_ret" for c in alpha_w.columns]
        
        return alpha_w

    # ---------------- Baseline estimation (ARIMA) ----------------
    def estimate_returns_ARIMA(self, daily_px: pd.DataFrame, tickers: List[str], span_days: int = 60, arima_order=(1,0,0)):
        """
        Estimate baseline returns using ARIMA model for each ticker.
        Returns a DataFrame with weekly ARIMA-forecasted returns.
        """
        import warnings
        warnings.filterwarnings("ignore")  # ARIMA can be noisy

        # Ensure all tickers are in the dataframe
        available_tickers = [t for t in tickers if t in daily_px.columns]
        if len(available_tickers) != len(tickers):
            missing = set(tickers) - set(available_tickers)
            logger.warning(f"Missing tickers in data: {missing}")
        
        # Calculate daily returns
        daily_ret = daily_px[available_tickers].pct_change(fill_method=None).dropna()
        weekly_idx = daily_ret.resample("W-FRI").last().index
        arima_forecasts = pd.DataFrame(index=weekly_idx, columns=available_tickers)
        
        # For each ticker, fit ARIMA model and forecast
        for ticker in available_tickers:
            series = daily_ret[ticker].dropna()
            for week_end in weekly_idx:
                # Use data up to the week before the forecast
                train = series.loc[:week_end - pd.Timedelta(days=1)]
                if len(train) < span_days:
                    arima_forecasts.at[week_end, ticker] = np.nan
                    continue
                try:
                    model = ARIMA(train, order=arima_order)
                    fit = model.fit()
                    # Forecast 5 steps ahead (1 week, if 5 trading days)
                    forecast = fit.forecast(steps=5)
                    # Use the mean of the 5-day forecast as the weekly return
                    arima_forecasts.at[week_end, ticker] = forecast.mean()
                except Exception as e:
                    logger.warning(f"ARIMA failed for {ticker} at {week_end}: {e}")
                    arima_forecasts.at[week_end, ticker] = np.nan
        
        # Rename columns
        arima_forecasts.columns = [f"{c}_baseline_ret" for c in arima_forecasts.columns]
        return arima_forecasts

    # ---------------- Prompt builder ----------------
    def prepare_prompt(self, row: pd.Series, default: float = 0.0) -> str:
        """Build prompt for LLM with current market data."""
        TEMPLATE = textwrap.dedent("""
        Based only on the data for week ending {date}: macro & sentiment shifts and commodity sector ETF stats, predict your variance_view (alpha vs baseline) for next week.

        ▌Macro snapshot
        • Dollar Index Δ…… {USD_Index_weekly_change:.3f}
        • ISM New Orders…… {ISM_NewOrders_weekly_change:.2f}
        • WTI Oil Δ ($/bbl)… {WTI_Oil_weekly_change:.3f}
        • Gold Δ ($/oz)…… {Gold_Spot_weekly_change:.3f}

        ▌Risk sentiment
        • VIX…… {VIX:.2f} (Δ {VIX_weekly_change:.3f})
        • MOVE… {MOVE:.2f} (Δ {MOVE_weekly_change:.3f})

        ▌Sector ETF table
        | Sector | ETF | adj_close | baseline_ret | ewma_1w | ewma_1m | vol_1m |
        | Energy               | {Energy_etf} | {Energy_px:.4f} | {Energy_baseline:.4f} | {Energy_ewma_1w:.4f} | {Energy_ewma_1m:.4f} | {Energy_vol_1m:.4f} |
        | Precious Metals      | {Precious_etf} | {Precious_px:.4f} | {Precious_baseline:.4f} | {Precious_ewma_1w:.4f} | {Precious_ewma_1m:.4f} | {Precious_vol_1m:.4f} |
        | Industrial Metals    | {Industrial_etf} | {Industrial_px:.4f} | {Industrial_baseline:.4f} | {Industrial_ewma_1w:.4f} | {Industrial_ewma_1m:.4f} | {Industrial_vol_1m:.4f} |
        | Agriculture          | {Agriculture_etf} | {Agriculture_px:.4f} | {Agriculture_baseline:.4f} | {Agriculture_ewma_1w:.4f} | {Agriculture_ewma_1m:.4f} | {Agriculture_vol_1m:.4f} |
        | Livestock            | {Livestock_etf} | {Livestock_px:.4f} | {Livestock_baseline:.4f} | {Livestock_ewma_1w:.4f} | {Livestock_ewma_1m:.4f} | {Livestock_vol_1m:.4f} |

        **Task**
        For every ETF provided above, please provide:
        1. Your**variance_view** , it should be a high conviction (can be aggressive, avoid neutral, conservative) view on the alpha you expect above/below baseline weekly return based on the analysis and reflect the relative outperformance or underperformance of the different ETF in the portfolio for next week as a decimal. 
        2. A confidence score between 0 and 1,
        3. A brief rationale for the prediction. reference the data we supplied. (ex. "INDPRO +0.6 % and VIX −12 % indicate risk-on rotation.")
        4. Provide an include a top-level field **overall_analysis** summarizing how the data inform your predictions.
        
        Each baseline_ret is the **expected 1-week total return** (decimal).
        Return variance_view in the *same* units (weekly total return).
        
        IMPORTANT: Your predictions should align with current volatility. In higher volatility regimes, your variance_view should be more aggressive (larger magnitude) and not too conservative. Use the volatility metrics as a guide for how bold your predictions should be.
                                          
        Do **NOT** adjust the baseline yourself – we'll add it afterwards.
        Include a top-level field **overall_analysis**.
                                          
        Respond **only** with valid JSON conforming to the provided schema.
        """)

        # Helper to safely convert values to float
        def safe_float(val, default_val=default):
            try:
                if isinstance(val, pd.Series):
                    # Take the first value if it's a Series
                    return float(val.iloc[0]) if not val.empty else default_val
                return float(val)
            except (ValueError, TypeError):
                return default_val

        # Build mapping dict for template
        mapping = {
            "date": row.name.date(),
            # Macro / sentiment
            "USD_Index_weekly_change": safe_float(row.get("USD_Index_weekly_change", default)),
            "ISM_NewOrders_weekly_change": safe_float(row.get("ISM_NewOrders_weekly_change", default)),
            "WTI_Oil_weekly_change": safe_float(row.get("WTI_Oil_weekly_change", default)),
            "Gold_Spot_weekly_change": safe_float(row.get("Gold_Spot_weekly_change", default)),
            "VIX": safe_float(row.get("VIX", default)),
            "VIX_weekly_change": safe_float(row.get("VIX_weekly_change", default)),
            "MOVE": safe_float(row.get("MOVE", default)),
            "MOVE_weekly_change": safe_float(row.get("MOVE_weekly_change", default)),
        }
        
        # Helper to populate sector fields
        def sec(key, etf):
            mapping[f"{key}_etf"] = etf
            mapping[f"{key}_px"] = safe_float(row.get(etf, default))
            mapping[f"{key}_baseline"] = safe_float(row.get(f"{etf}_baseline_ret", default))
            mapping[f"{key}_ewma_1w"] = safe_float(row.get(f"{etf}_ewma_1w", default))
            mapping[f"{key}_ewma_1m"] = safe_float(row.get(f"{etf}_ewma_1m", default))
            mapping[f"{key}_vol_1m"] = safe_float(row.get(f"{etf}_vol_1m", default))
        
        # Populate sector data    
        sec("Energy", SECTOR_TO_ETF["Energy"])
        sec("Precious", SECTOR_TO_ETF["Precious Metals"])
        sec("Industrial", SECTOR_TO_ETF["Industrial Metals"])
        sec("Agriculture", SECTOR_TO_ETF["Agriculture"])
        sec("Livestock", SECTOR_TO_ETF["Livestock"])

        return TEMPLATE.format(**mapping)

    # ---------------- LLM query ----------------
    def _query_llm(self, prompt: str) -> dict:
        """Query LLM for commodity predictions."""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a commodities strategist forecasting 1 week alpha vs baseline for broad commodity sectors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
                temperature=0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "instruments": [
                    {
                        "instrument": s, 
                        "variance_view": 0.0, 
                        "confidence": 0.0, 
                        "rationale": f"Error {e}"
                    } for s in SECTOR_NAMES
                ], 
                "overall_analysis": "error"
            }

    # ---------------- Main pipeline ----------------
    def run_pipeline(self, start_date: str, end_date: str):
        """Run the commodity prediction pipeline."""
        logger.info(f"Using commodity ETF tickers: {TICKERS}")
        logger.info("Loading weekly commodity data...")
        
        # Load or collect weekly data
        if os.path.exists("data/features/commodity_combined_features_weekly.csv"):
            weekly = pd.read_csv("data/features/commodity_combined_features_weekly.csv", index_col=0, parse_dates=True)
        else:
            weekly = self.data_collector.collect_data(start_date, end_date)

        # Calculate baseline returns
        daily = pd.read_csv("data/features/commodity_combined_features_daily.csv", index_col=0, parse_dates=True)
        baseline = self.estimate_returns_ARIMA(daily, TICKERS).reindex(weekly.index)
        weekly = pd.concat([weekly, baseline], axis=1)

        # Process each week
        preds, dates = [], []
        for i, (dt, row) in enumerate(weekly.iterrows(), 1):
            logger.info(f"Week {i}/{len(weekly)} – {dt.date()}")
            prompt = self.prepare_prompt(row)
            preds.append(self._query_llm(prompt))
            dates.append(dt)
            time.sleep(1)
            
            # Save progress periodically
            if i % 5 == 0 or i == len(weekly):
                self._save_temp(preds, dates, i, weekly)

        # Build and save final output
        final_df = self._build_output(preds, dates, weekly)
        os.makedirs("data/predictions", exist_ok=True)
        final_df.to_csv("data/predictions/commodity_weekly_predictions.csv", index=False)
        logger.info("Commodity predictions saved → data/predictions/commodity_weekly_predictions.csv")
        
        return final_df

    # ---------------- Helper methods ----------------
    def _save_temp(self, preds, dates, idx, weekly):
        """Save temporary progress."""
        records = self._records(preds, dates, weekly)
        temp = f"data/temp/commodity_predictions_temp_{idx}.csv"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        pd.DataFrame(records).to_csv(temp, index=False)
        logger.info(f" ↳ progress saved ({idx} weeks)")

    def _records(self, preds, dates, weekly):
        """Convert predictions to records for DataFrame."""
        recs = []
        for dt, p in zip(dates, preds):
            overall = p.get("overall_analysis", "")
            for inst in p.get("instruments", []):
                sector = inst["instrument"]
                etf = SECTOR_TO_ETF[sector]
                baseline = weekly.loc[dt, f"{etf}_baseline_ret"]
                variance = inst["variance_view"]
                recs.append({
                    "date": dt,
                    "sector": sector,
                    "etf": etf,
                    "baseline_return": baseline,
                    "variance_view": variance,
                    "predicted_return": baseline + variance,
                    "confidence": inst["confidence"],
                    "rationale": inst["rationale"],
                    "overall_analysis": overall
                })
        return recs

    def _build_output(self, preds, dates, weekly):
        """Build final output DataFrame."""
        return pd.DataFrame(self._records(preds, dates, weekly))

# -----------------------------------------------------------------------------
# Stand‑alone execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)
    collector = CommodityDataCollector(
        full_start_date="2020-01-01", 
        target_start_date="2023-11-01", 
        end_date="2025-03-31"
    )
    agent = CommodityAgent(collector, client)
    df = agent.run_pipeline("2023-11-01", "2025-03-31")
    print(df.tail())
