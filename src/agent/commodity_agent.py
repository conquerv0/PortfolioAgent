import os
import sys
import time
import json
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred
from openai import OpenAI
import textwrap, re
from string import Formatter

from DataCollector import DataCollector
from PortfolioAgent import PortfolioAgent

# -----------------------------------------------------------------------------
# Environment & configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from config.settings import PORTFOLIOS  # noqa: E402

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# -----------------------------------------------------------------------------
# Portfolio helpers
# -----------------------------------------------------------------------------
SECTOR_DEFS: List[Dict] = PORTFOLIOS["commodity"]["sectors"]
SECTOR_TO_ETF: Dict[str, str] = {s["name"]: s["etf"][0] for s in SECTOR_DEFS}  # primary ETF per sector
TICKERS: List[str] = list(SECTOR_TO_ETF.values())
SECTOR_NAMES: List[str] = list(SECTOR_TO_ETF.keys())

# -----------------------------------------------------------------------------
# Prediction JSON schema (mirrors latest equity agent – variance_view)
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
                            "description": "Confidence 0‑1"
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
                "description": "Cross‑sector commodity narrative"
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

    # ---------------- Macro & sentiment helpers -------------------
    def _get_fred_series(self, start_date: str, end_date: str) -> pd.DataFrame:
        fred = Fred(api_key=self.FRED_API_KEY)
        series_map = {
            "DTWEXBGS": "USD_Index",        # Trade‑weighted dollar index
            "DCOILWTICO": "WTI_Oil",        # WTI spot $/bbl
            "GOLDAMGBD228NLBM": "Gold_Spot",# Gold London fix $/oz
            "PCOPPUSDM": "Copper_Spot",     # Copper LME $/mt
            "NAPMNOI": "ISM_NewOrders",     # ISM New Orders diffusion index
        }
        frames = []
        for sid, label in series_map.items():
            try:
                data = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
                frames.append(data.to_frame(name=label))
            except Exception as e:
                logger.warning(f"FRED {sid} failed: {e}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        return df.asfreq("D").ffill()

    # ---------------- Main collect_data ---------------------------
    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("Downloading commodity ETF adj close…")
        adj_close = self.get_etf_adj_close(self.tickers, self.full_start_date, self.end_date)
        if adj_close.empty:
            logger.error("No ETF data – abort")
            return pd.DataFrame()

        # Technical features
        returns = adj_close.pct_change().dropna()
        # momentum
        mom = pd.DataFrame(index=adj_close.index)
        for t in self.tickers:
            mom[f"{t}_mom_1m"] = adj_close[t].pct_change(21)
            mom[f"{t}_mom_3m"] = adj_close[t].pct_change(63)
            mom[f"{t}_mom_12m"] = adj_close[t].pct_change(252)
        # ewma 1m & 1w
        ewma_1m = self.get_ewma_factor(adj_close, span=21)
        ewma_1w = self.get_ewma_factor(adj_close, span=5)
        # rename columns to include horizon label
        ewma_1m.columns = [c.replace("_ewma_1m", "") + "_ewma_1m" if "ewma_1m" not in c else c for c in ewma_1m.columns]
        ewma_1w.columns = [c.replace("_ewma_1m", "") for c in ewma_1w.columns]  # ensure suffix
        ewma_1w.columns = [col.replace("_ewma", "_ewma_1w") for col in ewma_1w.columns]
        # volatility (1m,3m)
        vol = self.calculate_historical_volatility(adj_close, windows=[21, 63])

        macro = self._get_fred_series(self.full_start_date, self.end_date)
        risk = self.get_risk_sentiment_data(self.full_start_date, self.end_date)

        daily = pd.concat([adj_close, mom, ewma_1m, ewma_1w, vol, macro, risk], axis=1)
        daily = daily[self.full_start_date:self.end_date]
        os.makedirs("data", exist_ok=True)
        daily.to_csv("data/commodity_combined_features_daily.csv")
        logger.info("Saved daily commodity features")

        weekly = daily.resample("W-FRI").last()
        for col in ["USD_Index", "WTI_Oil", "Gold_Spot", "ISM_NewOrders", "VIX", "MOVE"]:
            if col in weekly.columns:
                weekly[f"{col}_weekly_change"] = weekly[col].pct_change()
        weekly = weekly[self.target_start_date:self.end_date]
        weekly.to_csv("data/commodity_combined_features_weekly.csv")
        logger.info("Saved weekly commodity features")
        return weekly

# -----------------------------------------------------------------------------
# CommodityAgent (mirrors latest EquityAgent structure)
# -----------------------------------------------------------------------------
class CommodityAgent(PortfolioAgent):
    def __init__(self, data_collector: CommodityDataCollector, llm_client: OpenAI):
        super().__init__("CommodityAgent", data_collector, llm_client)

    # ------- baseline estimation (EWMA 20d) -----------------------
    def estimate_returns(self, daily_px: pd.DataFrame, tickers: List[str], span_days: int = 20):
        daily_ret = daily_px[tickers].pct_change()
        ewma = daily_ret.ewm(span=span_days).mean()
        alpha_w = ewma.resample("W-FRI").last()
        alpha_w.columns = [f"{c}_baseline_ret" for c in alpha_w.columns]
        return alpha_w

    # ---------------- prompt builder ------------------------------
    def prepare_prompt(self, row: pd.Series, default: float = 0.0) -> str:
        TEMPLATE = textwrap.dedent("""
        Based only on the data for week ending **{date}**: macro & sentiment shifts and commodity‑sector ETF stats, predict your **variance_view** (alpha vs baseline) for next week.

        ▌Macro snapshot
        • Dollar Index Δ…… {USD_Index_weekly_change:.3f}
        • ISM New Orders…… {ISM_NewOrders_weekly_change:.2f}
        • WTI Oil Δ ($/bbl)… {WTI_Oil_weekly_change:.3f}
        • Gold Δ ($/oz)…… {Gold_Spot_weekly_change:.3f}

        ▌Risk sentiment
        • VIX…… {VIX:.2f} (Δ {VIX_weekly_change:.3f})
        • MOVE… {MOVE:.2f} (Δ {MOVE_weekly_change:.3f})

        ▌Sector ETF table
        | Sector | ETF | adj_close | baseline_ret | ewma_1w | ewma_1m |
        | Energy               | {Energy_etf} | {Energy_px:.4f} | {Energy_baseline:.4f} | {Energy_ewma_1w:.4f} | {Energy_ewma_1m:.4f} |
        | Precious Metals      | {Precious_etf} | {Precious_px:.4f} | {Precious_baseline:.4f} | {Precious_ewma_1w:.4f} | {Precious_ewma_1m:.4f} |
        | Industrial Metals    | {Industrial_etf} | {Industrial_px:.4f} | {Industrial_baseline:.4f} | {Industrial_ewma_1w:.4f} | {Industrial_ewma_1m:.4f} |
        | Agriculture          | {Agriculture_etf} | {Agriculture_px:.4f} | {Agriculture_baseline:.4f} | {Agriculture_ewma_1w:.4f} | {Agriculture_ewma_1m:.4f} |
        | Livestock            | {Livestock_etf} | {Livestock_px:.4f} | {Livestock_baseline:.4f} | {Livestock_ewma_1w:.4f} | {Livestock_ewma_1m:.4f} |

        **Task**
        For each sector, provide:
        1. **variance_view** – alpha vs baseline weekly return (decimal)
        2. confidence 0‑1
        3. rationale (1 sentence)
        Add **overall_analysis** summarising the drivers.
        Respond **only** with JSON matching the schema.
        """)

        # Build mapping dict
        mapping = {
            "date": row.name.date(),
            # Macro / sentiment
            "USD_Index_weekly_change": row.get("USD_Index_weekly_change", default),
            "ISM_NewOrders_weekly_change": row.get("ISM_NewOrders_weekly_change", default),
            "WTI_Oil_weekly_change": row.get("WTI_Oil_weekly_change", default),
            "Gold_Spot_weekly_change": row.get("Gold_Spot_weekly_change", default),
            "VIX": row.get("VIX", default),
            "VIX_weekly_change": row.get("VIX_weekly_change", default),
            "MOVE": row.get("MOVE", default),
            "MOVE_weekly_change": row.get("MOVE_weekly_change", default),
        }
        # Sector rows
        def sec(key, etf):
            mapping[f"{key}_etf"]          = etf
            mapping[f"{key}_px"]           = row.get(etf, default)
            mapping[f"{key}_baseline"]     = row.get(f"{etf}_baseline_ret", default)
            mapping[f"{key}_ewma_1w"]      = row.get(f"{etf}_ewma_1w", default)
            mapping[f"{key}_ewma_1m"]      = row.get(f"{etf}_ewma_1m", default)
        sec("Energy", SECTOR_TO_ETF["Energy"])
        sec("Precious", SECTOR_TO_ETF["Precious Metals"])
        sec("Industrial", SECTOR_TO_ETF["Industrial Metals"])
        sec("Agriculture", SECTOR_TO_ETF["Agriculture"])
        sec("Livestock", SECTOR_TO_ETF["Livestock"])

        return TEMPLATE.format(**mapping)

    # ---------------- GPT query -------------------------------
    def _query_llm(self, prompt: str) -> dict:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a commodities strategist forecasting 1‑week alpha vs baseline for broad commodity sectors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
                temperature=0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"instruments": [{"instrument": s, "variance_view": 0.0, "confidence": 0.0, "rationale": f"Error {e}"} for s in SECTOR_NAMES], "overall_analysis": "error"}

    # ---------------- pipeline -------------------------------
    def run_pipeline(self, start_date: str, end_date: str):
    
        logger.info("Loading weekly commodity data…")
        if os.path.exists("data/commodity_combined_features_weekly.csv"):
            weekly = pd.read_csv("data/commodity_combined_features_weekly.csv", index_col=0, parse_dates=True)
        else:
            weekly = self.data_collector.collect_data(start_date, end_date)

        # 1️⃣ baseline returns
        daily = pd.read_csv("data/commodity_combined_features_daily.csv", index_col=0, parse_dates=True)
        baseline = self.estimate_returns(daily, TICKERS).reindex(weekly.index)
        weekly   = pd.concat([weekly, baseline], axis=1)

        preds, dates = [], []
        for i, (dt, row) in enumerate(weekly.iterrows(), 1):
            logger.info(f"Week {i}/{len(weekly)} – {dt.date()}")
            prompt = self.prepare_prompt(row)
            preds.append(self._query_llm(prompt))
            dates.append(dt)
            time.sleep(1)
            if i % 5 == 0 or i == len(weekly):
                self._save_temp(preds, dates, i, weekly)

        final_df = self._build_output(preds, dates, weekly)
        final_df.to_csv("data/commodity_weekly_predictions.csv", index=False)
        logger.info("Commodity predictions saved → data/commodity_weekly_predictions.csv")
        return final_df

    # ---------------- helpers -------------------------------
    def _save_temp(self, preds, dates, idx, weekly):
        records = self._records(preds, dates, weekly)
        temp = f"data/temp/commodity_predictions_temp_{idx}.csv"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        pd.DataFrame(records).to_csv(temp, index=False)
        logger.info(f" ↳ progress saved ({idx} weeks)")

    def _records(self, preds, dates, weekly):
        recs = []
        for dt, p in zip(dates, preds):
            overall = p.get("overall_analysis", "")
            for inst in p.get("instruments", []):
                sector = inst["instrument"]
                etf    = SECTOR_TO_ETF[sector]
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
        return pd.DataFrame(self._records(preds, dates, weekly))

# -----------------------------------------------------------------------------
# Stand‑alone execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)
    collector = CommodityDataCollector(full_start_date="2020-01-01", target_start_date="2023-11-01", end_date="2025-03-31")
    agent = CommodityAgent(collector, client)
    df = agent.run_pipeline("2023-11-01", "2025-03-31")
    print(df.tail())
