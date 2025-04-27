import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_community.llms import OpenAI
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
from string import Formatter
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from config.settings import *

# ----------------------------
# Equity-Specific Implementations
# ----------------------------

PREDICTION_SCHEMA = {
    "name": "equity_prediction",
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
                                "Information Technology",
                                "Health Care",
                                "Financials",
                                "Consumer Discretionary",
                                "Communication Services",
                                "Industrials",
                                "Consumer Staples",
                                "Energy",
                                "Utilities",
                                "Real Estate",
                                "Materials"
                            ],
                            "description": "GICS sector name"
                            },
                        "variance_view": {
                            "type": "number",
                            "description": "Alpha adjustment vs. baseline return (decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence (0-1)"
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
                "description": "Cross-sector and macro narrative"
            }
        },
        "required": ["instruments", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}


class EquityDataCollector(DataCollector):
    """
    Equity data collector that retrieves historical price data for each ETF
    in the equity portfolio using yfinance.
    """
    def __init__(self, portfolio: list, full_start_date: str, target_start_date: str, end_date: str):
        super().__init__(full_start_date, target_start_date, end_date)
        self.portfolio = portfolio

    def get_fred_data(self, start_date: str, end_date: str, freq: str = 'D') -> pd.DataFrame:
        """
        Pulls a set of macroeconomic series from FRED and resamples to the given frequency.
        """
        fred = Fred(api_key=self.FRED_API_KEY)
        series_dict = {
            "INDPRO": "INDPRO",        # Industrial Production Index
            "RSAFS": "RSAFS",          # Retail Sales: Total (Excl Food Services)
            "HOUST": "HOUST",          # Housing Starts: Total
            "UNRATE": "UNRATE",        # Unemployment Rate
            "PCEPILFE": "PCEPILFE",    # Core PCE Price Index
            "CPIAUCSL": "CPIAUCSL",    # Consumer Price Index for All Urban Consumers
            "EFFR": "EFFR",    # Effective Federal Funds Rate
            "T10Y2Y": "T10Y2Y",        # 10-Year Treasury Constant Maturity Minus 2-Year
            "BAA10Y": "BAA10Y",        # Moody's BAA Corporate Bond Yield Spread
            "DCOILWTICO": "DCOILWTICO",# Crude Oil Prices: West Texas Intermediate
            "PCOPPUSDM": "PCOPPUSDM"   # Copper Spot Price
        }
        frames = []
        for label, sid in series_dict.items():
            try:
                data = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
                df = data.to_frame(name=label)
                frames.append(df)
                logger.info(f"Downloaded {label} from FRED.")
            except Exception as e:
                logger.error(f"Error downloading {sid}: {e}")
        if not frames:
            return pd.DataFrame()
        df_all = pd.concat(frames, axis=1)
        full_idx = pd.date_range(start=df_all.index.min(), end=df_all.index.max(), freq=freq)
        df_all = df_all.reindex(full_idx)
        df_all = df_all.fillna(method='ffill')
        return df_all

    def collect_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        # Macro features
        logger.info("Downloading FRED macro data...")
        macro = self.get_fred_data(self.full_start_date, self.end_date)
        if macro.empty:
            logger.warning("No macro data collected.")
        risk = self.get_risk_sentiment_data(self.full_start_date, self.end_date)

        # Extract ETF tickers
        tickers = [entry['etf'] for entry in self.portfolio]
        logger.info(f"Downloading adjusted close prices for {len(tickers)} sector ETFs...")
        tickers = [p["etf"] for p in self.portfolio]
        prices = self.get_etf_adj_close(tickers, self.full_start_date, self.end_date)
        # Combine all features
        if prices.empty:
            logger.error("Price frame empty – abort.")
            return pd.DataFrame()
        mom = self.get_momentum_factors(prices)
        ewma = self.get_ewma_factor(prices)
        vol = self.calculate_historical_volatility(prices)

        features = pd.concat([prices, mom, ewma, vol, risk, macro], axis=1)

        features.index = pd.to_datetime(features.index)
        daily = features[self.full_start_date:self.end_date]

        # Save daily
        os.makedirs('data', exist_ok=True)
        daily.to_csv('data/equity_combined_features_daily.csv')
        logger.info("Saved daily equity features.")

        # Weekly
        weekly = daily.resample("W-FRI").last()
        for col in ["VIX", "MOVE"]:
            if col in weekly.columns:
                weekly[f"{col}_weekly_change"] = weekly[col].pct_change()
        weekly = weekly[self.target_start_date:self.end_date]
        weekly.to_csv("data/equity_combined_features_weekly.csv")
        logger.info("Saved weekly equity features.")

        return weekly
    

class EquityAgent(PortfolioAgent):
    """
    Equity agent that uses an EquityDataCollector and a customized prompt template.
    The prompt instructs the LLM to provide a forecast view (expected return and confidence)
    for each equity ETF.
    """
    def __init__(self, data_collector: EquityDataCollector, llm_client: OpenAI):

        super().__init__(name="EquityAgent", data_collector=data_collector, llm_client=llm_client)
    
    # def prepare_prompt(self, row: pd.Series) -> str:
    #     """
    #     Creates a detailed prompt for the LLM based on the current market data.
    #     Includes sector ETF performance, momentum indicators, volatility metrics,
    #     market breadth, and macro indicators.
        
    #     Args:
    #         row (pd.Series): A row of data containing all features for the current timestamp
            
    #     Returns:
    #         str: Formatted prompt for the LLM
    #     """
    #     prompt = """
    #     Based on the following weekly U.S. macro indicators, risk-sentiment gauges, and sector-ETF prices, predict next week’s total return for each sector ETF.

    #     1. Macro Indicators  
    #     - Industrial Production (INDPRO): {INDPRO:.4f}  
    #     - Retail Sales ex-Food (RSAFS): {RSAFS:.4f}  
    #     - Housing Starts (HOUST): {HOUST:.4f}  
    #     - Unemployment Rate (UNRATE): {UNRATE:.4f}%  
    #     - Core PCE Price Index (PCEPILFE): {PCEPILFE:.2f}  
    #     - Consumer CPI (CPIAUCSL): {CPIAUCSL:.2f}  
    #     - Effective Federal Funds Rate (EFFR): {EFFR:.4f}
    #     - 10-y minus 2-y Treasury Spread (T10Y2Y): {T10Y2Y:.4f}%  
    #     - BAA Credit Spread (BAA10Y): {BAA10Y:.4f}%  
    #     - WTI Crude Oil ($/bbl, DCOILWTICO): {DCOILWTICO:.4f}  
    #     - Copper Price ($/t, PCOPPUSDM): {PCOPPUSDM:.4f}  

    #     2. Risk-Sentiment  
    #     - VIX Index: {VIX:.2f} (weekly Δ {VIX_weekly_change:.3f})  
    #     - MOVE Index: {MOVE:.2f} (weekly Δ {MOVE_weekly_change:.3f})  

    #     3. Sector ETFs:
    #     - Information Technology (XLK): {XLK:.4f} (mom_1m: {XLK_mom_1m:.4f}, mom_3m: {XLK_mom_3m:.4f}, mom_12m: {XLK_mom_12m:.4f}, ewma_1m: {XLK_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLK_vol_1m:.4f}, 3m: {XLK_vol_3m:.4f}
    #     - Health Care (XLV): {XLV:.4f} (mom_1m: {XLV_mom_1m:.4f}, mom_3m: {XLV_mom_3m:.4f}, mom_12m: {XLV_mom_12m:.4f}, ewma_1m: {XLV_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLV_vol_1m:.4f}, 3m: {XLV_vol_3m:.4f}
    #     - Financials (XLF): {XLF:.4f} (mom_1m: {XLF_mom_1m:.4f}, mom_3m: {XLF_mom_3m:.4f}, mom_12m: {XLF_mom_12m:.4f}, ewma_1m: {XLF_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLF_vol_1m:.4f}, 3m: {XLF_vol_3m:.4f}
    #     - Consumer Discretionary (XLY): {XLY:.4f} (mom_1m: {XLY_mom_1m:.4f}, mom_3m: {XLY_mom_3m:.4f}, mom_12m: {XLY_mom_12m:.4f}, ewma_1m: {XLY_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLY_vol_1m:.4f}, 3m: {XLY_vol_3m:.4f}
    #     - Communication Services (XLC): {XLC:.4f} (mom_1m: {XLC_mom_1m:.4f}, mom_3m: {XLC_mom_3m:.4f}, mom_12m: {XLC_mom_12m:.4f}, ewma_1m: {XLC_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLC_vol_1m:.4f}, 3m: {XLC_vol_3m:.4f}
    #     - Industrials (XLI): {XLI:.4f} (mom_1m: {XLI_mom_1m:.4f}, mom_3m: {XLI_mom_3m:.4f}, mom_12m: {XLI_mom_12m:.4f}, ewma_1m: {XLI_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLI_vol_1m:.4f}, 3m: {XLI_vol_3m:.4f}
    #     - Consumer Staples (XLP): {XLP:.4f} (mom_1m: {XLP_mom_1m:.4f}, mom_3m: {XLP_mom_3m:.4f}, mom_12m: {XLP_mom_12m:.4f}, ewma_1m: {XLP_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLP_vol_1m:.4f}, 3m: {XLP_vol_3m:.4f}
    #     - Energy (XLE): {XLE:.4f} (mom_1m: {XLE_mom_1m:.4f}, mom_3m: {XLE_mom_3m:.4f}, mom_12m: {XLE_mom_12m:.4f}, ewma_1m: {XLE_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLE_vol_1m:.4f}, 3m: {XLE_vol_3m:.4f}
    #     - Utilities (XLU): {XLU:.4f} (mom_1m: {XLU_mom_1m:.4f}, mom_3m: {XLU_mom_3m:.4f}, mom_12m: {XLU_mom_12m:.4f}, ewma_1m: {XLU_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLU_vol_1m:.4f}, 3m: {XLU_vol_3m:.4f}
    #     - Real Estate (XLRE): {XLRE:.4f} (mom_1m: {XLRE_mom_1m:.4f}, mom_3m: {XLRE_mom_3m:.4f}, mom_12m: {XLRE_mom_12m:.4f}, ewma_1m: {XLRE_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLRE_vol_1m:.4f}, 3m: {XLRE_vol_3m:.4f}
    #     - Materials (XLB): {XLB:.4f} (mom_1m: {XLB_mom_1m:.4f}, mom_3m: {XLB_mom_3m:.4f}, mom_12m: {XLB_mom_12m:.4f}, ewma_1m: {XLB_ewma_1m:.4f})
    #       Historical Vol: 1m: {XLB_vol_1m:.4f}, 3m: {XLB_vol_3m:.4f}

    #     For every ETF provided above, please provide:
    #     1. The predicted expected return for next week as a decimal,
    #     2. A confidence score between 0 and 1,
    #     3. A brief rationale for the prediction.
    #     4. Provide an include a top-level field **overall_analysis** summarizing how macro conditions, sector-specific drivers, and risk sentiment inform your predictions.

    #     Respond **only** with valid JSON conforming to the provided schema.
    #     """

    #     try:
    #         formatted_prompt = prompt.format(**row.to_dict())

    #     except KeyError as e:
    #         print(f"Warning: missing {e}; filling with 0.0.")
    #         row_copy = row.copy()

    #         required_cols = [
    #             # ---- macro series
    #             'INDPRO', 'RSAFS', 'HOUST', 'UNRATE',
    #             'PCEPILFE', 'CPIAUCSL', 'EFFR',
    #             'T10Y2Y', 'BAA10Y', 'DCOILWTICO', 'PCOPPUSDM',
    #             # ---- risk-sentiment
    #             'VIX', 'VIX_weekly_change', 'MOVE', 'MOVE_weekly_change',
    #             # ---- sector ETFs
    #             'XLK', 'XLV', 'XLF', 'XLY', 'XLC',
    #             'XLI', 'XLP', 'XLE', 'XLU', 'XLRE', 'XLB'
    #         ]

    #         for col in required_cols:
    #             if col not in row_copy or pd.isna(row_copy[col]):
    #                 row_copy[col] = 0.0

    #         formatted_prompt = prompt.format(**row_copy.to_dict())

    #     return formatted_prompt
    def estimate_returns(self, weekly_frame, tickers, span_weeks=4):
        # 1. Compute WEEKLY returns first
        prices = weekly_frame[tickers]            # Friday closes
        wk_ret = prices.pct_change()

        # 2. EWMA of weekly returns over `span_weeks`
        alpha = wk_ret.ewm(span=span_weeks).mean()

        alpha.columns = [f"{c}_baseline_ret" for c in alpha.columns]
        return alpha

    
    def prepare_prompt(self, row: pd.Series, default: float = 0.0) -> str:
        """
        Format PROMPT_TEMPLATE with the values in `row`, falling back to `default`
        where data are missing or NaN.
        """
        PROMPT_TEMPLATE = textwrap.dedent("""
        Based only on the given data for week ending **{date}**:U.S. macro indicator, risk-sentiment proxy and sector-ETF adjusted_close prices, a baseline return estimated by a quantitative mode
                                          and some technical features features of the etf returns of given period, predict your **variance_view** the alpha you expect above/below baseline 

        ▌Macro snapshot
        • Industrial Production INDPRO…… {INDPRO:.2f}
        • Retail Sales ex-Food RSAFS…… {RSAFS:.2f}
        • Housing Starts HOUST…………… {HOUST:.2f}
        • Unemployment Rate UNRATE…… {UNRATE:.2f} %
        • Core PCE PCEPILFE……………… {PCEPILFE:.2f}
        • CPI CPIAUCSL………………… {CPIAUCSL:.2f}
        • Fed Funds EFFR………………… {EFFR:.2f}
        • 10Y-2Y Spread………………… {T10Y2Y:.3f} %
        • BAA Credit Spread…………… {BAA10Y:.3f} %
        • WTI Oil $………………… {DCOILWTICO:.2f}
        • Copper $………………… {PCOPPUSDM:.2f}

        ▌Risk sentiment
        • VIX………… {VIX:.2f}  (Δ {VIX_weekly_change:.3f})
        • MOVE…… {MOVE:.2f} (Δ {MOVE_weekly_change:.3f})
        
        ▌Sector ETF data:                                                                    
        | Sector (ETF) | adj_close | baseline_ret | ewma_1w | ewma_1m |
        |--------------|-------|--------------|---------|---------|
        | Information Technology (XLK) | {XLK:.4f} | {XLK_baseline_ret:.4f} | {XLK_ewma_1w:.4f} | {XLK_ewma_1m:.4f} |
        | Health Care (XLV)            | {XLV:.4f} | {XLV_baseline_ret:.4f} | {XLV_ewma_1w:.4f} | {XLV_ewma_1m:.4f} |
        | Financials (XLF)             | {XLF:.4f} | {XLF_baseline_ret:.4f} | {XLF_ewma_1w:.4f} | {XLF_ewma_1m:.4f} |
        | Consumer Discretionary (XLY) | {XLY:.4f} | {XLY_baseline_ret:.4f} | {XLY_ewma_1w:.4f} | {XLY_ewma_1m:.4f} |
        | Communication Services (XLC) | {XLC:.4f} | {XLC_baseline_ret:.4f} | {XLC_ewma_1w:.4f} | {XLC_ewma_1m:.4f} |
        | Industrials (XLI)            | {XLI:.4f} | {XLI_baseline_ret:.4f} | {XLI_ewma_1w:.4f} | {XLI_ewma_1m:.4f} |
        | Consumer Staples (XLP)       | {XLP:.4f} | {XLP_baseline_ret:.4f} | {XLP_ewma_1w:.4f} | {XLP_ewma_1m:.4f} |
        | Energy (XLE)                 | {XLE:.4f} | {XLE_baseline_ret:.4f} | {XLE_ewma_1w:.4f} | {XLE_ewma_1m:.4f} |
        | Utilities (XLU)              | {XLU:.4f} | {XLU_baseline_ret:.4f} | {XLU_ewma_1w:.4f} | {XLU_ewma_1m:.4f} |
        | Real Estate (XLRE)           | {XLRE:.4f} | {XLRE_baseline_ret:.4f} | {XLRE_ewma_1w:.4f} | {XLRE_ewma_1m:.4f} |
        | Materials (XLB)              | {XLB:.4f} | {XLB_baseline_ret:.4f} | {XLB_ewma_1w:.4f} | {XLB_ewma_1m:.4f} |

        **Task**
        For every ETF provided above, please provide:
        1. Your**variance_view** – the alpha you expect above/below baseline weekly return based on the analysis for next week. a decimal, (ex. –0.001 means 10 bp below baseline, 0 implies neutral.)
        2. A confidence score between 0 and 1,
        3. A brief rationale for the prediction. (ex. "INDPRO +0.6 % and VIX −12 % indicate risk-on rotation.")
        4. Provide an include a top-level field **overall_analysis** summarizing how the data inform your predictions.
        
        Each baseline_ret is the **expected 1-week total return** (decimal).
        Return variance_view in the *same* units (weekly total return).
                                          
        Do **NOT** adjust the baseline yourself – we’ll add it afterwards.
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


    # ------------------------------------------------------------------
    #  Get GPT-4o prediction (sector ETFs)
    # ------------------------------------------------------------------
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
                            "You are an equity trading strategist. Provide 1-week return forecasts for "
                            "major GICS sector ETFs based on macro, risk-sentiment, price data and some technical signal."
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
                    {
                        "instrument": name,
                        "predicted_return": None,
                        "confidence": 0,
                        "rationale": f"Error: {str(e)}",
                    }
                    for name in [
                        "Information Technology",
                        "Health Care",
                        "Financials",
                        "Consumer Discretionary",
                        "Communication Services",
                        "Industrials",
                        "Consumer Staples",
                        "Energy",
                        "Utilities",
                        "Real Estate",
                        "Materials",
                    ]
                ],
                "overall_analysis": f"Failed to generate predictions due to error: {str(e)}",
            }

    # ------------------------------------------------------------------
    #  End-to-end weekly pipeline
    # ------------------------------------------------------------------
    def run_pipeline(self, start_date, end_date):
        print("Loading weekly equity data…")
        weekly_path = "data/equity_combined_features_weekly.csv"
        if os.path.exists(weekly_path):
            data_w = pd.read_csv(weekly_path, index_col=0, parse_dates=True)
        else:
            data_w = self.data_collector.collect_data(start_date, end_date)

        # ── 1.  add baseline_ret columns  ─────────────────────────────────
        tickers     = [p["etf"] for p in PORTFOLIOS["equity"]["sectors"]]
        baseline_w  = self.estimate_returns(data_w, tickers, span_weeks=4)             # <- ewma_1m
        data_w      = pd.concat([data_w, baseline_w], axis=1)

        # ── 2.  main loop  ───────────────────────────────────────────────
        predictions, dates = [], []
        total_weeks        = len(data_w)

        name_to_etf = {p["name"]: p["etf"] for p in PORTFOLIOS["equity"]["sectors"]}

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

        final_path = "data/equity_weekly_predictions.csv"
        final_df.to_csv(final_path, index=False)
        print(f"Weekly equity predictions saved to '{final_path}'")

        # raw JSON (optional – keeps whole LLM response)
        pd.DataFrame({"date": dates, "predictions": predictions}).to_json("data/equity_weekly_predictions_raw.json", orient="records")

        return final_df

    # ---------------------------------------------------------------------
    #  helper: save a temp checkpoint every 5 predictions
    # ---------------------------------------------------------------------
    def _save_temp(self, preds, dates, name_to_etf, data_w, idx):
        out = self._records_from_preds(preds, dates, name_to_etf, data_w)
        temp = f"data/temp/equity_weekly_predictions_temp_{idx}.csv"
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
                    "predicted_volatility": inst.get("predicted_volatility"),
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

if __name__ == "__main__":
    start_date = "2022-11-31"
    end_date = "2022-12-01"

    equity_portfolio = PORTFOLIOS["equity"]["sectors"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    equity_data_collector = EquityDataCollector(
        portfolio=equity_portfolio,
        full_start_date="2020-01-01", 
        target_start_date="2023-11-01", 
        end_date="2025-03-31")
    
    equity_data_collector.collect_data(start_date, end_date)
    equity_agent = EquityAgent(data_collector=equity_data_collector, llm_client=client)
    
    start_date = "2023-11-01"
    end_date = "2025-03-31"
    
    result_df = equity_agent.run_pipeline(start_date, end_date)
    print("Equity Agent Predictions:")
    print(result_df.tail())
