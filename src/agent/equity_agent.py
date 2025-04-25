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
                                "XLK", "XLV", "XLF", "XLY", "XLC",
                                "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB"
                            ],
                            "description": "Sector ETF ticker"
                        },
                        "predicted_return": {
                            "type": "number",
                            "description": "Predicted 1-week total return (decimal)"
                        },
                        "predicted_volatility": {
                            "type": "number",
                            "description": "Predicted 1-week volatility (decimal)"
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
                    "required": ["instrument", "predicted_return", "predicted_volatility", "confidence", "rationale"],
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
    
    def prepare_prompt(self, row: pd.Series) -> str:
        """
        Creates a detailed prompt for the LLM based on the current market data.
        Includes sector ETF performance, momentum indicators, volatility metrics,
        market breadth, and macro indicators.
        
        Args:
            row (pd.Series): A row of data containing all features for the current timestamp
            
        Returns:
            str: Formatted prompt for the LLM
        """
        prompt = """
        Based on the following weekly U.S. macro indicators, risk-sentiment gauges, and sector-ETF prices, predict next week’s total return for each sector ETF.

        1. Macro Indicators  
        - Industrial Production (INDPRO): {INDPRO:.4f}  
        - Retail Sales ex-Food (RSAFS): {RSAFS:.4f}  
        - Housing Starts (HOUST): {HOUST:.4f}  
        - Unemployment Rate (UNRATE): {UNRATE:.4f}%  
        - Core PCE Price Index (PCEPILFE): {PCEPILFE:.2f}  
        - Consumer CPI (CPIAUCSL): {CPIAUCSL:.2f}  
        - Effective Federal Funds Rate (EFFR): {EFFR:.4f}
        - 10-y minus 2-y Treasury Spread (T10Y2Y): {T10Y2Y:.4f}%  
        - BAA Credit Spread (BAA10Y): {BAA10Y:.4f}%  
        - WTI Crude Oil ($/bbl, DCOILWTICO): {DCOILWTICO:.4f}  
        - Copper Price ($/t, PCOPPUSDM): {PCOPPUSDM:.4f}  

        2. Risk-Sentiment  
        - VIX Index: {VIX:.2f} (weekly Δ {VIX_weekly_change:.3f})  
        - MOVE Index: {MOVE:.2f} (weekly Δ {MOVE_weekly_change:.3f})  

        3. Sector ETFs:
        - Information Technology (XLK): {XLK:.4f} (mom_1m: {XLK_mom_1m:.4f}, mom_3m: {XLK_mom_3m:.4f}, mom_12m: {XLK_mom_12m:.4f}, ewma_1m: {XLK_ewma_1m:.4f})
          Historical Vol: 1m: {XLK_vol_1m:.4f}, 3m: {XLK_vol_3m:.4f}
        - Health Care (XLV): {XLV:.4f} (mom_1m: {XLV_mom_1m:.4f}, mom_3m: {XLV_mom_3m:.4f}, mom_12m: {XLV_mom_12m:.4f}, ewma_1m: {XLV_ewma_1m:.4f})
          Historical Vol: 1m: {XLV_vol_1m:.4f}, 3m: {XLV_vol_3m:.4f}
        - Financials (XLF): {XLF:.4f} (mom_1m: {XLF_mom_1m:.4f}, mom_3m: {XLF_mom_3m:.4f}, mom_12m: {XLF_mom_12m:.4f}, ewma_1m: {XLF_ewma_1m:.4f})
          Historical Vol: 1m: {XLF_vol_1m:.4f}, 3m: {XLF_vol_3m:.4f}
        - Consumer Discretionary (XLY): {XLY:.4f} (mom_1m: {XLY_mom_1m:.4f}, mom_3m: {XLY_mom_3m:.4f}, mom_12m: {XLY_mom_12m:.4f}, ewma_1m: {XLY_ewma_1m:.4f})
          Historical Vol: 1m: {XLY_vol_1m:.4f}, 3m: {XLY_vol_3m:.4f}
        - Communication Services (XLC): {XLC:.4f} (mom_1m: {XLC_mom_1m:.4f}, mom_3m: {XLC_mom_3m:.4f}, mom_12m: {XLC_mom_12m:.4f}, ewma_1m: {XLC_ewma_1m:.4f})
          Historical Vol: 1m: {XLC_vol_1m:.4f}, 3m: {XLC_vol_3m:.4f}
        - Industrials (XLI): {XLI:.4f} (mom_1m: {XLI_mom_1m:.4f}, mom_3m: {XLI_mom_3m:.4f}, mom_12m: {XLI_mom_12m:.4f}, ewma_1m: {XLI_ewma_1m:.4f})
          Historical Vol: 1m: {XLI_vol_1m:.4f}, 3m: {XLI_vol_3m:.4f}
        - Consumer Staples (XLP): {XLP:.4f} (mom_1m: {XLP_mom_1m:.4f}, mom_3m: {XLP_mom_3m:.4f}, mom_12m: {XLP_mom_12m:.4f}, ewma_1m: {XLP_ewma_1m:.4f})
          Historical Vol: 1m: {XLP_vol_1m:.4f}, 3m: {XLP_vol_3m:.4f}
        - Energy (XLE): {XLE:.4f} (mom_1m: {XLE_mom_1m:.4f}, mom_3m: {XLE_mom_3m:.4f}, mom_12m: {XLE_mom_12m:.4f}, ewma_1m: {XLE_ewma_1m:.4f})
          Historical Vol: 1m: {XLE_vol_1m:.4f}, 3m: {XLE_vol_3m:.4f}
        - Utilities (XLU): {XLU:.4f} (mom_1m: {XLU_mom_1m:.4f}, mom_3m: {XLU_mom_3m:.4f}, mom_12m: {XLU_mom_12m:.4f}, ewma_1m: {XLU_ewma_1m:.4f})
          Historical Vol: 1m: {XLU_vol_1m:.4f}, 3m: {XLU_vol_3m:.4f}
        - Real Estate (XLRE): {XLRE:.4f} (mom_1m: {XLRE_mom_1m:.4f}, mom_3m: {XLRE_mom_3m:.4f}, mom_12m: {XLRE_mom_12m:.4f}, ewma_1m: {XLRE_ewma_1m:.4f})
          Historical Vol: 1m: {XLRE_vol_1m:.4f}, 3m: {XLRE_vol_3m:.4f}
        - Materials (XLB): {XLB:.4f} (mom_1m: {XLB_mom_1m:.4f}, mom_3m: {XLB_mom_3m:.4f}, mom_12m: {XLB_mom_12m:.4f}, ewma_1m: {XLB_ewma_1m:.4f})
          Historical Vol: 1m: {XLB_vol_1m:.4f}, 3m: {XLB_vol_3m:.4f}

        For every ETF provided above, please provide:
        1. The predicted expected return for next week as a decimal,
        2. A confidence score between 0 and 1,
        3. A brief rationale for the prediction.
        4. Provide an include a top-level field **overall_analysis** summarizing how macro conditions, sector-specific drivers, and risk sentiment inform your predictions.

        Respond **only** with valid JSON conforming to the provided schema.
        """

        try:
            formatted_prompt = prompt.format(**row.to_dict())

        except KeyError as e:
            print(f"Warning: missing {e}; filling with 0.0.")
            row_copy = row.copy()

            required_cols = [
                # ---- macro series
                'INDPRO', 'RSAFS', 'HOUST', 'UNRATE',
                'PCEPILFE', 'CPIAUCSL', 'EFFR',
                'T10Y2Y', 'BAA10Y', 'DCOILWTICO', 'PCOPPUSDM',
                # ---- risk-sentiment
                'VIX', 'VIX_weekly_change', 'MOVE', 'MOVE_weekly_change',
                # ---- sector ETFs
                'XLK', 'XLV', 'XLF', 'XLY', 'XLC',
                'XLI', 'XLP', 'XLE', 'XLU', 'XLRE', 'XLB'
            ]

            for col in required_cols:
                if col not in row_copy or pd.isna(row_copy[col]):
                    row_copy[col] = 0.0

            formatted_prompt = prompt.format(**row_copy.to_dict())

        return formatted_prompt


    # ------------------------------------------------------------------
    #  Get GPT-4o prediction (sector ETFs)
    # ------------------------------------------------------------------
    def get_gpt4o_prediction(self, prompt: str) -> dict:
        """
        Get prediction from GPT-4o API with structured output for equity sectors.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",                                   # GPT-4o
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an equity-sector strategist. Provide 1-week return forecasts for "
                            "major GICS sector ETFs based on macro, risk-sentiment and price data."
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
        print("Loading weekly equity data...")
        if os.path.exists("data/equity_combined_features_weekly.csv"):
            data = pd.read_csv("data/equity_combined_features_weekly.csv", index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            data = self.data_collector.collect_data(start_date, end_date)

        predictions = []
        dates = []
        total_weeks = len(data)

        # Mapping from instrument names to ETF tickers
        instrument_to_etf = {
            "Information Technology": "XLK",
            "Health Care": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
        }

        for i, (date, row) in enumerate(data.iterrows(), 1):
            print(f"Processing week ending {date} ({i}/{total_weeks})...")
            prompt = self.prepare_prompt(row)
            prediction = self.get_gpt4o_prediction(prompt)
            predictions.append(prediction)
            dates.append(date)
            time.sleep(1)  # Respect API rate limits

            if i % 5 == 0 or i == total_weeks:
                processed_data = {
                    "date": [],
                    "etf": [],
                    "instrument": [],
                    "predicted_return": [],
                    "confidence": [],
                    "rationale": [],
                    "overall_analysis": [],
                }
                for j, pred in enumerate(predictions):
                    pred_date = dates[j]
                    overall = pred.get("overall_analysis", "")
                    for inst in pred.get("instruments", []):
                        instrument = inst.get("instrument", "")
                        processed_data["date"].append(pred_date)
                        processed_data["etf"].append(instrument_to_etf.get(instrument, ""))
                        processed_data["instrument"].append(instrument)
                        processed_data["predicted_return"].append(inst.get("predicted_return"))
                        processed_data["confidence"].append(inst.get("confidence"))
                        processed_data["rationale"].append(inst.get("rationale", ""))
                        processed_data["overall_analysis"].append(overall)

                temp_filename = f"data/temp/equity_weekly_predictions_temp_{i}.csv"
                os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
                pd.DataFrame(processed_data).to_csv(temp_filename, index=False)
                print(f"Progress saved: {i}/{total_weeks} predictions")

        processed_data = {
            "date": [],
            "etf": [],
            "instrument": [],
            "predicted_return": [],
            'predicted_volatility': [],
            "confidence": [],
            "rationale": [],
            "overall_analysis": [],
        }
        for j, pred in enumerate(predictions):
            pred_date = dates[j]
            overall = pred.get("overall_analysis", "")
            for inst in pred.get("instruments", []):
                instrument = inst.get("instrument", "")
                processed_data["date"].append(pred_date)
                processed_data["etf"].append(instrument_to_etf.get(instrument, ""))
                processed_data["instrument"].append(instrument)
                processed_data["predicted_return"].append(inst.get("predicted_return"))
                processed_data["predicted_volatility"].append(inst.get("predicted_volatility"))
                processed_data["confidence"].append(inst.get("confidence"))
                processed_data["rationale"].append(inst.get("rationale", ""))
                processed_data["overall_analysis"].append(overall)

        final_df = pd.DataFrame(processed_data)
        final_csv = "data/equity_weekly_predictions.csv"
        final_df.to_csv(final_csv, index=False)
        print(f"Weekly equity predictions completed and saved to '{final_csv}'")

        raw_df = pd.DataFrame({"date": dates, "predictions": predictions})
        raw_json = "data/equity_weekly_predictions_raw.json"
        raw_df.to_json(raw_json, orient="records")
        print(f"Raw predictions saved to '{raw_json}'")

        return final_df

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
