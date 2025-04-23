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
                            "description": "Predicted next period return (decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Prediction confidence level (0-1)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reasoning behind prediction"
                        }
                    },
                    "required": ["instrument", "predicted_return", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Overall market and sector analysis"
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
            "FEDFUNDS": "FEDFUNDS",    # Effective Federal Funds Rate
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

        # Extract ETF tickers
        tickers = [entry['etf'] for entry in self.portfolio]
        logger.info(f"Downloading adjusted close prices for {len(tickers)} sector ETFs...")
        # tickers = [entry["etf"] for entry in self.portfolio]
        # logger.info("Starting fixed income data collection...")
        etf_data = self.get_etf_data(tickers, self.full_start_date, self.end_date)
        adj_close = self.get_etf_adj_close(tickers, self.full_start_date, self.end_date)
        if adj_close.empty:
            logger.error("No adjusted close data downloaded for sector ETFs.")
            return pd.DataFrame()

        # Price-based features
        returns = adj_close.pct_change().dropna()
        momentum = pd.DataFrame(index=adj_close.index)
        for col in adj_close.columns:
            momentum[f"{col}_mom_1m"] = adj_close[col].pct_change(periods=21)
            momentum[f"{col}_mom_3m"] = adj_close[col].pct_change(periods=63)
            momentum[f"{col}_mom_12m"] = adj_close[col].pct_change(periods=252)
        volatility = pd.DataFrame(index=returns.index)
        for window, label in [(21, '1m'), (63, '3m'), (252, '12m')]:
            vol = returns.rolling(window=window).std() * np.sqrt(252)
            vol.columns = [f"{col}_vol_{label}" for col in returns.columns]
            volatility = pd.concat([volatility, vol], axis=1)

        # Combine all features
        features = pd.concat([adj_close, returns, momentum, volatility, macro], axis=1)
        features.index = pd.to_datetime(features.index)
        daily = features[self.full_start_date:self.end_date]

        # Save daily
        os.makedirs('data', exist_ok=True)
        daily.to_csv('data/equity_combined_features_daily.csv')
        logger.info("Saved daily equity features.")

        # Weekly
        weekly = daily.resample('W-FRI').last()
        weekly = weekly[self.target_start_date:self.end_date]
        weekly.to_csv('data/equity_combined_features_weekly.csv')
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
        Based on the following weekly financial market data, predict next week's returns for each sector ETF.

        1. Sector ETF Performance:"""

            # Process each sector ETF
        for sector in self.data_collector.portfolio.get("sectors", []):
                ticker = sector["etf"]
                name = sector["name"]
                prompt += f"""
        - {name} ({ticker}):
        • Price: {row.get(ticker, 0):.2f}
        • Weekly Return: {row.get(f'{ticker}_return', 0)*100:.2f}%
        • Momentum: 1m: {row.get(f'{ticker}_mom_1m', 0)*100:.2f}%, 3m: {row.get(f'{ticker}_mom_3m', 0)*100:.2f}%, 6m: {row.get(f'{ticker}_mom_6m', 0)*100:.2f}%
        • Volatility: {row.get(f'{ticker}_volatility', 0)*100:.2f}%
        • Relative Strength vs SPY: {row.get(f'{ticker}_rel_strength', 0):.3f}"""

            prompt += """

        2. Market Risk Indicators:
        - VIX Index: {VIX:.2f} (Weekly Change: {VIX_weekly_change:.2f}%)
        - MOVE Index: {MOVE:.2f} (Weekly Change: {MOVE_weekly_change:.2f}%)

        3. Market Breadth:
        - SPY Trend: {SPY_trend:.3f}
        - SPY Momentum: {SPY_momentum:.2f}%

        4. Macro Indicators:
        - Industrial Production (INDPRO): {INDPRO:.2f}
        - Retail Sales ex-Food (RSAFS): {RSAFS:.2f}
        - Housing Starts (HOUST): {HOUST:.2f}
        - Unemployment Rate (UNRATE): {UNRATE:.2f}%
        - Core PCE (PCEPILFE): {PCEPILFE:.2f}
        - CPI (CPIAUCSL): {CPIAUCSL:.2f}
        - Fed Funds Rate (FEDFUNDS): {FEDFUNDS:.2f}%
        - 10Y–2Y Yield Curve (T10Y2Y): {T10Y2Y:.2f}%
        - BAA Credit Spread (BAA10Y): {BAA10Y:.2f}%
        - WTI Oil (DCOILWTICO): {DCOILWTICO:.2f}
        - Copper Price (PCOPPUSDM): {PCOPPUSDM:.2f}

        Based on this comprehensive market data, please provide:
        1. For each sector ETF:
        - Predicted return for next week (as a decimal)
        - Confidence level (0-1)
        - Brief rationale considering sector-specific factors, momentum, and macro environment
        2. An overall market analysis incorporating:
        - Sector rotation trends
        - Risk sentiment
        - Macro implications
        - Key catalysts to watch

        Your response must be structured in the required JSON format with:
        - "instruments" array containing predictions for each sector ETF
        - "overall_analysis" providing market-wide insights"""

        try:
            formatted_prompt = prompt.format(**row.to_dict())
        except KeyError as e:
            logger.warning(f"Missing data for key {e}. Using default values.")
            # Create a copy with missing values filled
            row_copy = row.copy()
            
            # Required fields that should have defaults if missing
            required_fields = [
                # Market indicators
                'VIX', 'VIX_weekly_change', 'MOVE', 'MOVE_weekly_change',
                'SPY_trend', 'SPY_momentum',
                
                # Macro indicators
                'INDPRO', 'RSAFS', 'HOUST', 'UNRATE', 'PCEPILFE', 'CPIAUCSL',
                'FEDFUNDS', 'T10Y2Y', 'BAA10Y', 'DCOILWTICO', 'PCOPPUSDM'
            ]
            
            # Add ETF-specific fields for each sector
            for sector in self.data_collector.portfolio.get("sectors", []):
                ticker = sector["etf"]
                required_fields.extend([
                    ticker,
                    f'{ticker}_return',
                    f'{ticker}_mom_1m',
                    f'{ticker}_mom_3m',
                    f'{ticker}_mom_6m',
                    f'{ticker}_volatility',
                    f'{ticker}_rel_strength'
                ])
            
            # Fill missing values with 0.0
            for field in required_fields:
                if field not in row_copy or pd.isna(row_copy[field]):
                    row_copy[field] = 0.0
                    
            formatted_prompt = prompt.format(**row_copy.to_dict())
        
        return formatted_prompt

    def get_gpt4o_prediction(self, prompt: str) -> dict:
        try:
            resp = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a sector investment analyst."},
                    {"role":"user","content":prompt}
                ],
                response_format={"type":"json_schema","json_schema":PREDICTION_SCHEMA},
                temperature=0
            )
            content=resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # fallback empty
            insts=[]
            for e in self.data_collector.portfolio:
                insts.append({"instrument":e['etf'],"predicted_return":None,"confidence":0.0,"rationale":str(e)})
            return {"instruments":insts,"overall_analysis":"Error."}

    def run_pipeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        # load or collect
        if os.path.exists('data/equity_combined_features_weekly.csv'):
            data=pd.read_csv('data/equity_combined_features_weekly.csv',index_col=0,parse_dates=True)
        else:
            data=self.data_collector.collect_data(start_date,end_date)
        preds=[]; dates=[]
        total=len(data)
        for i,(dt,row) in enumerate(data.iterrows(),1):
            prompt=self.prepare_prompt(row)
            out=self.get_gpt4o_prediction(prompt)
            preds.append(out); dates.append(dt)
            if i%5==0 or i==total:
                logger.info(f"Processed {i}/{total}")
        # flatten
        records={"date":[],"instrument":[],"predicted_return":[],"confidence":[],"rationale":[],"overall_analysis":[]}
        for dt,p in zip(dates,preds):
            oa=p.get('overall_analysis','')
            for inst in p.get('instruments',[]):
                records['date'].append(dt)
                records['instrument'].append(inst['instrument'])
                records['predicted_return'].append(inst['predicted_return'])
                records['confidence'].append(inst['confidence'])
                records['rationale'].append(inst['rationale'])
                records['overall_analysis'].append(oa)
        df=pd.DataFrame(records)
        df.to_csv('data/equity_predictions.csv',index=False)
        return df

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
    # equity_agent = EquityAgent(data_collector=equity_data_collector, llm_client=client)
    
    # start_date = "2023-11-01"
    # end_date = "2025-03-31"
    
    # result_df = equity_agent.run_pipeline(start_date, end_date)
    # print("Equity Agent Predictions:")
    # print(result_df.tail())
