import os
import pandas as pd
from fredapi import Fred
import yfinance as yf
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_fred_series(api_key, start_date="2020-01-01", end_date=None, freq='W'):
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
        'IRLTLT01GBM156N': 'GBP_T10Y'   # UK 10-Year Rate
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
    yield_cols = ["3M_Yield", "6M_Yield", "1Y_Yield", "2Y_Yield", "5Y_Yield", "10Y_Yield", "EUR_T10Y", "JPY_T10Y", "GBP_T10Y"]
    for col in yield_cols:
        if col in df.columns:
            series = df[col].ffill()
            momentum_data[f'{col}_mom_1m'] = series.pct_change(periods=21)
            momentum_data[f'{col}_mom_3m'] = series.pct_change(periods=63)
            momentum_data[f'{col}_mom_12m'] = series.pct_change(periods=252)
            logger.info(f"Computed momentum for {col}.")
    return momentum_data

def get_risk_sentiment_data(start_date='2020-01-01', end_date='2025-03-31'):
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

# New function to download treasury ETF data
def get_etf_data(tickers, start_date='2020-01-01', end_date='2025-03-31'):
    logger.info(f"Downloading data for {len(tickers)} ETFs...")
    data = yf.download(tickers, start=start_date, end=end_date)
    if data.empty:
        logger.error("Error: No ETF data available")
        return pd.DataFrame()
    etf_data = data['Close'].copy()
    missing_data = etf_data.isnull().sum()
    if missing_data.any():
        logger.warning("Missing data points per ETF:")
        for ticker, count in missing_data.items():
            if count > 0:
                logger.warning(f"{ticker}: {count} missing points")
    return etf_data

def main():
    # Import configuration variables (assumes you have a config or .env file with your FRED API key)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")

    # Set date ranges
    full_start_date = '2020-01-01'  # For momentum calculations
    target_start_date = '2023-11-01'  # Target period start for predictions
    end_date = '2025-03-31'
    
    logger.info("Starting fixed income data collection...")
    
    # Download macro and yield data from FRED
    logger.info("Downloading FRED series...")
    fred_data = get_fred_series(FRED_API_KEY, start_date=full_start_date, end_date=end_date)
    logger.info(f"Downloaded FRED data with shape: {fred_data.shape}")
    
    # Compute momentum for yield series
    logger.info("Computing yield momentum factors...")
    fred_with_momentum = get_yield_momentum(fred_data)
    logger.info(f"Data with momentum factors shape: {fred_with_momentum.shape}")
    
    # Get risk sentiment data (VIX and MOVE)
    logger.info("Downloading risk sentiment data...")
    risk_data = get_risk_sentiment_data(start_date=full_start_date, end_date=end_date)
    if risk_data.empty:
        logger.warning("No risk sentiment data collected.")
    else:
        logger.info(f"Risk sentiment data shape: {risk_data.shape}")
    
    # Get Treasury ETF data for the fixed income portfolio
    logger.info("Downloading Treasury ETF data...")
    treasury_etf_data = get_etf_data(start_date=full_start_date, end_date=end_date)
    if treasury_etf_data.empty:
        logger.warning("No Treasury ETF data collected.")
    else:
        logger.info(f"Treasury ETF data shape: {treasury_etf_data.shape}")
    
    # Combine all data: macro, momentum, risk sentiment, and Treasury ETF prices
    logger.info("Combining FRED, risk sentiment, and Treasury ETF data...")
    combined_data = pd.concat([fred_with_momentum, risk_data, treasury_etf_data], axis=1)
    logger.info(f"Combined daily data shape: {combined_data.shape}")
    
    # Filter for target period
    combined_data = combined_data[target_start_date:end_date]
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

if __name__ == "__main__":
    main()