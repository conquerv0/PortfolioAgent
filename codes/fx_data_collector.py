import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from fredapi import Fred

# Configure FRED API
FRED_API_KEY = '4a2b876c9abde725340e62b99b0ddb57'
fred = Fred(api_key=FRED_API_KEY)

def get_currency_etf_data(start_date='2010-01-01', end_date='2025-02-28'):
    """Get currency ETF data"""
    tickers = ['UUP', 'FXE', 'FXY']
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

def get_momentum_factors(df):
    """Calculate momentum factors"""
    # Calculate various momentum indicators
    for ticker in df.columns:
        # 1-month momentum
        df[f'{ticker}_mom_1m'] = df[ticker].pct_change(periods=21)
        # 3-month momentum
        df[f'{ticker}_mom_3m'] = df[ticker].pct_change(periods=63)
        # 12-month momentum
        df[f'{ticker}_mom_12m'] = df[ticker].pct_change(periods=252)
    return df

def get_risk_sentiment_data(start_date='2010-01-01', end_date='2025-02-28'):
    """Get risk sentiment data"""
    # Download VIX data
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    # Create a DataFrame with VIX data
    risk_data = pd.DataFrame()
    risk_data['VIX'] = vix
    
    # Download other risk indicators (SPY for market returns)
    spy = yf.download('SPY', start=start_date, end=end_date)['Close']
    risk_data['SPY_returns'] = spy.pct_change()
    
    return risk_data

def get_macro_data(start_date='2010-01-01', end_date='2025-02-28'):
    """Get macro economic data using FRED API"""
    try:
        # List of FRED series to download
        series = {
            'GDP': 'GDP',           # Gross Domestic Product
            'CPIAUCSL': 'CPI',      # Consumer Price Index
            'PAYEMS': 'Payrolls',   # Total Nonfarm Payrolls
            'DFF': 'FedRate',       # Federal Funds Rate
            'T10Y2Y': 'YieldSpread' # 10Y-2Y Treasury Spread
        }
        
        macro_data = pd.DataFrame()
        
        for code, name in series.items():
            # Get data using fredapi
            data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            macro_data[name] = data
        
        # Convert index to datetime if not already
        macro_data.index = pd.to_datetime(macro_data.index)
        
        # Calculate month-over-month changes for relevant metrics
        for col in ['CPI', 'Payrolls', 'GDP']:
            if col in macro_data.columns:
                macro_data[f'{col}_MoM'] = macro_data[col].pct_change()
        
        # Forward fill missing values (since some data is quarterly/monthly)
        macro_data = macro_data.ffill()
        
        # Resample to daily frequency
        macro_data = macro_data.resample('D').ffill()
        
        return macro_data
    
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return pd.DataFrame()

def main():
    # Set date range
    start_date = '2010-01-01'
    end_date = '2025-02-28'
    
    # Get currency ETF data
    etf_data = get_currency_etf_data(start_date, end_date)
    
    # Calculate momentum factors
    data_with_momentum = get_momentum_factors(etf_data)
    
    # Get risk sentiment data
    risk_data = get_risk_sentiment_data(start_date, end_date)
    
    # Get macro data
    macro_data = get_macro_data(start_date, end_date)
    
    # Combine all data
    combined_data = pd.concat([
        data_with_momentum,
        risk_data,
        macro_data
    ], axis=1)
    
    # Save to CSV
    combined_data.to_csv('data/fx_combined_features.csv')
    print("Data collection completed and saved to 'data/fx_combined_features.csv'")

if __name__ == "__main__":
    main() 