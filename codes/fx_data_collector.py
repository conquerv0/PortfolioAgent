import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from fredapi import Fred
import sys
import os

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRED_API_KEY

# Configure FRED API
fred = Fred(api_key=FRED_API_KEY)

def get_currency_etf_data(start_date='2020-01-01', end_date='2025-02-28'):
    """Get currency ETF data"""
    tickers = ['UUP']
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

def get_exchange_rates(start_date='2020-01-01', end_date='2025-02-28'):
    """Get exchange rates for major currencies against USD"""
    # Define currency pairs
    pairs = [
        'EURUSD=X',  # Euro
        'JPYUSD=X',  # Japanese Yen
        'GBPUSD=X',  # British Pound
    ]
    
    # Download exchange rate data
    data = yf.download(pairs, start=start_date, end=end_date)
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

def get_risk_sentiment_data(start_date='2020-01-01', end_date='2025-02-28'):
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

def get_interest_rates(start_date='2020-01-01', end_date='2025-02-28'):
    """Get 10-year interest rates for major economies"""
    try:
        # List of FRED series to download (focusing on 10-year rates)
        series = {
            # US Rates
            'DGS10': 'US_T10Y',           # US 10-Year Treasury Rate
            
            # Euro Area Rates
            'IRLTLT01EZM156N': 'EUR_T10Y',  # Euro Area 10-Year Rate
            
            # Japan Rates
            'IRLTLT01JPM156N': 'JPY_T10Y',  # Japan 10-Year Rate
            
            # UK Rates
            'IRLTLT01GBM156N': 'GBP_T10Y',  # UK 10-Year Rate
        }
        
        rates_data = pd.DataFrame()
        
        for code, name in series.items():
            try:
                data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
                rates_data[name] = data
            except Exception as e:
                print(f"Error fetching {code}: {e}")
                continue
        
        # Convert index to datetime if not already
        rates_data.index = pd.to_datetime(rates_data.index)
        
        # Calculate daily changes for all rates
        for col in rates_data.columns:  # Use actual columns from the DataFrame
            rates_data[f'{col}_change'] = rates_data[col].diff()
        
        # Forward fill missing values (some rates might have gaps)
        rates_data = rates_data.ffill()
        
        return rates_data
    
    except Exception as e:
        print(f"Error fetching interest rates data: {e}")
        return pd.DataFrame()

def get_macro_data(start_date='2020-01-01', end_date='2025-02-28'):
    """Get US macro indicators"""
    try:
        # List of FRED series to download
        series = {
            'EFFR': 'FedRate',        # Effective Federal Funds Rate
            'T10Y2Y': 'YieldSpread'   # 10Y-2Y Treasury Spread
        }
        
        macro_data = pd.DataFrame()
        
        for code, name in series.items():
            try:
                data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
                macro_data[name] = data
            except Exception as e:
                print(f"Error fetching {code}: {e}")
                continue
        
        # Convert index to datetime if not already
        macro_data.index = pd.to_datetime(macro_data.index)
        
        # Calculate daily changes
        for col in macro_data.columns:
            macro_data[f'{col}_change'] = macro_data[col].diff()
        
        # Forward fill missing values
        macro_data = macro_data.ffill()
        
        return macro_data
    
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return pd.DataFrame()

def main():
    # Set date ranges
    full_start_date = '2020-01-01'  # Start from 2020 for momentum calculation
    target_start_date = '2023-11-01'  # Target period start
    end_date = '2025-02-28'
    
    # Get currency ETF data
    etf_data = get_currency_etf_data(full_start_date, end_date)
    
    # Calculate momentum factors
    data_with_momentum = get_momentum_factors(etf_data)
    
    # Get exchange rates
    exchange_rates = get_exchange_rates(full_start_date, end_date)
    
    # Get risk sentiment data
    risk_data = get_risk_sentiment_data(full_start_date, end_date)
    
    # Get interest rates data
    rates_data = get_interest_rates(full_start_date, end_date)
    
    # Get macro data
    macro_data = get_macro_data(full_start_date, end_date)
    
    # Combine all data
    combined_data = pd.concat([
        data_with_momentum,
        exchange_rates,
        risk_data,
        rates_data,
        macro_data
    ], axis=1)
    
    # Filter for target period
    combined_data = combined_data[target_start_date:end_date]
    
    # Drop rows where UUP is NaN
    combined_data = combined_data.dropna(subset=['UUP'])
    
    # Save to CSV
    combined_data.to_csv('data/fx_combined_features.csv')
    print("Data collection completed and saved to 'data/fx_combined_features.csv'")

if __name__ == "__main__":
    main() 