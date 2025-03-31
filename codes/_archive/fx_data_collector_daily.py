import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import sys
import os

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_currency_etf_data(start_date='2020-01-01', end_date='2025-02-28'):
    """Get currency ETF data for G5 currencies"""
    tickers = [
        'FXE',  # Euro
        'FXB',  # British Pound
        'FXY',  # Japanese Yen
        'FXF',  # Swiss Franc
        'FXC',  # Canadian Dollar
    ]
    
    print(f"Downloading data for {len(tickers)} ETFs...")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if data.empty:
        print("Error: No ETF data available")
        return pd.DataFrame()
    
    # Get Close prices and handle missing data
    etf_data = data['Close'].copy()
    
    # Report missing data
    missing_data = etf_data.isnull().sum()
    if missing_data.any():
        print("\nMissing data points per ETF:")
        for ticker, count in missing_data.items():
            if count > 0:
                print(f"{ticker}: {count} missing points")
    
    return etf_data

def get_exchange_rates(start_date='2020-01-01', end_date='2025-02-28'):
    """Get exchange rates for G5 currencies against USD"""
    # Define currency pairs
    pairs = [
        'EURUSD=X',  # Euro
        'GBPUSD=X',  # British Pound
        'USDJPY=X',  # Japanese Yen
        'USDCHF=X',  # Swiss Franc
        'USDCAD=X',  # Canadian Dollar
    ]
    
    # Download exchange rate data
    data = yf.download(pairs, start=start_date, end=end_date)
    return data['Close']

def get_momentum_factors(df):
    """Calculate momentum factors"""
    momentum_data = df.copy()
    
    # Calculate various momentum indicators
    for ticker in df.columns:
        # Fill NaN values before calculating momentum
        series = df[ticker].ffill()
        # 1-month momentum
        momentum_data[f'{ticker}_mom_1m'] = series.pct_change(periods=21, fill_method=None)
        # 3-month momentum
        momentum_data[f'{ticker}_mom_3m'] = series.pct_change(periods=63, fill_method=None)
        # 12-month momentum
        momentum_data[f'{ticker}_mom_12m'] = series.pct_change(periods=252, fill_method=None)
    
    return momentum_data

def get_risk_sentiment_data(start_date='2020-01-01', end_date='2025-02-28'):
    """Get risk sentiment data"""
    # Download market indices
    indices = {
        '^VIX': 'VIX',           # Volatility Index
        '^MOVE': 'MOVE'          # ICE BofA MOVE Index
    }
    
    risk_data = pd.DataFrame()
    
    for ticker, name in indices.items():
        try:
            print(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if data.empty:
                print(f"Warning: No data received for {ticker}")
                continue
            risk_data[name] = data
            print(f"Successfully downloaded {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    if risk_data.empty:
        print("Warning: No risk sentiment data was collected")
    else:
        print(f"Collected risk sentiment data with columns: {risk_data.columns.tolist()}")
    
    return risk_data

def get_interest_rates(start_date='2020-01-01', end_date='2025-02-28'):
    """Get 10-year interest rates for G5 economies from local CSV files"""
    try:
        # Define file paths and column mappings for G5 countries
        rate_files = {
            'US_T10Y': 'data/Investing Government Bond Yield Data/United States 10-Year Bond Yield Historical Data.csv',
            'GBP_T10Y': 'data/Investing Government Bond Yield Data/United Kingdom 10-Year Bond Yield Historical Data.csv',
            'JPY_T10Y': 'data/Investing Government Bond Yield Data/Japan 10-Year Bond Yield Historical Data.csv',
            'EUR_T10Y': 'data/Investing Government Bond Yield Data/Germany 10-Year Bond Yield Historical Data.csv',
            'CHF_T10Y': 'data/Investing Government Bond Yield Data/Switzerland 10-Year Bond Yield Historical Data.csv',
            'CAD_T10Y': 'data/Investing Government Bond Yield Data/Canada 10-Year Bond Yield Historical Data.csv'
        }
        
        rates_data = pd.DataFrame()
        
        # Read each file and combine the data
        for name, file_path in rate_files.items():
            try:
                # Read CSV file
                data = pd.read_csv(file_path)
                
                # Convert Date column to datetime
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Set Date as index and sort
                data.set_index('Date', inplace=True)
                data.sort_index(inplace=True)
                
                # Get the Price column (which contains the yield)
                rates_data[name] = data['Price']
                
                print(f"Successfully read {name}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if not rates_data.empty:
            # Ensure index is sorted
            rates_data.sort_index(inplace=True)
            
            # Filter for the desired date range
            mask = (rates_data.index >= pd.to_datetime(start_date)) & (rates_data.index <= pd.to_datetime(end_date))
            rates_data = rates_data[mask]
            
            # Calculate daily changes for all rates
            for col in rates_data.columns:
                rates_data[f'{col}_change'] = rates_data[col].diff()
            
            # Forward fill missing values (some rates might have gaps)
            rates_data = rates_data.ffill()
            
            print(f"Successfully processed interest rates data with shape: {rates_data.shape}")
        else:
            print("Warning: No interest rates data was collected")
        
        return rates_data
    
    except Exception as e:
        print(f"Error processing interest rates data: {e}")
        return pd.DataFrame()

def main():
    # Set date ranges
    full_start_date = '2020-01-01'  # Start from 2020 for momentum calculation
    target_start_date = '2023-11-01'  # Target period start
    end_date = '2025-02-28'
    
    print("Starting data collection...")
    
    # Get currency ETF data
    print("\nCollecting currency ETF data...")
    etf_data = get_currency_etf_data(full_start_date, end_date)
    if etf_data.empty:
        print("Error: No ETF data collected")
        return
    print(f"Collected ETF data with shape: {etf_data.shape}")
    
    # Calculate momentum factors
    print("\nCalculating momentum factors...")
    data_with_momentum = get_momentum_factors(etf_data)
    print(f"Added momentum factors. Shape: {data_with_momentum.shape}")
    
    # Get exchange rates
    print("\nCollecting exchange rates...")
    exchange_rates = get_exchange_rates(full_start_date, end_date)
    if exchange_rates.empty:
        print("Error: No exchange rate data collected")
        return
    print(f"Collected exchange rates. Shape: {exchange_rates.shape}")
    
    # Get risk sentiment data
    print("\nCollecting risk sentiment data...")
    risk_data = get_risk_sentiment_data(full_start_date, end_date)
    if risk_data.empty:
        print("Warning: No risk sentiment data collected")
    else:
        print(f"Collected risk sentiment data. Shape: {risk_data.shape}")
    
    # Get interest rates data
    print("\nCollecting interest rates data...")
    rates_data = get_interest_rates(full_start_date, end_date)
    if not rates_data.empty:
        print(f"Collected interest rates data. Shape: {rates_data.shape}")
    
    # Combine all data
    print("\nCombining all data...")
    combined_data = pd.concat([
        data_with_momentum,
        exchange_rates,
        risk_data,
        rates_data
    ], axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Filter for target period
    combined_data = combined_data[target_start_date:end_date]
    print(f"Data filtered to target period. Shape: {combined_data.shape}")
    
    # Drop rows with missing ETF data
    initial_rows = len(combined_data)
    combined_data = combined_data.dropna(subset=etf_data.columns)
    dropped_rows = initial_rows - len(combined_data)
    print(f"Dropped {dropped_rows} rows with missing ETF data. Remaining rows: {len(combined_data)}")
    
    if len(combined_data) == 0:
        print("Error: No data remaining after filtering")
        return
    
    # Save to CSV
    print("\nSaving data to CSV...")
    combined_data.to_csv('data/fx_combined_features.csv')
    print(f"Data collection completed. Final dataset shape: {combined_data.shape}")
    print("Data saved to 'data/fx_combined_features.csv'")

if __name__ == "__main__":
    main() 