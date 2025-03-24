import yfinance as yf
import pandas as pd

# Define a list of ETFs representing major asset classes
etfs = [
    # Equities (Global, US, Emerging Markets, Developed Markets ex-US)
    "SPY",  # S&P 500 (US Large-Cap)
    "VT",   # Vanguard Total World Stock ETF
    "EFA",  # iShares MSCI EAFE (Developed Markets ex-US)
    "EEM",  # iShares MSCI Emerging Markets
    
    # Fixed Income (Treasuries, Investment Grade, High Yield, Emerging Market Bonds)
    "TLT",  # iShares 20+ Year Treasury Bond
    "LQD",  # iShares Investment Grade Corporate Bond
    "HYG",  # iShares High Yield Corporate Bond (Junk Bonds)
    "EMB",  # iShares J.P. Morgan USD Emerging Markets Bond
    
    # Commodities (Gold, Oil, Broad Commodities)
    "GLD",  # SPDR Gold Trust
    "USO",  # United States Oil Fund
    "DBC",  # Invesco DB Commodity Index Tracking Fund
    
    # Currencies (USD Index)
    "UUP",  # Invesco DB US Dollar Index Bullish Fund
    
    # Alternatives (Real Estate, Volatility)
    "VNQ",  # Vanguard Real Estate ETF (REITs)
    "VXX"   # iPath S&P 500 VIX Short-Term Futures (Volatility)
]

# Define the start date for data collection
start_date = "2023-11-01"

# Download daily historical data from Yahoo Finance
# The interval is set to "1d" (daily)
data = yf.download(etfs, start=start_date, end=None, interval="1d")

# Extract adjusted closing prices (to account for splits and dividends)
adj_close_prices = data['Adj Close']

print(adj_close_prices)