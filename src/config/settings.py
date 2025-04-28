from pathlib import Path
import os
from datetime import date

DATA_DIR = os.path.join(Path(__file__).parent.parent, "data")
DATA_PATHS = {
    "raw": os.path.join(DATA_DIR, "raw/"),
    "processed": os.path.join(DATA_DIR, "processed"),
}

# Default date ranges

'''
DATE_RANGES = {
    "training": {
        "start_date": date(2020, 1, 1),
        "end_date": date(2022, 12, 31),
    },
    "testing": {
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 12, 31),
    },
    "validation": {
        "start_date": date(2024, 1, 1),
        "end_date": date.today(),
    }
}  
'''

DATE_RANGES = {
    "test_period": {
        "start_date": date(2023, 11, 1),
        "end_date": date(2025, 3, 31),
    }
}

# Default portfolio definitions
PORTFOLIOS = {
    # Equity sectors based on GICS classification
    "equity": {
            "sectors": [
                {"name": "Information Technology", "etf": "XLK", "weight": 0.0},
                {"name": "Health Care", "etf": "XLV", "weight": 0.0},
                {"name": "Financials", "etf": "XLF", "weight": 0.0},
                {"name": "Consumer Discretionary", "etf": "XLY", "weight": 0.0},
                {"name": "Communication Services", "etf": "XLC", "weight": 0.0},
                {"name": "Industrials", "etf": "XLI", "weight": 0.0},
                {"name": "Consumer Staples", "etf": "XLP", "weight": 0.0},
                {"name": "Energy", "etf": "XLE", "weight": 0.0},
                {"name": "Utilities", "etf": "XLU", "weight": 0.0},
                {"name": "Real Estate", "etf": "XLRE", "weight": 0.0},
                {"name": "Materials", "etf": "XLB", "weight": 0.0},
            ],
            "single_names": [
                {"ticker": "AAPL", "name": "Apple Inc.", "weight": 0.0},
                {"ticker": "MSFT", "name": "Microsoft Corp.", "weight": 0.0},
                {"ticker": "JPM", "name": "JPMorgan Chase", "weight": 0.0},
                {"ticker": "UNH", "name": "UnitedHealth Group", "weight": 0.0},
                {"ticker": "NVDA", "name": "NVIDIA Corp.", "weight": 0.0},
                {"ticker": "PG", "name": "Procter & Gamble", "weight": 0.0},
                {"ticker": "TSLA", "name": "Tesla Inc.", "weight": 0.0},
                {"ticker": "XOM", "name": "ExxonMobil", "weight": 0.0},
                {"ticker": "HD", "name": "Home Depot", "weight": 0.0},
                {"ticker": "VZ", "name": "Verizon Communications", "weight": 0.0}
            ],
            "factors": [
                {"name": "Value", "etf": "VLUE", "weight": 0.0},
                {"name": "Momentum", "etf": "MTUM", "weight": 0.0},
                {"name": "Quality", "etf": "QUAL", "weight": 0.0},
                {"name": "Low Volatility", "etf": "USMV", "weight": 0.0},
                {"name": "Size", "etf": "IJR", "weight": 0.0},
                {"name": "Growth", "etf": "IWF", "weight": 0.0}
            ]
        },
    # Bond maturity buckets and credit quality segments
    "bond": {
        "treasuries": [
            {"name": "Short-Term Treasury", "etf": "SHV", "maturity": "0-1yr", "weight": 0.0},
            {"name": "1-3 Year Treasury", "etf": "SHY", "maturity": "1-3yr", "weight": 0.0},
            {"name": "3-7 Year Treasury", "etf": "IEI", "maturity": "3-7yr", "weight": 0.0},
            {"name": "7-10 Year Treasury", "etf": "IEF", "maturity": "7-10yr", "weight": 0.0},
            {"name": "10-20 Year Treasury", "etf": "TLH", "maturity": "10-20yr", "weight": 0.0},
            {"name": "20+ Year Treasury", "etf": "TLT", "maturity": "20+yr", "weight": 0.0},
        ],
        "credit": [
            {"name": "Investment Grade Corporate", "etf": "LQD", "weight": 0.0},
            {"name": "High Yield Corporate", "etf": "HYG", "weight": 0.0},
            {"name": "Mortgage-Backed Securities", "etf": "MBB", "weight": 0.0},
            {"name": "Emerging Market Bonds", "etf": "EMB", "weight": 0.0},
        ],
    },
    
    # Commodity sectors based on Bloomberg Commodity Index (BCOM) classification
    "commodity": {
        "sectors": [
            {"name": "Energy", "etf": "USO", "weight": 0.0},
            {"name": "Precious Metals", "etf": "GLD", "weight": 0.0},
            {"name": "Industrial Metals", "etf": "CPER", "weight": 0.0},
            {"name": "Agriculture", "etf": "CORN", "weight": 0.0},
            {"name": "Livestock", "etf": "COW", "weight": 0.0},
        ],
    },
    
    # G5 currencies
    "fx": {
        "currencies": [
            {"name": "EUR/USD", "etf": "FXE", "weight": 0.2},
            {"name": "GBP/USD", "etf": "FXB", "weight": 0.2},
            {"name": "USD/JPY", "etf": "FXY", "weight": 0.2},
            {"name": "USD/CHF", "etf": "FXF", "weight": 0.2},
            {"name": "USD/CAD", "etf": "FXC", "weight": 0.2},
        ],
    }
}