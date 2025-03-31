# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 20:30:30 2024
@author: conquerv0
"""
import numpy as np
import pandas as pd
import wrds
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import time
from datetime import datetime, timedelta
import yfinance as yf
import logging
from pathlib import Path
from fredapi import Fred

# Add the project directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration settings
from src.config.settings import DATA_PATHS, PORTFOLIOS, DATE_RANGES, API_KEYS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_all_etf_tickers():
    """
    Extract all ETF tickers from the PORTFOLIOS configuration.
    
    Returns:
        dict: Dictionary mapping asset classes to their respective ETF tickers
    """
    etf_tickers = {}
    
    # Extract equity ETFs
    equity_etfs = [item["etf"] for item in PORTFOLIOS["equity"]["sectors"]]
    etf_tickers["equity"] = equity_etfs
    
    # Extract bond ETFs
    treasury_etfs = [item["etf"] for item in PORTFOLIOS["bond"]["treasuries"]]
    credit_etfs = [item["etf"] for item in PORTFOLIOS["bond"]["credit"]]
    etf_tickers["bond"] = treasury_etfs + credit_etfs
    
    # Extract commodity ETFs (which can have multiple ETFs per sector)
    commodity_etfs = []
    for sector in PORTFOLIOS["commodity"]["sectors"]:
        commodity_etfs.extend(sector["etfs"])
    etf_tickers["commodity"] = commodity_etfs
    
    # Extract FX ETFs
    fx_etfs = [item["etf"] for item in PORTFOLIOS["fx"]["currencies"]]
    etf_tickers["fx"] = fx_etfs
    
    return etf_tickers

def stocks_download_crsp_daily(start_year,end_year,vars_stocks,cusips=None):
    db = wrds.Connection(wrds_username = API_KEYS["WRDS"])   
    years = range(start_year, end_year+1)
    stocks_list_fnct = pd.DataFrame()    
    comma = ","
    vars_select = comma.join(map(str, vars_stocks))
    if cusips is not None:
        cusips_select = ','.join([f"'{s}'" for s in cusips])

    for temp_year in years:
        start_time = time.time()
        temp_year_start = str(temp_year)+"-01-01"
        temp_year_end = str(temp_year)+"-12-31"
        if cusips is None:
            df_query_list = str("SELECT " + vars_select + " FROM crsp_a_stock.wrds_dsfv2_query WHERE (DlyCalDt BETWEEN '" + str(temp_year_start) + "' AND '" + str(temp_year_end) + "')")
        else: 
            df_query_list = str("SELECT " + vars_select + " FROM crsp_a_stock.wrds_dsfv2_query WHERE (DlyCalDt BETWEEN '" + str(temp_year_start) + "' AND '" + str(temp_year_end) + "') AND (cusip IN (" + cusips_select + "))")
        temp_df = db.raw_sql(df_query_list)
        stocks_list_fnct = pd.concat([stocks_list_fnct, temp_df])        
        elapsed_time = (time.time() - start_time)
        print(str(str(elapsed_time) + " seconds elapsed for year " + str(temp_year)))    
    db.close()
    return stocks_list_fnct

def ff_download(start_year,end_year,vars_name):
    db = wrds.Connection(wrds_username = API_KEYS["WRDS"])    
    years = range(start_year, end_year+1)
    ff_data = pd.DataFrame()   
    comma = ","
    vars_select = comma.join(map(str, vars_name))
    
    for temp_year in years:
        start_time = time.time()
        temp_year_start = str(temp_year)+"-01-01"
        temp_year_end = str(temp_year)+"-12-31"
        df_query_list = str("SELECT " + vars_select + " FROM ff_all.factors_daily WHERE (date BETWEEN '" + str(temp_year_start) + "' AND '" + str(temp_year_end) + "')")
        temp_df = db.raw_sql(df_query_list)
        ff_data = pd.concat([ff_data, temp_df])        
        elapsed_time = (time.time() - start_time)
        print(str(str(elapsed_time) + " seconds elapsed for year " + str(temp_year)))  
        
    db.close()
    return ff_data

# def ravenpack_download(start_year,end_year,variables):
#     db = wrds.Connection(wrds_username = USER_NAME)   
#     years = range(start_year, end_year+1)
#     return_df = pd.DataFrame()    
#     comma = ","
#     vars_select = comma.join(map(str, variables))
    
#     for temp_year in years:
#         start_time = time.time()
#         #temp_year_start = str(temp_year)+"-01-01"
#         #temp_year_end = str(temp_year)+"-12-31"
#         df_query_list = str("SELECT " + vars_select + " FROM ravenpack_dj.rpa_djpr_equities_" + str(temp_year))
#         temp_df = db.raw_sql(df_query_list)
#         return_df = pd.concat([return_df, temp_df])        
#         elapsed_time = (time.time() - start_time)
#         print(str(str(elapsed_time) + " seconds elapsed for year " + str(temp_year)))  
        
#     db.close()
#     return return_df

def download_daily_fred_series(api_key, start_date="2020-01-01", end_date=None):
    fred = Fred(api_key=api_key)
    
    # Define the FRED series to download
    series_dict = {
        "EFFR": "EFFR",          # Effective Federal Funds Rate
        "Headline_PCE": "PCE",   # Headline Personal Consumption Expenditures Price Index
        "Core_PCE": "PCEPILFE",  # Core PCE Price Index (excludes food and energy)
        "3M_Yield": "DGS3MO",    # 3-Month Treasury Constant Maturity Rate
        "6M_Yield": "DGS6MO",    # 6-Month Treasury Constant Maturity Rate
        "1Y_Yield": "DGS1",      # 1-Year Treasury Constant Maturity Rate
        "2Y_Yield": "DGS2",      # 2-Year Treasury Constant Maturity Rate
        "5Y_Yield": "DGS5",      # 5-Year Treasury Constant Maturity Rate
        "10Y_Yield": "DGS10",     # 10-Year Treasury Constant Maturity Rate
        'IRLTLT01EZM156N': 'EUR_T10Y',  # Euro Area 10-Year Rate
        'IRLTLT01JPM156N': 'JPY_T10Y',  # Japan 10-Year Rate
        'IRLTLT01GBM156N': 'GBP_T10Y',  # UK 10-Year Rate
    }
    
    data_frames = {}
    for label, series_id in series_dict.items():
        try:
            series_data = fred.get_series(series_id, start_date, end_date)
            # Convert to DataFrame and rename column to the label
            df_series = series_data.to_frame(name=label)
            data_frames[label] = df_series
        except Exception as e:
            print(f"Error downloading {series_id}: {e}")
    
    df = pd.concat(data_frames.values(), axis=1)
    
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_index)
    df = df.fillna(method='ffill')
    
    return df

def download_etf_full_data(tickers, start_date, end_date):
    logger.info(f"Downloading full price data for {len(tickers)} ETFs from {start_date} to {end_date}")
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Check if data is empty
        if data.empty:
            logger.error("No data downloaded")
            return pd.DataFrame()
            
        # Extract the necessary price columns
        if len(tickers) == 1:
            # For a single ticker, structure is different
            ticker = tickers[0]
            price_data = pd.DataFrame({
                (ticker, 'Open'): data['Open'],
                (ticker, 'High'): data['High'],
                (ticker, 'Low'): data['Low'],
                (ticker, 'Close'): data['Close'],
                (ticker, 'Volume'): data['Volume']
            })
            
            # Calculate VWAP using the simplified formula (H+L+C)/3
            price_data[(ticker, 'VWAP')] = (data['High'] + data['Low'] + data['Close']) / 3
        else:
            # For multiple tickers, we have a MultiIndex DataFrame
            price_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Calculate VWAP for each ticker
            for ticker in tickers:
                if (ticker, 'High') in price_data.columns:
                    price_data[(ticker, 'VWAP')] = (
                        price_data[(ticker, 'High')] + 
                        price_data[(ticker, 'Low')] + 
                        price_data[(ticker, 'Close')]
                    ) / 3
        
        # Check for missing data
        available_tickers = set([ticker for ticker, _ in price_data.columns])
        missing_tickers = [ticker for ticker in tickers if ticker not in available_tickers]
        if missing_tickers:
            logger.warning(f"Could not find data for tickers: {', '.join(missing_tickers)}")
        
        return price_data
    
    except Exception as e:
        logger.error(f"Error downloading ETF data: {str(e)}")
        return pd.DataFrame()

def download_all_etf_full_data(start_date=None, end_date=None):
    # Set default dates if not provided
    if start_date is None:
        start_date = DATE_RANGES["training"]["start_date"]
    if end_date is None:
        end_date = datetime.now().date()
    
    # Extract all ETF tickers
    etf_tickers = extract_all_etf_tickers()
    
    # Download data for each asset class
    etf_data = {}
    for asset_class, tickers in etf_tickers.items():
        logger.info(f"Downloading {asset_class} ETF full price data...")
        
        # Skip if there are no tickers for this asset class
        if not tickers:
            logger.warning(f"No ETFs defined for {asset_class}")
            continue
            
        # Download data
        class_data = download_etf_full_data(tickers, start_date, end_date)
        
        # Store data if successful
        if not class_data.empty:
            etf_data[asset_class] = class_data
            logger.info(f"Successfully downloaded data for {len(set([ticker for ticker, _ in class_data.columns]))} {asset_class} ETFs")
        else:
            logger.error(f"Failed to download data for {asset_class} ETFs")
    
    return etf_data

def extract_etf_tickers(portfolio: dict, key: str = "treasuries") -> list:
    """
    Extracts ETF tickers from a portfolio dictionary.

    Parameters:
        portfolio (dict): Portfolio dictionary structured with a key (e.g. "treasuries") 
                          mapping to a list of asset dictionaries.
                          Example:
                          {
                              "treasuries": [
                                  {"name": "Short-Term Treasury", "etf": "SHV", "maturity": "0-1yr", "weight": 0.0},
                                  {"name": "1-3 Year Treasury", "etf": "SHY", "maturity": "1-3yr", "weight": 0.0},
                                  ...
                              ]
                          }
        key (str): The key in the dictionary where the asset list is stored (default "treasuries").

    Returns:
        list: A list of ETF tickers extracted from the portfolio.
    """
    tickers = []
    assets = portfolio.get(key, [])
    for asset in assets:
        ticker = asset.get("etf")
        if ticker:
            tickers.append(ticker)
    return tickers

if __name__=='__main__':
    start_year_crsp = 2020
    start_year = 2020
    end_year = 2024
    path_data = 'C:/Users/conqu/OneDrive/Documents/2025Spring/Thesis/codes/data' #To-Dos: Change this to the path where you want to save the data
    # variables_stocks = ['DlyCalDt','PERMNO','CUSIP9','SecurityNm','PERMCO','securitysubtype','securitytype','SICCD','ICBIndustry','DlyOpen','DlyClose','DlyHigh','DlyLow','DlyBid','DlyAsk','DlyPrc','DlyOrdDivAmt','DlyNonOrdDivAmt','DlyFacPrc','DisFacPr','DlyRet','DlyVol','ShrOut','DlyCap']
    # fundamental_stocks = ['datadate','CUSIP','RDQ','gvkey']
    # xpressfeed_stocks = ['datadate','CUSIP','RDQ','gvkey','PDATEQ','FQTR','RP','SALEQ','DLCQ','DLTTQ','CHEQ','MKVALTQ','OIBDPQ','CAPXY','OANCFY','PRSTKCY','XRDY','WCAPCHY']
    # vars_ff = ['date','mktrf','smb','hml','rf','umd']
    # vars_rp = ['relevance','css','rpa_date_utc','rpa_time_utc','timestamp_utc','rp_story_id','rp_entity_id','entity_name','topic','type','sub_type','category','news_type','rp_source_id','nip','bee','bmq','bam','bca','ber','anl_chg','event_similarity_key','event_similarity_days','event_relevance','event_sentiment_score']
    # current_time = datetime.now().strftime("%Y-%m-%d-%H:%M")

    # crsp_data = stocks_download_crsp_daily(start_year_crsp,end_year,variables_stocks)
    # crsp_data.rename(columns={'dlycaldt': 'date'}, inplace=True)
    # crsp_data.rename(columns={'cusip9': 'cusip'}, inplace=True)
    # crsp_data.to_csv(path_data + 'crsp_data_raw.csv')
    
    # ff_data = ff_download(start_year_crsp,end_year,vars_ff)
    # ff_data.to_csv(path_data + 'ff_data_raw.csv')

    #     ## RP DOWNLOAD
    # for year in range(start_year, end_year, 2):
    #     next_year = year + 1
    #     rp_data = ravenpack_download(year, next_year, vars_rp)
    #     var_name = f'rp_data{str(year)[-2:]}{str(next_year)[-2:]}'
    #     locals()[var_name] = rp_data        
    #     csv_filename = f'rp_data{str(year)[-2:]}{str(next_year)[-2:]}.csv'
    #     rp_data.to_csv(os.path.join(path_data, csv_filename))
    fred_data = download_daily_fred_series(API_KEYS["FRED"], start_date="2023-01-01")