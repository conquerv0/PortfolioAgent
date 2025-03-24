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
import data_downloader as dd

USER_NAME = 'vince_astra'

def stocks_download_crsp_daily(start_year,end_year,vars_stocks,cusips=None):
    db = wrds.Connection(wrds_username = USER_NAME)   
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
    db = wrds.Connection(wrds_username = USER_NAME)    
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

if __name__=='__main__':
    start_year_crsp = 2020
    start_year = 2020
    end_year = 2024
    path_data = 'C:/Users/conqu/OneDrive/Documents/2025Spring/Thesis/codes/data/raw' #To-Dos: Change this to the path where you want to save the data
    variables_stocks = ['DlyCalDt','PERMNO','CUSIP9','SecurityNm','PERMCO','securitysubtype','securitytype','SICCD','ICBIndustry','DlyOpen','DlyClose','DlyHigh','DlyLow','DlyBid','DlyAsk','DlyPrc','DlyOrdDivAmt','DlyNonOrdDivAmt','DlyFacPrc','DisFacPr','DlyRet','DlyVol','ShrOut','DlyCap']
    fundamental_stocks = ['datadate','CUSIP','RDQ','gvkey']
    xpressfeed_stocks = ['datadate','CUSIP','RDQ','gvkey','PDATEQ','FQTR','RP','SALEQ','DLCQ','DLTTQ','CHEQ','MKVALTQ','OIBDPQ','CAPXY','OANCFY','PRSTKCY','XRDY','WCAPCHY']
    vars_ff = ['date','mktrf','smb','hml','rf','umd']
    vars_rp = ['relevance','css','rpa_date_utc','rpa_time_utc','timestamp_utc','rp_story_id','rp_entity_id','entity_name','topic','type','sub_type','category','news_type','rp_source_id','nip','bee','bmq','bam','bca','ber','anl_chg','event_similarity_key','event_similarity_days','event_relevance','event_sentiment_score']
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M")

    crsp_data = stocks_download_crsp_daily(start_year_crsp,end_year,variables_stocks)
    crsp_data.rename(columns={'dlycaldt': 'date'}, inplace=True)
    crsp_data.rename(columns={'cusip9': 'cusip'}, inplace=True)
    crsp_data.to_csv(path_data + 'crsp_data_raw.csv')
    
    ff_data = dd.ff_download(start_year_crsp,end_year,vars_ff)
    ff_data.to_csv(path_data + 'ff_data_raw.csv')

    #     ## RP DOWNLOAD
    # for year in range(start_year, end_year, 2):
    #     next_year = year + 1
    #     rp_data = ravenpack_download(year, next_year, vars_rp)
    #     var_name = f'rp_data{str(year)[-2:]}{str(next_year)[-2:]}'
    #     locals()[var_name] = rp_data        
    #     csv_filename = f'rp_data{str(year)[-2:]}{str(next_year)[-2:]}.csv'
    #     rp_data.to_csv(os.path.join(path_data, csv_filename))