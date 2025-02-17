# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 20:30:30 2024
@author: conquerv0
"""
import numpy as np
import pandas as pd
import wrds
import time
from datetime import datetime, timedelta

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

def ravenpack_download(start_year,end_year,variables):
    db = wrds.Connection(wrds_username = USER_NAME)   
    years = range(start_year, end_year+1)
    return_df = pd.DataFrame()    
    comma = ","
    vars_select = comma.join(map(str, variables))
    
    for temp_year in years:
        start_time = time.time()
        #temp_year_start = str(temp_year)+"-01-01"
        #temp_year_end = str(temp_year)+"-12-31"
        df_query_list = str("SELECT " + vars_select + " FROM ravenpack_dj.rpa_djpr_equities_" + str(temp_year))
        temp_df = db.raw_sql(df_query_list)
        return_df = pd.concat([return_df, temp_df])        
        elapsed_time = (time.time() - start_time)
        print(str(str(elapsed_time) + " seconds elapsed for year " + str(temp_year)))  
        
    db.close()
    return return_df