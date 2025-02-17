import os
import pandas as pd
from datetime import timedelta, datetime
import os
import pandas as pd
from datetime import timedelta, datetime
from data_downloader import stocks_download_crsp_daily

def update_crsp_data(existing_data_path, start_date, end_date=None):
    """
    Incrementally updates an existing CSV file with new CRSP daily data.
    If the file does not exist, it calls on the main download routine to
    download the complete dataset for the given period.
    
    The end_date parameter defaults to today's date if not provided.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if not os.path.exists(existing_data_path):
        print("CRSP data file not found. Downloading complete dataset using main download routine...")
        crsp_data = stocks_download_crsp_daily(start_date.year, end_date.year, VARIABLES_STOCKS)
        
        if 'DlyCalDt' in crsp_data.columns:
            crsp_data = crsp_data.rename(columns={'DlyCalDt': 'date'})
        if 'CUSIP9' in crsp_data.columns:
            crsp_data = crsp_data.rename(columns={'CUSIP9': 'cusip'})
        if 'date' in crsp_data.columns:
            crsp_data['date'] = pd.to_datetime(crsp_data['date'])
        
        mask = (crsp_data['date'] >= start_date) & (crsp_data['date'] <= end_date)
        crsp_data = crsp_data.loc[mask]

        if crsp_data.empty:
            print("No data downloaded for the given date range.")
            return
        
        crsp_data.to_csv(existing_data_path, index=False)
        print(f"Data downloaded and saved. Total records: {len(crsp_data)}")
        return

    crsp_data = pd.read_csv(existing_data_path)
    crsp_data['date'] = pd.to_datetime(crsp_data['date'])
    last_date = crsp_data['date'].max()
    next_date = max(last_date + timedelta(days=1), start_date)

    if next_date > end_date:
        print("Data is already up-to-date.")
        return

    download_start_year = next_date.year
    download_end_year = end_date.year
    print(f"Downloading new CRSP data from {download_start_year} to {download_end_year}...")

    new_data = stocks_download_crsp_daily(download_start_year, download_end_year, VARIABLES_STOCKS)
    
    if 'DlyCalDt' in new_data.columns:
        new_data = new_data.rename(columns={'DlyCalDt': 'date'})
    if 'CUSIP9' in new_data.columns:
        new_data = new_data.rename(columns={'CUSIP9': 'cusip'})
    if 'date' in new_data.columns:
        new_data['date'] = pd.to_datetime(new_data['date'])
    
    mask = (new_data['date'] >= next_date) & (new_data['date'] <= end_date)
    new_data = new_data.loc[mask]
    
    if new_data.empty:
        print("No new data found for the given date range.")
        return
    
    crsp_data = pd.concat([crsp_data, new_data], ignore_index=True)
    crsp_data.to_csv(existing_data_path, index=False)
    print(f"Data updated. Total records: {len(crsp_data)}")


if __name__ == '__main__':
    update_crsp_data(
        r'{path_to_data}\crsp_data.csv',
        '2023-01-01'
    )
    VARIABLES_STOCKS = [
        'DlyCalDt','PERMNO','CUSIP9','SecurityNm','PERMCO','securitysubtype',
        'securitytype','SICCD','ICBIndustry','DlyOpen','DlyClose','DlyHigh',
        'DlyLow','DlyBid','DlyAsk','DlyPrc','DlyOrdDivAmt','DlyNonOrdDivAmt',
        'DlyFacPrc','DisFacPr','DlyRet','DlyVol','ShrOut','DlyCap'
    ]