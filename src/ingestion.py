# src/ingestion.py --> This type of file convert data from raw to processed
import pandas as pd
import os
from pathlib import Path 
import logging

# Logging setup — console pe messages dikhenge 
logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__ )

#Paths 
RAW_PATH = Path('data/raw')
PROCESSED_PATH = Path('data/processed')
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

#Function 1: UCI Data Load karo
def load_uci_data(filepath: str) -> pd.DataFrame:
    """
    UCI Online Retail II Excel file ko read karo.
    Dono sheets (2009-10 aur 2010-11) ko combine karo.
    """
    log.info(f'Loading UCI data from {filepath}...') 
    # Dono sheets padho
    df1 = pd.read_excel(filepath, sheet_name='Year 2009-2010',dtype={'Customer ID': str})
    df2 = pd.read_excel(filepath, sheet_name='Year 2010-2011', dtype={'Customer ID': str})
    #Combine karo
    df = pd.concat([df1, df2], ignore_index=True)
    log.info(f'UCI data loaded: {df.shape[0]:,} rows,{df.shape[1]} columns') 
    return df

#Function 2: UCI Data Clean karo
def clean_uci_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: nulls hatao, wrong values fix karo. 
    """
    original_rows = len(df)
    # Column names standardize karo
    df.columns = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'price','customer_id', 'country']
    df["stock_code"]=df["stock_code"].astype(str)
    # Nulls hatao (customer_id wale rows mostly returns hain) 
    df = df.dropna(subset=['customer_id'])
    # Negative quantities = returns, inhe alag rakhenge 
    df_returns = df[df['quantity'] < 0].copy()
    df = df[df['quantity'] > 0].copy()
    # Negative/zero prices hatao
    df = df[df['price'] > 0]
    # Test transactions hatao (invoice 'C' se shuru wale = cancelled)
    df = df[~df['invoice_no'].astype(str).str.startswith('C')]
    # Total amount column add karo
    df['total_amount'] = df['quantity'] * df['price'] 
    # Date column properly parse karo
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['year'] = df['invoice_date'].dt.year 
    df['month'] = df['invoice_date'].dt.month 
    df['day'] = df['invoice_date'].dt.day
    cleaned_rows = len(df)
    log.info(f'Cleaned: {original_rows:,} -> {cleaned_rows:,} rows ' f'({original_rows - cleaned_rows:,} removed)')
    return df

#Function 3: Parquet mein save karo
def save_to_parquet(df: pd.DataFrame, filename: str):
    """
    Parquet format mein save karo — fast aur compressed.
    """
    output_path = PROCESSED_PATH / filename 
    df.to_parquet(output_path, index=False, compression='snappy')
    size_mb = output_path.stat().st_size / (1024*1024) 
    log.info(f'Saved: {output_path} ({size_mb:.1f} MB)')

#Function 4: RetailRocket Load karo 
def load_retailrocket_data(filepath: str) -> pd.DataFrame: 
    '''
    RetailRocket events.csv load karo.
    '''
    log.info('Loading RetailRocket events...') 
    df = pd.read_csv(filepath)
    # Timestamp milliseconds se datetime mein convert karo
    df['event_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'visitorid': 'visitor_id','itemid': 'item_id'})
    log.info(f'RetailRocket loaded: {df.shape[0]:,} events')
    return df

#Main function
if __name__=='__main__':
    # UCI pipeline
    uci_file = RAW_PATH / 'online_retail_II.xlsx' 
    df_uci = load_uci_data(str(uci_file))
    df_uci_clean = clean_uci_data(df_uci)
    save_to_parquet(df_uci_clean, 'uci_clean.parquet') 
    #RetailRocket pipeline
    rr_file = RAW_PATH / 'events.csv'
    df_rr = load_retailrocket_data(str(rr_file)) 
    save_to_parquet(df_rr, 'retailrocket_events.parquet')
    print('Data ingestion complete!')
    print(f'UCI shape: {df_uci_clean.shape}')
    print(f'RetailRocket shape: {df_rr.shape}')

