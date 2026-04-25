# src/ingestion.py
# This file handles raw data ingestion and preprocessing for the NeuralRetail pipeline.
# It loads the UCI Online Retail Excel file (both sheets), cleans and validates the data,
# and loads the RetailRocket events dataset.
# All outputs are saved as compressed Parquet files in the data/processed/ folder.

import pandas as pd
import os
from pathlib import Path
import logging

# logging setup — messages will appear on the console
logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# paths
RAW_PATH = Path('data/raw')
PROCESSED_PATH = Path('data/processed')
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# load UCI data
def load_uci_data(filepath: str) -> pd.DataFrame:
    """
    Read the Online Retail Excel file.
    Combine both sheets (2009-10 and 2010-11).
    """
    log.info(f'Loading UCI data from {filepath}...')
    # read both sheets
    df1 = pd.read_excel(filepath, sheet_name='Year 2009-2010', dtype={'Customer ID': str})
    df2 = pd.read_excel(filepath, sheet_name='Year 2010-2011', dtype={'Customer ID': str})
    # combine both sheets
    df = pd.concat([df1, df2], ignore_index=True)
    log.info(f'UCI data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns')
    return df

# clean UCI data
def clean_uci_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove nulls and fix invalid values.
    """
    original_rows = len(df)

    # standardize column names
    df.columns = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'price', 'customer_id', 'country']
    df["stock_code"] = df["stock_code"].astype(str)

    # remove null customer_id rows
    df = df.dropna(subset=['customer_id'])

    # remove nulls from quantity and price
    df = df[df['quantity'].notnull()]
    df = df[df['price'].notnull()]

    # convert quantity and price to numeric safely
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['quantity', 'price'])

    # remove invalid values
    df = df[df['quantity'] > 0]
    df = df[df['price'] >= 0.01]

    # remove extreme outliers
    df = df[(df['quantity'] <= 10000)]
    df = df[(df['price'] <= 5000)]

    # final sanity check
    assert (df['quantity'] > 0).all()
    assert (df['price'] > 0).all()

    # remove cancelled transactions — invoices starting with 'C'
    df = df[~df['invoice_no'].astype(str).str.startswith('C')]

    # add total amount column
    df['total_amount'] = df['quantity'] * df['price']

    # parse invoice date and extract year, month, day
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['year']  = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day']   = df['invoice_date'].dt.day

    cleaned_rows = len(df)
    log.info(f'Cleaned: {original_rows:,} -> {cleaned_rows:,} rows '
             f'({original_rows - cleaned_rows:,} removed)')
    return df

# save to parquet
def save_to_parquet(df: pd.DataFrame, filename: str):
    """
    Save in Parquet format — fast and compressed.
    """
    output_path = PROCESSED_PATH / filename
    df.to_parquet(output_path, index=False, compression='snappy')
    size_mb = output_path.stat().st_size / (1024*1024)
    log.info(f'Saved: {output_path} ({size_mb:.1f} MB)')

# load RetailRocket dataset
def load_retailrocket_data(filepath: str) -> pd.DataFrame:
    """
    Load RetailRocket events.csv.
    """
    log.info('Loading RetailRocket events...')
    df = pd.read_csv(filepath)
    # convert timestamp from milliseconds to datetime
    df['event_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'visitorid': 'visitor_id', 'itemid': 'item_id'})
    log.info(f'RetailRocket loaded: {df.shape[0]:,} events')
    return df

# main block
if __name__ == '__main__':

    # UCI pipeline
    uci_file = RAW_PATH / 'online_retail_II.xlsx'
    df_uci = load_uci_data(str(uci_file))
    df_uci_clean = clean_uci_data(df_uci)
    save_to_parquet(df_uci_clean, 'uci_clean.parquet')

    # RetailRocket pipeline
    rr_file = RAW_PATH / 'events.csv'
    df_rr = load_retailrocket_data(str(rr_file))
    save_to_parquet(df_rr, 'retailrocket_events.parquet')

    print('Data ingestion complete!')
    print(f'UCI shape: {df_uci_clean.shape}')
    print(f'RetailRocket shape: {df_rr.shape}')