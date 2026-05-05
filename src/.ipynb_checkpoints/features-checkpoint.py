# src/features.py
# This file engineers two types of features from the cleaned UCI dataset.
# RFM features (Recency, Frequency, Monetary) are computed at the customer level
# for churn prediction and segmentation.
# Demand features include lag, rolling, and seasonal variables for forecasting.
# Both outputs are saved as parquet files in the data/features/ folder.

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_PATH = Path('data/features')
FEATURE_PATH.mkdir(parents=True, exist_ok=True)

def compute_rfm(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df['invoice_date'].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.Timestamp(snapshot_date)

    rfm = df.groupby('customer_id').agg(
        recency=('invoice_date', lambda x: (snapshot_date - x.max()).days),
        frequency=('invoice_no', 'nunique'),
        monetary=('total_amount', 'sum')
    ).reset_index()

    rfm['monetary_log'] = np.log1p(rfm['monetary'])
    rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5])

    rfm['rfm_score'] = (
        rfm['r_score'].astype(int) +
        rfm['f_score'].astype(int) +
        rfm['m_score'].astype(int)
    )

    def segment(row):
        if row['r_score'] >= 4 and row['f_score'] >= 4:
            return 'Champions'
        elif row['r_score'] >= 3 and row['f_score'] >= 3:
            return 'Loyal Customers'
        elif row['r_score'] >= 4 and row['f_score'] < 2:
            return 'New Customers'
        elif row['r_score'] <= 2 and row['f_score'] >= 3:
            return 'At Risk'
        elif row['r_score'] <= 2 and row['f_score'] <= 2:
            return 'Lost'
        else:
            return 'Potential'

    rfm['segment'] = rfm.apply(segment, axis=1)
    return rfm

def compute_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df['invoice_date'].dt.date)['quantity'].sum().reset_index()
    daily.columns = ['date', 'demand']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    # lag features
    daily['lag_1']  = daily['demand'].shift(1)
    daily['lag_7']  = daily['demand'].shift(7)
    daily['lag_14'] = daily['demand'].shift(14)

    # rolling features
    daily['rolling_mean_7']  = daily['demand'].rolling(7).mean()
    daily['rolling_mean_30'] = daily['demand'].rolling(30).mean()
    daily['rolling_std_7']   = daily['demand'].rolling(7).std()

    # date features
    daily['day_of_week']  = daily['date'].dt.dayofweek
    daily['month']        = daily['date'].dt.month
    daily['week_of_year'] = daily['date'].dt.isocalendar().week.astype(int)
    daily['is_weekend']   = (daily['day_of_week'] >= 5).astype(int)

    daily = daily.dropna()
    return daily

if __name__ == "__main__":
    df = pd.read_parquet('data/processed/uci_clean.parquet')

    # compute and save RFM features
    rfm = compute_rfm(df)
    rfm.to_parquet('data/features/rfm_features.parquet', index=False)

    # compute and save demand features
    demand_df = compute_demand_features(df)
    demand_df.to_parquet('data/features/demand_features.parquet', index=False)

    print("RFM DONE")
    print(rfm.head())
    print("DEMAND DONE")
    print(demand_df.head())