# src/data_quality.py
# This file runs automated data quality checks on the UCI Online Retail dataset
# using Great Expectations. It validates columns like customer_id, price, quantity,
# invoice_date, and country against predefined rules.
# A JSON report is saved to reports/dq_report.json after every run.

import great_expectations as ge
import pandas as pd
import json
from pathlib import Path

# validate data using Great Expectations
def validate_uci_data(df: pd.DataFrame):
    """
    Run data quality checks on the dataset.
    Return: Pass/Fail status and details.
    """
    # convert the DataFrame into a Great Expectations context
    gdf = ge.from_pandas(df)
    results = {}

    # customer_id must not be null
    r1 = gdf.expect_column_values_to_not_be_null('customer_id')
    results['no_null_customer_id'] = r1['success']

    # quantity must be positive
    r2 = gdf.expect_column_values_to_be_between(
        'quantity', min_value=1, max_value=10000
    )
    results['quantity_positive'] = r2['success']

    # price must be positive
    r3 = gdf.expect_column_values_to_be_between(
        'price', min_value=0.01, max_value=5000
    )
    results['price_positive'] = r3['success']

    # invoice_date must be within a valid range
    r4 = gdf.expect_column_values_to_be_between(
        'invoice_date',
        min_value='2009-01-01',
        max_value='2012-12-31'
    )
    results['valid_date_range'] = r4['success']

    # country must be in the valid set
    valid_countries = [
        'United Kingdom', 'Germany', 'France', 'EIRE',
        'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal'
    ]
    r5 = gdf.expect_column_values_to_be_in_set(
        'country', valid_countries, mostly=0.90
    )
    results['valid_countries'] = r5['success']

    # row count must be within expected range
    r6 = gdf.expect_table_row_count_to_be_between(
        min_value=500000, max_value=1200000
    )
    results['row_count_ok'] = r6['success']

    # print summary report
    print("\n=== DATA QUALITY REPORT ===")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total} checks")
    print(f"DQ Score: {(passed/total)*100:.1f}%")
    for check, status in results.items():
        icon = 'PASS' if status else 'FAIL'
        print(f"[{icon}] {check}")

    return results

# main block
if __name__ == "__main__":
    df = pd.read_parquet('data/processed/uci_clean.parquet')
    results = validate_uci_data(df)

    # save the report as JSON
    Path("reports").mkdir(exist_ok=True)
    with open('reports/dq_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDQ report saved to reports/dq_report.json")