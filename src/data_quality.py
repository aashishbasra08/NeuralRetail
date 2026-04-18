import great_expectations as ge
import pandas as pd
import json
from pathlib import Path


def validate_uci_data(df: pd.DataFrame) -> dict:
    """
    UCI dataset pe data quality checks chalao.
    Return karo: pass/fail status aur details.
    """

    # DataFrame ko GE context me convert karo
    gdf = ge.from_pandas(df)

    results = {}

    # Check 1: customer_id null nahi hona chahiye
    r1 = gdf.expect_column_values_to_not_be_null('customer_id')
    results['no_null_customer_id'] = r1['success']

    # Check 2: quantity positive honi chahiye
    r2 = gdf.expect_column_values_to_be_between(
        'quantity', min_value=1, max_value=10000
    )
    results['quantity_positive'] = r2['success']

    # Check 3: price positive hona chahiye
    r3 = gdf.expect_column_values_to_be_between(
        'price', min_value=0.01, max_value=5000
    )
    results['price_positive'] = r3['success']

    # Check 4: invoice_date valid range me hona chahiye
    r4 = gdf.expect_column_values_to_be_between(
        'invoice_date',
        min_value='2009-01-01',
        max_value='2012-12-31'
    )
    results['valid_date_range'] = r4['success']

    # Check 5: country valid set me hona chahiye
    valid_countries = [
        'United Kingdom', 'Germany', 'France', 'EIRE',
        'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal'
    ]

    r5 = gdf.expect_column_values_to_be_in_set(
        'country', valid_countries, mostly=0.90
    )
    results['valid_countries'] = r5['success']

    # Check 6: row count expected range me hona chahiye
    r6 = gdf.expect_table_row_count_to_be_between(
        min_value=500000, max_value=1200000
    )
    results['row_count_ok'] = r6['success']

    # Summary print
    print("\n=== DATA QUALITY REPORT ===")

    passed = sum(results.values())
    total = len(results)

    print(f"Passed: {passed}/{total} checks")
    print(f"DQ Score: {(passed/total)*100:.1f}%")

    for check, status in results.items():
        icon = 'PASS' if status else 'FAIL'
        print(f"[{icon}] {check}")

    return results


# MAIN BLOCK (ye bahut important hai — sahi indentation)
if __name__ == "__main__":
    df = pd.read_parquet('data/processed/uci_clean.parquet')

    results = validate_uci_data(df)

    # Report save karo
    Path("reports").mkdir(exist_ok=True)

    with open('reports/dq_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDQ report saved to reports/dq_report.json")
