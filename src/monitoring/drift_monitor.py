# src/monitoring/drift_monitor.py
# This file monitors data and model drift in production using Evidently.
# It generates HTML drift reports, calculates PSI manually for feature stability,
# and triggers a retraining alert (logged to MLflow) if significant drift is detected.

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    RegressionPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
    DatasetMissingValuesMetric
)
from pathlib import Path
from datetime import datetime
import logging
import json

log = logging.getLogger(__name__)

REPORTS_PATH = Path('reports/drift')
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


def run_data_drift_report(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    report_name:  str = None
) -> dict:
    """
    Generate a data drift report comparing reference (training) data
    against current (production) data using Evidently.
    Returns a dict with drift status and per-column results.
    """
    if report_name is None:
        report_name = f'drift_{datetime.now().strftime("%Y%m%d_%H%M")}'

    data_drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name='recency'),
        ColumnDriftMetric(column_name='frequency'),
        ColumnDriftMetric(column_name='monetary'),
    ])

    data_drift_report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # Save full drift report as an HTML file for visual inspection
    html_path = REPORTS_PATH / f'{report_name}_data_drift.html'
    data_drift_report.save_html(str(html_path))
    log.info(f'Data drift report saved: {html_path}')

    # Extract top-level drift summary from report output
    report_dict     = data_drift_report.as_dict()
    drift_share     = report_dict['metrics'][0]['result']['drift_share']
    dataset_drifted = report_dict['metrics'][0]['result']['dataset_drift']

    print(f'\n=== DATA DRIFT REPORT ===')
    print(f'Dataset Drifted: {dataset_drifted}')
    print(f'Drift Share: {drift_share:.2%} of features drifted')

    results = {
        'dataset_drifted': dataset_drifted,
        'drift_share':     drift_share,
        'column_drift':    {}
    }

    # Extract per-column drift status and PSI threshold for RFM features
    for col in ['recency', 'frequency', 'monetary']:
        col_result = [
            m for m in report_dict['metrics']
            if m.get('metric') == 'ColumnDriftMetric'
            and m['result'].get('column_name') == col
        ]
        if col_result:
            is_drifted = col_result[0]['result']['drift_detected']
            psi        = col_result[0]['result'].get('stattest_threshold', 0)
            results['column_drift'][col] = {
                'drifted': is_drifted,
                'psi':     psi
            }
            status = "DRIFTED" if is_drifted else "OK"
            print(f'  {col}: {status} (PSI = {psi})')

    return results


def run_model_drift_report(
    reference_df:   pd.DataFrame,
    current_df:     pd.DataFrame,
    target_col:     str,
    prediction_col: str,
    model_type:     str = 'classification'
) -> dict:
    """
    Generate a model performance drift report using Evidently.
    Supports both classification and regression model types.
    """
    if model_type == 'classification':
        report = Report(metrics=[ClassificationPreset()])
    else:
        report = Report(metrics=[RegressionPreset()])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping={
            'target':     target_col,
            'prediction': prediction_col
        }
    )

    # Save model performance report as HTML
    html_path = REPORTS_PATH / f'model_perf_{datetime.now().strftime("%Y%m%d")}.html'
    report.save_html(str(html_path))
    log.info(f'Model performance report: {html_path}')

    return report.as_dict()


def calculate_psi(
    reference: np.ndarray,
    current:   np.ndarray,
    bins:      int = 10
) -> float:
    """
    Manually calculate Population Stability Index (PSI) between two distributions.
    PSI < 0.1  : Stable — no action needed
    PSI 0.1-0.2: Monitor — slight shift detected
    PSI > 0.2  : Retrain — significant distribution change
    """
    _, bin_edges  = np.histogram(reference, bins=bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    ref_pct    = ref_counts / len(reference)
    ref_pct    = np.where(ref_pct == 0, 0.0001, ref_pct)  # Avoid log(0) by replacing zeros

    cur_counts = np.histogram(current, bins=bin_edges)[0]
    cur_pct    = cur_counts / len(current)
    cur_pct    = np.where(cur_pct == 0, 0.0001, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def check_retrain_trigger(
    drift_results: dict,
    current_mape:  float = None,
    baseline_mape: float = 10.0
) -> bool:
    """
    Decide whether model retraining should be triggered based on:
    1. Dataset drift detected by Evidently
    2. MAPE has degraded more than 15% above the baseline threshold
    Logs a drift alert run to MLflow if retraining is required.
    """
    should_retrain = False
    reasons        = []

    if drift_results.get('dataset_drifted'):
        should_retrain = True
        reasons.append(
            f"Dataset drift detected ({drift_results['drift_share']:.0%} features)"
        )

    if current_mape and current_mape > baseline_mape * 1.15:
        should_retrain = True
        reasons.append(
            f"MAPE degraded: {current_mape:.1f}% > threshold {baseline_mape * 1.15:.1f}%"
        )

    if should_retrain:
        print(f'RETRAIN TRIGGERED: {", ".join(reasons)}')
        try:
            import mlflow
            with mlflow.start_run(run_name='Drift_Alert'):
                mlflow.log_param('retrain_reason', str(reasons))
                mlflow.log_metric('drift_share', drift_results['drift_share'])
                if current_mape:
                    mlflow.log_metric('current_mape', current_mape)
            print('MLflow run logged: Drift_Alert')
        except Exception as e:
            log.warning(f'MLflow logging failed: {e}')
    else:
        print('No retrain needed — model is stable.')

    return should_retrain


if __name__ == '__main__':
    # Load reference RFM data (training baseline)
    reference = pd.read_parquet('data/features/rfm_features.parquet')

    # Simulate production drift by inflating recency values by 30%
    current            = reference.copy()
    current['recency'] = current['recency'] * 1.3

    results = run_data_drift_report(reference, current)
    check_retrain_trigger(results, current_mape=11.5)