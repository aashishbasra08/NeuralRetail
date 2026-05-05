# airflow_dags.py
# NeuralRetail — Airflow DAG for automated ML pipeline
# Pipeline: ingestion → features → churn → segmentation → drift monitor → retrain trigger
# Schedule: Daily at 2 AM
# Run locally: airflow dags trigger neuralretail_pipeline
# Docs: http://localhost:8080

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
import subprocess
import logging

log = logging.getLogger(__name__)

# ── Default Args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "aashish_basra",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "start_date":       datetime(2026, 4, 1),
}


# ── Python Callables ──────────────────────────────────────────────────────────
def run_ingestion():
    log.info("Running data ingestion pipeline...")
    result = subprocess.run(["python", "src/ingestion.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Ingestion failed:\n{result.stderr}")
    log.info("Ingestion complete!")


def run_data_quality():
    log.info("Running data quality checks...")
    result = subprocess.run(["python", "src/Data_Quality.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Data Quality failed:\n{result.stderr}")
    # Check DQ score
    if "100.0%" not in result.stdout and "Passed: 6/6" not in result.stdout:
        log.warning("DQ score below 100% — check report!")
    log.info("Data Quality complete!")


def run_features():
    log.info("Running feature engineering...")
    result = subprocess.run(["python", "src/features.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Feature engineering failed:\n{result.stderr}")
    log.info("Features complete!")


def run_churn_model():
    log.info("Training churn model...")
    result = subprocess.run(["python", "src/models/churn.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Churn model failed:\n{result.stderr}")
    log.info("Churn model complete!")


def run_segmentation():
    log.info("Running customer segmentation...")
    result = subprocess.run(["python", "src/models/segmentation.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Segmentation failed:\n{result.stderr}")
    log.info("Segmentation complete!")


def run_shap():
    log.info("Running SHAP analysis...")
    result = subprocess.run(["python", "src/models/shap_analysis.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"SHAP analysis failed:\n{result.stderr}")
    log.info("SHAP complete!")


def run_drift_monitor(**context):
    """Run drift monitor and push result to XCom for branching."""
    log.info("Running drift monitoring...")
    result = subprocess.run(["python", "src/monitoring/drift_monitor.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Drift monitor failed:\n{result.stderr}")

    # Check if retrain is needed
    drift_detected = "retrain needed" in result.stdout.lower()
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    log.info(f"Drift detected: {drift_detected}")
    return drift_detected


def check_drift_branch(**context):
    """Branch: retrain if drift detected, else skip."""
    drift_detected = context["ti"].xcom_pull(key="drift_detected", task_ids="drift_monitor")
    if drift_detected:
        log.info("Drift detected — triggering retrain!")
        return "retrain_models"
    else:
        log.info("No drift — skipping retrain.")
        return "no_retrain_needed"


def retrain_models():
    """Retrain churn + LSTM models when drift is detected."""
    log.info("Retraining models due to drift...")
    for script in ["src/models/churn.py", "src/models/lstm_forecasting.py"]:
        result = subprocess.run(["python", script], capture_output=True, text=True)
        log.info(result.stdout)
        if result.returncode != 0:
            raise Exception(f"Retrain failed for {script}:\n{result.stderr}")
    log.info("Retrain complete!")


def run_model_validation():
    """Run CI model validation gate."""
    log.info("Running model validation gate...")
    result = subprocess.run(["python", "tests/validate_models.py"], capture_output=True, text=True)
    log.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Model validation FAILED:\n{result.stderr}")
    log.info("Model validation PASSED!")


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="neuralretail_pipeline",
    default_args=default_args,
    description="NeuralRetail — Full ML pipeline: ingest → features → train → monitor → retrain",
    schedule_interval="0 2 * * *",   # Daily at 2 AM
    catchup=False,
    tags=["neuralretail", "mlops", "amdox"],
) as dag:

    # ── Start ─────────────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Data Layer ────────────────────────────────────────────────────────────
    ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
    )

    data_quality = PythonOperator(
        task_id="data_quality",
        python_callable=run_data_quality,
    )

    features = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_features,
    )

    # ── Model Layer ───────────────────────────────────────────────────────────
    churn = PythonOperator(
        task_id="churn_model",
        python_callable=run_churn_model,
    )

    segmentation = PythonOperator(
        task_id="segmentation",
        python_callable=run_segmentation,
    )

    shap = PythonOperator(
        task_id="shap_analysis",
        python_callable=run_shap,
    )

    # ── Monitoring Layer ──────────────────────────────────────────────────────
    drift_monitor = PythonOperator(
        task_id="drift_monitor",
        python_callable=run_drift_monitor,
        provide_context=True,
    )

    # ── Branch: Retrain or Skip ───────────────────────────────────────────────
    drift_branch = BranchPythonOperator(
        task_id="drift_branch",
        python_callable=check_drift_branch,
        provide_context=True,
    )

    retrain = PythonOperator(
        task_id="retrain_models",
        python_callable=retrain_models,
    )

    no_retrain = EmptyOperator(task_id="no_retrain_needed")

    # ── Validation Gate ───────────────────────────────────────────────────────
    validate = PythonOperator(
        task_id="model_validation",
        python_callable=run_model_validation,
        trigger_rule="none_failed_min_one_success",
    )

    # ── End ───────────────────────────────────────────────────────────────────
    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # ── Pipeline Order ────────────────────────────────────────────────────────
    start >> ingestion >> data_quality >> features
    features >> [churn, segmentation]
    churn >> shap
    [shap, segmentation] >> drift_monitor
    drift_monitor >> drift_branch
    drift_branch >> [retrain, no_retrain]
    [retrain, no_retrain] >> validate >> end