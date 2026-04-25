# src/models/xgboost_forecasting.py
# This file trains an XGBoost regression model for demand forecasting.
# It performs feature engineering (lag, rolling, date features), splits data
# chronologically for time series, logs parameters and metrics to MLflow,
# saves the trained model, and runs SHAP analysis on test predictions.

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from shap_analysis import run_shap_analysis
import joblib
import os

mlflow.set_tracking_uri('http://localhost:5000')


def train_xgboost(demand_parquet_path: str):
    mlflow.set_experiment('NeuralRetail_Demand_Forecasting')

    # Load raw demand data from parquet file
    df = pd.read_parquet(demand_parquet_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Sort by date to maintain chronological order (required for time series)
    df = df.sort_values('date').reset_index(drop=True)

    # Lag features — capture demand from previous days
    df['lag_1'] = df['demand'].shift(1)
    df['lag_2'] = df['demand'].shift(2)
    df['lag_3'] = df['demand'].shift(3)
    df['lag_7'] = df['demand'].shift(7)

    # Rolling average features — smooth out short-term noise
    df['rolling_mean_3'] = df['demand'].rolling(3).mean()
    df['rolling_mean_7'] = df['demand'].rolling(7).mean()

    # Demand difference — captures day-over-day change (trend signal)
    df['demand_diff'] = df['demand'] - df['demand'].shift(1)

    # Date-based features — help model learn weekly and monthly seasonality
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month']       = pd.to_datetime(df['date']).dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # Drop rows with NaN values introduced by lag and rolling operations
    df.dropna(inplace=True)
    print('After FE Columns:', df.columns.tolist())

    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'rolling_mean_3', 'rolling_mean_7',
        'demand_diff',
        'day_of_week', 'month', 'is_weekend'
    ]

    X = df[feature_cols]
    y = df['demand']

    # Chronological split — no shuffling to avoid data leakage in time series
    n = len(X)
    X_train = X[:int(n * 0.70)]
    y_train = y[:int(n * 0.70)]
    X_val   = X[int(n * 0.70):int(n * 0.85)]
    y_val   = y[int(n * 0.70):int(n * 0.85)]
    X_test  = X[int(n * 0.85):]
    y_test  = y[int(n * 0.85):]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    with mlflow.start_run(run_name='XGBoost_tuned'):

        params = {
            'n_estimators':    500,
            'max_depth':       5,
            'learning_rate':   0.05,
            'subsample':       0.9,
            'colsample_bytree': 0.9,
            'random_state':    42,
        }
        mlflow.log_params(params)

        # Train the XGBoost regression model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Save model locally as a .pkl file
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgb_demand.pkl")
        print("Model saved at models/xgb_demand.pkl")

        # Generate predictions on the held-out test set
        preds   = model.predict(X_test)
        actuals = y_test

        # Compute MAPE and RMSE — no scaler needed since target is raw demand
        mape = np.mean(np.abs((actuals - preds) / actuals + 1e-5)) * 100
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))

        mlflow.log_metric('MAPE', mape)
        mlflow.log_metric('RMSE', rmse)

        # Print feature importances ranked by contribution to model predictions
        importances = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        print("\nFeature Importance:")
        print(importances)

        # Register the model in MLflow Model Registry for versioning and deployment
        mlflow.xgboost.log_model(
            model,
            name='xgboost_model',
            registered_model_name='DemandForecast_XGBoost'
        )

        print(f'\nXGBoost Results: MAPE={mape:.2f}%, RMSE={rmse:.0f}')

        return {
            'mape':          mape,
            'rmse':          rmse,
            'model':         model,
            'X_test':        X_test,
            'feature_names': feature_cols
        }


if __name__ == '__main__':
    results = train_xgboost(r'data/features/demand_features.parquet')

    # Reconstruct X_test as DataFrame with correct column names for SHAP
    X_test = pd.DataFrame(results['X_test'], columns=results['feature_names'])

    # Run SHAP explainability analysis on the demand model
    run_shap_analysis(X_test, model_path="models/xgb_demand.pkl")