# src/models/xgboost_lightgbm_churn_model.py
# This file trains and compares three churn prediction models — XGBoost, LightGBM,
# and a 50/50 ensemble of both. It uses RFM-based features, logs all metrics to MLflow,
# saves the XGBoost model as a .pkl file, and generates confusion matrix and ROC curve plots.

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score
)

import matplotlib.pyplot as plt

# Create output folders for reports and models
os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("data",    exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("NeuralRetail_Churn_Prediction")

# Load RFM feature data from parquet
rfm = pd.read_parquet(r"data/features/rfm_features.parquet")

# Define churn label — customer is churned if inactive for 120+ days and ordered rarely
rfm["churned"] = ((rfm["recency"] > 120) & (rfm["frequency"] <= 2)).astype(int)

# Add average order value as an additional feature
rfm["avg_order_value"] = rfm["monetary"] / (rfm["frequency"] + 1)

# One-hot encode segment column if present
if "segment" in rfm.columns:
    rfm = pd.get_dummies(rfm, columns=["segment"], drop_first=True)

feature_cols = [
    "frequency",
    "monetary_log",
    "f_score",
    "m_score",
    "avg_order_value"
]

X = rfm[feature_cols].astype(float)
y = rfm["churned"]

# Stratified split to preserve class distribution in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save X_test to disk so SHAP analysis script can load it independently
X_test.to_csv("data/X_test.csv", index=False)

# Class weight to handle imbalanced churn labels
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


# XGBoost model — trained with class weight adjustment for imbalanced data
with mlflow.start_run(run_name="XGBoost_Churn"):

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Save trained XGBoost model locally for inference and SHAP analysis
    joblib.dump(model, "models/xgb_churn.pkl")
    print("Model saved successfully")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy  = accuracy_score(y_test, y_pred) * 100

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_metric("AUC_ROC",  round(auc_score, 4))
    mlflow.log_metric("Accuracy", round(accuracy, 2))

    # Confusion matrix — shows true vs predicted churn labels
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"])
    disp.plot()
    plt.title("Confusion Matrix - XGBoost")
    plt.savefig("reports/confusion_matrix_xgb.png")
    plt.close()

    # ROC curve — measures model's ability to separate churned vs active customers
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc     = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend()
    plt.savefig("reports/roc_curve_xgb.png")
    plt.close()


# LightGBM model — trained as a comparison baseline against XGBoost
with mlflow.start_run(run_name="LightGBM_Churn"):

    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    lgb_model.fit(X_train, y_train)

    lgb_preds = lgb_model.predict(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    lgb_auc = roc_auc_score(y_test, lgb_proba)
    lgb_acc = accuracy_score(y_test, lgb_preds) * 100

    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_metric("AUC_ROC",  round(lgb_auc, 4))
    mlflow.log_metric("Accuracy", round(lgb_acc, 2))


# Ensemble — averages XGBoost and LightGBM probabilities with equal weight
xgb_proba  = model.predict_proba(X_test)[:, 1]
final_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
final_preds = (final_proba > 0.5).astype(int)

ensemble_auc = roc_auc_score(y_test, final_proba)
ensemble_acc = accuracy_score(y_test, final_preds) * 100

# Save ensemble ROC curve to reports folder
fpr, tpr, _ = roc_curve(y_test, final_proba)
roc_auc     = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Ensemble")
plt.legend()
plt.savefig("reports/roc_curve_ensemble.png")
plt.close()

# Print final comparison of all three models
print("=" * 50)
print("CHURN MODEL RESULTS")
print("=" * 50)
print(f"XGBoost  AUC      : {auc_score:.4f}")
print(f"LightGBM AUC      : {lgb_auc:.4f}")
print(f"Ensemble AUC      : {ensemble_auc:.4f}")
print(f"Ensemble Accuracy : {ensemble_acc:.2f}%")
print("=" * 50)


if __name__ == "__main__":
    print("Starting Churn Model Pipeline")