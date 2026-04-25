\# Model Card — NeuralRetail ML Models



\## Churn Prediction Model



\### Model Details

\- Name: ChurnModel\_XGBoost\_LightGBM\_Ensemble

\- Version: v1.0

\- Type: Binary Classification

\- Framework: XGBoost 2.1 + LightGBM 4.3



\### Training Data

\- Dataset: UCI Online Retail II (2009-2011)

\- Samples: 4,372 customers

\- Features: RFM scores, recency, frequency, monetary

\- Target: Churn — recency greater than 90 days

\- Split: 80/20 stratified



\### Performance

\- AUC-ROC: 0.923 — Target >= 0.90 — ACHIEVED

\- Precision: 0.881

\- Recall: 0.763

\- F1-Score: 0.818

\- Decision Threshold: 0.35 tuned for high recall



\### Intended Use

\- Identify customers at risk of churning

\- Generate retention recommendations

\- CRM campaign targeting



\### Limitations

\- Trained on UK retail data — may not generalize to other markets

\- Seasonal patterns like Christmas may affect recency based features

\- Requires retraining if customer behavior shifts significantly



\### Bias Evaluation

\- Evaluated across UK vs non-UK customer groups — similar performance

\- No demographic proxies in features — GDPR compliant



\### Maintenance

\- Drift monitoring via Evidently AI — PSI threshold 0.2

\- Retraining schedule — weekly or on drift alert



\---



\## Demand Forecast Model



\### Model Details

\- Name: DemandForecast\_XGBoost

\- Version: v1.0

\- Type: Regression

\- Framework: XGBoost 2.1



\### Training Data

\- Dataset: UCI Online Retail II (2009-2011)

\- Features: Lag features, rolling averages, date features

\- Target: Daily demand units

\- Split: 70% train, 15% val, 15% test — chronological



\### Performance

\- MAPE: 8.74% — Target <= 10% — ACHIEVED

\- RMSE: 2156 units



\### Intended Use

\- Daily and weekly demand forecasting per SKU

\- Inventory reorder point calculation

\- Revenue simulation with price elasticity



\### Limitations

\- Trained on historical data — may not capture sudden demand shocks

\- No external signals like holidays or promotions included



\### Maintenance

\- Drift monitoring via PSI on demand distribution

\- Retraining triggered when MAPE exceeds 11.5%

