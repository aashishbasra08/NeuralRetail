# Model Card — NeuralRetail ML Models

## Churn Prediction Model

### Model Details
- Name: ChurnModel_XGBoost_LightGBM_Ensemble
- Version: v1.0
- Type: Binary Classification
- Framework: XGBoost 2.1 + LightGBM 4.3

### Training Data
- Dataset: UCI Online Retail II (2009-2011)
- Samples: 4,372 customers
- Features: RFM scores, recency, frequency, monetary
- Target: Churn — recency > 90 days
- Split: 80/20 stratified

### Performance
- AUC-ROC: 0.923 — Target >= 0.90 — ✅ ACHIEVED
- Precision: 0.881
- Recall: 0.763
- F1-Score: 0.818
- Decision Threshold: 0.35 (tuned for high recall)

### Intended Use
- Identify customers at risk of churning
- Generate retention recommendations
- CRM campaign targeting

### Limitations
- Trained on UK retail data — may not generalize to other markets
- Seasonal patterns (Christmas) may affect recency-based features
- Requires retraining if customer behavior shifts significantly

### Bias Evaluation
- Evaluated across UK vs non-UK customer groups — similar performance
- No demographic proxies in features — GDPR compliant

### Maintenance
- Drift monitoring via Evidently AI — PSI threshold 0.2
- Retraining schedule — weekly or on drift alert

---

## Demand Forecast Model

### Model Details
- Name: DemandForecast_LSTM_Prophet_Ensemble
- Version: v1.0
- Type: Regression (Time Series)
- Framework: PyTorch LSTM + Prophet + Optuna ensemble

### Training Data
- Dataset: M5 Forecasting (Walmart sales)
- Features: Lag features, rolling averages, date features
- Target: Daily demand units
- Split: 70% train, 15% val, 15% test — chronological

### Performance
- Ensemble MAPE: 10.35% — Target <= 10% — ⚠️ Near target
- LSTM alone MAPE: 12.02%
- Prophet alone MAPE: 22.03%
- Best weights: LSTM 73.5% + Prophet 26.5% (Optuna tuned)
- RMSE: 5,383 | MAE: 4,647

### Intended Use
- Daily and weekly demand forecasting per SKU
- Inventory reorder point calculation
- Revenue simulation with price elasticity

### Limitations
- Trained on historical data — may not capture sudden demand shocks
- No external signals like holidays or promotions included

### Maintenance
- Drift monitoring via PSI on demand distribution
- Retraining triggered when MAPE exceeds 11.5%

---

## Customer Segmentation Model

### Model Details
- Name: CustomerSegmentation_KMeans_PCA
- Version: v1.0
- Type: Clustering
- Framework: Scikit-learn KMeans + PCA

### Training Data
- Dataset: M5 RFM features (store/dept level)
- Features: recency_log, frequency_log, monetary_log
- Scaler: RobustScaler
- Dimensionality Reduction: PCA 2D (100% variance explained)

### Performance
- Silhouette Score: 0.6226 — Target >= 0.55 — ✅ ACHIEVED
- Davies-Bouldin: 0.4761
- Best K: 3 (via silhouette search k=3 to 9)
- Status: PRODUCTION_READY

### Segment Labels
- Champions (17) — High frequency, high monetary
- Loyal Customers (15) — Medium frequency, medium monetary
- Hibernating (32) — Low frequency, low monetary
- At Risk (3) — Low frequency, high monetary
- Potential (3) — High frequency, low monetary

### Maintenance
- Retrain quarterly or when segment distribution shifts >20%
