# NeuralRetail - Model Card

## Overview
NeuralRetail is an AI-powered retail analytics system designed to solve three core business problems:
- Demand Forecasting
- Customer Churn Prediction
- Customer Segmentation

The system combines classical ML, deep learning, and statistical models for production-grade predictions.

---

## 1. Demand Forecasting Model

### Model Type
- LSTM (Deep Learning)
- XGBoost (Baseline)
- Ensemble (Final)

### Objective
Predict future product demand to optimize inventory and reduce stockouts.

### Input Features
- Historical demand
- Lag features (t-1, t-7, t-14)
- Rolling averages
- Seasonal indicators

### Output
- Predicted demand (next time step / future horizon)

### Performance
- MAPE: **8.74%** (Target: ≤10%)

### Business Impact
- Reduced overstocking
- Improved supply chain planning

---

## 2. Churn Prediction Model

### Model Type
- XGBoost + LightGBM (Ensemble)

### Objective
Identify customers likely to stop purchasing.

### Input Features
- RFM features (Recency, Frequency, Monetary)
- Purchase history
- Customer behavior metrics

### Output
- Probability of churn (0–1)

### Performance
- AUC-ROC: **0.923** (Target: ≥0.90)

### Business Impact
- Targeted retention campaigns
- Increased customer lifetime value

---

## 3. Customer Segmentation

### Model Type
- KMeans / DBSCAN / GMM

### Objective
Segment customers into meaningful groups for marketing strategies.

### Input Features
- RFM metrics
- Behavioral patterns

### Output
- Cluster labels (customer segments)

### Performance
- Silhouette Score: **0.623** (Target: ≥0.55)

### Business Impact
- Personalized marketing
- Better customer targeting

---

## Data

### Source
- RetailRocket Dataset
- Synthetic demand dataset (M5-inspired)

### Preprocessing
- Missing value handling
- Feature engineering
- Scaling (MinMax / StandardScaler)

---

## Limitations

- LSTM performance depends on sequence length and data quality
- Churn model assumes historical behavior consistency
- Segmentation sensitive to feature scaling

---

## Future Improvements

- Add real-time data pipeline (Kafka)
- Hyperparameter tuning with Optuna
- Online learning models
- Deep ensemble models

---

## Deployment

- FastAPI (Model Serving)
- Streamlit (Dashboard)
- Docker + Railway (Production)
- MLflow (Experiment Tracking)

---

## Author

Aashish Basra  
MCA | Data Science Intern @ Amdox Technologies  
GitHub: https://github.com/Aashishbasra08