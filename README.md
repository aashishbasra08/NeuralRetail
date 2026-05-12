# NeuralRetail 🚀
### AI-Powered Retail Intelligence Platform
End-to-end MLOps system for **Demand Forecasting, Churn Prediction, and Customer Segmentation** built using Machine Learning, Deep Learning, and production-grade deployment tools.

---

## 🚀 Live Demo
- 🌐 **Dashboard:** https://neuralretail-aashish.streamlit.app
- ⚡ **API Docs:** Deployed on Railway (run locally for full access — see setup below)
---

## 🔐 Demo Credentials

| Role     | Email                        | Password    | Access         |
|----------|------------------------------|-------------|----------------|
| Admin    | aashishbasra8@gmail.com      | admin123    | Full access    |
| Analyst  | analyst@neuralretail.com     | analyst123  | Analytics only |
| Viewer   | viewer@neuralretail.com      | view123     | Read only      |

> **Recommended for evaluators:** Use `viewer@neuralretail.com` / `view123`

---

## 🧠 Problem Statement
Retail businesses struggle with:
- ❌ Stockouts and overstocking
- ❌ Customer churn
- ❌ Inefficient marketing

👉 NeuralRetail solves these using AI-driven insights.

---

## ✨ Key Features
- 📈 Demand Forecasting (Prophet + LSTM Ensemble) — MAPE ≤ 8.74%
- 👥 Customer Churn Prediction (XGBoost + LightGBM) — AUC-ROC 0.923
- 🧩 Customer Segmentation (KMeans, DBSCAN, GMM)
- ⚡ FastAPI-based real-time prediction API
- 📊 Interactive Streamlit dashboard (5 pages, role-based access)
- 📦 Dockerized deployment on Railway
- 📉 Drift monitoring with Evidently AI
- 🔬 Experiment tracking with MLflow

---

## 🏗️ Architecture

```text
Data Sources → Spark + Delta Lake → Feature Engineering (Feast)
     → ML Models (Prophet/LSTM/XGBoost) → MLflow Registry
     → FastAPI Scoring API → Streamlit Dashboard
     → Evidently AI Monitoring → Auto-Retrain (Airflow)
```

---

## 📊 Model Performance

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| Demand Forecasting | MAPE | ≤ 10% | **8.74%** ✅ |
| Churn Prediction | AUC-ROC | ≥ 0.90 | **0.923** ✅ |
| Customer Segmentation | Silhouette | ≥ 0.55 | **0.57** ✅ |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Forecasting | Prophet + LSTM (PyTorch Lightning) |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | Railway + Streamlit Cloud |

---

## 🚀 Local Setup

```bash
# Clone the repo
git clone https://github.com/aashishbasra08/NeuralRetail.git
cd NeuralRetail

# Create virtual environment
python -m venv neuralretail_env
neuralretail_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Start FastAPI
uvicorn src.api.main:app --reload --port 8000

# Start Streamlit
streamlit run dashboard/Streamlit_Home.py
```

---

## 📁 Project Structure

```
NeuralRetail/
├── dashboard/          # Streamlit multi-page app
│   ├── Streamlit_Home.py
│   └── pages/
│       ├── customers.py
│       ├── Demand.py
│       ├── Inventory.py
│       └── MLOps.py
├── src/
│   ├── api/            # FastAPI endpoints
│   ├── models/         # ML model training
│   └── monitoring/     # Evidently AI drift
├── notebooks/          # EDA + model training
├── data/               # Processed datasets
├── mlruns/             # MLflow experiments
├── Dockerfile.fastapi
├── Dockerfile.streamlit
└── docker-compose.prod.yml
```

---

## 👨‍💻 Author

**Aashish Basra**
Data Science & Analytics Intern — Amdox Technologies
- GitHub: https://github.com/aashishbasra08
- Project Code: AMX-DS-2026-04

---

*NeuralRetail © 2026 | Amdox Technologies | Confidential*