\# NeuralRetail — AI Sales Intelligence Platform



> AI-powered demand forecasting, churn prediction and inventory optimization

> for retail companies | Amdox Technologies Internship | April 2026



\## Live Demo

\- Dashboard: https://aashishbasra08-neuralretail.streamlit.app

\- API Docs: https://neuralretail-api.railway.app/docs

\- Demo Video: https://youtu.be/YOUR\_VIDEO\_ID



\## Architecture

Data Sources → Feature Engineering → ML Models → FastAPI Serving → Streamlit Dashboard → Monitoring



\## Model Performance

| Model             | Metric  | Score  | Target  |

|-------------------|---------|--------|---------|

| Demand Forecast   | MAPE    | 8.74%  | <= 10%  |

| Churn Prediction  | AUC-ROC | 0.923  | >= 0.90 |



\## Tech Stack

Python 3.12 | XGBoost | LightGBM | MLflow | Streamlit | FastAPI | Evidently AI | Docker | GitHub Actions



\## Quick Start

git clone https://github.com/Aashishbasra08/NeuralRetail.git

cd NeuralRetail

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

mlflow server --port 5000

streamlit run dashboard/streamlit\_home.py



\## Project Structure

NeuralRetail/

├── data/raw/               # Raw datasets (not in git)

├── data/processed/         # Cleaned Parquet files

├── data/features/          # Feature engineered files

├── src/models/             # ML model training code

├── src/api/                # FastAPI endpoints

├── src/monitoring/         # Evidently AI drift detection

├── dashboard/              # Streamlit pages

├── notebooks/              # EDA and experiments

├── models/                 # Saved model pkl files

├── reports/                # Drift reports and SHAP plots

├── tests/                  # CI validation scripts

└── monitoring/             # Prometheus and Grafana config



\## Author

Aashish Basra | MCA | Amdox DS Intern

GitHub: github.com/Aashishbasra08

