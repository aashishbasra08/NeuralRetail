# NeuralRetail - AI Sales Intelligence Platform

AI-powered demand forecasting, churn prediction and inventory optimization  
for retail companies | Amdox Technologies Internship | April 2026

## Live Demo
- Dashboard: https://neuralretail-aashish.streamlit.app
- API Docs: https://neuralretail-production-1218.up.railway.app/docs
- Demo Video: https://youtu.be/YOUR_VIDEO_ID

## Architecture
Data Sources -> Feature Engineering -> ML Models -> FastAPI Serving -> Streamlit Dashboard -> Monitoring

## Model Performance
| Model            | Metric     | Score | Target |
|------------------|------------|-------|--------|
| Demand Forecast  | MAPE       | 8.74% | <=10%  |
| Churn Prediction | AUC-ROC    | 0.923 | >=0.90 |
| Segmentation     | Silhouette | 0.623 | >=0.55 |

## Tech Stack
Python 3.12 | XGBoost | LightGBM | Prophet | LSTM | MLflow | Streamlit | FastAPI | Evidently AI | Docker | GitHub Actions

## Quick Start
git clone https://github.com/Aashishbasra08/NeuralRetail.git
cd NeuralRetail
python -m venv neuralretail_env
neuralretail_env\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/Streamlit_Home.py

## Project Structure
NeuralRetail/
- data/raw/
- data/processed/
- src/models/
- src/api/
- dashboard/
- models/
- reports/
- monitoring/

## Author
Aashish Basra | MCA | Amdox DS Intern  
GitHub: https://github.com/Aashishbasra08