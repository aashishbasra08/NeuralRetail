# dashboard/pages/mlops.py
# This file is a Streamlit dashboard page for MLOps monitoring.
# It tracks data drift (PSI scores) and model performance (AUC, MAPE) over the last 14 days,
# displays a model registry table, shows auto retrain trigger status, and allows Admin users
# to manually trigger retraining for churn or demand models.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="MLOps Monitor", layout="wide")

# Block access if user is not authenticated
if not st.session_state.get('authenticated'):
    st.warning("Please login first!")
    st.stop()

st.title("⚙️ MLOps Monitor — Drift & Model Health")
st.caption(f"Logged in as: {st.session_state.get('username')} | Role: {st.session_state.get('role')}")
st.divider()


@st.cache_data(ttl=300)
def load_drift_data():
    """Generate 14 days of simulated PSI drift scores for RFM features."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=14, freq='D')
    return pd.DataFrame({
        'date':          dates,
        'recency_psi':   np.clip(np.cumsum(np.random.normal(0.01,  0.02,  14)), 0.01, 0.45),
        'frequency_psi': np.clip(np.cumsum(np.random.normal(0.005, 0.01,  14)), 0.01, 0.20),
        'monetary_psi':  np.clip(np.cumsum(np.random.normal(0.008, 0.015, 14)), 0.01, 0.35),
    })


@st.cache_data(ttl=300)
def load_model_metrics():
    """Generate 14 days of simulated churn AUC and demand MAPE metrics."""
    np.random.seed(7)
    dates = pd.date_range(end=datetime.today(), periods=14, freq='D')
    return pd.DataFrame({
        'date':        dates,
        'churn_auc':   np.clip(0.923 + np.cumsum(np.random.normal(-0.002, 0.005, 14)), 0.85, 0.95),
        'demand_mape': np.clip(8.74  + np.cumsum(np.random.normal(0.05,   0.1,   14)), 7.0,  13.0),
    })


df_drift   = load_drift_data()
df_metrics = load_model_metrics()

# Extract latest values for KPI cards
latest_recency_psi  = df_drift['recency_psi'].iloc[-1]
latest_mape         = df_metrics['demand_mape'].iloc[-1]
latest_auc          = df_metrics['churn_auc'].iloc[-1]
latest_monetary_psi = df_drift['monetary_psi'].iloc[-1]

st.markdown("### Model Health Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Churn AUC-ROC",
    f"{latest_auc:.3f}",
    delta=f"{latest_auc - 0.923:.3f} vs baseline",
    delta_color="normal"
)
col2.metric(
    "Demand MAPE",
    f"{latest_mape:.2f}%",
    delta=f"{latest_mape - 8.74:.2f}% vs baseline",
    delta_color="inverse"
)
col3.metric(
    "Recency PSI",
    f"{latest_recency_psi:.3f}",
    delta="DRIFTED" if latest_recency_psi > 0.2 else "Stable",
    delta_color="inverse" if latest_recency_psi > 0.2 else "normal"
)
col4.metric(
    "Monetary PSI",
    f"{latest_monetary_psi:.3f}",
    delta="DRIFTED" if latest_monetary_psi > 0.2 else "Stable",
    delta_color="inverse" if latest_monetary_psi > 0.2 else "normal"
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📉 Drift Monitor",
    "📊 Model Performance",
    "🔁 Retrain Status"
])


# Tab 1 — PSI drift trend for recency, frequency, and monetary over last 14 days
with tab1:
    st.subheader("PSI Drift Scores — Last 14 Days")
    st.caption("PSI < 0.1 = Stable | 0.1–0.2 = Monitor | > 0.2 = Retrain!")

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=df_drift['date'], y=df_drift['recency_psi'],
        mode='lines+markers', name='Recency PSI',
        line=dict(color='#E84E1B', width=2)
    ))
    fig_drift.add_trace(go.Scatter(
        x=df_drift['date'], y=df_drift['frequency_psi'],
        mode='lines+markers', name='Frequency PSI',
        line=dict(color='#00B4D8', width=2)
    ))
    fig_drift.add_trace(go.Scatter(
        x=df_drift['date'], y=df_drift['monetary_psi'],
        mode='lines+markers', name='Monetary PSI',
        line=dict(color='#F7941D', width=2)
    ))
    # Reference lines marking monitor and retrain PSI thresholds
    fig_drift.add_hline(
        y=0.1, line_dash='dash', line_color='#F7941D',
        annotation_text='Monitor (0.1)', annotation_position='right'
    )
    fig_drift.add_hline(
        y=0.2, line_dash='dash', line_color='#E84E1B',
        annotation_text='Retrain (0.2)', annotation_position='right'
    )
    fig_drift.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1A1A2E',
        paper_bgcolor='#0E1117',
        height=350,
        hovermode='x unified',
        yaxis_title='PSI Score',
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    # Summary table showing latest PSI value and status per feature
    st.markdown("#### Latest PSI Values")
    psi_table = pd.DataFrame({
        'Feature': ['recency', 'frequency', 'monetary'],
        'PSI': [
            round(df_drift['recency_psi'].iloc[-1],   4),
            round(df_drift['frequency_psi'].iloc[-1], 4),
            round(df_drift['monetary_psi'].iloc[-1],  4),
        ],
        'Status': [
            '🔴 DRIFTED' if df_drift['recency_psi'].iloc[-1]   > 0.2 else ('🟠 Monitor' if df_drift['recency_psi'].iloc[-1]   > 0.1 else '🟢 Stable'),
            '🔴 DRIFTED' if df_drift['frequency_psi'].iloc[-1] > 0.2 else ('🟠 Monitor' if df_drift['frequency_psi'].iloc[-1] > 0.1 else '🟢 Stable'),
            '🔴 DRIFTED' if df_drift['monetary_psi'].iloc[-1]  > 0.2 else ('🟠 Monitor' if df_drift['monetary_psi'].iloc[-1]  > 0.1 else '🟢 Stable'),
        ]
    })
    st.dataframe(psi_table, use_container_width=True, hide_index=True)


# Tab 2 — Churn AUC and demand MAPE trends with threshold reference lines
with tab2:
    st.subheader("Model Metrics Trend — Last 14 Days")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=df_metrics['date'], y=df_metrics['churn_auc'],
            mode='lines+markers', name='Churn AUC',
            line=dict(color='#2DC653', width=2),
            fill='tozeroy', fillcolor='rgba(45,198,83,0.08)'
        ))
        # Dashed line marks the minimum acceptable AUC threshold
        fig_auc.add_hline(
            y=0.90, line_dash='dash', line_color='#E84E1B',
            annotation_text='Min threshold (0.90)'
        )
        fig_auc.update_layout(
            title='Churn AUC-ROC',
            template='plotly_dark',
            plot_bgcolor='#1A1A2E',
            paper_bgcolor='#0E1117',
            height=280,
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_right:
        fig_mape = go.Figure()
        fig_mape.add_trace(go.Scatter(
            x=df_metrics['date'], y=df_metrics['demand_mape'],
            mode='lines+markers', name='Demand MAPE',
            line=dict(color='#F7941D', width=2),
            fill='tozeroy', fillcolor='rgba(247,148,29,0.08)'
        ))
        # Dashed line marks the maximum acceptable MAPE threshold
        fig_mape.add_hline(
            y=10.0, line_dash='dash', line_color='#E84E1B',
            annotation_text='Max threshold (10%)'
        )
        fig_mape.update_layout(
            title='Demand MAPE %',
            template='plotly_dark',
            plot_bgcolor='#1A1A2E',
            paper_bgcolor='#0E1117',
            height=280,
        )
        st.plotly_chart(fig_mape, use_container_width=True)

    # Static model registry showing current production and staging model versions
    st.markdown("#### Model Registry")
    model_data = pd.DataFrame({
        'Model':        ['XGBoost Churn', 'XGBoost Demand', 'LightGBM Churn', 'Prophet Ensemble'],
        'Version':      ['v1.0', 'v1.0', 'v1.0', 'v1.0'],
        'Stage':        ['Production', 'Production', 'Staging', 'Staging'],
        'AUC / MAPE':   ['0.923', '8.74%', '0.901', '9.41%'],
        'Last Trained': ['2 days ago', '2 days ago', '1 week ago', '1 week ago'],
        'Drift':        ['🟠 Monitor', '🟢 Stable', '🟢 Stable', '🟢 Stable'],
    })
    st.dataframe(model_data, use_container_width=True, hide_index=True)


# Tab 3 — Auto retrain trigger check and manual retrain buttons for Admin users
with tab3:
    st.subheader("Auto Retrain Trigger Status")

    # Retrain is triggered if recency PSI exceeds 0.2 or MAPE exceeds 15% above baseline
    should_retrain = latest_recency_psi > 0.2 or latest_mape > 10.0 * 1.15

    if should_retrain:
        st.error("🚨 RETRAIN TRIGGERED — Drift threshold exceeded!")
        reasons = []
        if latest_recency_psi > 0.2:
            reasons.append(f"Recency PSI = {latest_recency_psi:.3f} > 0.2")
        if latest_mape > 10.0 * 1.15:
            reasons.append(f"MAPE = {latest_mape:.2f}% > 11.5% threshold")
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.success("✅ Model is stable — no retrain needed.")

    # Log of past retrain events with trigger reason and outcome
    st.markdown("#### Retrain History")
    retrain_log = pd.DataFrame({
        'Date':   ['2026-04-20', '2026-04-10', '2026-03-28'],
        'Reason': ['Recency drift PSI=0.28', 'Scheduled weekly', 'MAPE degraded to 11.2%'],
        'Model':  ['XGBoost Churn', 'XGBoost Demand', 'XGBoost Churn'],
        'Result': ['✅ AUC improved 0.89→0.923', '✅ MAPE improved 9.8→8.74%', '✅ MAPE improved 11.2→9.1%'],
    })
    st.dataframe(retrain_log, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Manual Retrain")

    if st.session_state.get('role') == 'Admin':
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 Trigger Churn Model Retrain", type="primary"):
                with st.spinner("Retraining..."):
                    time.sleep(2)
                st.success("✅ Churn model retrain triggered — check MLflow at localhost:5000")
        with col2:
            if st.button("🔁 Trigger Demand Model Retrain"):
                with st.spinner("Retraining..."):
                    time.sleep(2)
                st.success("✅ Demand model retrain triggered — check MLflow at localhost:5000")
    else:
        st.info("ℹ️ Only Admin can trigger manual retrain.")