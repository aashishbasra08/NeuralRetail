# dashboard/pages/customers.py
# This file is a Streamlit dashboard page for churn intelligence.
# It displays a churn risk heatmap by customer segment, a detailed Customer 360 view
# with SHAP-based explanations, and a filterable CRM export tool for high-risk customers.
# Requires authentication via session state before rendering any content.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="Customer Hub", layout="wide")

# Block access if user is not logged in
if not st.session_state.get('authenticated'):
    st.warning("Please login first!")
    st.stop()

st.title("👥 Customer Hub — Churn Intelligence")
st.caption(f"Logged in as: {st.session_state.get('username')} | Role: {st.session_state.get('role')}")
st.divider()


@st.cache_data
def load_rfm_data():
    """
    Load RFM feature data from parquet file.
    Falls back to synthetically generated demo data if the file is not found.
    """
    try:
        rfm = pd.read_parquet('../data/features/rfm_with_segments.parquet')
        return rfm
    except Exception:
        # Generate realistic demo data with five customer segments
        np.random.seed(42)
        n            = 500
        customer_ids = [f"C{i:04d}" for i in range(1, n + 1)]
        segments     = np.random.choice(
            ['Champions', 'Loyal', 'At Risk', 'Lost', 'New Customer'],
            n, p=[0.15, 0.25, 0.30, 0.20, 0.10]
        )
        recency   = np.random.randint(1, 365, n)
        frequency = np.random.randint(1, 30, n)
        monetary  = np.round(np.random.exponential(250, n), 2)

        rfm = pd.DataFrame({
            'customer_id':     customer_ids,
            'segment_name':    segments,
            'recency':         recency,
            'frequency':       frequency,
            'monetary':        monetary,
            'monetary_log':    np.log1p(monetary),
            'r_score':         pd.qcut(recency, 5, labels=[5,4,3,2,1]).astype(int),
            'f_score':         pd.qcut(frequency, 5, labels=[1,2,3,4,5], duplicates='drop').astype(int),
            'm_score':         pd.qcut(monetary, 5, labels=[1,2,3,4,5], duplicates='drop').astype(int),
            'avg_order_value': np.round(monetary / (frequency + 1), 2)
        })
        return rfm


@st.cache_resource
def load_churn_model():
    """Load the trained XGBoost churn model from disk. Returns None if not found."""
    try:
        model = joblib.load(r'models/xgb_churn.pkl')
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


rfm   = load_rfm_data()
model = load_churn_model()

feature_cols = ['frequency', 'monetary_log', 'f_score', 'm_score', 'avg_order_value']

# Compute churn probability using model if available, otherwise use beta distribution as demo
X = rfm[feature_cols].astype(float)
if model:
    rfm['churn_prob'] = model.predict_proba(X)[:, 1]
else:
    np.random.seed(42)
    rfm['churn_prob'] = np.random.beta(3, 3, len(rfm))

# Assign risk decile from 1 (lowest) to 10 (highest) based on churn probability
rfm['risk_decile'] = pd.qcut(
    rfm['churn_prob'], 10, labels=range(1, 11)
).astype(int)

if model:
    st.success("✅ XGBoost churn model loaded from disk — live predictions active.")
else:
    st.info("ℹ️ Model not found — running in demo mode with synthetic churn scores.")

st.markdown("<br>", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs([
    "🔥 Churn Heatmap",
    "🔍 Customer 360",
    "📤 CRM Export",
])


# Tab 1 — Heatmap showing customer count per segment and risk decile bucket
with tab1:
    st.subheader("Churn Risk Heatmap — Segment × Risk Decile")
    st.caption("Darker red = more customers at high churn risk in that segment-decile bucket.")

    heatmap_data = (
        rfm.groupby(['segment_name', 'risk_decile'])
        .size()
        .unstack(fill_value=0)
    )

    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale=['#0D0221', '#1B1464', '#5C0099', '#A00000', '#E84E1B'],
        title='Customer Count by Segment & Risk Decile',
        labels={'x': 'Risk Decile (10 = Highest Risk)', 'y': 'Segment', 'color': 'Customers'},
        template='plotly_dark',
        text_auto=True,
    )
    fig_heat.update_layout(
        plot_bgcolor='#1A1A2E',
        paper_bgcolor='#0E1117',
        font=dict(family='monospace', size=11),
        height=380,
    )
    fig_heat.update_coloraxes(showscale=True)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    high_risk = rfm[rfm['churn_prob'] > 0.7]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk Customers (>70%)",  f"{len(high_risk):,}", delta=f"{len(high_risk)/len(rfm)*100:.1f}% of base")
    col2.metric("Avg Churn Probability",        f"{rfm['churn_prob'].mean()*100:.1f}%")
    col3.metric("Potential Revenue at Risk",    f"£{high_risk['monetary'].sum():,.0f}")
    col4.metric("Critical Risk (>90%)",         f"{len(rfm[rfm['churn_prob'] > 0.9]):,}", delta="Needs immediate action", delta_color="inverse")

    st.markdown("#### Avg Churn Probability by Segment")
    seg_churn = (
        rfm.groupby('segment_name')['churn_prob']
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_seg = px.bar(
        seg_churn,
        x='churn_prob', y='segment_name',
        orientation='h',
        color='churn_prob',
        color_continuous_scale=['#0D0221', '#1B1464', '#5C0099', '#A00000', '#E84E1B'],
        labels={'churn_prob': 'Avg Churn Prob', 'segment_name': 'Segment'},
        template='plotly_dark',
        text=seg_churn['churn_prob'].apply(lambda x: f"{x*100:.1f}%"),
    )
    fig_seg.update_traces(textposition='outside')
    fig_seg.update_layout(
        plot_bgcolor='#1A1A2E',
        paper_bgcolor='#0E1117',
        height=280,
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_seg, use_container_width=True)


# Tab 2 — Customer 360 view with RFM metrics, SHAP explanation, and retention recommendations
with tab2:
    st.subheader("Customer 360 View")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        # Default dropdown shows top 100 customers sorted by highest churn risk
        customer_id = st.selectbox(
            "Select Customer (sorted by highest churn risk)",
            rfm.sort_values('churn_prob', ascending=False)['customer_id'].head(100).tolist()
        )
    with col_f2:
        show_all = st.checkbox("Show all customers (not just top 100)")
        if show_all:
            customer_id = st.selectbox(
                "Select from all",
                rfm.sort_values('churn_prob', ascending=False)['customer_id'].tolist(),
                key='all_customers'
            )

    customer  = rfm[rfm['customer_id'] == customer_id].iloc[0]
    churn_pct = customer['churn_prob'] * 100

    # Assign risk label and badge color based on churn probability threshold
    if churn_pct >= 80:
        risk_label, risk_color = "CRITICAL", "#FF4444"
    elif churn_pct >= 60:
        risk_label, risk_color = "HIGH", "#FF8800"
    elif churn_pct >= 40:
        risk_label, risk_color = "MEDIUM", "#FFBB00"
    else:
        risk_label, risk_color = "LOW", "#2DC653"

    # Customer profile card with segment and risk badge
    st.markdown(f"""
    <div style="background:#111318; border:1px solid rgba(232,78,27,0.2);
                border-radius:12px; padding:20px 24px; margin:12px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:20px; font-weight:700; color:#F0F2F5;">
                    Customer #{customer_id}
                </div>
                <div style="font-size:13px; color:#7A8090; margin-top:4px;">
                    Segment: <b style="color:#F7941D;">{customer['segment_name']}</b>
                </div>
            </div>
            <div style="background:{risk_color}22; border:1px solid {risk_color};
                        border-radius:8px; padding:8px 20px; text-align:center;">
                <div style="font-size:24px; font-weight:700; color:{risk_color};">
                    {churn_pct:.1f}%
                </div>
                <div style="font-size:11px; color:{risk_color}; letter-spacing:2px;">
                    {risk_label} RISK
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recency",        f"{customer['recency']:.0f} days",  "Since last purchase")
    col2.metric("Frequency",      f"{customer['frequency']:.0f} orders")
    col3.metric("Lifetime Value", f"£{customer['monetary']:,.0f}")
    col4.metric("RFM Score",
                f"{customer['r_score']}/{customer['f_score']}/{customer['m_score']}",
                "R/F/M out of 5")

    st.divider()

    if model:
        # Compute SHAP values for the selected customer using the loaded XGBoost model
        cust_features = rfm[rfm['customer_id'] == customer_id][feature_cols]
        explainer     = shap.TreeExplainer(model)
        shap_vals     = explainer.shap_values(cust_features)

        sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        if hasattr(sv, 'shape') and len(sv.shape) > 1:
            sv = sv[:, 1]

        shap_df = pd.DataFrame({
            'Feature':       feature_cols,
            'SHAP Value':    sv,
            'Feature Value': cust_features.values[0],
        }).sort_values('SHAP Value', ascending=True)

        st.subheader("🔍 Why is this customer at risk? (SHAP Explanation)")
        fig_shap = px.bar(
            shap_df,
            x='SHAP Value', y='Feature',
            orientation='h',
            color='SHAP Value',
            color_continuous_scale=['#2DC653', '#FFFFFF', '#E84E1B'],
            color_continuous_midpoint=0,
            title='Feature Contribution to Churn Risk',
            template='plotly_dark',
            hover_data={'Feature Value': True},
        )
        fig_shap.update_layout(
            plot_bgcolor='#1A1A2E',
            paper_bgcolor='#0E1117',
            height=320,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    else:
        # Show approximate SHAP-style chart when model is not loaded (demo mode)
        st.subheader("🔍 Why is this customer at risk? (Demo SHAP)")
        demo_shap = pd.DataFrame({
            'Feature':    feature_cols,
            'SHAP Value': [
                -customer['frequency'] / 30,
                -customer['monetary_log'] / 10,
                -(customer['f_score'] - 3) * 0.06,
                -(customer['m_score'] - 3) * 0.04,
                -customer['avg_order_value'] / 100,
            ],
        }).sort_values('SHAP Value', ascending=True)

        fig_shap_demo = px.bar(
            demo_shap,
            x='SHAP Value', y='Feature',
            orientation='h',
            color='SHAP Value',
            color_continuous_scale=['#2DC653', '#FFFFFF', '#E84E1B'],
            color_continuous_midpoint=0,
            title='Feature Contribution to Churn Risk (Demo)',
            template='plotly_dark',
        )
        fig_shap_demo.update_layout(
            plot_bgcolor='#1A1A2E',
            paper_bgcolor='#0E1117',
            height=300,
        )
        st.plotly_chart(fig_shap_demo, use_container_width=True)
        st.caption("⚠️ Demo mode — load xgb_churn.pkl for real SHAP values.")

    st.divider()
    st.subheader("💡 Retention Recommendations")

    if customer['churn_prob'] > 0.7:
        st.error("🚨 HIGH RISK: Immediate Action Required!")
        recs = []
        if customer['recency'] > 90:
            recs.append("📧 Send win-back email with **15% discount** code — last purchase was too long ago.")
        if customer['frequency'] < 3:
            recs.append("🎁 Offer **loyalty points** for next purchase to increase engagement frequency.")
        if customer['monetary'] > 500:
            recs.append("👑 Assign a **VIP customer success manager** — high-value customer at risk.")
        recs.append("📱 Send personalised product recommendations based on past purchases.")
        for r in recs:
            st.markdown(f"- {r}")

    elif customer['churn_prob'] > 0.4:
        st.warning("⚠️ MEDIUM RISK: Proactive Engagement Recommended")
        st.markdown("- 📩 Include in next **email newsletter** with personalised content.")
        st.markdown("- ⭐ Enrol in **loyalty rewards program** to increase stickiness.")

    else:
        st.success("✅ LOW RISK: Customer is healthy — maintain regular engagement.")
        st.markdown("- 🔔 Continue standard **monthly touchpoints**.")
        st.markdown("- 📊 Monitor RFM scores quarterly.")


# Tab 3 — Export high-risk customers to CSV for CRM and marketing campaign use
with tab3:
    st.subheader("📤 CRM Export — High Risk Customers")
    st.caption("Filter by risk threshold and export to CSV for marketing campaigns.")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_threshold = st.slider(
            "Churn Risk Threshold",
            min_value=0.5, max_value=0.95,
            value=0.7, step=0.05,
            format="%.0f%%",
            help="Only export customers above this churn probability"
        )
    with col2:
        segment_filter = st.multiselect(
            "Filter by Segment",
            options=rfm['segment_name'].unique().tolist(),
            default=rfm['segment_name'].unique().tolist(),
        )
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["churn_prob", "monetary", "recency"],
            format_func=lambda x: {
                "churn_prob": "Churn Risk (High→Low)",
                "monetary":   "Lifetime Value (High→Low)",
                "recency":    "Recency (Highest→Lowest)",
            }[x]
        )

    # Apply filters and sort the export dataframe
    export_df = (
        rfm[
            (rfm['churn_prob'] >= risk_threshold) &
            (rfm['segment_name'].isin(segment_filter))
        ][[
            'customer_id', 'segment_name', 'churn_prob',
            'recency', 'frequency', 'monetary'
        ]]
        .sort_values(sort_by, ascending=False)
        .copy()
    )
    export_df['churn_prob'] = export_df['churn_prob'].round(4)
    export_df['monetary']   = export_df['monetary'].round(2)

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Customers to Export",  f"{len(export_df):,}")
    col_s2.metric("Revenue at Risk",       f"£{export_df['monetary'].sum():,.0f}")
    col_s3.metric("Avg Churn Probability", f"{export_df['churn_prob'].mean()*100:.1f}%")

    # Color-code rows by urgency level based on churn probability
    def urgency_color(val):
        if val >= 0.8:
            return 'background-color:#FF444433; color:#FF6666'
        elif val >= 0.6:
            return 'background-color:#FF880033; color:#FFAA44'
        return 'background-color:#FFBB0022; color:#FFCC55'

    st.markdown(f"**Showing top 20 of {len(export_df):,} customers**")
    st.dataframe(
        export_df.head(20).style.map(urgency_color, subset=['churn_prob']),
        use_container_width=True,
        height=360,
    )

    st.download_button(
        label=f"⬇️  Export {len(export_df):,} High-Risk Customers to CSV",
        data=export_df.to_csv(index=False).encode('utf-8'),
        file_name=f'high_risk_customers_threshold_{int(risk_threshold*100)}pct.csv',
        mime='text/csv',
        type='primary',
    )

    st.caption("💡 Tip: Import this CSV into Salesforce / HubSpot and trigger an automated email campaign.")
