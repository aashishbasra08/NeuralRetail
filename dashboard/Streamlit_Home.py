# dashboard/streamlit_home.py
# This is the main entry point for the NeuralRetail Streamlit dashboard.
# It handles user authentication via session state, displays top-level KPI cards,
# renders a monthly revenue trend chart from processed sales data, and provides
# sidebar navigation with a logout button. All other pages branch from this file.

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
sys.path.append('..')

st.set_page_config(
    page_title="NeuralRetail — AI Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for formal corporate styling — clean whites, accent borders, metric colors
st.markdown("""
<style>
.main { background-color: #F5F5F5; }

.stMetric {
    background: #FFFFFF;
    border: 0.5px solid #E0E0E0;
    border-left: 3px solid #B83A1A;
    border-radius: 8px;
    padding: 12px 16px !important;
}
.stMetric label { color: #6B6B6B !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.4px; }
[data-testid="stMetricValue"] { color: #1A1A1A !important; font-size: 24px !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

h1 { color: #1A1A1A !important; font-size: 22px !important; font-weight: 600 !important; }
h2, h3 { color: #1A1A1A !important; }

[data-testid="stSidebar"] { background-color: #1C1C1C !important; }
[data-testid="stSidebar"] * { color: #EFEFEF !important; }
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-size: 13px !important; letter-spacing: 0.6px; }

div[data-testid="stButton"] button {
    background-color: #B83A1A;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    padding: 0.45rem 1.5rem;
}
div[data-testid="stButton"] button:hover { background-color: #993217; }
</style>
""", unsafe_allow_html=True)

# Initialise session state keys for authentication on first load
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role          = None
    st.session_state.username      = None


def login():
    """Render login form and validate credentials against the hardcoded user registry."""

    # Hardcoded user registry with plaintext passwords and role assignments
    users = {
        'aashishbasra8@gmail.com':  {'password': 'admin123',   'role': 'Admin'},
        'analyst@neuralretail.com': {'password': 'analyst123', 'role': 'Analyst'},
        'viewer@neuralretail.com':  {'password': 'view123',    'role': 'Viewer'},
    }

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        # Brand header on login screen
        st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <div style="display:inline-flex; align-items:center; gap:10px;">
                <div style="background:#B83A1A; border-radius:8px; width:36px; height:36px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="color:white; font-size:18px; font-weight:700;">N</span>
                </div>
                <div style="text-align:left;">
                    <div style="font-size:17px; font-weight:600; color:#1A1A1A; letter-spacing:-0.3px;">NeuralRetail</div>
                    <div style="font-size:10px; color:#888; letter-spacing:0.8px; text-transform:uppercase;">AI Sales Intelligence</div>
                </div>
            </div>
            <p style="margin-top:1.5rem; font-size:18px; font-weight:500; color:#1A1A1A;">Sign in to your account</p>
            <p style="font-size:13px; color:#777; margin-top:0.25rem;">Enter your credentials to access the dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Email address", placeholder="you@company.com", label_visibility="visible")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign In", type="primary", use_container_width=True):
            if username in users and users[username]['password'] == password:
                st.session_state.authenticated = True
                st.session_state.role          = users[username]['role']
                st.session_state.username      = username
                st.rerun()
            else:
                # Subtle inline message — no red st.error flash
                st.warning("Wrong password or email. Please check your credentials.")

        st.markdown("""
        <div style="text-align:center; margin-top:2rem; padding-top:1rem;
                    border-top:0.5px solid #E0E0E0; font-size:11.5px; color:#AAAAAA;">
            NeuralRetail &copy; 2026 &nbsp;|&nbsp; Confidential
        </div>
        """, unsafe_allow_html=True)


# Show login screen and halt execution if user is not authenticated
if not st.session_state.authenticated:
    login()
    st.stop()

# Dashboard header with user info
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("NeuralRetail — AI Sales Intelligence")
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; padding-top:0.6rem;">
        <span style="font-size:12px; color:#777;">Logged in as </span>
        <span style="font-size:12px; color:#1A1A1A; font-weight:500;">{st.session_state.username}</span>
        <span style="font-size:11px; background:#FAECE7; color:#993C1D; border-radius:12px;
                     padding:2px 10px; margin-left:8px;">{st.session_state.role}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Top-level KPI cards summarising key model and business metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Forecast MAPE",      "8.74%", delta="-10.56% vs baseline")
with col2:
    st.metric("Churn AUC-ROC",      "0.923", delta="+0.141 vs baseline")
with col3:
    st.metric("Active Customers",   "4,372", delta="+234 this week")
with col4:
    st.metric("Stockout Risk SKUs", "23",    delta="-15 vs last week", delta_color="inverse")

st.divider()


@st.cache_data
def load_sales_data():
    """
    Load cleaned UCI sales data from parquet, parse invoice dates,
    and aggregate total revenue by calendar month.
    """
    BASE_DIR  = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "processed" / "uci_clean.parquet"

    df = pd.read_parquet(data_path)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])

    # Group by month and sum revenue
    monthly = df.groupby(
        df['invoice_date'].dt.to_period('M')
    )['total_amount'].sum().reset_index()

    # Convert period to string for Plotly compatibility
    monthly['invoice_date'] = monthly['invoice_date'].astype(str)
    return monthly


df_sales = load_sales_data()

# Monthly revenue line chart with formal light theme
fig = px.line(
    df_sales,
    x='invoice_date',
    y='total_amount',
    title='Monthly Revenue Trend',
    template='plotly_white',
    color_discrete_sequence=['#B83A1A']
)
fig.update_layout(
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#F5F5F5',
    xaxis_title='Month',
    yaxis_title='Revenue (GBP)',
    font=dict(color='#1A1A1A', size=12),
    title_font=dict(size=14, color='#1A1A1A'),
    xaxis=dict(showgrid=True, gridcolor='#EEEEEE'),
    yaxis=dict(showgrid=True, gridcolor='#EEEEEE'),
)
st.plotly_chart(fig, use_container_width=True)

# Sidebar with brand logo and logout
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
            <div style="background:#B83A1A; border-radius:6px; width:28px; height:28px;
                        display:flex; align-items:center; justify-content:center;">
                <span style="color:white; font-size:14px; font-weight:700;">N</span>
            </div>
            <span style="font-size:14px; font-weight:600; color:#FFFFFF;">NeuralRetail</span>
        </div>
        <div style="font-size:10px; color:#888; letter-spacing:0.8px; text-transform:uppercase; padding-left:2px;">
            AI Sales Intelligence
        </div>
    </div>
    <hr style="border:none; border-top:0.5px solid #333; margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()