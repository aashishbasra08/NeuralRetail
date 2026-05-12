# dashboard/streamlit_home.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
sys.path.append('..')

st.set_page_config(
    page_title="NeuralRetail — AI Sales Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {background-color: #0E1117 !important;}
.main .block-container {background-color: #0E1117 !important;}
section[data-testid="stMain"] {background-color: #0E1117 !important;}

.stMetric {
    background: #1E1E2E;
    border: 0.5px solid #333;
    border-left: 3px solid #B83A1A;
    border-radius: 8px;
    padding: 12px 16px !important;
}
.stMetric label { color: #AAAAAA !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.4px; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

h1 { color: #FFFFFF !important; font-size: 22px !important; font-weight: 600 !important; }
h2, h3 { color: #FFFFFF !important; }
p, label, span { color: #EEEEEE !important; }

[data-testid="stSidebar"] { background-color: #1C1C1C !important; }
[data-testid="stSidebar"] * { color: #EFEFEF !important; }

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

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role          = None
    st.session_state.username      = None


def login():
    users = {
        'aashishbasra8@gmail.com':  {'password': 'admin123',   'role': 'Admin'},
        'analyst@neuralretail.com': {'password': 'analyst123', 'role': 'Analyst'},
        'viewer@neuralretail.com':  {'password': 'view123',    'role': 'Viewer'},
    }

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <div style="display:inline-flex; align-items:center; gap:10px;">
                <div style="background:#B83A1A; border-radius:8px; width:36px; height:36px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="color:white; font-size:18px; font-weight:700;">N</span>
                </div>
                <div style="text-align:left;">
                    <div style="font-size:17px; font-weight:600; color:#FFFFFF; letter-spacing:-0.3px;">NeuralRetail</div>
                    <div style="font-size:10px; color:#888; letter-spacing:0.8px; text-transform:uppercase;">AI Sales Intelligence</div>
                </div>
            </div>
            <p style="margin-top:1.5rem; font-size:18px; font-weight:500; color:#FFFFFF;">Sign in to your account</p>
            <p style="font-size:13px; color:#AAAAAA; margin-top:0.25rem;">Enter your credentials to access the dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Email address", placeholder="you@company.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign In", type="primary", use_container_width=True):
            if username in users and users[username]['password'] == password:
                st.session_state.authenticated = True
                st.session_state.role          = users[username]['role']
                st.session_state.username      = username
                st.rerun()
            else:
                st.warning("Wrong password or email. Please check your credentials.")

        st.markdown("""
        <div style="text-align:center; margin-top:2rem; padding-top:1rem;
                    border-top:0.5px solid #333; font-size:11.5px; color:#666;">
            NeuralRetail &copy; 2026 &nbsp;|&nbsp; Confidential
        </div>
        """, unsafe_allow_html=True)


if not st.session_state.authenticated:
    login()
    st.stop()

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("NeuralRetail — AI Sales Intelligence")
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; padding-top:0.6rem;">
        <span style="font-size:12px; color:#AAAAAA;">Logged in as </span>
        <span style="font-size:12px; color:#FFFFFF; font-weight:500;">{st.session_state.username}</span>
        <span style="font-size:11px; background:#3A1A0E; color:#FF6B3D; border-radius:12px;
                     padding:2px 10px; margin-left:8px;">{st.session_state.role}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

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
    BASE_DIR  = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "processed" / "uci_clean.parquet"
    df = pd.read_parquet(data_path)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    monthly = df.groupby(
        df['invoice_date'].dt.to_period('M')
    )['total_amount'].sum().reset_index()
    monthly['invoice_date'] = monthly['invoice_date'].astype(str)
    return monthly


df_sales = load_sales_data()

fig = px.line(
    df_sales,
    x='invoice_date',
    y='total_amount',
    title='Monthly Revenue Trend',
    template='plotly_dark',
    color_discrete_sequence=['#E84E1B']
)
fig.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    xaxis_title='Month',
    yaxis_title='Revenue (GBP)',
    font=dict(color='#FFFFFF', size=12),
    title_font=dict(size=14, color='#FFFFFF'),
    xaxis=dict(showgrid=True, gridcolor='#2A2A3A'),
    yaxis=dict(showgrid=True, gridcolor='#2A2A3A'),
    margin=dict(l=40, r=40, t=50, b=40),
)
st.plotly_chart(fig, use_container_width=True)

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