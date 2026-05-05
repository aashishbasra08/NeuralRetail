# dashboard/pages/inventory.py
# This file is a Streamlit dashboard page for inventory health and optimization.
# It provides an EOQ (Economic Order Quantity) calculator with safety stock and reorder point,
# a sensitivity chart showing how costs vary with order quantity, and a reorder alert table
# that flags SKUs currently below their safety stock level with urgency color coding.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math

st.set_page_config(page_title="Inventory Health", layout="wide")
st.title("Inventory Health — Optimization Dashboard")

# Block access if user is not authenticated
if not st.session_state.get('authenticated'):
    st.warning("Please login first!")
    st.stop()

# EOQ calculator — user inputs annual demand, costs, lead time, and service level
st.subheader("EOQ Calculator — Optimal Reorder Quantity")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Input Parameters")
    annual_demand    = st.number_input("Annual Demand (units)",          value=10000, step=500)
    order_cost       = st.number_input("Order Cost per PO (GBP)",        value=150.0, step=10.0)
    holding_cost_pct = st.slider("Holding Cost (% of unit price)",       10, 40, 25)
    unit_price       = st.number_input("Unit Price (GBP)",               value=12.50, step=0.5)
    lead_time_days   = st.number_input("Supplier Lead Time (days)",      value=7,     step=1)
    service_level    = st.selectbox("Service Level", [90, 95, 99],       index=1)

# Compute EOQ using the standard Wilson formula: sqrt(2DS / H)
holding_cost = unit_price * holding_cost_pct / 100
eoq          = math.sqrt((2 * annual_demand * order_cost) / holding_cost)

# Compute safety stock using z-score, daily demand std dev, and lead time
demand_per_day = annual_demand / 365
sigma_daily    = demand_per_day * 0.2
z_scores       = {90: 1.28, 95: 1.65, 99: 2.33}
z              = z_scores[service_level]
safety_stock   = z * sigma_daily * math.sqrt(lead_time_days)
reorder_point  = (demand_per_day * lead_time_days) + safety_stock

# Compute annual ordering and holding costs at the optimal EOQ
orders_per_year      = annual_demand / eoq
annual_ordering_cost = orders_per_year * order_cost
annual_holding_cost  = (eoq / 2) * holding_cost
total_cost           = annual_ordering_cost + annual_holding_cost

with col2:
    st.markdown("#### Results")
    st.metric("Optimal Order Quantity (EOQ)", f"{eoq:.0f} units")
    st.metric("Safety Stock",                 f"{safety_stock:.0f} units")
    st.metric("Reorder Point",                f"{reorder_point:.0f} units")
    st.metric("Orders per Year",              f"{orders_per_year:.1f}")
    st.metric("Total Annual Cost",            f"GBP {total_cost:,.0f}")

# Sensitivity chart — shows how ordering, holding, and total cost change across order quantities
st.subheader("EOQ Sensitivity — How Cost Changes with Order Quantity")
quantities        = np.arange(50, eoq * 3, 20)
ordering_costs    = (annual_demand / quantities) * order_cost
holding_costs_arr = (quantities / 2) * holding_cost
total_costs_arr   = ordering_costs + holding_costs_arr

fig = px.line(
    x=quantities,
    y=[ordering_costs, holding_costs_arr, total_costs_arr],
    labels={'x': 'Order Quantity', 'value': 'Annual Cost (GBP)'},
    title='Total Cost vs Order Quantity',
    template='plotly_dark',
    color_discrete_sequence=['#E84E1B', '#00B4D8', '#2DC653']
)
# Vertical dashed line marks the optimal EOQ point on the chart
fig.add_vline(
    x=eoq, line_dash='dash', line_color='#FBBA13',
    annotation_text=f'EOQ={eoq:.0f}', annotation_position='top'
)
fig.update_layout(plot_bgcolor='#1A1A2E', paper_bgcolor='#0E1117')
st.plotly_chart(fig, use_container_width=True)
st.divider()

# Reorder alert table — lists SKUs currently below safety stock with urgency classification
st.subheader("Reorder Alerts — SKUs Below Safety Stock")


@st.cache_data
def generate_inventory_alerts():
    """
    Generate synthetic inventory data for 50 SKUs.
    Flags SKUs below safety stock and classifies urgency by days until stockout.
    """
    np.random.seed(42)
    skus = [f'SKU-{i:04d}' for i in range(1, 51)]
    data = {
        'sku':           skus,
        'description':   [f'Product {i}' for i in range(1, 51)],
        'current_stock': np.random.randint(0, 500, 50),
        'safety_stock':  np.random.randint(100, 300, 50),
        'reorder_point': np.random.randint(200, 400, 50),
        'daily_demand':  np.random.randint(10, 80, 50),
        'lead_time':     np.random.randint(3, 14, 50),
    }
    df = pd.DataFrame(data)

    # Estimate how many days of stock remain at current daily demand rate
    df['days_until_stockout'] = (df['current_stock'] / df['daily_demand']).round(1)

    # Classify urgency based on days until stockout thresholds
    df['urgency'] = pd.cut(
        df['days_until_stockout'],
        bins=[0, 3, 7, 14, float('inf')],
        labels=['CRITICAL', 'HIGH', 'MEDIUM', 'OK']
    )
    df['below_safety'] = df['current_stock'] < df['safety_stock']

    # Return only SKUs that have fallen below their safety stock threshold
    return df[df['below_safety']].sort_values('days_until_stockout')


alerts_df = generate_inventory_alerts()
st.warning(f"{len(alerts_df)} SKUs are below safety stock level!")


def color_urgency(val):
    """Apply background color to urgency column based on risk level."""
    colors_map = {
        'CRITICAL': 'background-color:#FF4444; color:white',
        'HIGH':     'background-color:#FF8800; color:white',
        'MEDIUM':   'background-color:#FFBB00; color:black',
        'OK':       'background-color:#44BB44; color:white',
    }
    return colors_map.get(val, '')


styled_df = alerts_df[[
    'sku', 'description', 'current_stock',
    'safety_stock', 'days_until_stockout', 'urgency'
]].style.applymap(color_urgency, subset=['urgency'])

st.dataframe(styled_df, use_container_width=True, height=350)

st.download_button(
    "Download Reorder List",
    alerts_df.to_csv(index=False).encode(),
    file_name='reorder_alerts.csv',
    mime='text/csv'
)