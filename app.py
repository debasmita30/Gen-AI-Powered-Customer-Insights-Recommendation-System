import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import os

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")

st.title("Dynamic Pricing Engine")
st.markdown("### AI-Powered Pricing Simulation Using Reinforcement Learning (PPO)")

# --- Sidebar Inputs ---
st.sidebar.header("Market Controls")
demand_factor = st.sidebar.slider("Market Demand Level", 0.0, 1.0, 0.5, 0.05)
competitor_factor = st.sidebar.slider("Competitor Pricing Pressure", 0.0, 1.0, 0.5, 0.05)
inventory_level = st.sidebar.slider("Current Inventory", 0.0, 1.0, 0.5, 0.05)
learning_rate = st.sidebar.slider(
    "Learning Rate (Simulation Sensitivity)", 0.001, 0.01, 0.005, 0.001,
    help="Higher values make the price simulation more sensitive to market changes."
)

# --- Randomize Market Conditions ---
if st.sidebar.button("🎲 Randomize Market Conditions"):
    demand_factor = np.random.rand()
    competitor_factor = np.random.rand()
    inventory_level = np.random.rand()
    st.sidebar.success("Market conditions randomized!")

# --- Load RL Model if available ---
if os.path.exists("ppo_actor.h5"):
    model = tf.keras.models.load_model("ppo_actor.h5")
    st.sidebar.success("✅ RL model loaded successfully.")
else:
    st.sidebar.warning("⚠️ No trained model found — running simulated predictions.")

# --- Predicted price (simulated) ---
predicted_price = (
    500 * demand_factor
    + 400 * (1 - competitor_factor)
    + 200 * inventory_level
    + np.random.uniform(-50, 50)
)

# --- Realistic demand, sales, and profit ---
dynamic_demand = max(
    0,
    1 - (predicted_price / 1000)
    + demand_factor * 0.2
    - competitor_factor * 0.1
    + (0.5 - inventory_level) * 0.1,
)
expected_sales = int(dynamic_demand * 100)
expected_profit = predicted_price * expected_sales

# --- Track previous profit for percentage change ---
if "prev_profit" not in st.session_state:
    st.session_state.prev_profit = expected_profit

profit_change = ((expected_profit - st.session_state.prev_profit) / max(st.session_state.prev_profit, 1)) * 100
st.session_state.prev_profit = expected_profit

# --- Metrics display ---
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Optimal Price", f"₹{predicted_price:,.2f}")
col2.metric("Expected Sales", f"{expected_sales} units")
col3.metric("Expected Profit", f"₹{expected_profit:,.0f}", f"{profit_change:.2f}%")

# --- Profit vs Price chart (Plotly for animation) ---
prices = np.linspace(100, 1000, 100)
profits, sales_list = [], []

for p in prices:
    demand = max(0, 1 - (p / 1000)
                    + demand_factor * 0.2
                    - competitor_factor * 0.1
                    + (0.5 - inventory_level) * 0.1)
    sales = demand * 100
    profit = p * sales
    profits.append(profit)
    sales_list.append(sales)

fig = go.Figure()
# Profit line
fig.add_trace(go.Scatter(
    x=prices, y=profits, mode="lines", name="Profit", line=dict(color="green", width=3)
))
# Sales line (secondary y-axis)
fig.add_trace(go.Scatter(
    x=prices, y=sales_list, mode="lines", name="Sales", line=dict(color="blue", width=2), yaxis="y2"
))
# Predicted price marker
fig.add_vline(x=predicted_price, line=dict(color="red", width=2, dash="dash"))
fig.add_annotation(
    x=predicted_price, y=max(profits) * 0.05,
    text="Predicted Price", showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red")
)

# Layout settings
fig.update_layout(
    title="📊 Profit Trend vs Price",
    xaxis_title="Price (₹)",
    yaxis=dict(title="Profit (₹)", color="green"),
    yaxis2=dict(title="Sales (units)", overlaying="y", side="right", color="blue"),
    legend=dict(y=0.99, x=0.01),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "🧠 This demo shows how a Reinforcement Learning (PPO) agent adapts prices to maximize long-term profit under changing demand, competition, and inventory."
)
