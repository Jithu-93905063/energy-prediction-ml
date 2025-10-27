import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# STREAMLIT APP CONFIG
# =============================
st.set_page_config(page_title="Smart Energy Predictor", layout="wide")
st.title("🏭 Smart Energy Consumption Predictor Dashboard")

st.markdown("""
Predict **Hourly** and **Monthly** Energy Consumption for industrial or commercial buildings  
based on temperature, humidity, and other factors.  
*(Optimized version – realistic monthly variations & faster computation!)* ⚡
""")

# =============================
# SIDEBAR INPUTS
# =============================
st.sidebar.header("⚙️ Input Parameters")
sqft = st.sidebar.number_input("🏢 Building Area (sqft)", min_value=100.0, value=5000.0)
temp = st.sidebar.slider("🌡️ Air Temperature (°C)", -10.0, 45.0, 25.0)
dew = st.sidebar.slider("💧 Dew Temperature (°C)", -10.0, 35.0, 18.0)
hour = st.sidebar.slider("🕒 Hour of Day", 0, 23, 12)
dayofweek = st.sidebar.slider("📅 Day of Week (0=Mon)", 0, 6, 3)
month = st.sidebar.slider("🗓️ Month", 1, 12, 6)

# =============================
# FUNCTION TO ESTIMATE MONTHLY ENERGY (Optimized & Cached)
# =============================
@st.cache_data
def estimate_monthly_energy(_rf_hour, sqft, month, temp, dew):
    # Vectorized prediction for all hour-day combinations
    hours = np.repeat(np.arange(24), 7)
    days = np.tile(np.arange(7), 24)
    X = np.column_stack((
        np.full(24*7, sqft),
        np.full(24*7, temp),
        np.full(24*7, dew),
        hours,
        days,
        np.full(24*7, month)
    ))
    preds = _rf_hour.predict(X)
    return preds.sum()

# =============================
# PREDICTION SECTION
# =============================
if st.sidebar.button("🔮 Predict Energy Consumption"):
    # Load trained hourly model
    try:
        rf_hour = pickle.load(open('rf_hour.pkl', 'rb'))
    except:
        st.error("❌ Model file 'rf_hour.pkl' not found! Place it in the same folder as this script.")
        st.stop()

    # Predict Hourly Energy
    X_hour = np.array([[sqft, temp, dew, hour, dayofweek, month]])
    pred_hour = rf_hour.predict(X_hour)[0]

    # =============================
    # Realistic monthly temperature/dew values
    # =============================
    avg_monthly_temp = [10, 12, 15, 20, 25, 30, 32, 31, 28, 22, 16, 12]  # Jan → Dec
    avg_monthly_dew = [2, 3, 5, 8, 12, 15, 17, 16, 14, 10, 6, 3]

    # Predict Monthly Energy using realistic temps
    months = np.arange(1, 13)
    monthly_preds = [
        estimate_monthly_energy(rf_hour, sqft, m, avg_monthly_temp[m-1], avg_monthly_dew[m-1])
        for m in months
    ]

    # Predicted energy for selected month (from sidebar)
    pred_month = monthly_preds[month-1]

    st.markdown(f"### ⚡ Predicted Hourly Energy: **{pred_hour:.2f} kWh**")
    st.markdown(f"### 📆 Estimated Monthly Energy: **{pred_month:.2f} kWh**")

    # =============================
    # 1️⃣ Monthly Energy Trend (Line + Area)
    # =============================
    st.subheader("📈 Monthly Energy Consumption Trend")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, monthly_preds, color='royalblue', marker='o', linewidth=2.5, label='Predicted Energy')
    ax.fill_between(months, monthly_preds, color='lightblue', alpha=0.4)

    # Month names
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Energy (kWh)")
    ax.set_title("Monthly Energy Consumption Trend")

    # Highlight peak and low months
    max_idx = np.argmax(monthly_preds)
    min_idx = np.argmin(monthly_preds)
    ax.scatter(months[max_idx], monthly_preds[max_idx], color='red', s=100, label='Peak Month')
    ax.scatter(months[min_idx], monthly_preds[min_idx], color='green', s=100, label='Lowest Month')
    ax.legend()
    st.pyplot(fig)

    st.info(f"🌞 Peak Month: **{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][max_idx]}** → {monthly_preds[max_idx]:.2f} kWh")
    st.info(f"❄️ Lowest Month: **{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][min_idx]}** → {monthly_preds[min_idx]:.2f} kWh")

    # =============================
    # 2️⃣ Temperature Sensitivity Line Chart
    # =============================
    st.subheader("🌡️ Temperature vs Hourly Energy Usage")
    temp_range = np.arange(temp - 5, temp + 6)
    temp_preds = [rf_hour.predict(np.array([[sqft, t, dew, hour, dayofweek, month]]))[0] for t in temp_range]
    fig2, ax2 = plt.subplots()
    ax2.plot(temp_range, temp_preds, marker='o', color='orange')
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Predicted Hourly Energy (kWh)")
    ax2.set_title("Temperature Impact on Energy Consumption")
    st.pyplot(fig2)

    # =============================
    # 3️⃣ Heatmap: Hour vs Day Pattern
    # =============================
    st.subheader("🔥 Hour vs Day Energy Pattern")
    hours = np.arange(0, 24)
    days = np.arange(0, 7)
    heat_data = np.zeros((7, 24))
    for d in days:
        for h in hours:
            heat_data[d, h] = rf_hour.predict(np.array([[sqft, temp, dew, h, d, month]]))[0]
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_data, cmap="YlOrRd", xticklabels=hours, yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Day of Week")
    ax3.set_title("Energy Consumption Heatmap")
    st.pyplot(fig3)

    # =============================
    # 4️⃣ Pie Chart: Month-wise Energy Share
    # =============================
    st.subheader("🌀 Month-wise Energy Share")
    fig4, ax4 = plt.subplots()
    ax4.pie(monthly_preds, labels=[f'M{m}' for m in months], autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
    ax4.set_title("Energy Usage Share by Month")
    st.pyplot(fig4)

    st.success("✅ Dashboard generated successfully!")
