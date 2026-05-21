import threading
import time
import psutil

from datetime import datetime

from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

from streamlit_autorefresh import st_autorefresh

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import sqlite3

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Real-Time System Monitoring",
    layout="wide"
)

# -----------------------------------
# BACKGROUND TELEMETRY COLLECTION
# -----------------------------------

def collect_metrics():

    conn = sqlite3.connect(
        "system_monitor.db",
        check_same_thread=False
    )

    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_stats (
        timestamp TEXT,
        cpu REAL,
        memory REAL,
        disk REAL
    )
    """)

    conn.commit()

    while True:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cpu = psutil.cpu_percent()

        memory = psutil.virtual_memory().percent

        disk = psutil.disk_usage('/').percent

        cursor.execute("""
        INSERT INTO system_stats
        (timestamp, cpu, memory, disk)
        VALUES (?, ?, ?, ?)
        """, (timestamp, cpu, memory, disk))

        conn.commit()

        time.sleep(3)

# -----------------------------------
# START BACKGROUND THREAD
# -----------------------------------

if 'threads_started' not in st.session_state:

    telemetry_thread = threading.Thread(
        target=collect_metrics,
        daemon=True
    )

    telemetry_thread.start()

    st.session_state['threads_started'] = True

# -----------------------------------
# TITLE
# -----------------------------------

st.title(
    "Real-Time System Monitoring & ML-Based Forecasting Dashboard"
)

# -----------------------------------
# AUTO REFRESH
# -----------------------------------

st_autorefresh(
    interval=5000,
    key="datarefresh"
)

# -----------------------------------
# DATABASE CONNECTION
# -----------------------------------

conn = sqlite3.connect(
    "system_monitor.db"
)

query = "SELECT * FROM system_stats"

df = pd.read_sql_query(query, conn)

# -----------------------------------
# DATASETS
# -----------------------------------

# Smaller optimized ML dataset
train_df = df.tail(1500).copy()

# Dashboard display dataset
display_df = df.tail(100).copy()

# -----------------------------------
# HANDLE EMPTY DATA
# -----------------------------------

if train_df.empty:

    st.error(
        "No telemetry data available yet."
    )

    st.stop()

# -----------------------------------
# LATEST METRICS
# -----------------------------------

st.subheader("Latest System Metrics")

st.write(display_df.tail())

# -----------------------------------
# TELEMETRY FRESHNESS CHECK
# -----------------------------------

latest_timestamp = train_df['timestamp'].iloc[-1]

latest_time = datetime.strptime(
    latest_timestamp,
    "%Y-%m-%d %H:%M:%S"
)

current_time = datetime.now()

time_diff = (
    current_time - latest_time
).total_seconds()

if time_diff > 10:

    st.error(
        "⚠ No recent telemetry detected. "
        "Predictions may be unreliable."
    )

else:

    st.success(
        "✅ Telemetry stream active."
    )

# -----------------------------------
# LIVE MONITORING GRAPHS
# -----------------------------------

st.subheader("CPU Usage Over Time")

st.line_chart(display_df["cpu"])

st.subheader("Memory Usage Over Time")

st.line_chart(display_df["memory"])

st.subheader("Disk Usage Over Time")

st.line_chart(display_df["disk"])

# -----------------------------------
# ANOMALY DETECTION
# -----------------------------------

st.subheader("Anomaly Detection")

features = train_df[['cpu', 'memory']]

anomaly_model = IsolationForest(
    n_estimators=40,
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)

train_df['anomaly'] = anomaly_model.fit_predict(
    features
)

anomalies = train_df[
    train_df['anomaly'] == -1
]

st.write("Detected System Anomalies")

st.write(
    anomalies.tail(10)
)

# -----------------------------------
# PREDICTIVE ANALYTICS
# -----------------------------------

st.subheader("Predictive Analytics")

# Lag Features
train_df['cpu_lag1'] = (
    train_df['cpu'].shift(1)
)

train_df['cpu_lag2'] = (
    train_df['cpu'].shift(2)
)

train_df['cpu_lag3'] = (
    train_df['cpu'].shift(3)
)

# Rolling Mean Feature
train_df['cpu_rolling_mean'] = (
    train_df['cpu']
    .rolling(window=5)
    .mean()
)

# Drop missing rows
train_df = train_df.dropna()

# -----------------------------------
# ENSURE ENOUGH DATA
# -----------------------------------

if len(train_df) < 20:

    st.warning(
        "Not enough telemetry history "
        "for forecasting yet."
    )

    st.stop()

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = train_df[[
    'cpu_lag1',
    'cpu_lag2',
    'cpu_lag3',
    'cpu_rolling_mean'
]]

y = train_df['cpu']

# -----------------------------------
# TRAIN XGBOOST MODEL
# -----------------------------------

prediction_model = XGBRegressor(
    n_estimators=60,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

prediction_model.fit(X, y)

# -----------------------------------
# NEXT CPU PREDICTION
# -----------------------------------

latest_features = [[
    train_df['cpu'].iloc[-1],
    train_df['cpu'].iloc[-2],
    train_df['cpu'].iloc[-3],
    train_df['cpu'].tail(5).mean()
]]

predicted_cpu = prediction_model.predict(
    latest_features
)[0]

st.write(
    f"Predicted CPU Usage "
    f"(Next Interval): "
    f"{predicted_cpu:.2f}%"
)

# -----------------------------------
# CPU FORECAST GRAPH
# -----------------------------------

st.subheader("CPU Prediction Graph")

actual_cpu = display_df['cpu'].tolist()

last_values = train_df['cpu'].tolist()[-5:]

future_predictions = []

for i in range(5):

    rolling_mean = (
        sum(last_values[-5:]) / 5
    )

    future_input = [[
        last_values[-1],
        last_values[-2],
        last_values[-3],
        rolling_mean
    ]]

    future_pred = prediction_model.predict(
        future_input
    )[0]

    future_predictions.append(
        future_pred
    )

    last_values.append(
        future_pred
    )

future_x = list(
    range(
        len(actual_cpu),
        len(actual_cpu) + 5
    )
)

# -----------------------------------
# PLOT GRAPH
# -----------------------------------

fig, ax = plt.subplots()

# Actual CPU
ax.plot(
    range(len(actual_cpu)),
    actual_cpu,
    label="Actual CPU Usage"
)

# Predicted CPU
ax.plot(
    future_x,
    future_predictions,
    linestyle='dashed',
    marker='o',
    color='red',
    label='Predicted Future CPU'
)

# Connect line
ax.plot(
    [len(actual_cpu)-1, future_x[0]],
    [
        actual_cpu[-1],
        future_predictions[0]
    ],
    linestyle='dashed',
    color='red'
)

ax.set_xlabel("Time")

ax.set_ylabel("CPU Usage (%)")

ax.set_title("CPU Forecast")

ax.legend()

st.pyplot(fig)

# Memory fix
plt.close(fig)

# -----------------------------------
# LATEST VALUES
# -----------------------------------

latest_cpu = train_df['cpu'].iloc[-1]

latest_memory = train_df['memory'].iloc[-1]

# -----------------------------------
# PREDICTION ALERT
# -----------------------------------

if predicted_cpu > 80:

    st.error(
        "⚠ Potential CPU overload "
        "risk detected!"
    )

# -----------------------------------
# SYSTEM HEALTH SCORE
# -----------------------------------

st.subheader(
    "System Health Score"
)

health_score = 100

health_score -= latest_cpu * 0.3

health_score -= latest_memory * 0.2

if not anomalies.empty:

    health_score -= 15

if predicted_cpu > 80:

    health_score -= 20

health_score = max(
    0,
    round(health_score)
)

if health_score >= 80:

    st.success(
        f"🟢 System Health Score: "
        f"{health_score}/100"
    )

elif health_score >= 50:

    st.warning(
        f"🟡 System Health Score: "
        f"{health_score}/100"
    )

else:

    st.error(
        f"🔴 System Health Score: "
        f"{health_score}/100"
    )

# -----------------------------------
# SYSTEM ALERTS
# -----------------------------------

st.subheader("System Alerts")

# CPU Alert
if latest_cpu > 80:

    st.error(
        f"⚠ High CPU Usage Detected: "
        f"{latest_cpu}%"
    )

# Memory Alert
if latest_memory > 80:

    st.warning(
        f"⚠ High Memory Usage Detected: "
        f"{latest_memory}%"
    )

# Anomaly Alert
if not anomalies.empty:

    latest_anomaly = anomalies.iloc[-1]

    anomaly_cpu = latest_anomaly['cpu']

    anomaly_memory = latest_anomaly['memory']

    anomaly_time = latest_anomaly['timestamp']

    st.error(
        f"""
⚠ Anomalous telemetry pattern detected at
{anomaly_time}

CPU Usage: {anomaly_cpu}%
Memory Usage: {anomaly_memory}%

The observed system behaviour deviates
from learned operational patterns.
"""
    )