import threading
import random
import time
import psutil

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

from streamlit_autorefresh import st_autorefresh

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import sqlite3

# -----------------------------------
# BACKGROUND TELEMETRY COLLECTION
# -----------------------------------

def collect_metrics():

    conn = sqlite3.connect(
        "system_monitor.db",
        check_same_thread=False
    )

    cursor = conn.cursor()

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
# BACKGROUND LOG GENERATION
# -----------------------------------

def generate_logs():

    conn = sqlite3.connect(
        "system_monitor.db",
        check_same_thread=False
    )

    cursor = conn.cursor()

    log_levels = [
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL"
    ]

    log_messages = [
        "User login successful",
        "High memory usage detected",
        "Database timeout occurred",
        "Failed login attempt",
        "API response delayed",
        "Disk nearing capacity",
        "Network timeout detected",
        "Service temporarily unavailable",
        "CPU spike detected",
        "Application restarted"
    ]

    while True:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        level = random.choice(log_levels)

        message = random.choice(log_messages)

        cursor.execute("""
        INSERT INTO logs
        (timestamp, level, message)
        VALUES (?, ?, ?)
        """, (timestamp, level, message))

        conn.commit()

        time.sleep(3)

# -----------------------------------
# START BACKGROUND THREADS
# -----------------------------------

if 'threads_started' not in st.session_state:

    telemetry_thread = threading.Thread(
        target=collect_metrics,
        daemon=True
    )

    log_thread = threading.Thread(
        target=generate_logs,
        daemon=True
    )

    telemetry_thread.start()

    log_thread.start()

    st.session_state['threads_started'] = True

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.title("AI System Monitoring Dashboard")

# Auto refresh every 3 seconds
st_autorefresh(interval=3000, key="datarefresh")

# -----------------------------------
# DATABASE CONNECTION
# -----------------------------------

conn = sqlite3.connect("system_monitor.db")

query = "SELECT * FROM system_stats"

df = pd.read_sql_query(query, conn)

# Keep latest 100 rows only
df = df.tail(100)

# -----------------------------------
# LATEST METRICS
# -----------------------------------

st.subheader("Latest System Metrics")
st.write(df.tail())
# -----------------------------------
# TELEMETRY FRESHNESS CHECK
# -----------------------------------

latest_timestamp = df['timestamp'].iloc[-1]

latest_time = datetime.strptime(
    latest_timestamp,
    "%Y-%m-%d %H:%M:%S"
)

current_time = datetime.now()

# Time difference in seconds
time_diff = (
    current_time - latest_time
).total_seconds()

# Detect stale telemetry
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
st.line_chart(df["cpu"])

st.subheader("Memory Usage Over Time")
st.line_chart(df["memory"])

st.subheader("Disk Usage Over Time")
st.line_chart(df["disk"])

# -----------------------------------
# ANOMALY DETECTION
# -----------------------------------

st.subheader("Anomaly Detection")

features = df[['cpu', 'memory']]

anomaly_model = IsolationForest(
    contamination=0.1,
    random_state=42
)

df['anomaly'] = anomaly_model.fit_predict(features)

anomalies = df[df['anomaly'] == -1]

st.write("Detected System Anomalies")
st.write(anomalies)

# -----------------------------------
# PREDICTIVE ANALYTICS
# -----------------------------------

st.subheader("Predictive Analytics")

# Create lag features
df['cpu_lag1'] = df['cpu'].shift(1)
df['cpu_lag2'] = df['cpu'].shift(2)
df['cpu_lag3'] = df['cpu'].shift(3)

# Remove empty rows
df = df.dropna()

# Features and target
X = df[['cpu_lag1', 'cpu_lag2', 'cpu_lag3']]

y = df['cpu']

# -----------------------------------
# TRAIN RANDOM FOREST MODEL
# -----------------------------------

prediction_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

prediction_model.fit(X, y)

# -----------------------------------
# PREDICT NEXT CPU VALUE
# -----------------------------------

latest_features = [[
    df['cpu'].iloc[-1],
    df['cpu'].iloc[-2],
    df['cpu'].iloc[-3]
]]

predicted_cpu = prediction_model.predict(latest_features)[0]

# Display prediction
st.write(
    f"Predicted CPU Usage (Next Interval): {predicted_cpu:.2f}%"
)

# -----------------------------------
# CPU PREDICTION GRAPH
# -----------------------------------

st.subheader("CPU Prediction Graph")

actual_cpu = df['cpu'].tolist()

# Start with latest CPU values
last_values = actual_cpu[-3:]

future_predictions = []

# Predict next 5 future CPU values
for i in range(5):

    # Use previous CPU values
    future_input = [last_values[-3:]]

    # Predict next CPU
    future_pred = prediction_model.predict(future_input)[0]

    future_predictions.append(future_pred)

    # Feed prediction back into history
    last_values.append(future_pred)

# Future x-axis
future_x = list(
    range(len(actual_cpu), len(actual_cpu) + 5)
)

# -----------------------------------
# PLOT GRAPH
# -----------------------------------

fig, ax = plt.subplots()

# Actual CPU usage
ax.plot(
    range(len(actual_cpu)),
    actual_cpu,
    label="Actual CPU Usage"
)

# Predicted future CPU
ax.plot(
    future_x,
    future_predictions,
    linestyle='dashed',
    marker='o',
    color='red',
    label='Predicted Future CPU'
)

# Connect actual to prediction
ax.plot(
    [len(actual_cpu)-1, future_x[0]],
    [actual_cpu[-1], future_predictions[0]],
    linestyle='dashed',
    color='red'
)

# Labels
ax.set_xlabel("Time")
ax.set_ylabel("CPU Usage (%)")
ax.set_title("5-Step CPU Forecast")

ax.legend()

# Show graph
st.pyplot(fig)

# IMPORTANT MEMORY FIX
plt.close(fig)

latest_cpu = df['cpu'].iloc[-1]
latest_memory = df['memory'].iloc[-1]
# -----------------------------------
# PREDICTION ALERT
# -----------------------------------

if predicted_cpu > 80:
    st.error("⚠ Potential CPU overload risk detected!")

# -----------------------------------
# SYSTEM HEALTH SCORE
# -----------------------------------

st.subheader("System Health Score")

health_score = 100

# Reduce score for high CPU
health_score -= latest_cpu * 0.3

# Reduce score for high memory
health_score -= latest_memory * 0.2

# Reduce score if anomalies exist
if not anomalies.empty:
    health_score -= 15

# Reduce score for dangerous predictions
if predicted_cpu > 80:
    health_score -= 20

# Prevent negative score
health_score = max(0, round(health_score))

# Display health score
if health_score >= 80:
    st.success(f"🟢 System Health Score: {health_score}/100")

elif health_score >= 50:
    st.warning(f"🟡 System Health Score: {health_score}/100")

else:
    st.error(f"🔴 System Health Score: {health_score}/100")


# -----------------------------------
# INTELLIGENT ALERTS
# -----------------------------------

st.subheader("System Alerts")

latest_cpu = df['cpu'].iloc[-1]
latest_memory = df['memory'].iloc[-1]

# CPU Alert
if latest_cpu > 80:
    st.error(
        f"⚠ High CPU Usage Detected: {latest_cpu}%"
    )

# Memory Alert
if latest_memory > 80:
    st.warning(
        f"⚠ High Memory Usage Detected: {latest_memory}%"
    )

# AI Anomaly Alert
if not anomalies.empty:

    latest_anomaly = anomalies.iloc[-1]

    anomaly_cpu = latest_anomaly['cpu']
    anomaly_memory = latest_anomaly['memory']
    anomaly_time = latest_anomaly['timestamp']

    st.error(
        f"""
⚠ AI detected anomalous behaviour at {anomaly_time}

CPU Usage: {anomaly_cpu}%
Memory Usage: {anomaly_memory}%

The observed system behaviour deviates from learned normal operational patterns.
"""
    )

# -----------------------------------
# LIVE LOG ANALYTICS
# -----------------------------------

st.subheader("Live System Logs")

# Read logs table
log_query = "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20"

logs_df = pd.read_sql_query(log_query, conn)

# Show logs
st.dataframe(logs_df)

# -----------------------------------
# LOG INTELLIGENCE ANALYTICS
# -----------------------------------

st.subheader("Log Intelligence Analytics")

# Count log severities
error_count = len(
    logs_df[logs_df['level'] == 'ERROR']
)

critical_count = len(
    logs_df[logs_df['level'] == 'CRITICAL']
)

warning_count = len(
    logs_df[logs_df['level'] == 'WARNING']
)

# Failed login detection
failed_login_count = len(
    logs_df[
        logs_df['message']
        .str.contains("login", case=False)
    ]
)

# Display metrics
st.write(f"ERROR Logs: {error_count}")
st.write(f"CRITICAL Logs: {critical_count}")
st.write(f"WARNING Logs: {warning_count}")
st.write(f"Failed Login Events: {failed_login_count}")

# Intelligent log alerts

if critical_count >= 5:
    st.error(
        "🚨 High number of CRITICAL system events detected!"
    )

if error_count >= 5:
    st.warning(
        "⚠ Elevated ERROR log frequency detected!"
    )

if failed_login_count >= 3:
    st.warning(
        "⚠ Multiple failed login attempts detected!"
    )

# -----------------------------------
# ML-BASED LOG ANOMALY DETECTION
# -----------------------------------

st.subheader("ML-Based Log Anomaly Detection")

# Create numerical log features

logs_df['is_error'] = (
    logs_df['level'] == 'ERROR'
).astype(int)

logs_df['is_critical'] = (
    logs_df['level'] == 'CRITICAL'
).astype(int)

logs_df['is_warning'] = (
    logs_df['level'] == 'WARNING'
).astype(int)

logs_df['failed_login'] = (
    logs_df['message']
    .str.contains("login", case=False)
).astype(int)

# Feature set
log_features = logs_df[[
    'is_error',
    'is_critical',
    'is_warning',
    'failed_login'
]]

# Train anomaly model
log_anomaly_model = IsolationForest(
    contamination=0.2,
    random_state=42
)

# Predict anomalies
logs_df['log_anomaly'] = (
    log_anomaly_model.fit_predict(log_features)
)

# Extract anomalous logs
log_anomalies = logs_df[
    logs_df['log_anomaly'] == -1
]

# Show anomalies
st.write("Detected Log Anomalies")

st.dataframe(
    log_anomalies[[
        'timestamp',
        'level',
        'message'
    ]]
)

# Intelligent alerts

if not log_anomalies.empty:

    st.error(
        "🚨 AI detected anomalous operational log patterns!"
    )