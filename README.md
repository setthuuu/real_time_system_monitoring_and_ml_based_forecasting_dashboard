\# Real-Time System Monitoring \& ML-Based Forecasting System



\## Overview



A real-time telemetry monitoring system built using Python, Streamlit, SQLite, XGBoost, and Isolation Forest for system analytics, anomaly detection, and forecasting.



The system continuously collects CPU, memory, and disk telemetry metrics, stores them in SQLite, performs anomaly detection using Isolation Forest, and predicts future CPU utilization trends using XGBoost with lag-based forecasting.



\---



\## Features



\- Real-time telemetry collection

\- CPU, memory, and disk monitoring

\- Auto-refreshing Streamlit dashboard

\- ML-based anomaly detection

\- XGBoost forecasting pipeline

\- Lag feature engineering

\- Rolling window feature generation

\- Intelligent alert system

\- System health scoring



\---



\## Tech Stack



\- Python

\- Pandas

\- SQLite

\- Streamlit

\- Scikit-learn

\- XGBoost

\- Matplotlib

\- psutil



\---



\## System Architecture



Telemetry Collection (psutil)

↓

SQLite Storage

↓

Feature Engineering

↓

Isolation Forest Anomaly Detection

↓

XGBoost Forecasting

↓

Streamlit Visualization Dashboard



\---



\## Dashboard Preview



(Add screenshot here later)



\---



\## Installation



Clone the repository:



```bash

git clone YOUR\_GITHUB\_LINK

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run app.py

```



\---



\## Future Improvements



\- LSTM-based deep learning forecasting for sequential telemetry prediction

\- Log analytics integration

\- Multi-system monitoring

\- Cloud deployment

\- Advanced time-series forecasting

\- Alert notifications

