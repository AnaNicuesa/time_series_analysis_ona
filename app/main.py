import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
DATA_PATH = OUTPUTS_DIR / "timeseries_with_features.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features_for_date(history_df, target_date, feature_columns):
    history_df = history_df.copy().sort_values("date").reset_index(drop=True)
    target_date = pd.Timestamp(target_date)

    max_lag = 30
    if len(history_df) < max_lag:
        raise ValueError(f"At least {max_lag} historical rows are required to build lag features.")

    features = {}
    features["day"] = target_date.day
    features["month"] = target_date.month
    features["dayofweek"] = target_date.dayofweek
    features["is_weekend"] = int(target_date.dayofweek >= 5)
    features["week_of_year"] = int(target_date.isocalendar()[1])
    

    for lag in [1, 7, 14, 30]:
        features[f"lag_{lag}"] = history_df["unit_sales"].iloc[-lag]

    features["rolling_7d_mean"] = history_df["unit_sales"].iloc[-7:].mean()
    features["rolling_14d_mean"] = history_df["unit_sales"].iloc[-14:].mean()
    features["rolling_30d_mean"] = history_df["unit_sales"].iloc[-30:].mean()
    features["rolling_7d_std"] = history_df["unit_sales"].iloc[-7:].std()

    for col in ["oil_lag_1", "oil_lag_7", "oil_rolling_7d_mean"]:
        if col in history_df.columns:
            features[col] = history_df[col].iloc[-1]

    for col in feature_columns:
        if col not in features:
            features[col] = 0

    return pd.DataFrame([features])[feature_columns]


def forecast_next_days(history_df, start_date, horizon, model, feature_columns):
    temp_history = history_df.copy().sort_values("date").reset_index(drop=True)
    start_date = pd.Timestamp(start_date)
    forecasts = []

    for step in range(horizon):
        forecast_date = start_date + pd.Timedelta(days=step)
        X_step = build_features_for_date(temp_history, forecast_date, feature_columns)
        y_hat = float(model.predict(X_step)[0])

        forecasts.append({"date": forecast_date, "forecast": y_hat})

        new_row = {col: np.nan for col in temp_history.columns}
        new_row["date"] = forecast_date
        new_row["unit_sales"] = y_hat
        temp_history = pd.concat([temp_history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(forecasts)


st.title("Sales Forecasting App")
st.write("Forecast future unit sales using the trained model artifacts from the project.")

model = load_model()
feature_columns = load_features()
metrics_df = load_metrics()
df = load_data()

with st.sidebar:
    st.header("Forecast settings")
    default_start = df["date"].max().date() + pd.Timedelta(days=1)
    start_date = st.date_input("Forecast start date", value=default_start)
    horizon = st.slider("Forecast horizon", min_value=1, max_value=30, value=14)
    history_window = st.slider("Historical window shown", min_value=30, max_value=365, value=180)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Last historical date", str(df["date"].max().date()))
col3.metric("Model features", len(feature_columns))

if not metrics_df.empty:
    st.subheader("Saved Model Metrics")
    st.dataframe(metrics_df)

forecast_df = forecast_next_days(df, start_date=start_date, horizon=horizon, model=model, feature_columns=feature_columns)

st.subheader("Forecast Table")
st.dataframe(forecast_df)

plot_history = df.tail(history_window)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_history["date"], plot_history["unit_sales"], label="Historical unit sales")
ax.plot(forecast_df["date"], forecast_df["forecast"], marker="o", label="Forecast")
ax.set_title(f"{horizon}-Day Unit Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Unit sales")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download forecast as CSV",
    data=csv,
    file_name="forecast_output.csv",
    mime="text/csv",
)
