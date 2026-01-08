import streamlit as st
import pandas as pd

# ---- Matplotlib cloud-safe setup ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(
    page_title="Delhi Air Quality Anomaly Dashboard",
    layout="wide"
)

# ---- Data loading ----
@st.cache_data
def load_data():
    # CSV is in the SAME folder as this app.py
    DATA_PATH = Path(__file__).parent / "delhi_air_anomaly_merged_if_lstm.csv"

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    df["iforest_label_bool"] = df["iforest_label_bool"].astype(bool)
    df["lstm_label_bool"] = df["lstm_label_bool"].astype(bool)

    return df


df = load_data()

# ---- Safety check (prevents 503 crashes) ----
required_cols = [
    "date", "aqi",
    "iforest_label_bool", "lstm_label_bool",
    "iforest_score", "lstm_score"
]

missing = set(required_cols) - set(df.columns)
if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

# ---- UI ----
st.title("Delhi Air Quality Anomaly Dashboard")
st.write("Isolation Forest vs LSTM anomalies on AQI time series.")

# ---- Controls ----
year = st.selectbox(
    "Select year",
    sorted(df["date"].dt.year.unique())
)

show_if = st.checkbox("Show IsolationForest anomalies", value=True)
show_lstm = st.checkbox("Show LSTM anomalies", value=True)

sub = df[df["date"].dt.year == year]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sub["date"], sub["aqi"], label="AQI", color="blue")

if show_if:
    ax.scatter(
        sub.loc[sub["iforest_label_bool"], "date"],
        sub.loc[sub["iforest_label_bool"], "aqi"],
        color="red",
        label="IF anomalies",
        s=25
    )

if show_lstm:
    ax.scatter(
        sub.loc[sub["lstm_label_bool"], "date"],
        sub.loc[sub["lstm_label_bool"], "aqi"],
        facecolors="none",
        edgecolors="green",
        label="LSTM anomalies",
        s=40
    )

ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.set_title(f"Delhi AQI with Anomalies ({year})")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

# ---- Top anomalies table ----
st.subheader("Top anomalous days")

model_choice = st.radio(
    "Model",
    ["IsolationForest", "LSTM", "Both"]
)

if model_choice == "IsolationForest":
    temp = df[df["iforest_label_bool"]].sort_values(
        "iforest_score", ascending=False
    )
elif model_choice == "LSTM":
    temp = df[df["lstm_label_bool"]].sort_values(
        "lstm_score", ascending=False
    )
else:
    temp = df[
        df["iforest_label_bool"] & df["lstm_label_bool"]
    ].sort_values(
        ["iforest_score", "lstm_score"],
        ascending=False
    )

st.dataframe(
    temp[
        [
            "date", "aqi", "pm2.5",
            "fire_count_all_india",
            "temperature_celsius",
            "wind_kph",
            "iforest_score",
            "lstm_score"
        ]
    ].head(30),
    use_container_width=True
)
