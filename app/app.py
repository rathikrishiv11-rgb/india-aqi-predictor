import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ── Load model and data ────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/rf_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_aqi.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model, feature_cols = load_model()
df = load_data()

# ── Health risk logic ──────────────────────────────────
def health_risk(aqi):
    if aqi <= 50:    return "Good",          "🟢", "#00b300"
    elif aqi <= 100: return "Satisfactory",  "🟡", "#99cc00"
    elif aqi <= 200: return "Moderate",      "🟠", "#ffaa00"
    elif aqi <= 300: return "Poor",          "🔴", "#ff0000"
    elif aqi <= 400: return "Very Poor",     "🟣", "#990066"
    else:            return "Severe",        "⚫", "#4d0000"

def outdoor_advice(category):
    advice = {
        "Good":         "✅ Safe for all outdoor activities.",
        "Satisfactory": "✅ Generally safe. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        "Moderate":     "⚠️ Sensitive groups (children, elderly, asthma patients) should limit outdoor activity.",
        "Poor":         "🚫 Avoid prolonged outdoor exertion. Wear a mask if going outside.",
        "Very Poor":    "🚫 Avoid all outdoor activity. Keep windows closed.",
        "Severe":       "🚨 Health emergency level. Stay indoors. Seek medical advice if experiencing symptoms."
    }
    return advice[category]

# ── UI ─────────────────────────────────────────────────
st.title("🌫️ India AQI Predictor")
st.markdown("Real-time air quality insights and next-day AQI forecasting for Indian cities.")
st.divider()

# City selector
cities = sorted(df["City"].unique().tolist())
selected_city = st.selectbox("Select a city", cities)

city_df = df[df["City"] == selected_city].sort_values("Date")

# ── Section 1: Today's AQI ─────────────────────────────
st.subheader("📍 Latest AQI Reading")

latest = city_df.dropna(subset=["AQI"]).iloc[-1]
latest_aqi = latest["AQI"]
latest_date = latest["Date"].strftime("%d %B %Y")
category, emoji, color = health_risk(latest_aqi)

col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"AQI as of {latest_date}", value=int(latest_aqi))
with col2:
    st.markdown(
        f"<div style='background-color:{color};padding:16px;border-radius:8px;"
        f"text-align:center;color:white;font-size:20px;font-weight:bold'>"
        f"{emoji} {category}</div>",
        unsafe_allow_html=True
    )

st.markdown(f"**Outdoor advisory:** {outdoor_advice(category)}")
st.divider()

# ── Section 2: Tomorrow's Prediction ──────────────────
st.subheader("🔮 Tomorrow's AQI Prediction")

# Build input row using latest available values
# This is what the model actually needs — same features as training
def build_input_row(city_df, feature_cols):
    latest = city_df.dropna(subset=["AQI"]).iloc[-1]
    aqi_series = city_df["AQI"].dropna().values

    row = {}

    # Pollutants — use latest reading
    for col in ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]:
        row[col] = latest[col] if col in city_df.columns else 0

    # Lag features
    row["AQI_lag_1"] = aqi_series[-1] if len(aqi_series) >= 1 else 0
    row["AQI_lag_3"] = aqi_series[-3] if len(aqi_series) >= 3 else 0
    row["AQI_lag_7"] = aqi_series[-7] if len(aqi_series) >= 7 else 0

    # Rolling averages
    row["AQI_roll_3"] = np.mean(aqi_series[-3:]) if len(aqi_series) >= 3 else aqi_series[-1]
    row["AQI_roll_7"] = np.mean(aqi_series[-7:]) if len(aqi_series) >= 7 else aqi_series[-1]

    # Date features for tomorrow
    tomorrow = latest["Date"] + pd.Timedelta(days=1)
    row["month"]       = tomorrow.month
    row["day_of_week"] = tomorrow.dayofweek
    row["day_of_year"] = tomorrow.timetuple().tm_yday

    # Season encoding
    month = tomorrow.month
    row["season_Post_Monsoon"] = 1 if month in [10, 11] else 0
    row["season_Summer"]       = 1 if month in [3, 4, 5] else 0
    row["season_Winter"]       = 1 if month in [12, 1, 2] else 0

    # City encoding — all zeros except selected city
    for col in feature_cols:
        if col.startswith("City_"):
            row[col] = 0
    city_col = f"City_{selected_city}"
    if city_col in feature_cols:
        row[city_col] = 1

    # Build dataframe in exact column order model expects
    input_df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_cols]

input_row = build_input_row(city_df, feature_cols)
predicted_aqi = model.predict(input_row)[0]
predicted_aqi = max(0, round(predicted_aqi, 1))

pred_category, pred_emoji, pred_color = health_risk(predicted_aqi)

col3, col4 = st.columns(2)
with col3:
    st.metric(label="Predicted AQI (Tomorrow)", value=predicted_aqi)
with col4:
    st.markdown(
        f"<div style='background-color:{pred_color};padding:16px;border-radius:8px;"
        f"text-align:center;color:white;font-size:20px;font-weight:bold'>"
        f"{pred_emoji} {pred_category}</div>",
        unsafe_allow_html=True
    )

st.markdown(f"**Outdoor advisory:** {outdoor_advice(pred_category)}")
st.divider()

# ── Section 3: Historical trend ────────────────────────
st.subheader("📈 AQI Trend (Last 90 Days)")

recent = city_df.dropna(subset=["AQI"]).tail(90)
fig = px.line(
    recent, x="Date", y="AQI",
    title=f"AQI Trend — {selected_city}",
    labels={"AQI": "AQI", "Date": "Date"}
)
fig.add_hline(y=100, line_dash="dash", line_color="orange",
              annotation_text="Moderate threshold")
fig.add_hline(y=200, line_dash="dash", line_color="red",
              annotation_text="Poor threshold")
st.plotly_chart(fig, use_container_width=True)

# ── Section 4: Manual Prediction (Real-time input) ────
st.subheader("🎛️ Predict from Today's Readings")
st.markdown("Enter current pollutant values from any AQI monitor or weather site to get tomorrow's forecast.")

col5, col6 = st.columns(2)
with col5:
    input_pm25  = st.slider("PM2.5 (µg/m³)",  0, 500, 52)
    input_pm10  = st.slider("PM10 (µg/m³)",   0, 500, 65)
    input_no2   = st.slider("NO2 (µg/m³)",    0, 200, 41)
    input_co    = st.slider("CO (mg/m³)",      0, 50,  1)
with col6:
    input_so2   = st.slider("SO2 (µg/m³)",    0, 200, 20)
    input_o3    = st.slider("O3 (µg/m³)",     0, 200, 30)
    input_no    = st.slider("NO (µg/m³)",     0, 200, 20)
    input_nox   = st.slider("NOx (µg/m³)",    0, 200, 30)
    input_nh3   = st.slider("NH3 (µg/m³)",    0, 200, 15)

if st.button("🔮 Predict Tomorrow's AQI"):
    # Build input row manually using slider values
    # Lag and rolling features use the city's historical data
    # since we don't have yesterday's manual readings
    aqi_series = city_df["AQI"].dropna().values
    tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)

    manual_row = {}

    # Pollutants from sliders
    manual_row["PM2.5"] = input_pm25
    manual_row["PM10"]  = input_pm10
    manual_row["NO"]    = input_no
    manual_row["NO2"]   = input_no2
    manual_row["NOx"]   = input_nox
    manual_row["NH3"]   = input_nh3
    manual_row["CO"]    = input_co
    manual_row["SO2"]   = input_so2
    manual_row["O3"]    = input_o3

    # Lag features from city historical data
    manual_row["AQI_lag_1"]   = aqi_series[-1] if len(aqi_series) >= 1 else 0
    manual_row["AQI_lag_3"]   = aqi_series[-3] if len(aqi_series) >= 3 else 0
    manual_row["AQI_lag_7"]   = aqi_series[-7] if len(aqi_series) >= 7 else 0
    manual_row["AQI_roll_3"]  = np.mean(aqi_series[-3:]) if len(aqi_series) >= 3 else aqi_series[-1]
    manual_row["AQI_roll_7"]  = np.mean(aqi_series[-7:]) if len(aqi_series) >= 7 else aqi_series[-1]

    # Date features for tomorrow
    manual_row["month"]       = tomorrow.month
    manual_row["day_of_week"] = tomorrow.dayofweek
    manual_row["day_of_year"] = tomorrow.timetuple().tm_yday

    # Season encoding
    month = tomorrow.month
    manual_row["season_Post_Monsoon"] = 1 if month in [10, 11] else 0
    manual_row["season_Summer"]       = 1 if month in [3, 4, 5] else 0
    manual_row["season_Winter"]       = 1 if month in [12, 1, 2] else 0

    # City encoding
    for col in feature_cols:
        if col.startswith("City_"):
            manual_row[col] = 0
    city_col = f"City_{selected_city}"
    if city_col in feature_cols:
        manual_row[city_col] = 1

    # Build dataframe in exact column order
    manual_df = pd.DataFrame([manual_row])
    for col in feature_cols:
        if col not in manual_df.columns:
            manual_df[col] = 0
    manual_df = manual_df[feature_cols]

    # Predict
    manual_aqi = model.predict(manual_df)[0]
    manual_aqi = max(0, round(manual_aqi, 1))
    manual_cat, manual_emoji, manual_color = health_risk(manual_aqi)

    # Display result
    st.markdown("---")
    col7, col8 = st.columns(2)
    with col7:
        st.metric("Predicted AQI (Tomorrow)", manual_aqi)
    with col8:
        st.markdown(
            f"<div style='background-color:{manual_color};padding:16px;"
            f"border-radius:8px;text-align:center;color:white;"
            f"font-size:20px;font-weight:bold'>"
            f"{manual_emoji} {manual_cat}</div>",
            unsafe_allow_html=True
        )
    st.markdown(f"**Outdoor advisory:** {outdoor_advice(manual_cat)}")
    st.info("ℹ️ Lag features use historical city data as baseline. "
            "For full accuracy, connect to a live CPCB API feed.")
    
# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.caption("Data source: CPCB via Kaggle (2015–2020). "
           "Model: Random Forest Regressor. "
           "For real-time predictions, connect to live CPCB API feeds.")