import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────
CLEAN_PATH = "data/clean_aqi.csv"
FEATURES_PATH = "data/features.csv"

# ── Load cleaned data ──────────────────────────────────
df = pd.read_csv(CLEAN_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)
print("Loaded clean data:", df.shape)

# ── Lag features ───────────────────────────────────────
# For each city separately — you never want Delhi's yesterday
# bleeding into Chennai's today. groupby ensures city boundary.
for lag in [1, 3, 7]:
    df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)

# ── Rolling average features ───────────────────────────
# min_periods=1 means it won't produce NaN if window isn't full yet
# but we'll drop early rows anyway below
for window in [3, 7]:
    df[f"AQI_roll_{window}"] = (
        df.groupby("City")["AQI"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# ── Date features ──────────────────────────────────────
df["month"] = df["Date"].dt.month
df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["day_of_year"] = df["Date"].dt.dayofyear

# Season: Indian meteorological seasons
def get_season(month):
    if month in [12, 1, 2]:  return "Winter"
    elif month in [3, 4, 5]: return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else:                     return "Post_Monsoon"  # Oct, Nov

df["season"] = df["month"].apply(get_season)
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# ── CPCB Health Risk Category ──────────────────────────
def health_risk(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

df["AQI_Category"] = df["AQI"].apply(health_risk)

# ── City encoding ──────────────────────────────────────
df = pd.get_dummies(df, columns=["City"], drop_first=True)

# ── Drop rows with NaN in lag columns ─────────────────
# First 7 rows per city have no valid lag_7 value
before = len(df)
df.dropna(subset=["AQI_lag_1", "AQI_lag_3", "AQI_lag_7"], inplace=True)
print(f"Dropped {before - len(df)} rows due to lag NaN (expected ~7 per city)")

# ── Drop Date column — not a model feature ─────────────
df.drop(columns=["Date"], inplace=True)

# ── Save ───────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv(FEATURES_PATH, index=False)
print(f"Features saved to {FEATURES_PATH}")
print("Final shape:", df.shape)
print("Columns:", df.columns.tolist())