import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────
RAW_PATH = "data/raw/city_day.csv"
CLEAN_PATH = "data/clean_aqi.csv"

# ── Load ───────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

# ── Drop unreliable columns ────────────────────────────
df.drop(columns=["Xylene", "Toluene", "Benzene", "AQI_Bucket"], inplace=True)

# ── Keep only cities with 1000+ AQI readings ──────────
reliable_cities = df.groupby("City")["AQI"].count()
reliable_cities = reliable_cities[reliable_cities >= 1000].index.tolist()
df = df[df["City"].isin(reliable_cities)].copy()
print("Cities kept:", reliable_cities)
print("Shape after city filter:", df.shape)

# ── Parse and sort dates ───────────────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

# ── Impute pollutants using each city's own median ─────
pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NOx", "NO", "NH3"]
for col in pollutants:
    df[col] = df.groupby("City")[col].transform(
        lambda x: x.fillna(x.median())
    )

# ── Drop rows where AQI is missing (target variable) ───
before = len(df)
df.dropna(subset=["AQI"], inplace=True)
print(f"Dropped {before - len(df)} rows with missing AQI")

# ── Save cleaned data ──────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print(f"Cleaned data saved to {CLEAN_PATH}")
print("Final shape:", df.shape)