import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Paths ──────────────────────────────────────────────
FEATURES_PATH = "data/features.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/rf_model.pkl"

# ── Load ───────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
print("Loaded features:", df.shape)

# ── Reconstruct date from day_of_year + year ───────────
# We dropped Date in features.py but kept day_of_year.
# We need year to do temporal split — reload it from clean data.
clean = pd.read_csv("data/clean_aqi.csv")
clean["Date"] = pd.to_datetime(clean["Date"])
clean = clean.sort_values(["City", "Date"]).reset_index(drop=True)
clean = clean.dropna(subset=["AQI"])

# Align years with features dataframe (same rows after lag drop)
clean = clean.iloc[len(clean) - len(df):].reset_index(drop=True)
df["year"] = clean["Date"].dt.year.values

print("Year distribution:\n", df["year"].value_counts().sort_index())

# ── Define features and target ─────────────────────────
# Drop AQI_Category — it's derived from AQI, keeping it would be leakage
# Drop AQI itself from features — that's your target
TARGET = "AQI"
DROP_COLS = ["AQI", "AQI_Category", "year"]

feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols]
y = df[TARGET]

# ── Temporal train / validation / test split ───────────
train_mask = df["year"] <= 2018
val_mask   = df["year"] == 2019
test_mask  = df["year"] == 2020

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Train Random Forest ────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1      # use all CPU cores
)
rf.fit(X_train, y_train)
print("\nModel trained.")

# ── Evaluate on validation set ─────────────────────────
val_preds = rf.predict(X_val)
val_mae   = mean_absolute_error(y_val, val_preds)
val_rmse  = root_mean_squared_error(y_val, val_preds)
print(f"\nValidation MAE  : {val_mae:.2f}")
print(f"Validation RMSE : {val_rmse:.2f}")

# ── Evaluate on test set (once, final) ─────────────────
test_preds = rf.predict(X_test)
test_mae   = mean_absolute_error(y_test, test_preds)
test_rmse  = root_mean_squared_error(y_test, test_preds)
print(f"\nTest MAE  : {test_mae:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")

# ── Feature importance ─────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

# ── Save model ─────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ── Save feature column names ──────────────────────────
# Streamlit needs to know exact column order when predicting
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Feature columns saved to models/feature_cols.pkl")