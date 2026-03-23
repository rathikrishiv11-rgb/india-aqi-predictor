# 🌫️ India AQI Predictor

A machine learning web application that forecasts next-day Air Quality Index 
(AQI) for Indian cities and provides actionable health advisories based on 
CPCB standards.

**Live App:** [Link coming after deployment]

---

## What It Does

- Shows the latest AQI reading for 14 major Indian cities
- Predicts tomorrow's AQI using a trained Random Forest model
- Accepts today's real pollutant readings via interactive sliders
  for live inference
- Classifies air quality into CPCB health risk categories 
  (Good → Severe) with outdoor safety advisories
- Visualizes AQI trends over the last 90 days per city

---

## Demo

> Select a city → Enter today's pollutant readings → Get tomorrow's 
> AQI forecast with health advisory

---

## Model Performance

| Metric | Validation (2019) | Test (2020) |
|---|---|---|
| MAE | 21.34 | 15.31 |
| RMSE | 54.09 | 33.90 |

**Primary metric: MAE (Mean Absolute Error)**  
An MAE of 15.31 means predictions are on average ~15 AQI points from 
the actual value. Since CPCB categories span 50–100 points, predictions 
are accurate enough to reliably identify the correct health risk category.

**Note on Test MAE:** Lower test MAE vs validation is attributable to 
COVID-19 lockdowns in 2020 reducing pollution variability, making 
patterns more predictable during that period.

**Top predictive features:**
1. AQI yesterday (lag_1) — 63.6% importance
2. CO concentration — 18.2%
3. PM2.5 — 9.6%

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn
- **Model** — Random Forest Regressor (100 estimators)
- **App** — Streamlit + Plotly
- **Data** — CPCB via Kaggle (2015–2020)

---

## Project Structure
```
india-aqi-predictor/
├── data/
│   └── raw/          ← place city_day.csv here (see Setup)
├── src/
│   ├── clean.py      ← data cleaning pipeline
│   ├── features.py   ← feature engineering
│   └── train.py      ← model training + evaluation
├── app/
│   └── app.py        ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/rathikrishiv11-rgb/india-aqi-predictor
cd india-aqi-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**  
Download `city_day.csv` from 
[Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
Place it in `data/raw/city_day.csv`

**4. Run the pipeline**
```bash
python src/clean.py
python src/features.py
python src/train.py
```

**5. Launch the app**
```bash
streamlit run app/app.py
```

---

## Methodology

### Data Cleaning
- Dropped columns with >20% missing values (Xylene, Toluene, Benzene)
- Retained 14 cities with 1000+ AQI readings for reliability
- Imputed missing pollutant values using each city's own median
  — not the global median — to preserve city-specific baselines
- Dropped rows where AQI itself was missing (target variable cannot 
  be imputed)

### Feature Engineering
- **Lag features** — AQI from 1, 3, and 7 days prior per city
- **Rolling averages** — 3-day and 7-day rolling mean of AQI
- **Date features** — month, day of week, day of year, Indian 
  meteorological season
- All rolling features use `shift(1)` to prevent target leakage

### Train / Validation / Test Split
Temporal split by year — not random — to prevent data leakage:
- Train: 2015–2018 (11,677 rows)
- Validation: 2019 (4,319 rows)
- Test: 2020 (2,175 rows)

Random splitting would allow future data to leak into training,
artificially inflating performance metrics.

---

## Limitations

1. **Data cutoff:** Dataset ends July 2020. The manual input mode 
   allows real-time inference using today's readings from any AQI 
   monitor. For fully automated real-time forecasting, the pipeline 
   would need to connect to the live CPCB API.

2. **Lag feature baseline:** When using manual input mode, lag 
   features fall back to the city's historical data rather than 
   actual recent readings, which may reduce prediction accuracy.

3. **City coverage:** Limited to 14 cities with sufficient 
   historical data. Smaller cities and rural areas are not covered.

4. **Post-COVID patterns:** Air quality patterns may have shifted 
   after 2020 due to structural changes in traffic and industry. 
   Model retraining on recent data is recommended for production use.

---

## What I Would Improve With More Time

- Connect to live CPCB API for real-time automated data ingestion
- Add XGBoost and compare against Random Forest
- Expand to more cities using interpolation for sparse data
- Add particulate matter forecasting breakdown (PM2.5 vs PM10 
  separately)
- Retrain on post-2020 data when available

---

## Data Source

Rohan Rao. *Air Quality Data in India (2015–2020)*. Kaggle.  
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Original source: Central Pollution Control Board (CPCB), Government 
of India.