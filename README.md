# 🌦️ Weather Predictor: any city, any date & time

Choose **any city in the world**, a **date** and an **hour**, and get the predicted temperature. For each city, a Linear Regression model is trained on that city's own 3-year hourly history, then compared with Open-Meteo's official forecast.

**🔗 Live app:** _<paste your Streamlit Cloud link here>_

_<add 1-2 screenshots here>_

## Features
- 🔎 Search any city (Open-Meteo geocoding API)
- 📅 Pick any future date/hour up to **7 days ahead**, or a past date to see observed temperature
- 🧹 Preprocessing with Pandas: gap filling, continuous hourly index, lag/anomaly/seasonality features
- 🎯 **Feature selection** (SelectKBest F-test, K chosen by validation error); the app shows each feature's score
- 🤖 Linear Regression model trained **per city, on demand** (about 10-30 s the first time, cached afterwards)
- ⚖️ Side-by-side comparison with Open-Meteo's forecast
- 📊 Honest accuracy report: MAE by lead time vs. two baselines (climatology, persistence)
- 🌍 Climate tab: average temperature and rainfall by month

## How the model works
1. **Climatology**: average temperature for each (month, hour of day) in that city.
2. **Anomaly**: how much warmer/colder than normal it is *right now*.
3. Anomalies fade with time, so Linear Regression predicts the **future anomaly** from the current anomaly (scaled by `exp(-h/24)` and `exp(-h/72)`), 6 h trend, humidity, wind, 24 h rainfall, pressure change, hour of day and season.
4. **Feature selection** keeps only the most informative features (`SelectKBest`, K picked on a validation slice).
5. **Prediction = climatology(target time) + predicted anomaly.** One model covers all lead times from 1 h to 168 h.

**Evaluation:** time-based split (no shuffling), so the model is tested on forecasts made *after* its training period. Compared with climatology-only and persistence ("same as now") baselines.

## Results
Run `python train.py "<your city>"` and paste the table here (mean absolute error in °C):

| Lead time | Our model | Climatology only | Persistence |
|---|---|---|---|
| _fill_ | _fill_ | _fill_ | _fill_ |

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python train.py "Bengaluru"      # optional: prints accuracy table
streamlit run app.py
```

## Project structure
| File | Purpose |
|---|---|
| `weather_utils.py` | Open-Meteo API calls (geocoding, history, live) |
| `model.py` | Features, climatology, training, evaluation, prediction |
| `app.py` | Streamlit interface |
| `train.py` | Command-line training/evaluation report |

## Tech stack
Python · Pandas · NumPy · Scikit-Learn · Streamlit · Open-Meteo API

## Limitations
- Official forecasts use physics-based models and are generally more accurate; this project demonstrates an end-to-end ML pipeline.
- Beyond ~3 days the model has little skill over plain climatology (shown in the accuracy tab).
- Predicts temperature only; trained on a single city's history at a time.
- The free API is rate-limited, so many different cities in a short time may hit the limit.

## Ideas for improvement
Gradient Boosting / Random Forest comparison · more variables (cloud cover, dew point) · predict rainfall · train one global model across many cities.
