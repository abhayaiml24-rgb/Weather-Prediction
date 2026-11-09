"""Hourly temperature model that works for ANY city.

Idea (easy to explain in a viva):
  1. Climatology  = the city's average temperature for each (month, hour of day).
  2. Anomaly      = how much warmer/colder than climatology it is right now.
  3. Weather anomalies fade over time, so we predict the FUTURE anomaly with
     Linear Regression from the current anomaly (scaled by exp(-h/24), exp(-h/72)),
     recent trend, humidity, wind, rain, pressure change, hour of day, season.
  4. Prediction = climatology(target time) + predicted anomaly.

One direct model handles every lead time h = 1..168 hours (7 days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72, 96, 120, 168]  # lead times used in training
REPORT_HORIZONS = [1, 6, 12, 24, 48, 72, 120, 168]
MAX_HORIZON = 168          # hours (7 days)
ORIGIN_STEP = 3            # use every 3rd hour as a forecast origin (keeps training fast)
K_OPTIONS = (6, 9, 13)     # candidate numbers of features for feature selection

ORIGIN_COLS = ["anom_now", "anom_lag6", "anom_lag24", "humid_now", "wind_now", "precip24", "dpres6"]
FEATURES = [
    "anom_now_d1", "anom_now_d2", "anom_trend_d1", "anom_lag24_d2",
    "humid_d1", "wind_d1", "precip24_d1", "dpres6_d1",
    "tgt_hour_sin", "tgt_hour_cos", "tgt_doy_sin", "tgt_doy_cos", "horizon_scaled",
]


# ---------- preprocessing ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, make the hourly index continuous, fill short gaps (<=6h). Keeps remaining NaNs."""
    d = df.sort_values("time").drop_duplicates("time").set_index("time")
    d = d.asfreq(pd.offsets.Hour(1))
    d = d.interpolate(limit=6)
    return d.reset_index()


def build_climatology(df: pd.DataFrame) -> np.ndarray:
    """Mean temperature for each (month, hour) -> array of 12*24 values."""
    g = df.groupby([df["time"].dt.month, df["time"].dt.hour])["temp"].mean()
    arr = np.full(12 * 24, np.nan)
    for (m, h), v in g.items():
        arr[(m - 1) * 24 + h] = v
    return np.where(np.isnan(arr), np.nanmean(arr), arr)


def clim_at(clim: np.ndarray, times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return clim[(t.month - 1) * 24 + t.hour]


def origin_frame(df: pd.DataFrame, clim: np.ndarray) -> pd.DataFrame:
    """Features describing the weather 'right now' at every hour (look-back only)."""
    d = df.sort_values("time").reset_index(drop=True).copy()
    d["anom_now"] = d["temp"] - clim_at(clim, d["time"])
    d["anom_lag6"] = d["anom_now"].shift(6)
    d["anom_lag24"] = d["anom_now"].shift(24)
    d["humid_now"] = d["humid"]
    d["wind_now"] = d["wind"]
    d["precip24"] = d["precip"].rolling(24).sum()
    d["dpres6"] = d["pressure"] - d["pressure"].shift(6)
    return d


def horizon_features(o: pd.DataFrame, h, target_times) -> pd.DataFrame:
    """Combine origin features with lead time h and the target's hour/season."""
    n = len(o)
    h = np.broadcast_to(np.asarray(h, dtype=float), (n,))
    d1, d2 = np.exp(-h / 24), np.exp(-h / 72)
    tt = pd.DatetimeIndex(target_times)
    hr, doy = tt.hour.values, tt.dayofyear.values
    v = lambda c: o[c].values
    X = pd.DataFrame({
        "anom_now_d1": v("anom_now") * d1,
        "anom_now_d2": v("anom_now") * d2,
        "anom_trend_d1": (v("anom_now") - v("anom_lag6")) * d1,
        "anom_lag24_d2": v("anom_lag24") * d2,
        "humid_d1": v("humid_now") * d1,
        "wind_d1": v("wind_now") * d1,
        "precip24_d1": v("precip24") * d1,
        "dpres6_d1": v("dpres6") * d1,
        "tgt_hour_sin": np.sin(2 * np.pi * hr / 24),
        "tgt_hour_cos": np.cos(2 * np.pi * hr / 24),
        "tgt_doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "tgt_doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "horizon_scaled": h / MAX_HORIZON,
    })
    return X[FEATURES]


def make_dataset(d: pd.DataFrame, clim: np.ndarray) -> pd.DataFrame:
    """One row per (origin hour, lead time h): features + true future temperature."""
    base = d.dropna(subset=ORIGIN_COLS).iloc[::ORIGIN_STEP]
    temp = d.set_index("time")["temp"]
    parts = []
    for h in HORIZONS:
        tt = pd.DatetimeIndex(base["time"] + pd.Timedelta(hours=h))
        y = temp.reindex(tt).values
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        b, tt_ok = base[ok], tt[ok]
        X = horizon_features(b, h, tt_ok)
        X["h"] = h
        X["origin"] = b["time"].values
        X["temp_target"] = y[ok]
        X["clim_target"] = clim_at(clim, tt_ok)
        X["temp_now"] = b["temp"].values
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


# ---------- training & evaluation ----------
def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def evaluate(model, test: pd.DataFrame) -> pd.DataFrame:
    """MAE (°C) by lead time: our model vs two naive baselines."""
    t = test.assign(pred=test["clim_target"] + model.predict(test[FEATURES]))
    rows = []
    for h in [None] + REPORT_HORIZONS:
        s = t if h is None else t[t["h"] == h]
        if s.empty:
            continue
        rows.append({
            "Lead time": "All (1h-7d)" if h is None else f"+{h}h",
            "Our model": _mae(s["temp_target"], s["pred"]),
            "Climatology only": _mae(s["temp_target"], s["clim_target"]),
            "Persistence (= now)": _mae(s["temp_target"], s["temp_now"]),
        })
    return pd.DataFrame(rows)


def _pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_regression, k=k)),   # keep the k most informative features
        ("lr", LinearRegression()),
    ])


def train_city(df_raw: pd.DataFrame) -> dict:
    """Train + evaluate a model for one city. Returns everything the app needs."""
    df = clean(df_raw)
    split_time = df["time"].iloc[int(len(df) * 0.8)]

    clim = build_climatology(df[df["time"] < split_time])      # climatology from training period only
    d = origin_frame(df, clim)
    data = make_dataset(d, clim)

    data["target_time"] = data["origin"] + pd.to_timedelta(data["h"], unit="h")
    train = data[data["target_time"] < split_time]              # train: everything strictly before split
    test = data[data["origin"] >= split_time]                   # test: forecasts made after the split

    # ---- feature selection: SelectKBest (F-test), K chosen on a validation slice of the training period ----
    cut = train["origin"].sort_values().iloc[int(len(train) * 0.8)]
    fit_part = train[train["target_time"] < cut]
    val_part = train[train["origin"] >= cut]
    y_fit = fit_part["temp_target"] - fit_part["clim_target"]
    best_k, best_err = K_OPTIONS[-1], np.inf
    for k in K_OPTIONS:                                         # ascending: ties keep the simpler model
        p = _pipeline(k).fit(fit_part[FEATURES], y_fit)
        err = _mae(val_part["temp_target"], val_part["clim_target"] + p.predict(val_part[FEATURES]))
        if err < best_err - 1e-6:
            best_k, best_err = k, err

    model = _pipeline(best_k).fit(train[FEATURES], train["temp_target"] - train["clim_target"])
    sel = model.named_steps["select"]
    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "F-score": np.round(sel.scores_, 1),
        "Selected": sel.get_support(),
    }).sort_values("F-score", ascending=False).reset_index(drop=True)

    years = max(len(df) / 8766, 0.5)
    monthly = pd.DataFrame({
        "Avg temperature (°C)": df.groupby(df["time"].dt.month)["temp"].mean(),
        "Avg rainfall (mm/month)": df.groupby(df["time"].dt.month)["precip"].sum() / years,
    }).round(1)

    return {
        "model": model,
        "clim": clim,
        "metrics": evaluate(model, test),
        "feature_table": feature_table,
        "k": best_k,
        "monthly": monthly,
        "n_train": len(train),
        "n_test": len(test),
        "history": (str(df["time"].min().date()), str(df["time"].max().date())),
        "split": str(split_time.date()),
    }


# ---------- prediction ----------
def predict_many(bundle: dict, live_df: pd.DataFrame, now_local: pd.Timestamp, target_times):
    """Predict temperature at several future times from the latest observed hour.

    Returns (predictions, origin_time). Targets outside 1..168h of the origin are NaN.
    """
    d = origin_frame(clean(live_df), bundle["clim"])
    d = d[d["time"] <= now_local.floor("h")].dropna(subset=ORIGIN_COLS)
    if d.empty:
        raise ValueError("Not enough recent data to make a prediction for this location.")
    o = d.iloc[[-1]]
    origin_time = o["time"].iloc[0]

    tt = pd.DatetimeIndex(target_times)
    h = np.asarray((tt - origin_time) / pd.Timedelta(hours=1), dtype=float)
    valid = (h >= 1) & (h <= MAX_HORIZON)
    out = np.full(len(tt), np.nan)
    if valid.any():
        oo = pd.concat([o] * int(valid.sum()), ignore_index=True)
        X = horizon_features(oo, h[valid], tt[valid])
        out[valid] = clim_at(bundle["clim"], tt[valid]) + bundle["model"].predict(X)
    return out, origin_time
