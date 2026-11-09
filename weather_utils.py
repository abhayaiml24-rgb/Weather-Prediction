"""API helpers: city search (geocoding) + hourly weather (history and live).

All data comes from Open-Meteo (free, no API key).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo name -> short name used in this project
HOURLY_VARS = {
    "temperature_2m": "temp",           # °C
    "relative_humidity_2m": "humid",    # %
    "surface_pressure": "pressure",     # hPa
    "wind_speed_10m": "wind",           # km/h
    "precipitation": "precip",          # mm
}
TRAIN_YEARS = 3


def _get(url: str, params: dict, timeout: int):
    """requests.get with retries: free APIs sometimes drop connections or rate-limit."""
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def search_cities(query: str, count: int = 8) -> list[dict]:
    """Find places by name. Returns [{label, lat, lon}, ...]."""
    if len(query.strip()) < 2:
        return []
    r = _get(
        GEOCODE_URL,
        params={"name": query.strip(), "count": count, "language": "en", "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    out = []
    for x in r.json().get("results") or []:
        label = ", ".join(p for p in (x.get("name"), x.get("admin1"), x.get("country")) if p)
        out.append({"label": label, "lat": x["latitude"], "lon": x["longitude"]})
    return out


def _to_frame(payload: dict) -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame(payload["hourly"]).rename(columns=HOURLY_VARS)
    df["time"] = pd.to_datetime(df["time"])  # city-local time (timezone=auto)
    return df, payload.get("timezone", "UTC")


def fetch_archive(lat: float, lon: float, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Hourly historical weather between two ISO dates (local time of the city)."""
    r = _get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY_VARS), "timezone": "auto",
        },
        timeout=90,
    )
    r.raise_for_status()
    return _to_frame(r.json())


def fetch_training_history(lat: float, lon: float) -> tuple[pd.DataFrame, str]:
    """~3 years of hourly history ending a few days ago (the archive lags slightly)."""
    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=365 * TRAIN_YEARS)
    return fetch_archive(lat, lon, start.isoformat(), end.isoformat())


def fetch_live(lat: float, lon: float, past_days: int = 7, forecast_days: int = 8) -> tuple[pd.DataFrame, str]:
    """Last week of hourly data + Open-Meteo's own forecast for the next 7 days."""
    r = _get(
        FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(HOURLY_VARS), "timezone": "auto",
            "past_days": past_days, "forecast_days": forecast_days,
        },
        timeout=30,
    )
    r.raise_for_status()
    return _to_frame(r.json())