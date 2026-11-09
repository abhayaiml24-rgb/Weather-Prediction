"""Streamlit app.  Run:  streamlit run app.py"""
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from model import MAX_HORIZON, predict_many, train_city
from weather_utils import (fetch_archive, fetch_live, fetch_training_history,
                           search_cities)

st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="wide")


def show_chart(df: pd.DataFrame) -> None:
    """Line chart whose y-axis fits the data (st.line_chart always starts at 0)."""
    long = df.rename_axis("time").reset_index().melt("time", var_name="Series", value_name="temp").dropna()
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("time:T", title=None),
            y=alt.Y("temp:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None)),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str):
    return search_cities(query)


@st.cache_resource(show_spinner=False)
def cached_bundle(lat: float, lon: float):
    df, _tz = fetch_training_history(lat, lon)
    return train_city(df)


@st.cache_data(ttl=600, show_spinner=False)
def cached_live(lat: float, lon: float):
    return fetch_live(lat, lon)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_day(lat: float, lon: float, day_iso: str):
    df, _tz = fetch_archive(lat, lon, day_iso, day_iso)
    return df


# ------------------------------ sidebar inputs ------------------------------
st.title("🌦️ Weather Predictor")
st.caption("Pick any city, date and time. A Linear Regression model is trained on that city's own history.")

with st.sidebar:
    st.header("Choose location & time")
    query = st.text_input("City", value="Bengaluru", help="Type a city name, e.g. Mumbai, London, Tokyo")
    try:
        matches = cached_search(query) if query else []
    except Exception as exc:
        st.error(f"City search failed: {exc}")
        st.stop()
    if not matches:
        st.warning("No matching place found. Try another spelling.")
        st.stop()
    idx = st.selectbox("Select location", list(range(len(matches))), format_func=lambda i: matches[i]["label"])
    place = matches[idx]
    sel_date = st.date_input(
        "Date", value=date.today() + timedelta(days=1),
        min_value=date(2015, 1, 1), max_value=date.today() + timedelta(days=MAX_HORIZON // 24),
    )
    sel_hour = st.selectbox("Time (city's local time)", list(range(24)), index=12, format_func=lambda h: f"{h:02d}:00")
    st.caption("Future dates: up to 7 days ahead (prediction). Past dates: observed data.")

# ------------------------------ data + model ------------------------------
try:
    with st.spinner(f"Training a model for {place['label']} (first time takes ~10-30 seconds)..."):
        bundle = cached_bundle(place["lat"], place["lon"])
        live, tz = cached_live(place["lat"], place["lon"])
except Exception as exc:
    st.error(f"Could not fetch weather data (the free API may be rate-limiting): {exc}")
    st.stop()

now_local = pd.Timestamp.now(tz=tz).tz_localize(None)
target = pd.Timestamp(sel_date) + pd.Timedelta(hours=sel_hour)
day_times = pd.date_range(pd.Timestamp(sel_date), periods=24, freq="h")
live_temp = live.set_index("time")["temp"]

tab_pred, tab_acc, tab_climate = st.tabs(["Prediction", "Model accuracy", "Climate"])

# ------------------------------ prediction tab ------------------------------
with tab_pred:
    st.subheader(f"{place['label']}  •  {target:%a, %d %b %Y  %H:%M}  (local time: now {now_local:%d %b %H:%M})")

    if target <= now_local.floor("h"):
        # ---- past / current hour: show observed data ----
        if target >= live_temp.index.min():
            series = live_temp.reindex(day_times)
        else:
            try:
                day_df = cached_day(place["lat"], place["lon"], sel_date.isoformat())
                series = day_df.set_index("time")["temp"].reindex(day_times)
            except Exception as exc:
                st.error(f"Could not load data for that date: {exc}")
                st.stop()
        value = series.get(target)
        c1, c2 = st.columns(2)
        c1.metric("Observed temperature", "n/a" if pd.isna(value) else f"{value:.1f} °C")
        c2.metric("Day range", f"{series.min():.1f} to {series.max():.1f} °C")
        show_chart(series.rename("Observed temperature").to_frame())
        st.caption("This time is in the past, so no prediction is needed. Choose a future time to see the model's forecast.")
    else:
        # ---- future: model prediction vs Open-Meteo forecast ----
        try:
            preds, origin = predict_many(bundle, live, now_local, day_times)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        pred = float(pd.Series(preds, index=day_times).get(target))
        om = live_temp.get(target)
        lead = int((target - origin) / pd.Timedelta(hours=1))

        c1, c2, c3 = st.columns(3)
        if pd.isna(pred):
            c1.metric("Our prediction", "n/a")
            st.warning("This time is beyond the model's 7-day range.")
        else:
            delta = None if (om is None or pd.isna(om)) else f"{pred - om:+.1f} °C vs Open-Meteo"
            c1.metric("Our model's prediction", f"{pred:.1f} °C", delta)
        c2.metric("Open-Meteo forecast (for comparison)", "n/a" if om is None or pd.isna(om) else f"{om:.1f} °C")
        c3.metric("Lead time", f"{lead} h")

        chart = pd.DataFrame({
            "Open-Meteo forecast": live_temp.reindex(day_times),
            "Our model": preds,
        }, index=day_times)
        show_chart(chart)
        if lead > 72:
            st.caption("⚠️ Beyond ~3 days, a simple model has little skill over the city's average climate. See the accuracy tab.")
        st.caption(f"Prediction made from the latest observed hour ({origin:%d %b %H:%M} local).")

# ------------------------------ accuracy tab ------------------------------
with tab_acc:
    st.write(
        f"Trained on {bundle['n_train']:,} forecast examples from {bundle['history'][0]} to {bundle['split']}. "
        f"Tested on {bundle['n_test']:,} forecasts made **after** {bundle['split']} (data the model never saw)."
    )
    m = bundle["metrics"]
    st.dataframe(m.round(2), hide_index=True)
    st.caption("Mean Absolute Error in °C, lower is better. A useful model should beat both baselines.")
    sub = m[m["Lead time"] != "All (1h-7d)"].rename(columns={"Lead time": "lead"})
    order = sub["lead"].tolist()
    mae_long = sub.melt("lead", var_name="Series", value_name="MAE")
    st.altair_chart(
        alt.Chart(mae_long).mark_line(point=True).encode(
            x=alt.X("lead:N", sort=order, title="Lead time"),
            y=alt.Y("MAE:Q", title="MAE (°C)"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None)),
        ).properties(height=300),
        use_container_width=True,
    )
    st.subheader("Feature selection")
    st.caption(f"SelectKBest (F-test) kept the {bundle['k']} most informative of {len(bundle['feature_table'])} features; "
               "the number was chosen by validation error on the training period.")
    st.dataframe(bundle["feature_table"], hide_index=True)
    st.info(
        "How it works: climatology (average for that month and hour) + a predicted anomaly. "
        "The anomaly is estimated by Linear Regression from current conditions, and its influence "
        "fades with lead time. Official forecasts (Open-Meteo, IMD) use physics-based weather models "
        "and are usually more accurate; this project demonstrates an ML pipeline, not a replacement."
    )

# ------------------------------ climate tab ------------------------------
with tab_climate:
    a, b = st.columns(2)
    a.subheader("Average temperature by month (°C)")
    a.bar_chart(bundle["monthly"]["Avg temperature (°C)"])
    b.subheader("Average rainfall by month (mm)")
    b.bar_chart(bundle["monthly"]["Avg rainfall (mm/month)"])
    st.caption(f"Based on hourly data from {bundle['history'][0]} to {bundle['history'][1]}.")