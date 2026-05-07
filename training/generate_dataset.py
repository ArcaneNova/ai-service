"""
generate_dataset.py
Generates a synthetic training dataset using real DTC route & stage CSVs.
Produces: data/demand_dataset.csv  and  data/delay_dataset.csv

Usage: python training/generate_dataset.py
"""

import os
import random
import math
import csv
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────

ROUTES_CSV = os.path.join(os.path.dirname(__file__), "../../routes.csv")
STAGES_CSV = os.path.join(os.path.dirname(__file__), "../../stages.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES_PER_ROUTE = 365   # one year of daily records per route
RANDOM_SEED       = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Helpers ────────────────────────────────────────────────────────────────

WEATHER_OPTIONS   = ["clear", "rain", "fog", "heatwave"]
WEATHER_WEIGHTS   = [0.65,    0.15,   0.12,  0.08]

HOUR_BASE_DEMAND  = {
    0:5, 1:3, 2:2, 3:2, 4:5,
    5:15, 6:40, 7:80, 8:120, 9:100, 10:60, 11:50,
    12:70, 13:65, 14:55, 15:60, 16:75, 17:110,
    18:130, 19:100, 20:70, 21:50, 22:35, 23:20,
}

WEATHER_DEMAND_FACTOR = {"clear":1.0, "rain":0.85, "fog":0.90, "heatwave":0.75}
WEATHER_DELAY_FACTOR  = {"clear":0.0, "rain":5.0,  "fog":4.0,  "heatwave":2.0}

DTC_HOLIDAYS_2024 = {
    "2024-01-26", "2024-03-25", "2024-04-14", "2024-04-17",
    "2024-05-23", "2024-08-15", "2024-10-02", "2024-10-12",
    "2024-11-15", "2024-12-25",
}

def is_holiday(date: datetime) -> bool:
    return date.strftime("%Y-%m-%d") in DTC_HOLIDAYS_2024

def is_weekend(date: datetime) -> bool:
    return date.weekday() >= 5

def random_weather() -> str:
    return random.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS)[0]

def demand_for(hour, weather, weekend, holiday, special_event, capacity):
    base   = HOUR_BASE_DEMAND.get(hour, 20)
    factor = WEATHER_DEMAND_FACTOR[weather]
    if weekend:       factor *= 0.75
    if holiday:       factor *= 0.60
    if special_event: factor *= 1.40
    # scale to route capacity
    scale = min(capacity, 60) / 60
    raw   = base * factor * scale
    noise = np.random.normal(0, raw * 0.10)
    return max(0, int(round(raw + noise)))

def delay_for(hour, weather, load_pct, duration_min, is_wknd):
    base = 0.0
    if hour in (8, 9, 17, 18, 19): base += 8
    if hour in (7, 16, 20):        base += 4
    base += WEATHER_DELAY_FACTOR[weather]
    if load_pct > 80:   base += 3
    if load_pct > 100:  base += 5
    if not is_wknd:     base += 2
    base += duration_min * 0.03
    noise = np.random.normal(0, 2.0)
    return max(0.0, round(base + noise, 1))


# ── Load real routes ────────────────────────────────────────────────────────

def load_routes():
    routes = []
    with open(ROUTES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                dist = float(row.get("distance_km", 10) or 10)
                stops= int(row.get("total_stages", 20) or 20)
                routes.append({
                    "url_route_id": row["url_route_id"],
                    "distance_km":  dist,
                    "total_stages": stops,
                })
            except (ValueError, KeyError):
                continue
    return routes


# ── Generate demand dataset ─────────────────────────────────────────────────

def generate_demand(routes):
    records = []
    start_date = datetime(2024, 1, 1)
    for route in routes:
        capacity = 60  # DTC standard bus capacity
        for d in range(SAMPLES_PER_ROUTE):
            date = start_date + timedelta(days=d)
            wknd = is_weekend(date)
            hday = is_holiday(date)
            wth  = random_weather()
            sevt = random.random() < 0.03   # 3% special events
            temp = np.random.normal(28 if date.month in range(4,10) else 18, 5)

            for hour in range(24):
                count = demand_for(hour, wth, wknd, hday, sevt, capacity)
                records.append({
                    "route_id":         route["url_route_id"],
                    "date":             date.strftime("%Y-%m-%d"),
                    "hour":             hour,
                    "day_of_week":      date.weekday(),
                    "is_weekend":       int(wknd),
                    "is_holiday":       int(hday),
                    "weather":          wth,
                    "avg_temp_c":       round(temp, 1),
                    "special_event":    int(sevt),
                    "month":            date.month,
                    "passenger_count":  count,
                    "crowd_level":      "low" if count<30 else "medium" if count<60 else "high" if count<90 else "critical",
                })

    df = pd.DataFrame(records)
    out = os.path.join(OUT_DIR, "demand_dataset.csv")
    df.to_csv(out, index=False)
    print(f"✅  Demand dataset: {len(df):,} rows → {out}")
    return df


# ── Generate delay dataset ──────────────────────────────────────────────────

def generate_delay(routes):
    records = []
    start_date = datetime(2024, 1, 1)
    for route in routes:
        duration = route["distance_km"] * 3  # ~3 min/km average
        for d in range(SAMPLES_PER_ROUTE):
            date = start_date + timedelta(days=d)
            wknd = is_weekend(date)
            hday = is_holiday(date)
            wth  = random_weather()
            temp = np.random.normal(28 if date.month in range(4,10) else 18, 5)

            for hour in range(5, 24):
                load_pct = random.uniform(20, 120)
                delay    = delay_for(hour, wth, load_pct, duration, wknd)
                records.append({
                    "route_id":                 route["url_route_id"],
                    "date":                     date.strftime("%Y-%m-%d"),
                    "hour":                     hour,
                    "day_of_week":              date.weekday(),
                    "is_weekend":               int(wknd),
                    "is_holiday":               int(hday),
                    "weather":                  wth,
                    "avg_temp_c":               round(temp, 1),
                    "passenger_load_pct":       round(load_pct, 1),
                    "scheduled_duration_min":   round(duration, 1),
                    "distance_km":              route["distance_km"],
                    "total_stops":              route["total_stages"],
                    "month":                    date.month,
                    "delay_minutes":            delay,
                    "is_delayed":               int(delay > 5),
                })

    df = pd.DataFrame(records)
    out = os.path.join(OUT_DIR, "delay_dataset.csv")
    df.to_csv(out, index=False)
    print(f"✅  Delay dataset: {len(df):,} rows → {out}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📊  Loading routes CSV…")
    routes = load_routes()
    print(f"    {len(routes)} routes loaded")

    # Use a sample for dev speed (remove slice for full dataset)
    sample = routes[:50]
    print(f"    Using {len(sample)} routes for dataset generation")

    print("\n⏳  Generating demand dataset…")
    generate_demand(sample)

    print("\n⏳  Generating delay dataset…")
    generate_delay(sample)

    print("\n🎉  Done! Datasets saved to ai-service/data/")
