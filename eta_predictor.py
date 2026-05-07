"""
eta_predictor.py — ML-based ETA prediction using Gradient Boosting.
Features: distance_remaining, hour, day_of_week, is_weekend, weather, avg_speed, load_pct
"""

import logging
import numpy as np
import os
import joblib
from typing import List, Optional

logger = logging.getLogger(__name__)

MODEL_DIR   = os.getenv("MODEL_DIR", "models/saved")
ETA_MODEL   = os.path.join(MODEL_DIR, "eta_regressor.pkl")
ETA_SCALER  = os.path.join(MODEL_DIR, "eta_scaler.pkl")

_eta_model  = None
_eta_scaler = None

WEATHER_MAP = {"clear": 0, "fog": 1, "rain": 2, "extreme": 3}


def load_eta_model():
    global _eta_model, _eta_scaler
    try:
        if os.path.exists(ETA_MODEL) and os.path.exists(ETA_SCALER):
            _eta_model  = joblib.load(ETA_MODEL)
            _eta_scaler = joblib.load(ETA_SCALER)
            logger.info("✅  ETA predictor loaded from disk")
        else:
            _build_and_save_default_eta_model()
    except Exception as e:
        logger.warning(f"ETA model load failed: {e} — building default")
        _build_and_save_default_eta_model()


def _build_and_save_default_eta_model():
    global _eta_model, _eta_scaler
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n = 5000

    # Synthetic features
    distance   = np.random.uniform(1, 30, n)        # km remaining
    hour       = np.random.randint(0, 24, n)
    dow        = np.random.randint(0, 7, n)
    is_weekend = ((dow >= 5).astype(float))
    weather    = np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.15, 0.2, 0.05])
    avg_speed  = np.random.uniform(15, 55, n)
    load_pct   = np.random.uniform(20, 100, n)

    # Target: time_min = (distance / speed) * 60 + weather_penalty + peak_penalty
    weather_penalty = weather * 3
    peak_mask = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
    peak_penalty = peak_mask.astype(float) * np.random.uniform(3, 10, n)
    time_min = (distance / avg_speed) * 60 + weather_penalty + peak_penalty + np.random.normal(0, 2, n)
    time_min = time_min.clip(1, 120)

    X = np.column_stack([distance, hour, dow, is_weekend, weather, avg_speed, load_pct])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_scaled, time_min)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  ETA_MODEL)
    joblib.dump(scaler, ETA_SCALER)
    _eta_model  = model
    _eta_scaler = scaler
    logger.info("✅  Default ETA model built and saved")


def predict_eta(
    distance_km:       float,
    hour:              int,
    day_of_week:       int,
    is_weekend:        bool = False,
    weather:           str  = "clear",
    avg_speed_kmh:     float = 30.0,
    passenger_load_pct: float = 60.0,
) -> dict:
    """
    Predict ETA in minutes for a bus given current conditions.

    Returns:
        {
            eta_minutes: float,
            eta_confidence: float,
            breakdown: { distance_time, traffic_penalty, weather_penalty }
        }
    """
    global _eta_model, _eta_scaler

    if _eta_model is None:
        load_eta_model()

    w_code = WEATHER_MAP.get(weather.lower(), 0)
    features = np.array([[
        distance_km, hour, day_of_week,
        float(is_weekend), w_code, avg_speed_kmh, passenger_load_pct,
    ]])

    try:
        X_scaled = _eta_scaler.transform(features)
        eta_min  = float(_eta_model.predict(X_scaled)[0])
        eta_min  = max(1.0, round(eta_min, 1))

        # Breakdown for display
        base_time = (distance_km / max(avg_speed_kmh, 5)) * 60
        traffic   = max(0, eta_min - base_time - w_code * 2)

        return {
            "eta_minutes":    eta_min,
            "eta_confidence": 0.82,     # fixed for synthetic model
            "breakdown": {
                "distance_time_min": round(base_time, 1),
                "traffic_penalty_min": round(traffic, 1),
                "weather_penalty_min": round(w_code * 2.0, 1),
            },
            "model": "GradientBoosting",
        }

    except Exception as e:
        logger.error(f"ETA prediction failed: {e}")
        # Haversine fallback
        eta_min = round((distance_km / max(avg_speed_kmh, 5)) * 60, 1)
        return {
            "eta_minutes":    eta_min,
            "eta_confidence": 0.5,
            "breakdown": {
                "distance_time_min": eta_min,
                "traffic_penalty_min": 0,
                "weather_penalty_min": 0,
            },
            "model": "haversine_fallback",
        }
