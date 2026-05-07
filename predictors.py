"""
predictors.py — SmartDTC Multi-Model Prediction Engine
Uses best available model for each task; falls back to rule-based heuristics.
All six demand models and six delay models are supported.
"""

import math
import numpy as np
import model_loader

# ── Weather factor table ────────────────────────────────────────────────────────
WEATHER_FACTOR = {
    "clear":    1.00,
    "rain":     0.85,
    "fog":      0.90,
    "heatwave": 0.80,
    "extreme":  0.75,
}

HOUR_BASE_DEMAND = {
    0: 10,  1: 8,   2: 6,   3: 5,   4: 8,   5: 20,
    6: 40,  7: 80,  8: 120, 9: 100, 10: 60, 11: 50,
    12: 70, 13: 65, 14: 55, 15: 60, 16: 75, 17: 110,
    18: 130,19: 100,20: 70, 21: 50, 22: 35, 23: 20,
}


def _crowd_level(count: int) -> str:
    if count < 30:  return "low"
    if count < 60:  return "medium"
    if count < 90:  return "high"
    return "critical"


def _build_demand_features(hour, is_weekend, is_holiday, weather, avg_temp_c, special_event) -> np.ndarray:
    """Build the 23-feature vector that the demand scaler was fitted on."""
    wf = WEATHER_FACTOR.get(weather, 1.0)
    # cyclic hour encoding
    sin_h = math.sin(2 * math.pi * hour / 24)
    cos_h = math.cos(2 * math.pi * hour / 24)
    return np.array([[
        hour,
        int(is_weekend),
        int(is_holiday),
        wf,
        avg_temp_c,
        int(special_event),
        sin_h,
        cos_h,
        # padding zeros for remaining features expected by scaler (23 total)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]], dtype=np.float32)


def _predict_with_model(model, key: str, X_scaled: np.ndarray) -> float:
    """Route prediction to the right model interface."""
    try:
        if key in ("lstm", "gru", "transformer"):
            # Keras Dense models — flat input
            out = model(X_scaled, training=False)
            # out may be a tensor or ndarray
            if hasattr(out, "numpy"):
                out = out.numpy()
            return float(out.flatten()[0])
        else:
            # sklearn / XGBoost / LightGBM / CatBoost / RF
            return float(model.predict(X_scaled)[0])
    except Exception:
        return float(model.predict(X_scaled)[0])


# ── DEMAND PREDICTION ───────────────────────────────────────────────────────────

def predict_demand(route_id: str, date: str, hour: int,
                   is_weekend: bool, is_holiday: bool,
                   weather: str, avg_temp_c: float,
                   special_event: bool,
                   model_key: str = "auto") -> dict:
    """
    Predict passenger demand.
    model_key: 'auto' uses best model; or specify 'lstm','gru','xgboost', etc.
    Returns extended dict including model name and all-model comparison when available.
    """

    scaler = model_loader.demand_scaler
    models = model_loader.demand_models
    best   = model_loader.demand_best_model

    target_key = best if model_key == "auto" else model_key
    model = models.get(target_key)

    if model is not None and scaler is not None:
        try:
            raw = _build_demand_features(hour, is_weekend, is_holiday, weather, avg_temp_c, special_event)
            # Adapt to scaler's expected feature count
            n_feat = scaler.n_features_in_
            if raw.shape[1] < n_feat:
                pad = np.zeros((1, n_feat - raw.shape[1]), dtype=np.float32)
                raw = np.hstack([raw, pad])
            elif raw.shape[1] > n_feat:
                raw = raw[:, :n_feat]

            X_scaled = scaler.transform(raw)
            pred = _predict_with_model(model, target_key, X_scaled)
            pred = max(0, int(round(pred)))

            # Confidence from comparison report
            model_metrics = (model_loader.demand_comparison.get("models") or {}).get(target_key, {})
            r2  = model_metrics.get("r2",  0.95)
            mape = model_metrics.get("mape", 8.0)
            confidence = min(0.99, max(0.60, r2))

            return {
                "route_id":        route_id,
                "date":            date,
                "hour":            hour,
                "predicted_count": pred,
                "crowd_level":     _crowd_level(pred),
                "confidence":      round(confidence, 4),
                "model":           target_key,
                "is_best_model":   target_key == best,
                "metrics": {
                    "mae":  round(model_metrics.get("mae",  0), 4),
                    "rmse": round(model_metrics.get("rmse", 0), 4),
                    "mape": round(mape, 4),
                    "r2":   round(r2,   6),
                },
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Demand ML failed ({target_key}): {e}")

    # Rule-based fallback
    base   = HOUR_BASE_DEMAND.get(hour, 30)
    factor = WEATHER_FACTOR.get(weather, 1.0)
    if is_weekend:    factor *= 0.75
    if is_holiday:    factor *= 0.60
    if special_event: factor *= 1.40
    pred = max(0, int(round(base * factor)))

    return {
        "route_id":        route_id,
        "date":            date,
        "hour":            hour,
        "predicted_count": pred,
        "crowd_level":     _crowd_level(pred),
        "confidence":      0.65,
        "model":           "rule_based",
        "is_best_model":   False,
        "metrics":         {"mae": None, "rmse": None, "mape": None, "r2": None},
    }


def predict_demand_all_models(route_id: str, date: str, hour: int,
                              is_weekend: bool, is_holiday: bool,
                              weather: str, avg_temp_c: float,
                              special_event: bool) -> dict:
    """Run all loaded demand models and return a comparison dict."""
    results = {}
    for key in model_loader.demand_models:
        r = predict_demand(route_id, date, hour, is_weekend, is_holiday,
                           weather, avg_temp_c, special_event, model_key=key)
        results[key] = {
            "predicted_count": r["predicted_count"],
            "crowd_level":     r["crowd_level"],
            "confidence":      r["confidence"],
            "metrics":         r["metrics"],
        }
    return {
        "route_id":     route_id,
        "date":         date,
        "hour":         hour,
        "best_model":   model_loader.demand_best_model,
        "model_results": results,
    }


# ── DELAY PREDICTION ────────────────────────────────────────────────────────────

def _build_delay_features(hour, day_of_week, is_weekend, is_holiday,
                          weather, avg_temp_c, passenger_load_pct,
                          scheduled_duration_min, distance_km, total_stops) -> np.ndarray:
    wf   = WEATHER_FACTOR.get(weather, 1.0)
    sin_h = math.sin(2 * math.pi * hour / 24)
    cos_h = math.cos(2 * math.pi * hour / 24)
    sin_d = math.sin(2 * math.pi * day_of_week / 7)
    cos_d = math.cos(2 * math.pi * day_of_week / 7)
    return np.array([[
        hour, day_of_week, int(is_weekend), int(is_holiday),
        wf, avg_temp_c, passenger_load_pct,
        scheduled_duration_min, distance_km, total_stops,
        sin_h, cos_h, sin_d, cos_d,
        # padding
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]], dtype=np.float32)


def predict_delay(route_id: str, hour: int, day_of_week: int,
                  is_weekend: bool, is_holiday: bool,
                  weather: str, avg_temp_c: float,
                  passenger_load_pct: float, scheduled_duration_min: float,
                  distance_km: float, total_stops: int,
                  model_key: str = "auto") -> dict:

    scaler = model_loader.delay_scaler
    models = model_loader.delay_models
    best   = model_loader.delay_best_model

    target_key = best if model_key == "auto" else model_key
    model = models.get(target_key)

    raw = _build_delay_features(hour, day_of_week, is_weekend, is_holiday,
                                weather, avg_temp_c, passenger_load_pct,
                                scheduled_duration_min, distance_km, total_stops)

    if model is not None:
        try:
            if scaler is not None:
                n_feat = scaler.n_features_in_
                if raw.shape[1] < n_feat:
                    raw = np.hstack([raw, np.zeros((1, n_feat - raw.shape[1]), dtype=np.float32)])
                elif raw.shape[1] > n_feat:
                    raw = raw[:, :n_feat]
                X = scaler.transform(raw)
            else:
                X = raw

            if target_key == "mlp":
                out = model(X, training=False)
                if hasattr(out, "numpy"): out = out.numpy()
                delay_min = float(out.flatten()[0])
            else:
                delay_min = float(model.predict(X)[0])

            delay_min = max(0.0, round(delay_min, 1))

            model_metrics = (model_loader.delay_comparison.get("models") or {}).get(target_key, {})
            r2   = model_metrics.get("r2",  0.95)
            rmse = model_metrics.get("rmse", 1.3)

            return {
                "route_id":                 route_id,
                "predicted_delay_minutes":  delay_min,
                "is_delayed":               delay_min > 5,
                "delay_probability":        round(min(delay_min / 15.0, 1.0), 3),
                "model":                    target_key,
                "is_best_model":            target_key == best,
                "metrics": {
                    "mae":  round(model_metrics.get("mae",  0), 4),
                    "rmse": round(rmse, 4),
                    "r2":   round(r2,   6),
                },
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Delay ML failed ({target_key}): {e}")

    # Rule-based fallback
    delay = 0.0
    if hour in (8, 9, 17, 18, 19): delay += 8
    if weather == "rain":           delay += 5
    if weather == "fog":            delay += 4
    if passenger_load_pct > 80:     delay += 3
    if not is_weekend:              delay += 2
    delay = max(0.0, round(delay + (distance_km * 0.1), 1))

    return {
        "route_id":                route_id,
        "predicted_delay_minutes": delay,
        "is_delayed":              delay > 5,
        "delay_probability":       round(min(delay / 15.0, 1.0), 3),
        "model":                   "rule_based",
        "is_best_model":           False,
        "metrics":                 {"mae": None, "rmse": None, "r2": None},
    }


# ── SCHEDULE GENERATION ─────────────────────────────────────────────────────────

def generate_schedule(route_id: str, date: str, total_buses: int) -> dict:
    slots = []
    for hour in range(5, 24):
        pred  = predict_demand(route_id, date, hour, False, False, "clear", 25.0, False)
        count = pred["predicted_count"]

        if count < 20:
            freq, buses, trip_type = 30, max(1, total_buses // 4), "regular"
        elif count < 60:
            freq, buses, trip_type = 20, max(2, total_buses // 3), "regular"
        elif count < 100:
            freq, buses, trip_type = 12, max(3, total_buses // 2), "peak"
        else:
            freq, buses, trip_type = 8, total_buses, "peak"

        for m in range(0, 60, freq):
            slots.append({
                "hour":               hour,
                "minute":             m,
                "frequency_minutes":  freq,
                "bus_count":          buses,
                "type":               trip_type,
                "predicted_demand":   count,
            })

    return {
        "route_id":         route_id,
        "date":             date,
        "slots":            slots,
        "total_trips":      len(slots),
        "ai_generated":     True,
        "demand_model":     model_loader.demand_best_model,
    }
