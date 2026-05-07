"""
model_loader.py — SmartDTC Multi-Model Loader
Loads ALL trained models (demand×6, delay×6, anomaly×6) at startup.
Falls back gracefully when any individual model file is absent.
"""

import os
import json
import logging
import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models/saved")

# ── DEMAND models ──────────────────────────────────────────────────────────────
demand_models: dict = {}          # key → sklearn/keras model
demand_scaler = None
demand_comparison: dict = {}      # loaded from demand_comparison_report.json
demand_best_model: str = "xgboost"  # key of best model (lowest MAE)

# ── DELAY models ───────────────────────────────────────────────────────────────
delay_models: dict = {}
delay_scaler = None
delay_comparison: dict = {}
delay_best_model: str = "xgboost"

# ── ANOMALY models ─────────────────────────────────────────────────────────────
anomaly_models: dict = {}
anomaly_scaler = None
anomaly_comparison: dict = {}
anomaly_best_model: str = "autoencoder"   # best F1
anomaly_ae_threshold: float = 0.009       # reconstruction threshold

# Legacy aliases (kept for backward-compat)
demand_model  = None   # points to best demand model
delay_model   = None   # points to best delay model
delay_clf     = None   # kept for compat (None now)


def _p(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)


def _load_keras(path: str, label: str):
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(path)
        logger.info(f"✅  Keras {label} loaded from {path}")
        return m
    except Exception as e:
        logger.warning(f"⚠️   Keras {label} load failed: {e}")
        return None


def _load_pkl(path: str, label: str):
    try:
        m = joblib.load(path)
        logger.info(f"✅  {label} loaded from {path}")
        return m
    except Exception as e:
        logger.warning(f"⚠️   {label} load failed: {e}")
        return None


def _load_report(filename: str) -> dict:
    path = _p(filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_models():
    global demand_models, demand_scaler, demand_comparison, demand_best_model, demand_model
    global delay_models, delay_scaler, delay_comparison, delay_best_model, delay_model
    global anomaly_models, anomaly_scaler, anomaly_comparison, anomaly_best_model, anomaly_ae_threshold

    logger.info("═══════════════════════════════════════")
    logger.info("🔄  Loading SmartDTC multi-model suite")
    logger.info("═══════════════════════════════════════")

    # ── Scalers ────────────────────────────────────────────────────────────────
    demand_scaler  = _load_pkl(_p("demand_scaler_multimodel.pkl"), "Demand scaler")
    delay_scaler   = _load_pkl(_p("delay_scaler_multimodel.pkl"),  "Delay scaler")
    anomaly_scaler = _load_pkl(_p("anomaly_scaler_multimodel.pkl"),"Anomaly scaler")

    # ── DEMAND models ─────────────────────────────────────────────────────────
    keras_demand = {
        "lstm":        "demand_lstm_multimodel.keras",
        "gru":         "demand_gru_multimodel.keras",
        "transformer": "demand_transformer_multimodel.keras",
    }
    pkl_demand = {
        "xgboost":      "demand_xgboost_multimodel.pkl",
        "lightgbm":     "demand_lightgbm_multimodel.pkl",
        "random_forest":"demand_rf_multimodel.pkl",
    }
    for key, fname in keras_demand.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_keras(path, f"Demand-{key}")
            if m: demand_models[key] = m
    for key, fname in pkl_demand.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_pkl(path, f"Demand-{key}")
            if m: demand_models[key] = m

    demand_comparison = _load_report("demand_comparison_report.json")
    if demand_comparison.get("models"):
        # Best = lowest MAE among loaded models
        loaded_keys = set(demand_models.keys())
        candidates = {k: v for k, v in demand_comparison["models"].items()
                      if k in loaded_keys and "error" not in v}
        if candidates:
            demand_best_model = min(candidates, key=lambda k: candidates[k].get("mae", 9999))
    # Point legacy alias to best
    demand_model = demand_models.get(demand_best_model)
    logger.info(f"🏆  Best demand model: {demand_best_model} "
                f"(loaded {len(demand_models)}/{len(keras_demand)+len(pkl_demand)} models)")

    # ── DELAY models ──────────────────────────────────────────────────────────
    keras_delay = {
        "mlp": "delay_mlp_multimodel.keras",
    }
    pkl_delay = {
        "xgboost":  "delay_xgboost_multimodel.pkl",
        "lightgbm": "delay_lightgbm_multimodel.pkl",
        "catboost": "delay_catboost_multimodel.pkl",
        "svr":      "delay_svr_multimodel.pkl",
        "ensemble": "delay_ensemble_multimodel.pkl",
    }
    for key, fname in keras_delay.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_keras(path, f"Delay-{key}")
            if m: delay_models[key] = m
    for key, fname in pkl_delay.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_pkl(path, f"Delay-{key}")
            if m: delay_models[key] = m

    delay_comparison = _load_report("delay_comparison_report.json")
    if delay_comparison.get("models"):
        loaded_keys = set(delay_models.keys())
        candidates = {k: v for k, v in delay_comparison["models"].items()
                      if k in loaded_keys and "error" not in v}
        if candidates:
            delay_best_model = min(candidates, key=lambda k: candidates[k].get("mae", 9999))
    delay_model = delay_models.get(delay_best_model)
    logger.info(f"🏆  Best delay model: {delay_best_model} "
                f"(loaded {len(delay_models)}/{len(keras_delay)+len(pkl_delay)} models)")

    # ── ANOMALY models ────────────────────────────────────────────────────────
    keras_anomaly = {
        "autoencoder": "anomaly_autoencoder_multimodel.keras",
    }
    pkl_anomaly = {
        "isolation_forest": "anomaly_isolation_forest_multimodel.pkl",
        "lof":              "anomaly_lof_multimodel.pkl",
        "ocsvm":            "anomaly_ocsvm_multimodel.pkl",
        "dbscan":           "anomaly_dbscan_multimodel.pkl",
    }
    for key, fname in keras_anomaly.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_keras(path, f"Anomaly-{key}")
            if m: anomaly_models[key] = m
    for key, fname in pkl_anomaly.items():
        path = _p(fname)
        if os.path.exists(path):
            m = _load_pkl(path, f"Anomaly-{key}")
            if m: anomaly_models[key] = m

    # AE threshold
    thr_path = _p("anomaly_ae_threshold.pkl")
    if os.path.exists(thr_path):
        try:
            anomaly_ae_threshold = float(joblib.load(thr_path))
            logger.info(f"✅  Autoencoder threshold: {anomaly_ae_threshold:.6f}")
        except Exception as e:
            logger.warning(f"⚠️   AE threshold load failed: {e}")

    anomaly_comparison = _load_report("anomaly_comparison_report.json")
    if anomaly_comparison.get("models"):
        loaded_keys = set(anomaly_models.keys())
        candidates = {k: v for k, v in anomaly_comparison["models"].items()
                      if k in loaded_keys and "error" not in v}
        if candidates:
            anomaly_best_model = max(candidates, key=lambda k: candidates[k].get("f1", 0))
    logger.info(f"🏆  Best anomaly model: {anomaly_best_model} "
                f"(loaded {len(anomaly_models)}/{len(keras_anomaly)+len(pkl_anomaly)} models)")

    logger.info("═══════════════════════════════════════")
    logger.info(f"✅  Model suite ready | Demand:{len(demand_models)} "
                f"Delay:{len(delay_models)} Anomaly:{len(anomaly_models)}")
    logger.info("═══════════════════════════════════════")
