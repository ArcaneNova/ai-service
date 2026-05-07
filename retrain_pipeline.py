"""
retrain_pipeline.py — Automated multi-model retraining pipeline (v2).
Backs up & saves using the *_multimodel.* naming convention.
Triggered via POST /admin/retrain or run as a cron job.
"""

import logging
import os
import shutil
import time
from datetime import datetime
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR  = os.getenv("MODEL_DIR", "models/saved")
BACKUP_DIR = os.path.join(MODEL_DIR, "backups")

# All multi-model filenames that should be backed up before retraining
MULTIMODEL_FILES = [
    # Demand
    "demand_lstm_multimodel.keras",
    "demand_gru_multimodel.keras",
    "demand_transformer_multimodel.keras",
    "demand_xgboost_multimodel.pkl",
    "demand_lightgbm_multimodel.pkl",
    "demand_rf_multimodel.pkl",
    "demand_scaler_multimodel.pkl",
    "demand_comparison_report.json",
    # Delay
    "delay_mlp_multimodel.keras",
    "delay_xgboost_multimodel.pkl",
    "delay_lightgbm_multimodel.pkl",
    "delay_catboost_multimodel.pkl",
    "delay_svr_multimodel.pkl",
    "delay_ensemble_multimodel.pkl",
    "delay_scaler_multimodel.pkl",
    "delay_comparison_report.json",
    # Anomaly
    "anomaly_autoencoder_multimodel.keras",
    "anomaly_isolation_forest_multimodel.pkl",
    "anomaly_lof_multimodel.pkl",
    "anomaly_ocsvm_multimodel.pkl",
    "anomaly_dbscan_multimodel.pkl",
    "anomaly_scaler_multimodel.pkl",
    "anomaly_ae_threshold.pkl",
    "anomaly_comparison_report.json",
    # ETA
    "eta_regressor.pkl",
    "eta_scaler.pkl",
]


def _backup_models() -> str:
    """Backup all multimodel files before retraining."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(backup_path, exist_ok=True)

    backed = []
    for fname in MULTIMODEL_FILES:
        src = os.path.join(MODEL_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(backup_path, fname)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            backed.append(fname)

    logger.info(f"✅  Backed up {len(backed)} model files to {backup_path}")
    return backup_path


def _retrain_demand_xgboost(data_path: Optional[str] = None) -> dict:
    """Retrain XGBoost demand model on fresh data → demand_xgboost_multimodel.pkl"""
    import joblib
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler

    logger.info("🔄  Retraining demand XGBoost…")
    start = time.time()

    if data_path and os.path.exists(data_path):
        import pandas as pd
        df   = pd.read_csv(data_path)
        cols = ["hour", "is_weekend", "is_holiday", "weather_factor",
                "avg_temp_c", "special_event", "sin_hour", "cos_hour",
                "day_of_week", "month", "quarter"]
        cols = [c for c in cols if c in df.columns]
        X    = df[cols].values
        y    = df["passenger_count"].values
    else:
        np.random.seed(int(time.time()) % 1000)
        n    = 5000
        hour = np.random.randint(0, 24, n)
        X    = np.column_stack([
            hour,
            np.random.randint(0, 2, n),  # is_weekend
            np.random.randint(0, 2, n),  # is_holiday
            np.random.uniform(0.75, 1.0, n),  # weather_factor
            np.random.uniform(15, 42, n),  # avg_temp_c
            np.random.randint(0, 2, n),  # special_event
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.zeros(n), np.zeros(n), np.zeros(n),  # pad to 11
        ])
        base = 30 + 80 * np.abs(np.sin(np.pi * hour / 24))
        y    = np.maximum(0, base + np.random.normal(0, 15, n))

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1)
    model.fit(X_s, y)

    tmp_m = os.path.join(MODEL_DIR, "_tmp_demand_xgboost.pkl")
    tmp_s = os.path.join(MODEL_DIR, "_tmp_demand_scaler.pkl")
    joblib.dump(model,  tmp_m)
    joblib.dump(scaler, tmp_s)
    os.replace(tmp_m, os.path.join(MODEL_DIR, "demand_xgboost_multimodel.pkl"))
    os.replace(tmp_s, os.path.join(MODEL_DIR, "demand_scaler_multimodel.pkl"))

    elapsed = round(time.time() - start, 1)
    logger.info(f"✅  Demand XGBoost retrained in {elapsed}s on {len(X)} samples")
    return {"model": "demand_xgboost_multimodel", "elapsed_sec": elapsed, "samples": len(X)}


def _retrain_delay_xgboost(data_path: Optional[str] = None) -> dict:
    """Retrain XGBoost delay model → delay_xgboost_multimodel.pkl"""
    import joblib
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler

    logger.info("🔄  Retraining delay XGBoost…")
    start = time.time()

    if data_path and os.path.exists(data_path):
        import pandas as pd
        df   = pd.read_csv(data_path)
        cols = ["hour", "day_of_week", "is_weekend", "is_holiday",
                "weather_factor", "avg_temp_c", "passenger_load_pct",
                "scheduled_duration_min", "distance_km", "total_stops",
                "sin_hour", "cos_hour", "sin_day", "cos_day"]
        cols = [c for c in cols if c in df.columns]
        X    = df[cols].values
        y    = df["delay_minutes"].values
    else:
        np.random.seed(int(time.time()) % 1000)
        n    = 5000
        hour = np.random.randint(0, 24, n)
        dow  = np.random.randint(0, 7, n)
        X    = np.column_stack([
            hour, dow,
            np.random.randint(0, 2, n),
            np.random.randint(0, 2, n),
            np.random.uniform(0.75, 1.0, n),
            np.random.uniform(15, 42, n),
            np.random.uniform(20, 100, n),
            np.random.uniform(30, 90, n),
            np.random.uniform(5, 30, n),
            np.random.randint(5, 40, n),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ])
        y = np.maximum(0, 3 + X[:, 4] * 4 + X[:, 6] / 25 + np.random.exponential(3, n))

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1)
    model.fit(X_s, y)

    tmp_m = os.path.join(MODEL_DIR, "_tmp_delay_xgboost.pkl")
    tmp_s = os.path.join(MODEL_DIR, "_tmp_delay_scaler.pkl")
    joblib.dump(model,  tmp_m)
    joblib.dump(scaler, tmp_s)
    os.replace(tmp_m, os.path.join(MODEL_DIR, "delay_xgboost_multimodel.pkl"))
    os.replace(tmp_s, os.path.join(MODEL_DIR, "delay_scaler_multimodel.pkl"))

    elapsed = round(time.time() - start, 1)
    logger.info(f"✅  Delay XGBoost retrained in {elapsed}s on {len(X)} samples")
    return {"model": "delay_xgboost_multimodel", "elapsed_sec": elapsed, "samples": len(X)}


def _retrain_anomaly(data_path: Optional[str] = None) -> dict:
    """Retrain Isolation Forest anomaly detector → anomaly_isolation_forest_multimodel.pkl"""
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    logger.info("🔄  Retraining anomaly Isolation Forest…")
    start = time.time()

    np.random.seed(int(time.time()) % 1000)
    n      = 6000
    speeds = np.random.uniform(5, 85, n)
    delays = np.random.exponential(4, n).clip(0, 30)
    loads  = np.random.uniform(10, 110, n)
    X      = np.column_stack([speeds, delays, loads])

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = IsolationForest(n_estimators=300, contamination=0.05,
                             random_state=42, n_jobs=-1)
    model.fit(X_s)

    tmp_m = os.path.join(MODEL_DIR, "_tmp_anomaly_iforest.pkl")
    tmp_s = os.path.join(MODEL_DIR, "_tmp_anomaly_scaler.pkl")
    joblib.dump(model,  tmp_m)
    joblib.dump(scaler, tmp_s)
    os.replace(tmp_m, os.path.join(MODEL_DIR, "anomaly_isolation_forest_multimodel.pkl"))
    os.replace(tmp_s, os.path.join(MODEL_DIR, "anomaly_scaler_multimodel.pkl"))

    elapsed = round(time.time() - start, 1)
    logger.info(f"✅  Anomaly IForest retrained in {elapsed}s on {n} samples")
    return {"model": "anomaly_isolation_forest_multimodel", "elapsed_sec": elapsed, "samples": n}


def run_retrain_pipeline(
    retrain_xgboost:  bool = True,
    retrain_lstm:     bool = False,
    retrain_anomaly:  bool = True,
    data_dir:         Optional[str] = None,
) -> dict:
    """
    Main entry point. Backs up current multimodel files, retrains selected models,
    reloads all models into memory. Returns a status report.
    """
    logger.info("═══════════════════════════════════════")
    logger.info("🚀  Multi-Model Retraining Pipeline Started")
    logger.info("═══════════════════════════════════════")
    start_total = time.time()
    results     = []
    errors      = []

    # 1. Backup
    backup_path = _backup_models()

    # 2. Retrain demand XGBoost
    if retrain_xgboost:
        try:
            d_path = os.path.join(data_dir, "demand_dataset.csv") if data_dir else None
            results.append(_retrain_demand_xgboost(d_path))
        except Exception as e:
            logger.error(f"Demand XGBoost retrain failed: {e}")
            errors.append({"model": "demand_xgboost", "error": str(e)})

        try:
            l_path = os.path.join(data_dir, "delay_dataset.csv") if data_dir else None
            results.append(_retrain_delay_xgboost(l_path))
        except Exception as e:
            logger.error(f"Delay XGBoost retrain failed: {e}")
            errors.append({"model": "delay_xgboost", "error": str(e)})

    # 3. Retrain anomaly
    if retrain_anomaly:
        try:
            results.append(_retrain_anomaly())
        except Exception as e:
            logger.error(f"Anomaly retrain failed: {e}")
            errors.append({"model": "anomaly", "error": str(e)})

    # 4. Reload all models into memory
    try:
        import model_loader
        model_loader.load_models()
        from anomaly_detector import load_anomaly_model
        load_anomaly_model()
        from eta_predictor import load_eta_model
        load_eta_model()
        results.append({"model": "reload", "status": "success"})
        logger.info("✅  All models reloaded into memory")
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        errors.append({"model": "reload", "error": str(e)})

    total = round(time.time() - start_total, 1)
    logger.info(f"═══ Pipeline done in {total}s — {len(results)} succeeded, {len(errors)} errors ═══")

    return {
        "status":             "completed" if not errors else "partial",
        "started_at":         datetime.now().isoformat(),
        "total_elapsed_sec":  total,
        "models_retrained":   results,
        "errors":             errors,
        "backup_path":        backup_path,
    }
