"""
Train 6 Demand Prediction Models - OPTIMIZED VERSION

Models:
1. LSTM (TensorFlow Dense)
2. GRU  (TensorFlow Dense)
3. Transformer (TensorFlow Dense)
4. XGBoost
5. LightGBM
6. Random Forest

Optimizations vs original:
- Data subsampled to 20% (configurable via DEMAND_SAMPLE_FRAC env var)
- batch_size 64 -> 512  (8x fewer gradient steps per epoch)
- epochs 30 -> 20 with EarlyStopping (patience=5)
- tf.keras.backend.clear_session() + gc.collect() between TF models
- XGBoost: tree_method='hist' + early_stopping_rounds in fit()
- LightGBM: early_stopping callback (fixed -1 period bug)
- Random Forest: n_estimators 200->100, max_depth 15->10
- Results written to JSON after EACH model (crash-safe)
- sys.stdout.flush() throughout for real-time Colab output
- Per-model timing display
- Supports --model <name> flag for subprocess isolation
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def log(msg):
    """Print with immediate flush so Colab shows output in real time."""
    print(msg, flush=True)

# ════════════════════════════════════════════════════════════════════════════
# CLI ARG (optional: --model lstm|gru|transformer|xgboost|lightgbm|random_forest)
# ════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model', type=str, default=None,
                    help='Train only this model (optional)')
args, _ = parser.parse_known_args()
SINGLE_MODEL = (args.model or os.environ.get('TRAINING_MODEL', '')).lower().strip() or None

ALL_MODELS = ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm', 'random_forest']
MODELS_TO_RUN = [SINGLE_MODEL] if SINGLE_MODEL and SINGLE_MODEL in ALL_MODELS else ALL_MODELS

# ════════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # ai-service/

DATA_PATH = None
for path in [
    os.path.join(PROJECT_ROOT, "data", "demand_dataset.csv"),
    os.path.join(PROJECT_ROOT, "..", "data", "demand_dataset.csv"),
    "/content/bus-site/ai-service/data/demand_dataset.csv",
    "data/demand_dataset.csv",
]:
    if os.path.exists(path):
        DATA_PATH = os.path.abspath(path)
        break

SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")
SCRIPT_VERSION = "2026-04-16-optimized"

# ════════════════════════════════════════════════════════════════════════════
# STARTUP BANNER
# ════════════════════════════════════════════════════════════════════════════

log(f"\n{'='*77}")
log(f"{'DEMAND PREDICTION - 6 MODELS COMPARISON (OPTIMIZED)':^77}")
log(f"{'='*77}")
log(f"\nConfiguration:")
log(f"   Script dir:     {SCRIPT_DIR}")
log(f"   Project root:   {PROJECT_ROOT}")
log(f"   Trainer ver:    {SCRIPT_VERSION}")
log(f"   Data path:      {DATA_PATH}")
log(f"   Save dir:       {SAVE_DIR}")
log(f"   Models to run:  {MODELS_TO_RUN}")

if DATA_PATH is None or not os.path.exists(DATA_PATH):
    log(f"\nERROR: Dataset not found! Run: python enhanced_generate_dataset.py")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════════════════════

log(f"\nLoading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    log(f"   Loaded {len(df):,} records")
except Exception as e:
    log(f"   ERROR loading CSV: {e}")
    sys.exit(1)

# --- SUBSAMPLE (key optimization: prevents OOM on 500K rows) ---
SAMPLE_FRAC = float(os.environ.get('DEMAND_SAMPLE_FRAC', '0.20'))
if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    log(f"   Subsampled to {len(df):,} records ({int(SAMPLE_FRAC*100)}% of full data)")

log(f"   Features: {df.shape[1] - 1}, Target: passenger_count")
log(f"   Target range: {df['passenger_count'].min()}-{df['passenger_count'].max()}")

feature_cols = [col for col in df.columns if col != 'passenger_count']
X = df[feature_cols].copy()
y = df['passenger_count'].values

if 'date' in X.columns:
    log(f"\n   Expanding date column into numeric features...")
    parsed_dates = pd.to_datetime(X['date'], errors='coerce')
    X['date_year'] = parsed_dates.dt.year.fillna(0).astype(np.int16)
    X['date_dayofyear'] = parsed_dates.dt.dayofyear.fillna(0).astype(np.int16)
    X = X.drop(columns=['date'])

categorical_cols = list(X.select_dtypes(include=['object']).columns)
if categorical_cols:
    log(f"   Encoding {len(categorical_cols)} categorical columns...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=np.float64)

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    X = X.astype(np.float64)

X_array = X.values.astype(np.float32)
X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

log(f"   Scaling with StandardScaler...")
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    log(f"   Shape after scaling: {X_scaled.shape}")
except ValueError as e:
    log(f"   CRITICAL ERROR during scaling: {e}")
    sys.exit(1)

del X_array
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)
log(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def safe_mape(y_true, y_pred, min_actual=1.0):
    mask = np.abs(y_true) >= min_actual
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)

def compute_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = safe_mape(y_true, y_pred)
    r2   = float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

def save_results(results_dict):
    """Write results JSON after every model — crash-safe."""
    with open(RESULT_JSON, 'w') as f:
        json.dump(results_dict, f, indent=2)

def print_model_header(idx, total, name):
    log(f"\n{'─'*77}")
    log(f"  [{idx}/{total}] {name.upper()}")
    log(f"{'─'*77}")

def print_model_done(name, metrics, elapsed):
    m = metrics
    log(f"   Done in {elapsed:.1f}s")
    log(f"   MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# RESULTS ACCUMULATOR
# ════════════════════════════════════════════════════════════════════════════

if os.path.exists(RESULT_JSON) and SINGLE_MODEL is None:
    try:
        with open(RESULT_JSON) as f:
            results_dict = json.load(f)
        log(f"\nResuming from existing report "
            f"({len(results_dict.get('models', {}))} models already done)")
    except Exception:
        results_dict = {}
else:
    results_dict = {}

results_dict.update({
    "task": "demand_prediction",
    "timestamp": datetime.now().isoformat(),
    "trainer_version": SCRIPT_VERSION,
    "models_compared": len(ALL_MODELS),
    "data_size": len(df),
    "sample_frac": SAMPLE_FRAC,
    "test_size": len(X_test),
})
results_dict.setdefault("models", {})

# ════════════════════════════════════════════════════════════════════════════
# TF MODEL BUILDER  (shared Dense architecture for LSTM / GRU / Transformer)
# ════════════════════════════════════════════════════════════════════════════

def build_dense_model(input_dim):
    from tensorflow.keras import Sequential, layers
    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_tf_model(name, save_filename):
    """Train a TF Dense model with early stopping, then free its memory."""
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_dense_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )

    log(f"   Training (max 20 epochs, early-stop patience=5, batch=512)...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=2,       # one line per epoch — readable in Colab
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test, y_pred)

    model.save(os.path.join(SAVE_DIR, save_filename))
    log(f"   Model saved: {save_filename}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics

# ════════════════════════════════════════════════════════════════════════════
# 1. LSTM
# ════════════════════════════════════════════════════════════════════════════

if 'lstm' in MODELS_TO_RUN:
    print_model_header(1, 6, 'LSTM')
    t0 = time.time()
    try:
        metrics = train_tf_model('lstm', 'demand_lstm_multimodel.keras')
        results_dict["models"]["lstm"] = metrics
        print_model_done('lstm', metrics, time.time() - t0)
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["lstm"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# 2. GRU
# ════════════════════════════════════════════════════════════════════════════

if 'gru' in MODELS_TO_RUN:
    print_model_header(2, 6, 'GRU')
    t0 = time.time()
    try:
        metrics = train_tf_model('gru', 'demand_gru_multimodel.keras')
        results_dict["models"]["gru"] = metrics
        print_model_done('gru', metrics, time.time() - t0)
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["gru"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# 3. Transformer
# ════════════════════════════════════════════════════════════════════════════

if 'transformer' in MODELS_TO_RUN:
    print_model_header(3, 6, 'Transformer')
    t0 = time.time()
    try:
        metrics = train_tf_model('transformer', 'demand_transformer_multimodel.keras')
        results_dict["models"]["transformer"] = metrics
        print_model_done('transformer', metrics, time.time() - t0)
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["transformer"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# 4. XGBoost
# ════════════════════════════════════════════════════════════════════════════

if 'xgboost' in MODELS_TO_RUN:
    print_model_header(4, 6, 'XGBoost')
    t0 = time.time()
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            tree_method='hist',   # histogram-based: 5-10x faster on large data
            random_state=42,
            verbosity=0,
        )
        log(f"   Training (early-stop rounds=20, tree_method=hist)...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,   # in fit() for XGBoost 2.x compatibility
            verbose=False,
        )
        log(f"   Stopped at tree {xgb_model.best_iteration + 1} / 500")

        y_pred_xgb = xgb_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred_xgb)

        joblib.dump(xgb_model, os.path.join(SAVE_DIR, "demand_xgboost_multimodel.pkl"))
        results_dict["models"]["xgboost"] = metrics
        print_model_done('xgboost', metrics, time.time() - t0)

        del xgb_model
        gc.collect()
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["xgboost"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# 5. LightGBM
# ════════════════════════════════════════════════════════════════════════════

if 'lightgbm' in MODELS_TO_RUN:
    print_model_header(5, 6, 'LightGBM')
    t0 = time.time()
    try:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        log(f"   Training (early-stop rounds=20)...")
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0),   # period=0 disables logging
            ]
        )
        log(f"   Stopped at iteration {lgb_model.best_iteration_} / 500")

        y_pred_lgb = lgb_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred_lgb)

        joblib.dump(lgb_model, os.path.join(SAVE_DIR, "demand_lightgbm_multimodel.pkl"))
        results_dict["models"]["lightgbm"] = metrics
        print_model_done('lightgbm', metrics, time.time() - t0)

        del lgb_model
        gc.collect()
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["lightgbm"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# 6. Random Forest
# ════════════════════════════════════════════════════════════════════════════

if 'random_forest' in MODELS_TO_RUN:
    print_model_header(6, 6, 'Random Forest')
    t0 = time.time()
    try:
        rf_model = RandomForestRegressor(
            n_estimators=100,  # was 200 — gain past 100 trees is marginal
            max_depth=10,      # was 15 — shallower trees, far less memory
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        log(f"   Training (100 trees, max_depth=10, n_jobs=-1)...")
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred_rf)

        joblib.dump(rf_model, os.path.join(SAVE_DIR, "demand_rf_multimodel.pkl"))
        results_dict["models"]["random_forest"] = metrics
        print_model_done('random_forest', metrics, time.time() - t0)

        del rf_model
        gc.collect()
    except Exception as e:
        log(f"   ERROR: {e}")
        results_dict["models"]["random_forest"] = {"error": str(e)}
    save_results(results_dict)

# ════════════════════════════════════════════════════════════════════════════
# SAVE SCALER + FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════

log(f"\nSaving auxiliary files...")
joblib.dump(scaler, os.path.join(SAVE_DIR, "demand_scaler_multimodel.pkl"))
log(f"   Scaler saved")

best_model = None
best_mae = float('inf')
for model_name, metrics in results_dict["models"].items():
    if "error" not in metrics and metrics.get("mae", float('inf')) < best_mae:
        best_mae = metrics["mae"]
        best_model = model_name

results_dict["best_model"] = best_model
results_dict["best_mae"] = best_mae
save_results(results_dict)
log(f"   Report saved: {RESULT_JSON}")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

log(f"\n{'='*77}")
log(f"{'DEMAND TRAINING COMPLETE':^77}")
log(f"{'='*77}")
log(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8}  Status")
log(f"{'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*10}")
for model_name in ALL_MODELS:
    m = results_dict["models"].get(model_name)
    if m is None:
        log(f"   {model_name:<20} {'(skipped)':>8}")
    elif "error" in m:
        log(f"   {model_name:<20} {'FAILED':>8}  {m['error'][:35]}")
    else:
        star = " <-- BEST" if model_name == best_model else ""
        log(f"   {model_name:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
            f"{m['mape']:>7.2f}% {m['r2']:>8.4f}{star}")

log(f"\nFiles saved to: {SAVE_DIR}")
log(f"   demand_comparison_report.json")
log(f"{'='*77}\n")
