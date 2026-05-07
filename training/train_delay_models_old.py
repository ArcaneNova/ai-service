"""
train_delay_models.py
Trains multiple models for delay prediction (regression + classification).

Regression Models (predict delay in minutes):
1. XGBoost - Gradient boosted trees
2. LightGBM - Faster gradient boosting
3. CatBoost - Handles categorical features better
4. Support Vector Regression - Robust kernel-based approach
5. Neural Network (MLP) - Deep learning baseline
6. Ensemble (Voting) - Combines top 3 models

Classification Models (predict is_delayed binary):
- Trained together with regression models (delay > 5 min = delayed)

Output: Individual model files + detailed comparison metrics

Usage: python training/train_delay_models.py
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────

# Resolve paths - works in both local and Colab environments
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # ai-service/

# Try multiple possible data paths
DATA_PATH = None
for possible_path in [
    os.path.join(project_root, "data/delay_dataset.csv"),
    "/content/bus-site/ai-service/data/delay_dataset.csv",
    "delay_dataset.csv",
]:
    if os.path.exists(possible_path):
        DATA_PATH = possible_path
        break

if DATA_PATH is None:
    raise FileNotFoundError("❌ delay_dataset.csv not found in any expected location")

SAVE_DIR = os.path.join(project_root, "models/saved")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_holiday",
    "weather_encoded", "avg_temp_c",
    "passenger_load_pct", "distance_km", "total_stops", "month", "quarter",
]
WEATHER_MAP = {"clear": 0, "light_rain": 1, "heavy_rain": 2, "fog": 3, "heatwave": 4, "extreme": 5}

TARGET_REG = "delay_minutes"
TARGET_CLF = "is_delayed"
TEST_SIZE = 0.20
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────
# Load & Preprocess

logger.info("🔄 Loading delay dataset...")
df = pd.read_csv(DATA_PATH)
logger.info(f"   Loaded {len(df):,} records")

# Encode weather
df["weather_encoded"] = df["weather"].map(WEATHER_MAP).fillna(0).astype(int)

X = df[FEATURES].values.astype(np.float32)
y_reg = df[TARGET_REG].values.astype(np.float32)
y_clf = df[TARGET_CLF].values.astype(np.int32)

logger.info(f"   Features: {len(FEATURES)}")
logger.info(f"   Delay range: {y_reg.min():.1f}-{y_reg.max():.1f} min")
logger.info(f"   Delayed samples: {y_clf.sum():,} ({y_clf.mean()*100:.1f}%)")

# Train/val/test split
X_train, X_temp, yr_train, yr_temp, yc_train, yc_temp = train_test_split(
    X, y_reg, y_clf, test_size=TEST_SIZE, shuffle=False, random_state=SEED
)
X_val, X_test, yr_val, yr_test, yc_val, yc_test = train_test_split(
    X_temp, yr_temp, yc_temp, test_size=0.50, shuffle=False, random_state=SEED
)

logger.info(f"   Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────
# Evaluation Function

def evaluate_regression(y_true, y_pred, model_name):
    """Evaluate regression model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"   {model_name} (Reg): MAE={mae:.2f}min, RMSE={rmse:.2f}min, R²={r2:.4f}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}

def evaluate_classification(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate classification model."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    logger.info(f"   {model_name} (Clf): Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

# ─────────────────────────────────────────────────────────────────────────
# Model Training

models_dict = {}
metrics_dict = {}

# ──────────────────────────────────────────────────────────────────
# 1. XGBoost
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building XGBoost model...")
xgb_reg = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="mae",
)

logger.info("   Training XGBoost Regressor...")
xgb_reg.fit(X_train_s, yr_train, eval_set=[(X_val_s, yr_val)], verbose=0)

yr_pred_xgb = xgb_reg.predict(X_test_s)
metrics_dict["XGBoost_reg"] = evaluate_regression(yr_test, yr_pred_xgb, "XGBoost")

# XGBoost classifier (using same model, threshold on probability)
xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="logloss",
    scale_pos_weight=(yc_train == 0).sum() / (yc_train == 1).sum(),  # Handle imbalance
)

logger.info("   Training XGBoost Classifier...")
xgb_clf.fit(X_train_s, yc_train, eval_set=[(X_val_s, yc_val)], verbose=0)

yc_pred_xgb = xgb_clf.predict(X_test_s)
yc_pred_proba_xgb = xgb_clf.predict_proba(X_test_s)[:, 1]
metrics_dict["XGBoost_clf"] = evaluate_classification(yc_test, yc_pred_xgb, yc_pred_proba_xgb, "XGBoost")

models_dict["xgboost"] = xgb_reg
joblib.dump(xgb_reg, os.path.join(SAVE_DIR, "delay_xgboost_reg_multimodel.pkl"))
joblib.dump(xgb_clf, os.path.join(SAVE_DIR, "delay_xgboost_clf_multimodel.pkl"))
logger.info("   ✅ Saved XGBoost models")

# ──────────────────────────────────────────────────────────────────
# 2. LightGBM
# ──────────────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
    
    logger.info("\n🔨 Building LightGBM model...")
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    
    logger.info("   Training LightGBM Regressor...")
    lgb_reg.fit(
        X_train_s, yr_train,
        eval_set=[(X_val_s, yr_val)],
        callbacks=[lgb.early_stopping(30)],
    )
    
    yr_pred_lgb = lgb_reg.predict(X_test_s)
    metrics_dict["LightGBM_reg"] = evaluate_regression(yr_test, yr_pred_lgb, "LightGBM")
    
    # LightGBM classifier
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=(yc_train == 0).sum() / (yc_train == 1).sum(),
    )
    
    logger.info("   Training LightGBM Classifier...")
    lgb_clf.fit(
        X_train_s, yc_train,
        eval_set=[(X_val_s, yc_val)],
        callbacks=[lgb.early_stopping(30)],
    )
    
    yc_pred_lgb = lgb_clf.predict(X_test_s)
    yc_pred_proba_lgb = lgb_clf.predict_proba(X_test_s)[:, 1]
    metrics_dict["LightGBM_clf"] = evaluate_classification(yc_test, yc_pred_lgb, yc_pred_proba_lgb, "LightGBM")
    
    models_dict["lightgbm"] = lgb_reg
    joblib.dump(lgb_reg, os.path.join(SAVE_DIR, "delay_lightgbm_reg_multimodel.pkl"))
    joblib.dump(lgb_clf, os.path.join(SAVE_DIR, "delay_lightgbm_clf_multimodel.pkl"))
    logger.info("   ✅ Saved LightGBM models")
    
except ImportError:
    logger.warning("   ⚠️  LightGBM not installed, skipping")

# ──────────────────────────────────────────────────────────────────
# 3. CatBoost
# ──────────────────────────────────────────────────────────────────

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    
    logger.info("\n🔨 Building CatBoost model...")
    cb_reg = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
        verbose=0,
        early_stopping_rounds=30,
    )
    
    logger.info("   Training CatBoost Regressor...")
    cb_reg.fit(X_train_s, yr_train, eval_set=(X_val_s, yr_val), verbose=False)
    
    yr_pred_cb = cb_reg.predict(X_test_s)
    metrics_dict["CatBoost_reg"] = evaluate_regression(yr_test, yr_pred_cb, "CatBoost")
    
    # CatBoost classifier
    cb_clf = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
        verbose=0,
        early_stopping_rounds=30,
        scale_pos_weight=(yc_train == 0).sum() / (yc_train == 1).sum(),
    )
    
    logger.info("   Training CatBoost Classifier...")
    cb_clf.fit(X_train_s, yc_train, eval_set=(X_val_s, yc_val), verbose=False)
    
    yc_pred_cb = cb_clf.predict(X_test_s)
    yc_pred_proba_cb = cb_clf.predict_proba(X_test_s)[:, 1]
    metrics_dict["CatBoost_clf"] = evaluate_classification(yc_test, yc_pred_cb, yc_pred_proba_cb, "CatBoost")
    
    models_dict["catboost"] = cb_reg
    joblib.dump(cb_reg, os.path.join(SAVE_DIR, "delay_catboost_reg_multimodel.pkl"))
    joblib.dump(cb_clf, os.path.join(SAVE_DIR, "delay_catboost_clf_multimodel.pkl"))
    logger.info("   ✅ Saved CatBoost models")
    
except ImportError:
    logger.warning("   ⚠️  CatBoost not installed, skipping")

# ──────────────────────────────────────────────────────────────────
# 4. Support Vector Regression
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Support Vector Regression model...")
svr_model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)

logger.info("   Training SVR...")
svr_model.fit(X_train_s, yr_train)

yr_pred_svr = svr_model.predict(X_test_s)
metrics_dict["SVR_reg"] = evaluate_regression(yr_test, yr_pred_svr, "SVR")

models_dict["svr"] = svr_model
joblib.dump(svr_model, os.path.join(SAVE_DIR, "delay_svr_multimodel.pkl"))
logger.info("   ✅ Saved SVR model")

# ──────────────────────────────────────────────────────────────────
# 5. Neural Network (MLP)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Neural Network (MLP) model...")
nn_model = Sequential([
    Dense(128, activation="relu", input_shape=(len(FEATURES),)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="relu"),
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)

logger.info("   Training MLP...")
nn_model.fit(
    X_train_s, yr_train,
    validation_data=(X_val_s, yr_val),
    epochs=200,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=0,
)

yr_pred_nn = nn_model.predict(X_test_s, verbose=0).flatten()
metrics_dict["MLP_reg"] = evaluate_regression(yr_test, yr_pred_nn, "MLP")

nn_model.save(os.path.join(SAVE_DIR, "delay_mlp_multimodel"))
logger.info("   ✅ Saved MLP model")

# ──────────────────────────────────────────────────────────────────
# 6. Ensemble (Voting Regressor)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Ensemble Voting model...")

ensemble_regressors = [
    ("xgb", models_dict["xgboost"]),
    ("svr", models_dict["svr"]),
]

if "lightgbm" in models_dict:
    ensemble_regressors.append(("lgb", models_dict["lightgbm"]))

ensemble = VotingRegressor(estimators=ensemble_regressors)

logger.info("   Ensemble includes: " + ", ".join([name for name, _ in ensemble_regressors]))

ensemble.fit(X_train_s, yr_train)
yr_pred_ensemble = ensemble.predict(X_test_s)
metrics_dict["Ensemble_reg"] = evaluate_regression(yr_test, yr_pred_ensemble, "Ensemble")

joblib.dump(ensemble, os.path.join(SAVE_DIR, "delay_ensemble_multimodel.pkl"))
logger.info("   ✅ Saved Ensemble model")

# Save scaler
joblib.dump(scaler, os.path.join(SAVE_DIR, "delay_scaler_multimodel.pkl"))

# ─────────────────────────────────────────────────────────────────────────
# Comparison Report

logger.info("\n" + "="*70)
logger.info("📊 DELAY PREDICTION - MODEL COMPARISON REPORT")
logger.info("="*70)

comparison = {
    "timestamp": datetime.now().isoformat(),
    "test_set_size": len(y_test),
    "regression_metrics": metrics_dict,
}

# Find best models
reg_models = {k: v for k, v in metrics_dict.items() if "reg" in k}
best_reg = min(reg_models.items(), key=lambda x: x[1]["mae"])

logger.info(f"\n🏆 Best Regression Model (by MAE): {best_reg[0]}")
logger.info(f"   MAE: {best_reg[1]['mae']:.2f} min")
logger.info(f"   RMSE: {best_reg[1]['rmse']:.2f} min")
logger.info(f"   R²: {best_reg[1]['r2']:.4f}")

clf_models = {k: v for k, v in metrics_dict.items() if "clf" in k}
if clf_models:
    best_clf = max(clf_models.items(), key=lambda x: x[1]["f1"])
    logger.info(f"\n🏆 Best Classification Model (by F1): {best_clf[0]}")
    logger.info(f"   F1: {best_clf[1]['f1']:.3f}")
    logger.info(f"   AUC: {best_clf[1]['auc']:.3f}")

# Save comparison
report_path = os.path.join(SAVE_DIR, "delay_comparison_report.json")
with open(report_path, "w") as f:
    json.dump(comparison, f, indent=2)

logger.info(f"\n📄 Full report saved to: {report_path}")
logger.info("\n✨ Delay model training complete!")
