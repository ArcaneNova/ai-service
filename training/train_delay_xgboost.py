"""
train_delay_xgboost.py
Trains XGBoost regressor (delay minutes) + classifier (is_delayed binary).

Usage: python training/train_delay_xgboost.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, classification_report
import xgboost as xgb

# ── Config ─────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/delay_dataset.csv")
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "../models/saved")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_holiday",
    "weather_encoded", "avg_temp_c",
    "passenger_load_pct", "scheduled_duration_min",
    "distance_km", "total_stops", "month",
]
TARGET_REG = "delay_minutes"
TARGET_CLF = "is_delayed"

# ── Load & Preprocess ──────────────────────────────────────────────────────

print("📂  Loading delay dataset…")
df = pd.read_csv(DATA_PATH)
print(f"    {len(df):,} rows")

weather_map = {"clear": 0, "rain": 1, "fog": 2, "heatwave": 3}
df["weather_encoded"] = df["weather"].map(weather_map).fillna(0)

X = df[FEATURES].values.astype(np.float32)
y_reg = df[TARGET_REG].values.astype(np.float32)
y_clf = df[TARGET_CLF].values.astype(np.int32)

X_train, X_temp, yr_train, yr_temp, yc_train, yc_temp = train_test_split(
    X, y_reg, y_clf, test_size=0.20, shuffle=False
)
X_val,   X_test, yr_val, yr_test, yc_val, yc_test = train_test_split(
    X_temp, yr_temp, yc_temp, test_size=0.50, shuffle=False
)

print(f"    Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── Train Regressor ────────────────────────────────────────────────────────

print("\n🏋️  Training XGBoost Regressor…")
reg = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="mae",
)
reg.fit(
    X_train_s, yr_train,
    eval_set=[(X_val_s, yr_val)],
    verbose=50,
)

yr_pred = reg.predict(X_test_s)
mae  = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(np.mean((yr_pred - yr_test) ** 2))
print(f"\n📊  Regressor — MAE: {mae:.2f} min   RMSE: {rmse:.2f} min")

# ── Train Classifier ───────────────────────────────────────────────────────

print("\n🏋️  Training XGBoost Classifier…")
scale_pos = (yc_train == 0).sum() / max((yc_train == 1).sum(), 1)
clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)
clf.fit(
    X_train_s, yc_train,
    eval_set=[(X_val_s, yc_val)],
    verbose=50,
)

yc_pred = clf.predict(X_test_s)
f1 = f1_score(yc_test, yc_pred)
print(f"\n📊  Classifier — F1: {f1:.3f}")
print(classification_report(yc_test, yc_pred, target_names=["on-time", "delayed"]))

# ── Feature Importance ─────────────────────────────────────────────────────

imp = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n📊  Top Features (Regressor):")
print(imp.head(8).to_string())

# ── Save ───────────────────────────────────────────────────────────────────

joblib.dump(reg,    os.path.join(SAVE_DIR, "delay_regressor.pkl"))
joblib.dump(clf,    os.path.join(SAVE_DIR, "delay_classifier.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "delay_scaler.pkl"))

print(f"\n✅  Models saved to {SAVE_DIR}/")
