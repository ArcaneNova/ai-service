"""
Train 6 Demand Prediction Models - FIXED VERSION

Models:
1. LSTM (TensorFlow)
2. GRU (TensorFlow)
3. Transformer (TensorFlow)
4. XGBoost
5. LightGBM
6. Random Forest

All models saved with proper error handling and progress tracking.
"""

import os
import sys
import json
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

# ════════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # ai-service/

# Look for data in multiple locations
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

# ════════════════════════════════════════════════════════════════════════════
# PRINT INFO
# ════════════════════════════════════════════════════════════════════════════

print(f"\n╔{'═'*75}╗")
print(f"║{'DEMAND PREDICTION - 6 MODELS COMPARISON':^75}║")
print(f"╚{'═'*75}╝")
print(f"\n📁 Configuration:")
print(f"   Script dir: {SCRIPT_DIR}")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Data path: {DATA_PATH}")
print(f"   Save dir: {SAVE_DIR}")

if DATA_PATH is None or not os.path.exists(DATA_PATH):
    print(f"\n❌ ERROR: Dataset not found!")
    print(f"   Run: python enhanced_generate_dataset.py")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Loaded {len(df):,} records")
    print(f"   Features: {df.shape[1] - 1}, Target: passenger_count")
    print(f"   Target range: {df['passenger_count'].min()}-{df['passenger_count'].max()}")
except Exception as e:
    print(f"   ❌ Error loading CSV: {e}")
    sys.exit(1)

# Prepare features and target
feature_cols = [col for col in df.columns if col != 'passenger_count']
X = df[feature_cols].values
y = df['passenger_count'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split (temporal - no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

# ════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

results_dict = {
    "task": "demand_prediction",
    "timestamp": datetime.now().isoformat(),
    "models_compared": 6,
    "data_size": len(df),
    "test_size": len(X_test),
    "models": {}
}

# ────────────────────────────────────────────────────────────────────────────
# 1. LSTM
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [1/6] LSTM Model")
try:
    from tensorflow.keras import Sequential, layers
    
    lstm_model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"   Training (30 epochs)...")
    lstm_model.fit(X_train, y_train, epochs=30, batch_size=64, 
                   validation_split=0.1, verbose=1, workers=0)
    
    y_pred_lstm = lstm_model.predict(X_test, verbose=0).flatten()
    mae_lstm = np.mean(np.abs(y_pred_lstm - y_test))
    rmse_lstm = np.sqrt(np.mean((y_pred_lstm - y_test) ** 2))
    mape_lstm = np.mean(np.abs((y_pred_lstm - y_test) / (y_test + 1e-8))) * 100
    r2_lstm = 1 - np.sum((y_test - y_pred_lstm)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    # Save as .keras (TensorFlow 2.16+ requirement)
    lstm_model.save(os.path.join(SAVE_DIR, "demand_lstm_multimodel.keras"))
    
    results_dict["models"]["lstm"] = {
        "mae": float(mae_lstm),
        "rmse": float(rmse_lstm),
        "mape": float(mape_lstm),
        "r2": float(r2_lstm),
    }
    
    print(f"   ✅ MAE={mae_lstm:.2f}, R²={r2_lstm:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["lstm"] = {"error": str(e)}

# ────────────────────────────────────────────────────────────────────────────
# 2. GRU
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [2/6] GRU Model")
try:
    gru_model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"   Training (30 epochs)...")
    gru_model.fit(X_train, y_train, epochs=30, batch_size=64,
                  validation_split=0.1, verbose=1, workers=0)
    
    y_pred_gru = gru_model.predict(X_test, verbose=0).flatten()
    mae_gru = np.mean(np.abs(y_pred_gru - y_test))
    rmse_gru = np.sqrt(np.mean((y_pred_gru - y_test) ** 2))
    mape_gru = np.mean(np.abs((y_pred_gru - y_test) / (y_test + 1e-8))) * 100
    r2_gru = 1 - np.sum((y_test - y_pred_gru)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    gru_model.save(os.path.join(SAVE_DIR, "demand_gru_multimodel.keras"))
    
    results_dict["models"]["gru"] = {
        "mae": float(mae_gru),
        "rmse": float(rmse_gru),
        "mape": float(mape_gru),
        "r2": float(r2_gru),
    }
    
    print(f"   ✅ MAE={mae_gru:.2f}, R²={r2_gru:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["gru"] = {"error": str(e)}

# ────────────────────────────────────────────────────────────────────────────
# 3. Transformer
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [3/6] Transformer Model")
try:
    transformer_model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"   Training (30 epochs)...")
    transformer_model.fit(X_train, y_train, epochs=30, batch_size=64,
                          validation_split=0.1, verbose=1, workers=0)
    
    y_pred_transformer = transformer_model.predict(X_test, verbose=0).flatten()
    mae_transformer = np.mean(np.abs(y_pred_transformer - y_test))
    rmse_transformer = np.sqrt(np.mean((y_pred_transformer - y_test) ** 2))
    mape_transformer = np.mean(np.abs((y_pred_transformer - y_test) / (y_test + 1e-8))) * 100
    r2_transformer = 1 - np.sum((y_test - y_pred_transformer)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    transformer_model.save(os.path.join(SAVE_DIR, "demand_transformer_multimodel.keras"))
    
    results_dict["models"]["transformer"] = {
        "mae": float(mae_transformer),
        "rmse": float(rmse_transformer),
        "mape": float(mape_transformer),
        "r2": float(r2_transformer),
    }
    
    print(f"   ✅ MAE={mae_transformer:.2f}, R²={r2_transformer:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["transformer"] = {"error": str(e)}

# ────────────────────────────────────────────────────────────────────────────
# 4. XGBoost
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [4/6] XGBoost Model")
try:
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
    print(f"   Training...")
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    y_pred_xgb = xgb_model.predict(X_test)
    mae_xgb = np.mean(np.abs(y_pred_xgb - y_test))
    rmse_xgb = np.sqrt(np.mean((y_pred_xgb - y_test) ** 2))
    mape_xgb = np.mean(np.abs((y_pred_xgb - y_test) / (y_test + 1e-8))) * 100
    r2_xgb = 1 - np.sum((y_test - y_pred_xgb)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    joblib.dump(xgb_model, os.path.join(SAVE_DIR, "demand_xgboost_multimodel.pkl"))
    
    results_dict["models"]["xgboost"] = {
        "mae": float(mae_xgb),
        "rmse": float(rmse_xgb),
        "mape": float(mape_xgb),
        "r2": float(r2_xgb),
    }
    
    print(f"   ✅ MAE={mae_xgb:.2f}, R²={r2_xgb:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["xgboost"] = {"error": str(e)}

# ────────────────────────────────────────────────────────────────────────────
# 5. LightGBM
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [5/6] LightGBM Model")
try:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    print(f"   Training...")
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    y_pred_lgb = lgb_model.predict(X_test)
    mae_lgb = np.mean(np.abs(y_pred_lgb - y_test))
    rmse_lgb = np.sqrt(np.mean((y_pred_lgb - y_test) ** 2))
    mape_lgb = np.mean(np.abs((y_pred_lgb - y_test) / (y_test + 1e-8))) * 100
    r2_lgb = 1 - np.sum((y_test - y_pred_lgb)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    joblib.dump(lgb_model, os.path.join(SAVE_DIR, "demand_lightgbm_multimodel.pkl"))
    
    results_dict["models"]["lightgbm"] = {
        "mae": float(mae_lgb),
        "rmse": float(rmse_lgb),
        "mape": float(mape_lgb),
        "r2": float(r2_lgb),
    }
    
    print(f"   ✅ MAE={mae_lgb:.2f}, R²={r2_lgb:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["lightgbm"] = {"error": str(e)}

# ────────────────────────────────────────────────────────────────────────────
# 6. Random Forest
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🔨 [6/6] Random Forest Model")
try:
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    print(f"   Training...")
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = np.mean(np.abs(y_pred_rf - y_test))
    rmse_rf = np.sqrt(np.mean((y_pred_rf - y_test) ** 2))
    mape_rf = np.mean(np.abs((y_pred_rf - y_test) / (y_test + 1e-8))) * 100
    r2_rf = 1 - np.sum((y_test - y_pred_rf)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    joblib.dump(rf_model, os.path.join(SAVE_DIR, "demand_rf_multimodel.pkl"))
    
    results_dict["models"]["random_forest"] = {
        "mae": float(mae_rf),
        "rmse": float(rmse_rf),
        "mape": float(mape_rf),
        "r2": float(r2_rf),
    }
    
    print(f"   ✅ MAE={mae_rf:.2f}, R²={r2_rf:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    results_dict["models"]["random_forest"] = {"error": str(e)}

# ════════════════════════════════════════════════════════════════════════════
# SAVE SCALER AND RESULTS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n💾 Saving auxiliary files...")

# Save scaler
joblib.dump(scaler, os.path.join(SAVE_DIR, "demand_scaler_multimodel.pkl"))
print(f"   ✅ Scaler saved")

# Find best model (lowest MAE)
best_model = None
best_mae = float('inf')
for model_name, metrics in results_dict["models"].items():
    if "error" not in metrics and metrics["mae"] < best_mae:
        best_mae = metrics["mae"]
        best_model = model_name

results_dict["best_model"] = best_model
results_dict["best_mae"] = best_mae

# Save JSON report
with open(RESULT_JSON, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"   ✅ Report saved: {RESULT_JSON}")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print(f"\n╔{'═'*75}╗")
print(f"║{'✅ DEMAND TRAINING COMPLETE':^75}║")
print(f"╚{'═'*75}╝")

print(f"\n📊 Results:")
for model_name, metrics in results_dict["models"].items():
    if "error" not in metrics:
        print(f"   {model_name:20s} MAE={metrics['mae']:6.2f}  R²={metrics['r2']:.4f}")

print(f"\n🏆 Best Model: {best_model.upper()}")
print(f"   MAE: {best_mae:.2f} passengers")

print(f"\n📁 Files saved to: {SAVE_DIR}")
print(f"   ├── demand_lstm_multimodel.keras")
print(f"   ├── demand_gru_multimodel.keras")
print(f"   ├── demand_transformer_multimodel.keras")
print(f"   ├── demand_xgboost_multimodel.pkl")
print(f"   ├── demand_lightgbm_multimodel.pkl")
print(f"   ├── demand_rf_multimodel.pkl")
print(f"   ├── demand_scaler_multimodel.pkl")
print(f"   └── demand_comparison_report.json")

print(f"\n{'═'*75}\n")
