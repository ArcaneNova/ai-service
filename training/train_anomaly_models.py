"""
Train Anomaly Detection Models

Models:
- Isolation Forest, LOF, One-Class SVM, Autoencoder, DBSCAN, Ensemble
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = None
for path in [
    os.path.join(PROJECT_ROOT, "data", "anomaly_dataset.csv"),
    "/content/bus-site/ai-service/data/anomaly_dataset.csv",
]:
    if os.path.exists(path):
        DATA_PATH = os.path.abspath(path)
        break

SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)
SCRIPT_VERSION = "2026-04-14-fix"

print(f"\n╔{'═'*75}╗")
print(f"║{'ANOMALY DETECTION - 6 METHODS COMPARISON':^75}║")
print(f"╚{'═'*75}╝\n")
print(f"   Trainer version: {SCRIPT_VERSION}")

if DATA_PATH is None or not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found. Run: python enhanced_generate_dataset.py")
    sys.exit(1)

# LOAD & PREPARE
print(f"🔄 Loading anomaly dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   ✅ Loaded {len(df):,} records")

# Separate normal and anomaly data
X_normal = df[df['anomaly_label'] == 0].copy()
X_anomaly = df[df['anomaly_label'] == 1].copy()

feature_cols = [col for col in df.columns if col not in ['anomaly_label']]
X_all = df[feature_cols].copy()
y_all = df['anomaly_label'].values

if 'date' in X_all.columns:
    print(f"\n   📅 Expanding raw date column into numeric features...")
    parsed_dates = pd.to_datetime(X_all['date'], errors='coerce')
    X_all['date_year'] = parsed_dates.dt.year.fillna(0).astype(np.int16)
    X_all['date_dayofyear'] = parsed_dates.dt.dayofyear.fillna(0).astype(np.int16)
    X_normal['date_year'] = pd.to_datetime(X_normal['date'], errors='coerce').dt.year.fillna(0).astype(np.int16)
    X_normal['date_dayofyear'] = pd.to_datetime(X_normal['date'], errors='coerce').dt.dayofyear.fillna(0).astype(np.int16)
    X_anomaly['date_year'] = pd.to_datetime(X_anomaly['date'], errors='coerce').dt.year.fillna(0).astype(np.int16)
    X_anomaly['date_dayofyear'] = pd.to_datetime(X_anomaly['date'], errors='coerce').dt.dayofyear.fillna(0).astype(np.int16)
    X_all = X_all.drop(columns=['date'])
    X_normal = X_normal.drop(columns=['date'])
    X_anomaly = X_anomaly.drop(columns=['date'])
    print(f"      Added: date_year, date_dayofyear; dropped raw date")

print(f"   Initial shape: {X_all.shape}")
print(f"   Initial columns: {X_all.columns.tolist()}")
print(f"   Dtypes:\n{X_all.dtypes}")

# CRITICAL: Detect and encode ALL categorical columns BEFORE scaling
print(f"\n   🔍 Detecting categorical columns...")
categorical_cols = list(X_all.select_dtypes(include=['object']).columns)
print(f"      Found: {categorical_cols}")

if len(categorical_cols) > 0:
    print(f"   🔄 Encoding {len(categorical_cols)} categorical columns with pd.get_dummies()...")
    for col in categorical_cols:
        unique_vals = X_all[col].unique()[:5].tolist()
        print(f"      • {col}: {unique_vals}")
    
    # ONE-HOT ENCODE - this is CRITICAL for all three datasets
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True, dtype=np.float64)
    X_normal = pd.get_dummies(X_normal, columns=categorical_cols, drop_first=True, dtype=np.float64)
    X_anomaly = pd.get_dummies(X_anomaly, columns=categorical_cols, drop_first=True, dtype=np.float64)
    print(f"   ✅ After encoding: X_all shape = {X_all.shape} (features now numeric)")
else:
    print(f"   ℹ️  No categorical columns found - all numeric")

# Verify all columns are numeric BEFORE converting to numpy
print(f"\n   🔄 Verifying all columns are numeric...")
non_numeric_all = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_norm = X_normal.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_anom = X_anomaly.select_dtypes(exclude=[np.number]).columns.tolist()

if len(non_numeric_all) + len(non_numeric_norm) + len(non_numeric_anom) > 0:
    print(f"      ⚠️  WARNING: Non-numeric columns still exist")
    print(f"      Forcing conversion to float64...")
    X_all = X_all.astype(np.float64)
    X_normal = X_normal.astype(np.float64)
    X_anomaly = X_anomaly.astype(np.float64)
else:
    print(f"      ✅ All columns are numeric")

# Convert to numpy array (MUST be numeric at this point)
print(f"   🔄 Converting to numpy arrays...")
X_all_array = X_all.values.astype(np.float32)
X_normal_array = X_normal.values.astype(np.float32)
X_anomaly_array = X_anomaly.values.astype(np.float32)

# Handle NaN/inf values
print(f"   🔄 Handling NaN and inf values...")
X_all_array = np.nan_to_num(X_all_array, nan=0.0, posinf=0.0, neginf=0.0)
X_normal_array = np.nan_to_num(X_normal_array, nan=0.0, posinf=0.0, neginf=0.0)
X_anomaly_array = np.nan_to_num(X_anomaly_array, nan=0.0, posinf=0.0, neginf=0.0)

# NOW scale (this should work since we have float32 arrays)
print(f"   🔄 Scaling with StandardScaler...")
try:
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all_array)
    X_normal = scaler.transform(X_normal_array)
    X_anomaly = scaler.transform(X_anomaly_array)
    print(f"   ✅ Scaling successful! Shapes: X_all={X_all.shape}, X_normal={X_normal.shape}, X_anomaly={X_anomaly.shape}")
except ValueError as e:
    print(f"   ❌ CRITICAL ERROR during scaling: {e}")
    print(f"      X_all_array: shape={X_all_array.shape}, dtype={X_all_array.dtype}")
    print(f"      X_normal_array: shape={X_normal_array.shape}, dtype={X_normal_array.dtype}")
    print(f"      X_anomaly_array: shape={X_anomaly_array.shape}, dtype={X_anomaly_array.dtype}")
    sys.exit(1)

results = {"task": "anomaly_detection", "timestamp": datetime.now().isoformat(), "models": {}}

print(f"   Normal: {len(X_normal):,}  Anomaly: {len(X_anomaly):,}\n")

# ════════════════════════════════════════════════════════════════════════════

print(f"🔨 [1/6] Isolation Forest")
if_model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
if_model.fit(X_normal)
y_pred_if = if_model.predict(X_all)
y_pred_if = np.where(y_pred_if == -1, 1, 0)
tp = np.sum((y_pred_if == 1) & (y_all == 1))
fp = np.sum((y_pred_if == 1) & (y_all == 0))
fn = np.sum((y_pred_if == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
joblib.dump(if_model, os.path.join(SAVE_DIR, "anomaly_isolation_forest_multimodel.pkl"))
results["models"]["isolation_forest"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

print(f"🔨 [2/6] Local Outlier Factor (LOF)")
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
y_pred_lof = lof_model.fit_predict(X_all)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
tp = np.sum((y_pred_lof == 1) & (y_all == 1))
fp = np.sum((y_pred_lof == 1) & (y_all == 0))
fn = np.sum((y_pred_lof == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
joblib.dump(lof_model, os.path.join(SAVE_DIR, "anomaly_lof_multimodel.pkl"))
results["models"]["lof"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

print(f"🔨 [3/6] One-Class SVM")
ocsvm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm_model.fit(X_normal)
y_pred_ocsvm = ocsvm_model.predict(X_all)
y_pred_ocsvm = np.where(y_pred_ocsvm == -1, 1, 0)
tp = np.sum((y_pred_ocsvm == 1) & (y_all == 1))
fp = np.sum((y_pred_ocsvm == 1) & (y_all == 0))
fn = np.sum((y_pred_ocsvm == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
joblib.dump(ocsvm_model, os.path.join(SAVE_DIR, "anomaly_ocsvm_multimodel.pkl"))
results["models"]["ocsvm"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

print(f"🔨 [4/6] Autoencoder (Neural Network)")
from tensorflow.keras import Sequential, layers
autoencoder = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_normal.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(X_normal.shape[1])
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
X_pred = autoencoder.predict(X_all, verbose=0)
mse = np.mean((X_all - X_pred) ** 2, axis=1)
threshold = np.percentile(mse, 95)
y_pred_ae = (mse > threshold).astype(int)
tp = np.sum((y_pred_ae == 1) & (y_all == 1))
fp = np.sum((y_pred_ae == 1) & (y_all == 0))
fn = np.sum((y_pred_ae == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
autoencoder.save(os.path.join(SAVE_DIR, "anomaly_autoencoder_multimodel.keras"))
joblib.dump(threshold, os.path.join(SAVE_DIR, "anomaly_ae_threshold.pkl"))
results["models"]["autoencoder"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

print(f"🔨 [5/6] DBSCAN")
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
y_pred_dbscan = dbscan_model.fit_predict(X_all)
y_pred_dbscan = np.where(y_pred_dbscan == -1, 1, 0)
tp = np.sum((y_pred_dbscan == 1) & (y_all == 1))
fp = np.sum((y_pred_dbscan == 1) & (y_all == 0))
fn = np.sum((y_pred_dbscan == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
joblib.dump(dbscan_model, os.path.join(SAVE_DIR, "anomaly_dbscan_multimodel.pkl"))
results["models"]["dbscan"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

print(f"🔨 [6/6] Ensemble Voting")
votes = y_pred_if + y_pred_lof + y_pred_ocsvm + y_pred_ae + y_pred_dbscan
y_pred_ensemble = (votes >= 3).astype(int)
tp = np.sum((y_pred_ensemble == 1) & (y_all == 1))
fp = np.sum((y_pred_ensemble == 1) & (y_all == 0))
fn = np.sum((y_pred_ensemble == 0) & (y_all == 1))
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
results["models"]["ensemble"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
print(f"   ✅ F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

# SAVE
joblib.dump(scaler, os.path.join(SAVE_DIR, "anomaly_scaler_multimodel.pkl"))
with open(os.path.join(SAVE_DIR, "anomaly_comparison_report.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"╔{'═'*75}╗")
print(f"║{'✅ ANOMALY DETECTION TRAINING COMPLETE':^75}║")
print(f"╚{'═'*75}╝\n")
print(f"📊 Results saved to: {SAVE_DIR}\n")
