"""
train_anomaly_models.py
Trains multiple anomaly detection models for bus behavior analysis.

Models trained:
1. Isolation Forest - Tree-based, very efficient
2. Local Outlier Factor (LOF) - Density-based detection
3. One-Class SVM - Kernel-based boundary detection
4. Autoencoder - Deep learning reconstruction-based
5. DBSCAN - Clustering-based anomalies
6. Ensemble - Voting of multiple methods

Output: Individual model files + comprehensive comparison metrics

Usage: python training/train_anomaly_models.py
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
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
    os.path.join(project_root, "data/anomaly_dataset.csv"),
    "/content/bus-site/ai-service/data/anomaly_dataset.csv",
    "anomaly_dataset.csv",
]:
    if os.path.exists(possible_path):
        DATA_PATH = possible_path
        break

if DATA_PATH is None:
    raise FileNotFoundError("❌ anomaly_dataset.csv not found in any expected location")

SAVE_DIR = os.path.join(project_root, "models/saved")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = ["speed_kmh", "delay_minutes", "passenger_load"]
TARGET = "is_anomaly"
TEST_SIZE = 0.20
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────
# Load & Preprocess

logger.info("🔄 Loading anomaly dataset...")
df = pd.read_csv(DATA_PATH)
logger.info(f"   Loaded {len(df):,} records")

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.int32)

anomaly_count = y.sum()
anomaly_pct = anomaly_count / len(y) * 100

logger.info(f"   Anomalies: {anomaly_count:,} ({anomaly_pct:.2f}%)")
logger.info(f"   Features: {len(FEATURES)}")

# Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, random_state=SEED, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, shuffle=True, random_state=SEED, stratify=y_train
)

logger.info(f"   Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────
# Evaluation Function

def evaluate_anomaly(y_true, y_pred, model_name):
    """Evaluate anomaly detection model."""
    # Filter to only anomalies for positive evaluation
    if y_true.sum() == 0:
        logger.warning(f"   {model_name}: No anomalies in test set!")
        return {"error": "No anomalies in test set"}
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # For AUC, use a soft score if possible
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = None
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(f"   {model_name}: Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, Spec={specificity:.3f}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

# ─────────────────────────────────────────────────────────────────────────
# Model Training

models_dict = {}
metrics_dict = {}
predictions_dict = {}

# ──────────────────────────────────────────────────────────────────
# 1. Isolation Forest
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Isolation Forest model...")
if_model = IsolationForest(
    n_estimators=200,
    contamination=anomaly_pct / 100,
    max_samples="auto",
    random_state=SEED,
    n_jobs=-1,
)

logger.info("   Training Isolation Forest...")
if_model.fit(X_train_s)

y_pred_if = if_model.predict(X_test_s)
y_pred_if = np.where(y_pred_if == -1, 1, 0)  # Convert -1 to 1 for anomaly
metrics_dict["IsolationForest"] = evaluate_anomaly(y_test, y_pred_if, "Isolation Forest")
predictions_dict["IsolationForest"] = y_pred_if

models_dict["isolation_forest"] = if_model
joblib.dump(if_model, os.path.join(SAVE_DIR, "anomaly_isolation_forest_multimodel.pkl"))
logger.info("   ✅ Saved Isolation Forest model")

# ──────────────────────────────────────────────────────────────────
# 2. Local Outlier Factor (LOF)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building LOF model...")
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=anomaly_pct / 100,
    novelty=False,
    n_jobs=-1,
)

logger.info("   Training LOF...")
lof_model.fit(X_train_s)

y_pred_lof = lof_model.predict(X_test_s)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
metrics_dict["LOF"] = evaluate_anomaly(y_test, y_pred_lof, "LOF")
predictions_dict["LOF"] = y_pred_lof

models_dict["lof"] = lof_model
joblib.dump(lof_model, os.path.join(SAVE_DIR, "anomaly_lof_multimodel.pkl"))
logger.info("   ✅ Saved LOF model")

# ──────────────────────────────────────────────────────────────────
# 3. One-Class SVM
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building One-Class SVM model...")
ocsvm_model = OneClassSVM(
    kernel="rbf",
    gamma="auto",
    nu=anomaly_pct / 100,
)

logger.info("   Training One-Class SVM...")
ocsvm_model.fit(X_train_s)

y_pred_ocsvm = ocsvm_model.predict(X_test_s)
y_pred_ocsvm = np.where(y_pred_ocsvm == -1, 1, 0)
metrics_dict["OneClassSVM"] = evaluate_anomaly(y_test, y_pred_ocsvm, "One-Class SVM")
predictions_dict["OneClassSVM"] = y_pred_ocsvm

models_dict["one_class_svm"] = ocsvm_model
joblib.dump(ocsvm_model, os.path.join(SAVE_DIR, "anomaly_ocsvm_multimodel.pkl"))
logger.info("   ✅ Saved One-Class SVM model")

# ──────────────────────────────────────────────────────────────────
# 4. Autoencoder (Deep Learning)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Autoencoder model...")

# Use only normal data for training
normal_indices = y_train == 0
X_train_normal = X_train_s[normal_indices]

encoding_dim = 2
encoder = Sequential([
    Dense(8, activation="relu", input_shape=(len(FEATURES),)),
    Dropout(0.2),
    Dense(encoding_dim, activation="relu"),
])

decoder = Sequential([
    Dense(8, activation="relu", input_shape=(encoding_dim,)),
    Dropout(0.2),
    Dense(len(FEATURES), activation="sigmoid"),
])

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

logger.info("   Training Autoencoder...")
autoencoder.fit(
    X_train_normal, X_train_normal,
    validation_data=(X_val_s, X_val_s),
    epochs=100,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
    verbose=0,
)

# Calculate reconstruction error
X_test_pred = autoencoder.predict(X_test_s, verbose=0)
reconstruction_errors = np.mean(np.abs(X_test_s - X_test_pred), axis=1)

# Threshold: 95th percentile of training errors
train_errors = np.mean(np.abs(X_train_normal - autoencoder.predict(X_train_normal, verbose=0)), axis=1)
threshold = np.percentile(train_errors, 95)

y_pred_ae = (reconstruction_errors > threshold).astype(int)
metrics_dict["Autoencoder"] = evaluate_anomaly(y_test, y_pred_ae, "Autoencoder")
predictions_dict["Autoencoder"] = y_pred_ae

autoencoder.save(os.path.join(SAVE_DIR, "anomaly_autoencoder_multimodel"))
logger.info("   ✅ Saved Autoencoder model")

# ──────────────────────────────────────────────────────────────────
# 5. DBSCAN (Clustering-based)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building DBSCAN model...")
# DBSCAN doesn't have a clear predict method, so we'll use it for outlier detection
# by finding noise points

logger.info("   Training DBSCAN...")
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(X_test_s)

y_pred_dbscan = (dbscan_labels == -1).astype(int)  # -1 indicates noise/outliers
metrics_dict["DBSCAN"] = evaluate_anomaly(y_test, y_pred_dbscan, "DBSCAN")
predictions_dict["DBSCAN"] = y_pred_dbscan

joblib.dump(dbscan_model, os.path.join(SAVE_DIR, "anomaly_dbscan_multimodel.pkl"))
logger.info("   ✅ Saved DBSCAN model")

# ──────────────────────────────────────────────────────────────────
# 6. Ensemble (Voting)
# ──────────────────────────────────────────────────────────────────

logger.info("\n🔨 Building Ensemble Voting model...")

# Combine predictions from all models (majority voting)
ensemble_predictions = np.column_stack([
    predictions_dict["IsolationForest"],
    predictions_dict["LOF"],
    predictions_dict["OneClassSVM"],
    predictions_dict["Autoencoder"],
    predictions_dict["DBSCAN"],
])

# Majority voting
y_pred_ensemble = (ensemble_predictions.sum(axis=1) >= 3).astype(int)
metrics_dict["Ensemble"] = evaluate_anomaly(y_test, y_pred_ensemble, "Ensemble")
predictions_dict["Ensemble"] = y_pred_ensemble

logger.info("   ✅ Ensemble created (majority voting of 5 models)")

# Save scaler
joblib.dump(scaler, os.path.join(SAVE_DIR, "anomaly_scaler_multimodel.pkl"))
joblib.dump({"threshold": float(threshold)}, os.path.join(SAVE_DIR, "anomaly_ae_threshold.pkl"))

# ─────────────────────────────────────────────────────────────────────────
# Comparison Report

logger.info("\n" + "="*70)
logger.info("📊 ANOMALY DETECTION - MODEL COMPARISON REPORT")
logger.info("="*70)

comparison = {
    "timestamp": datetime.now().isoformat(),
    "test_set_size": len(y_test),
    "anomaly_rate": float(anomaly_pct),
    "models": metrics_dict,
}

# Find best model
best_model = max(metrics_dict.items(), key=lambda x: x[1].get("f1", 0))

logger.info(f"\n🏆 Best Model (by F1): {best_model[0]}")
logger.info(f"   F1: {best_model[1]['f1']:.3f}")
logger.info(f"   Precision: {best_model[1]['precision']:.3f}")
logger.info(f"   Recall: {best_model[1]['recall']:.3f}")
logger.info(f"   Specificity: {best_model[1]['specificity']:.3f}")

# Save comparison
report_path = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")
with open(report_path, "w") as f:
    json.dump(comparison, f, indent=2)

logger.info(f"\n📄 Full report saved to: {report_path}")
logger.info("\n✨ Anomaly detection model training complete!")
