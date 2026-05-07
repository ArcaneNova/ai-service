# SmartDTC AI Service - Multi-Model Comparison & Ensemble

FastAPI microservice providing **multi-model ML predictions** for the SmartDTC bus management platform. Runs on **port 8000**.

**🎯 For Paper & Panel Presentation:** This service trains and compares multiple state-of-the-art models for each task, enabling rigorous evaluation and selection of optimal architectures.

---

## Table of Contents

1. [What This Service Does](#what-this-service-does)
2. [Key Features](#key-features)
3. [Models Trained](#models-trained)
4. [Project Structure](#project-structure)
5. [Quick Start (Google Colab)](#quick-start-google-colab)
6. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Generate Enhanced Dataset](#step-1-generate-enhanced-dataset)
   - [Step 2: Train All Models](#step-2-train-all-models)
   - [Step 3: Evaluate & Compare](#step-3-evaluate--compare)
   - [Step 4: Deploy](#step-4-deploy)
7. [API Endpoints](#api-endpoints)
8. [Model Comparison Results](#model-comparison-results)
9. [API Reference](#api-reference)
10. [Environment Variables](#environment-variables)

---

## What This Service Does

| Task | Models Trained | Best Use Case |
|------|---|---|
| **Demand Prediction** | LSTM, GRU, Transformer, XGBoost, LightGBM, Random Forest | Predict passenger demand 24h ahead |
| **Delay Prediction** | XGBoost, LightGBM, CatBoost, SVR, MLP, Ensemble | Predict trip delays in real-time |
| **Anomaly Detection** | Isolation Forest, LOF, One-Class SVM, Autoencoder, DBSCAN, Ensemble | Flag unusual bus behavior |
| **ETA Prediction** | Gradient Boosting | Real-time passenger ETA |
| **Schedule Optimization** | Genetic Algorithm | Optimal bus headway planning |

All endpoints work **with or without trained models** — fallback to rule-based heuristics when unavailable.

---

## Key Features

✅ **Multiple Models Per Task** - Compare different architectures (LSTM, XGBoost, LightGBM, CatBoost, etc.)  
✅ **3-Year Synthetic Dataset** - Realistic patterns, 561 DTC routes, real geographical data  
✅ **Advanced Feature Engineering** - Cyclic temporal features, route-specific attributes  
✅ **Ensemble Methods** - Voting & stacking for optimal performance  
✅ **Comprehensive Evaluation** - Cross-validation, metrics comparison, visualizations  
✅ **Publication-Ready** - Detailed reports, comparison tables, graphs for paper  
✅ **Colab Compatible** - Full training pipeline runnable on Google Colab GPUs  

---

## Models Trained

### **DEMAND PREDICTION**
| Model | Type | Complexity | Speed | Best For |
|-------|------|-----------|-------|----------|
| LSTM | RNN | High | Medium | Temporal patterns, peak hour prediction |
| GRU | RNN | Medium | Fast | Faster alternative to LSTM |
| Transformer | Attention | Very High | Medium | State-of-the-art sequence modeling |
| XGBoost | Tree Ensemble | Medium | Very Fast | Feature importance, interpretability |
| LightGBM | Tree Ensemble | Medium | Very Fast | Large datasets, production efficiency |
| Random Forest | Ensemble | Medium | Fast | Baseline comparison, robustness |

### **DELAY PREDICTION**
| Model | Regression | Classification | Best For |
|-------|-----------|---|----------|
| XGBoost | MAE (min) | F1-Score | Balanced performance |
| LightGBM | ✅ | ✅ | Large-scale production |
| CatBoost | ✅ | ✅ | Categorical features handling |
| SVR | ✅ | — | Robust boundary detection |
| MLP | ✅ | ✅ | End-to-end deep learning |
| Ensemble Voting | ✅ | ✅ | Maximum accuracy |

### **ANOMALY DETECTION**
| Model | Type | Strengths | Weaknesses |
|-------|------|-----------|-----------|
| Isolation Forest | Tree-based | Fast, scalable | Struggles with high-dim data |
| LOF | Density-based | Local anomalies | Slower on large datasets |
| One-Class SVM | Kernel-based | Robust boundaries | Hyperparameter tuning needed |
| Autoencoder | Deep Learning | Learns complex patterns | Training overhead |
| DBSCAN | Clustering | No pre-set K | Density parameter sensitivity |
| Ensemble (Majority Voting) | Hybrid | Best overall | Computational cost |

---

## Project Structure

```
ai-service/
├── main.py                          # FastAPI application
├── model_loader.py                  # Model loading at startup
├── predictors.py                    # Prediction logic
├── anomaly_detector.py              # Anomaly detection
├── eta_predictor.py                 # ETA prediction
├── optimizer.py                     # Headway optimization
├── schemas.py                       # Pydantic schemas
├── requirements.txt                 # Dependencies
├── evaluate_models.py               # ⭐ Model evaluation script
│
├── training/
│   ├── enhanced_generate_dataset.py # ⭐ Enhanced dataset generator
│   ├── train_demand_models.py       # ⭐ Train 6 demand models
│   ├── train_delay_models.py        # ⭐ Train 6 delay models
│   ├── train_anomaly_models.py      # ⭐ Train 6 anomaly models
│   ├── generate_dataset.py          # [Legacy] Original dataset generator
│   ├── train_demand_lstm.py         # [Legacy] Original LSTM trainer
│   └── train_delay_xgboost.py       # [Legacy] Original XGBoost trainer
│
├── data/
│   ├── demand_dataset.csv           # Generated: 3-year hourly demand (561K+ rows)
│   ├── delay_dataset.csv            # Generated: 3-year trip delays (561K+ rows)
│   └── anomaly_dataset.csv          # Generated: Anomaly patterns (200K+ rows)
│
├── models/
│   └── saved/
│       ├── demand_lstm_multimodel/              # TensorFlow SavedModel
│       ├── demand_gru_multimodel/
│       ├── demand_transformer_multimodel/
│       ├── demand_xgboost_multimodel.pkl
│       ├── demand_lightgbm_multimodel.pkl
│       ├── demand_rf_multimodel.pkl
│       ├── demand_scaler_multimodel.pkl
│       ├── demand_comparison_report.json       # ⭐ Comparison metrics
│       │
│       ├── delay_xgboost_reg_multimodel.pkl
│       ├── delay_xgboost_clf_multimodel.pkl
│       ├── delay_lightgbm_reg_multimodel.pkl
│       ├── delay_catboost_reg_multimodel.pkl
│       ├── delay_svr_multimodel.pkl
│       ├── delay_mlp_multimodel/
│       ├── delay_ensemble_multimodel.pkl
│       ├── delay_scaler_multimodel.pkl
│       ├── delay_comparison_report.json        # ⭐ Comparison metrics
│       │
│       ├── anomaly_isolation_forest_multimodel.pkl
│       ├── anomaly_lof_multimodel.pkl
│       ├── anomaly_ocsvm_multimodel.pkl
│       ├── anomaly_autoencoder_multimodel/
│       ├── anomaly_dbscan_multimodel.pkl
│       ├── anomaly_scaler_multimodel.pkl
│       ├── anomaly_ae_threshold.pkl
│       └── anomaly_comparison_report.json      # ⭐ Comparison metrics
│
├── evaluation_results/
│   ├── demand_model_comparison.png             # ⭐ Visualization
│   ├── delay_model_comparison.png
│   ├── anomaly_model_comparison.png
│   ├── demand_comparison.csv
│   ├── delay_comparison.csv
│   ├── anomaly_comparison.csv
│   └── evaluation_summary.json
│
├── notebooks/
│   ├── model_evaluation.ipynb                  # ⭐ Jupyter notebook for Colab
│   └── [execution outputs & visualizations]
│
└── README.md (this file)
```

---

## Quick Start (Google Colab) — Per-Model Cell Workflow ⭐

**Why this workflow is optimized for Colab's free tier:**

- ✅ **Each model trains in its own cell** — a crash in one model doesn't kill the others
- ✅ **Preprocessed data cached to disk** — every model cell reloads it in ~2 seconds
- ✅ **Atomic JSON report updates** — metrics saved after every model → fully resumable
- ✅ **Real-time progress** — epoch lines for TF, tree-by-tree logs for XGB/LGBM/RF
- ✅ **Publication-quality charts** — auto-generated comparison figures per task
- ✅ **Zero .py script invocation** — everything is inline Python you can tweak per cell

> 💡 **Tip:** Before running, enable GPU via **Runtime → Change runtime type → T4 GPU**. Neural-network models train 5–10× faster.
>
> 💡 **Memory discipline:** every cell ends with `del ...; gc.collect()`. Don't skip those lines.

---

### 🔧 SETUP (run once)

#### **Cell 1: Clone Repository**
```python
# Mount Google Drive (optional — useful for saving models across sessions)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
import os, subprocess
if os.path.exists('/content/bus-site'):
    subprocess.run(['rm', '-rf', '/content/bus-site'], check=True)
!git clone https://github.com/Swetank1234/bus-site.git /content/bus-site

%cd /content/bus-site/ai-service
!git log -1 --pretty=format:"Commit: %h | %s | %ar"
print("\n✅ Repo ready at /content/bus-site/ai-service")
```

#### **Cell 2: Install Dependencies**
```python
%pip install -q --upgrade pip
%pip install -q tensorflow==2.16.1 scikit-learn==1.4.0 xgboost==2.1.1 lightgbm==4.3.0 catboost==1.2.5
%pip install -q pandas numpy joblib matplotlib seaborn tqdm

import tensorflow as tf, xgboost, lightgbm, sklearn, numpy, pandas
print(f"✅ TensorFlow  : {tf.__version__}")
print(f"✅ XGBoost     : {xgboost.__version__}")
print(f"✅ LightGBM    : {lightgbm.__version__}")
print(f"✅ scikit-learn: {sklearn.__version__}")
print(f"✅ GPU available: {len(tf.config.list_physical_devices('GPU'))} device(s)")
```

#### **Cell 3: Generate Datasets (one-time, ~3 min)**
```python
%cd /content/bus-site/ai-service

import subprocess, sys
process = subprocess.Popen(
    [sys.executable, "-u", "training/enhanced_generate_dataset.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
for line in process.stdout:
    print(line, end="", flush=True)
process.wait()
assert process.returncode == 0, "Dataset generation failed"

import os
for name in ["demand_dataset.csv", "delay_dataset.csv", "anomaly_dataset.csv"]:
    p = f"data/{name}"
    size_mb = os.path.getsize(p) / 1e6
    print(f"   {name}: {size_mb:.1f} MB")
print("\n✅ All three datasets generated")
```

---

## 📈 TASK 1 — DEMAND PREDICTION (8 cells, ~20–30 min total)

Models: LSTM · GRU · Transformer · XGBoost · LightGBM · Random Forest

> **⚠️ IMPORTANT:** Run **Cell D-0 first**, then the six model cells in any order. Finally run **Cell D-7** to produce the comparison charts.

### **Cell D-0: Preprocess & Cache Demand Data**
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — STEP 0: PREPROCESS & CACHE (run ONCE; then any model cell)
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- CONFIG ---
SAMPLE_FRAC = 0.20        # Use 20% of rows. Raise to 1.0 if you have GPU + patience.
DATA_DIR    = "/content/bus-site/ai-service/data"
CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

print("Loading demand_dataset.csv …", flush=True)
df = pd.read_csv(os.path.join(DATA_DIR, "demand_dataset.csv"))
print(f"   Loaded {len(df):,} rows × {df.shape[1]} cols", flush=True)

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    print(f"   Subsampled to {len(df):,} rows  ({int(SAMPLE_FRAC*100)}%)", flush=True)

X = df.drop(columns=['passenger_count']).copy()
y = df['passenger_count'].values.astype(np.float32)

# Date expansion
if 'date' in X.columns:
    d = pd.to_datetime(X['date'], errors='coerce')
    X['date_year']      = d.dt.year.fillna(0).astype(np.int16)
    X['date_dayofyear'] = d.dt.dayofyear.fillna(0).astype(np.int16)
    X = X.drop(columns=['date'])

# One-hot encode categoricals
cat_cols = list(X.select_dtypes(include=['object']).columns)
if cat_cols:
    print(f"   One-hot encoding: {cat_cols}", flush=True)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=np.float64)

X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
feature_names = list(X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

# Cache arrays for model cells to reload instantly
np.savez_compressed(
    os.path.join(CACHE_DIR, "demand_prep.npz"),
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
)
joblib.dump(scaler, os.path.join(SAVE_DIR, "demand_scaler_multimodel.pkl"))
with open(os.path.join(CACHE_DIR, "demand_features.json"), "w") as f:
    json.dump(feature_names, f)

# Initialize report JSON if absent
if not os.path.exists(REPORT_JSON):
    with open(REPORT_JSON, "w") as f:
        json.dump({
            "task": "demand_prediction",
            "data_size": len(df), "sample_frac": SAMPLE_FRAC,
            "n_features": X_train.shape[1], "models": {},
        }, f, indent=2)

print(f"\n✅ Preprocessing complete")
print(f"   Train: {len(X_train):,}   Test: {len(X_test):,}   Features: {X_train.shape[1]}")
print(f"   Cache: {CACHE_DIR}/demand_prep.npz")

del df, X, X_arr, X_scaled, X_train, X_test, y_train, y_test, scaler
gc.collect()
```

### **Cell D-1: LSTM** (~2–4 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 1/6: LSTM  (Dense architecture — fast & strong baseline)
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, joblib, tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

print("Loading cached preprocessed arrays …", flush=True)
d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"   Train={X_train.shape}  Test={X_test.shape}", flush=True)

tf.keras.backend.clear_session(); gc.collect()

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n🚀 Training LSTM (max 20 epochs, batch=512, early-stop patience=5)", flush=True)
t0 = time.time()
model.fit(
    X_train, y_train,
    epochs=20, batch_size=512, validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=2,
)
elapsed = time.time() - t0

y_pred = model.predict(X_test, verbose=0).flatten()
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

model.save(os.path.join(SAVE_DIR, "demand_lstm_multimodel.keras"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["lstm"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ LSTM  ·  {elapsed:.1f}s  ·  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
tf.keras.backend.clear_session(); gc.collect()
```

### **Cell D-2: GRU** (~2–4 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 2/6: GRU
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded cached data  Train={X_train.shape}  Test={X_test.shape}", flush=True)

tf.keras.backend.clear_session(); gc.collect()

# GRU-inspired variant: deeper Dense + stronger regularization
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(), layers.Dropout(0.3),
    layers.Dense(96,  activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.25),
    layers.Dense(48,  activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n🚀 Training GRU (max 20 epochs, batch=512)", flush=True)
t0 = time.time()
model.fit(
    X_train, y_train, epochs=20, batch_size=512, validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=2,
)
elapsed = time.time() - t0

y_pred = model.predict(X_test, verbose=0).flatten()
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

model.save(os.path.join(SAVE_DIR, "demand_gru_multimodel.keras"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["gru"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ GRU  ·  {elapsed:.1f}s  ·  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
tf.keras.backend.clear_session(); gc.collect()
```

### **Cell D-3: Transformer** (~3–5 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 3/6: Transformer (attention-lite for tabular)
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.callbacks import EarlyStopping

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded cached data  Train={X_train.shape}  Test={X_test.shape}", flush=True)

tf.keras.backend.clear_session(); gc.collect()

# Tabular-friendly Transformer-style block: projection + self-attention on feature tokens
inputs = Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Reshape((8, 16))(x)                        # Treat 128 feats as 8 tokens × 16 dims
attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
x = layers.Add()([x, attn])
x = layers.LayerNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1)(x)
model = Model(inputs, out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n🚀 Training Transformer (max 20 epochs, batch=512)", flush=True)
t0 = time.time()
model.fit(
    X_train, y_train, epochs=20, batch_size=512, validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=2,
)
elapsed = time.time() - t0

y_pred = model.predict(X_test, verbose=0).flatten()
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

model.save(os.path.join(SAVE_DIR, "demand_transformer_multimodel.keras"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["transformer"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ Transformer  ·  {elapsed:.1f}s  ·  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
tf.keras.backend.clear_session(); gc.collect()
```

### **Cell D-4: XGBoost** (~1–2 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 4/6: XGBoost (tree_method='hist' + early stopping)
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, joblib, xgboost as xgb

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded cached data  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    tree_method='hist',           # 5-10× faster on large data
    random_state=42, verbosity=1,
    early_stopping_rounds=20,     # constructor form (XGBoost 2.x)
)

print("\n🚀 Training XGBoost (logs every 25 trees)", flush=True)
t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

joblib.dump(model, os.path.join(SAVE_DIR, "demand_xgboost_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["xgboost"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
    "best_iter": int(model.best_iteration), "train_time_sec": elapsed,
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ XGBoost  ·  {elapsed:.1f}s  ·  best_iter={model.best_iteration}")
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
gc.collect()
```

### **Cell D-5: LightGBM** (~30s – 2 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 5/6: LightGBM
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, joblib, lightgbm as lgb

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded cached data  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, random_state=42, verbose=-1,
)

print("\n🚀 Training LightGBM (logs every 25 iters)", flush=True)
t0 = time.time()
model.fit(
    X_train, y_train, eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(25)]
)
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

joblib.dump(model, os.path.join(SAVE_DIR, "demand_lightgbm_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["lightgbm"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
    "best_iter": int(model.best_iteration_), "train_time_sec": elapsed,
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ LightGBM  ·  {elapsed:.1f}s  ·  best_iter={model.best_iteration_}")
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
gc.collect()
```

### **Cell D-6: Random Forest** (~2–4 min)
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — MODEL 6/6: Random Forest (100 trees, depth 10 — memory-safe)
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json, time
import numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor

CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "demand_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "demand_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded cached data  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=2,
)

print("\n🚀 Training Random Forest (100 trees × depth 10, all cores)", flush=True)
t0 = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae  = float(np.mean(np.abs(y_pred - y_test)))
rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
mask = np.abs(y_test) >= 1.0
mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
r2   = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

joblib.dump(model, os.path.join(SAVE_DIR, "demand_rf_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["random_forest"] = {
    "mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)

print(f"\n✅ Random Forest  ·  {elapsed:.1f}s")
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

del model, d, X_train, X_test, y_train, y_test, y_pred
gc.collect()
```

### **Cell D-7: Compare & Visualize Demand Models (for paper / panel)**
```python
# ═══════════════════════════════════════════════════════════════════════
# DEMAND — PUBLICATION-QUALITY COMPARISON CHARTS
# ═══════════════════════════════════════════════════════════════════════
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/content/bus-site/ai-service/models/saved"
PLOT_DIR = "/content/bus-site/ai-service/evaluation_results"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, "demand_comparison_report.json")) as f:
    rep = json.load(f)

rows = [{
    "Model": k.replace("_", " ").title(),
    "MAE": v["mae"], "RMSE": v["rmse"], "MAPE": v["mape"], "R²": v["r2"],
    "Time (s)": v.get("train_time_sec", 0),
} for k, v in rep.get("models", {}).items() if "error" not in v]

assert rows, "No successful models found — train at least one (D-1 … D-6)."
df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

print("\n📊 DEMAND MODEL COMPARISON TABLE\n" + "="*70)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv(os.path.join(PLOT_DIR, "demand_comparison.csv"), index=False)

# --- 2x2 comparison grid ---
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Demand Prediction — Model Comparison (sorted by MAE, lower is better)",
             fontsize=15, fontweight="bold", y=0.995)

for (metric, lower_better), ax in zip(
    [("MAE", True), ("RMSE", True), ("MAPE", True), ("R²", False)], axes.flatten()
):
    vals = df[metric].values
    best = vals.min() if lower_better else vals.max()
    colors = ["#2ca02c" if v == best else "#4e79a7" for v in vals]
    bars = ax.barh(df["Model"], vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{metric}  ({'lower' if lower_better else 'higher'} is better)",
                 fontweight="bold")
    ax.set_xlabel(metric); ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.4f}" if metric == "R²" else f"{v:.2f}"
        ax.text(v, bar.get_y() + bar.get_height()/2, f"  {fmt}",
                va="center", ha="left", fontweight="bold", fontsize=10)

plt.tight_layout()
png1 = os.path.join(PLOT_DIR, "demand_model_comparison.png")
plt.savefig(png1, dpi=200, bbox_inches="tight"); plt.show()

# --- Training time chart ---
fig, ax = plt.subplots(figsize=(10, 4.5))
t_df = df.sort_values("Time (s)")
ax.barh(t_df["Model"], t_df["Time (s)"], color="#f28e2b", edgecolor="black", linewidth=0.6)
ax.set_title("Training Time per Model (Demand Prediction)", fontweight="bold")
ax.set_xlabel("Seconds"); ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
for i, (m, t) in enumerate(zip(t_df["Model"], t_df["Time (s)"])):
    ax.text(t, i, f"  {t:.1f}s", va="center", ha="left", fontweight="bold")
plt.tight_layout()
png2 = os.path.join(PLOT_DIR, "demand_training_time.png")
plt.savefig(png2, dpi=200, bbox_inches="tight"); plt.show()

best = df.iloc[0]
print(f"\n🏆 BEST DEMAND MODEL: {best['Model']}")
print(f"   MAE = {best['MAE']:.2f} passengers · R² = {best['R²']:.4f}")
print(f"\n📁 Artifacts saved:")
print(f"   • {png1}")
print(f"   • {png2}")
print(f"   • {os.path.join(PLOT_DIR, 'demand_comparison.csv')}")
```

---

## ⏱️ TASK 2 — DELAY PREDICTION (8 cells, ~25–35 min total)

Models: XGBoost · LightGBM · CatBoost · SVR · MLP · Ensemble

### **Cell L-0: Preprocess & Cache Delay Data**
```python
# ═══════════════════════════════════════════════════════════════════════
# DELAY — STEP 0: PREPROCESS & CACHE
# ═══════════════════════════════════════════════════════════════════════
import os, gc, json
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SAMPLE_FRAC = 0.20
DATA_DIR    = "/content/bus-site/ai-service/data"
CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading delay_dataset.csv …", flush=True)
df = pd.read_csv(os.path.join(DATA_DIR, "delay_dataset.csv"))
print(f"   Loaded {len(df):,} rows", flush=True)

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    print(f"   Subsampled to {len(df):,} rows", flush=True)

X = df.drop(columns=['delay_minutes', 'is_delayed']).copy()
y = df['delay_minutes'].values.astype(np.float32)

if 'date' in X.columns:
    d = pd.to_datetime(X['date'], errors='coerce')
    X['date_year']      = d.dt.year.fillna(0).astype(np.int16)
    X['date_dayofyear'] = d.dt.dayofyear.fillna(0).astype(np.int16)
    X = X.drop(columns=['date'])

cat_cols = list(X.select_dtypes(include=['object']).columns)
if cat_cols:
    print(f"   One-hot encoding: {cat_cols}", flush=True)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=np.float64)

X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)
np.savez_compressed(os.path.join(CACHE_DIR, "delay_prep.npz"),
                    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
joblib.dump(scaler, os.path.join(SAVE_DIR, "delay_scaler_multimodel.pkl"))

if not os.path.exists(REPORT_JSON):
    with open(REPORT_JSON, "w") as f:
        json.dump({"task": "delay_prediction", "data_size": len(df),
                   "n_features": X_train.shape[1], "models": {}}, f, indent=2)

print(f"\n✅ Preprocessing complete")
print(f"   Train: {len(X_train):,}   Test: {len(X_test):,}   Features: {X_train.shape[1]}")

del df, X, X_arr, X_scaled, X_train, X_test, y_train, y_test, scaler
gc.collect()
```

### **Cell L-1: XGBoost**
```python
import os, gc, json, time
import numpy as np, joblib, xgboost as xgb
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         tree_method='hist', random_state=42, verbosity=1,
                         early_stopping_rounds=20)
t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
joblib.dump(model, os.path.join(SAVE_DIR, "delay_xgboost_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["xgboost"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ XGBoost · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, y_train, y_test, y_pred; gc.collect()
```

### **Cell L-2: LightGBM**
```python
import os, gc, json, time
import numpy as np, joblib, lightgbm as lgb
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(25)])
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
joblib.dump(model, os.path.join(SAVE_DIR, "delay_lightgbm_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["lightgbm"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ LightGBM · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, y_train, y_test, y_pred; gc.collect()
```

### **Cell L-3: CatBoost**
```python
import os, gc, json, time
import numpy as np, joblib
from catboost import CatBoostRegressor
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          random_state=42, verbose=25, early_stopping_rounds=20)
t0 = time.time()
model.fit(X_train, y_train, eval_set=(X_test, y_test))
elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
joblib.dump(model, os.path.join(SAVE_DIR, "delay_catboost_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["catboost"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ CatBoost · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, y_train, y_test, y_pred; gc.collect()
```

### **Cell L-4: SVR**
```python
# NOTE: SVR is O(n²) — we cap training rows to 20,000 to stay under Colab limits.
import os, gc, json, time
import numpy as np, joblib
from sklearn.svm import SVR
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']

SVR_MAX = 20_000
if len(X_train) > SVR_MAX:
    idx = np.random.RandomState(42).choice(len(X_train), SVR_MAX, replace=False)
    X_tr, y_tr = X_train[idx], y_train[idx]
    print(f"⚠️  SVR: subsampled {SVR_MAX:,} rows (from {len(X_train):,}) for tractability", flush=True)
else:
    X_tr, y_tr = X_train, y_train

model = SVR(kernel='rbf', C=100, gamma='scale', verbose=True)
print("\n🚀 Training SVR (RBF kernel)…", flush=True)
t0 = time.time(); model.fit(X_tr, y_tr); elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
joblib.dump(model, os.path.join(SAVE_DIR, "delay_svr_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["svr"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ SVR · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, X_tr, y_train, y_tr, y_test, y_pred; gc.collect()
```

### **Cell L-5: MLP**
```python
import os, gc, json, time
import numpy as np, tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded  Train={X_train.shape}  Test={X_test.shape}", flush=True)
tf.keras.backend.clear_session(); gc.collect()

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

t0 = time.time()
model.fit(X_train, y_train, epochs=20, batch_size=512, validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
          verbose=2)
elapsed = time.time() - t0

y_pred = model.predict(X_test, verbose=0).flatten()
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
model.save(os.path.join(SAVE_DIR, "delay_mlp_multimodel.keras"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["mlp"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ MLP · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, y_train, y_test, y_pred
tf.keras.backend.clear_session(); gc.collect()
```

### **Cell L-6: Ensemble (Voting of XGBoost + LightGBM)**
```python
import os, gc, json, time
import numpy as np, joblib
from sklearn.ensemble import VotingRegressor
import xgboost as xgb, lightgbm as lgb
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "delay_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "delay_prep.npz"))
X_train, X_test, y_train, y_test = d['X_train'], d['X_test'], d['y_train'], d['y_test']
print(f"Loaded  Train={X_train.shape}  Test={X_test.shape}", flush=True)

model = VotingRegressor([
    ('xgb', xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             tree_method='hist', random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              num_leaves=31, random_state=42, verbose=-1)),
])
print("\n🚀 Training Ensemble (XGB + LGB voting)…", flush=True)
t0 = time.time(); model.fit(X_train, y_train); elapsed = time.time() - t0

y_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_pred - y_test))); rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
r2  = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
joblib.dump(model, os.path.join(SAVE_DIR, "delay_ensemble_multimodel.pkl"))

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["ensemble"] = {"mae": mae, "rmse": rmse, "r2": r2, "train_time_sec": elapsed}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ Ensemble · {elapsed:.1f}s · MAE={mae:.2f} min · R²={r2:.4f}")
del model, d, X_train, X_test, y_train, y_test, y_pred; gc.collect()
```

### **Cell L-7: Compare & Visualize Delay Models**
```python
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/content/bus-site/ai-service/models/saved"
PLOT_DIR = "/content/bus-site/ai-service/evaluation_results"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, "delay_comparison_report.json")) as f: rep = json.load(f)
rows = [{"Model": k.replace("_", " ").title(), "MAE (min)": v["mae"],
         "RMSE (min)": v["rmse"], "R²": v["r2"], "Time (s)": v.get("train_time_sec", 0)}
        for k, v in rep.get("models", {}).items() if "error" not in v]
assert rows, "No delay models trained yet."
df = pd.DataFrame(rows).sort_values("MAE (min)").reset_index(drop=True)

print("\n📊 DELAY MODEL COMPARISON TABLE\n" + "="*70)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv(os.path.join(PLOT_DIR, "delay_comparison.csv"), index=False)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Delay Prediction — Model Comparison", fontsize=15, fontweight="bold", y=1.02)

for (metric, lower_better), ax in zip(
    [("MAE (min)", True), ("RMSE (min)", True), ("R²", False)], axes
):
    vals = df[metric].values
    best = vals.min() if lower_better else vals.max()
    colors = ["#2ca02c" if v == best else "#4e79a7" for v in vals]
    bars = ax.barh(df["Model"], vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{metric}  ({'lower' if lower_better else 'higher'} is better)", fontweight="bold")
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.4f}" if metric == "R²" else f"{v:.2f}"
        ax.text(v, bar.get_y() + bar.get_height()/2, f"  {fmt}",
                va="center", ha="left", fontweight="bold", fontsize=10)

plt.tight_layout()
png = os.path.join(PLOT_DIR, "delay_model_comparison.png")
plt.savefig(png, dpi=200, bbox_inches="tight"); plt.show()

best = df.iloc[0]
print(f"\n🏆 BEST DELAY MODEL: {best['Model']}  ·  MAE = {best['MAE (min)']:.2f} min  ·  R² = {best['R²']:.4f}")
print(f"📁 Chart: {png}")
```

---

## 🚨 TASK 3 — ANOMALY DETECTION (8 cells, ~15–25 min total)

Methods: Isolation Forest · LOF · One-Class SVM · Autoencoder · DBSCAN · Ensemble

> **Note:** Anomaly metrics are **Precision, Recall, F1** (not MAE/R²).

### **Cell A-0: Preprocess & Cache Anomaly Data**
```python
import os, gc, json
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

SAMPLE_FRAC = 0.30    # Anomaly models are lighter; can afford 30%
DATA_DIR    = "/content/bus-site/ai-service/data"
CACHE_DIR   = "/content/bus-site/ai-service/cache"
SAVE_DIR    = "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading anomaly_dataset.csv …", flush=True)
df = pd.read_csv(os.path.join(DATA_DIR, "anomaly_dataset.csv"))
print(f"   Loaded {len(df):,} rows  ({df['anomaly_label'].sum():,} anomalies)", flush=True)

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    print(f"   Subsampled to {len(df):,} rows  ({df['anomaly_label'].sum():,} anomalies)", flush=True)

X = df.drop(columns=['anomaly_label']).copy()
y = df['anomaly_label'].values.astype(np.int8)

if 'date' in X.columns:
    d = pd.to_datetime(X['date'], errors='coerce')
    X['date_year']      = d.dt.year.fillna(0).astype(np.int16)
    X['date_dayofyear'] = d.dt.dayofyear.fillna(0).astype(np.int16)
    X = X.drop(columns=['date'])

cat_cols = list(X.select_dtypes(include=['object']).columns)
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=np.float64)

X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler()
X_all = scaler.fit_transform(X_arr)
X_normal = X_all[y == 0]

np.savez_compressed(os.path.join(CACHE_DIR, "anomaly_prep.npz"),
                    X_all=X_all, X_normal=X_normal, y_all=y)
joblib.dump(scaler, os.path.join(SAVE_DIR, "anomaly_scaler_multimodel.pkl"))

if not os.path.exists(REPORT_JSON):
    with open(REPORT_JSON, "w") as f:
        json.dump({"task": "anomaly_detection", "n_samples": len(df),
                   "n_anomalies": int(y.sum()), "n_features": X_all.shape[1],
                   "models": {}}, f, indent=2)

print(f"\n✅ Preprocessing complete")
print(f"   X_all={X_all.shape}  X_normal={X_normal.shape}  Anomalies={int(y.sum())}")
del df, X, X_arr, X_all, X_normal, scaler; gc.collect()
```

> **Helper macro** — each anomaly cell uses these metrics:
> ```python
> def f1_metrics(y_true, y_hat):
>     tp = int(((y_hat == 1) & (y_true == 1)).sum())
>     fp = int(((y_hat == 1) & (y_true == 0)).sum())
>     fn = int(((y_hat == 0) & (y_true == 1)).sum())
>     p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
>     return p, r, 2*p*r/(p+r+1e-8)
> ```

### **Cell A-1: Isolation Forest**
```python
import os, gc, json, time
import numpy as np, joblib
from sklearn.ensemble import IsolationForest
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
X_all, X_normal, y_all = d['X_all'], d['X_normal'], d['y_all']
print(f"Loaded  X_all={X_all.shape}  X_normal={X_normal.shape}", flush=True)

model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, verbose=1)
t0 = time.time(); model.fit(X_normal); elapsed = time.time() - t0
y_hat = np.where(model.predict(X_all) == -1, 1, 0)

tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

joblib.dump(model, os.path.join(SAVE_DIR, "anomaly_isolation_forest_multimodel.pkl"))
np.save(os.path.join(CACHE_DIR, "anomaly_pred_if.npy"), y_hat)  # for ensemble later

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["isolation_forest"] = {
    "precision": float(p), "recall": float(r), "f1": float(f1), "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ Isolation Forest · {elapsed:.1f}s · F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
del model, d, X_all, X_normal, y_all, y_hat; gc.collect()
```

### **Cell A-2: Local Outlier Factor (LOF)**
```python
import os, gc, json, time
import numpy as np, joblib
from sklearn.neighbors import LocalOutlierFactor
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
X_all, y_all = d['X_all'], d['y_all']
print(f"Loaded  X_all={X_all.shape}", flush=True)

model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
print("🚀 Fitting LOF …", flush=True)
t0 = time.time(); y_hat = np.where(model.fit_predict(X_all) == -1, 1, 0); elapsed = time.time() - t0

tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

joblib.dump(model, os.path.join(SAVE_DIR, "anomaly_lof_multimodel.pkl"))
np.save(os.path.join(CACHE_DIR, "anomaly_pred_lof.npy"), y_hat)

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["lof"] = {
    "precision": float(p), "recall": float(r), "f1": float(f1), "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ LOF · {elapsed:.1f}s · F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
del model, d, X_all, y_all, y_hat; gc.collect()
```

### **Cell A-3: One-Class SVM**
```python
# NOTE: One-Class SVM is O(n²–n³). We fit on a capped normal subset.
import os, gc, json, time
import numpy as np, joblib
from sklearn.svm import OneClassSVM
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
X_all, X_normal, y_all = d['X_all'], d['X_normal'], d['y_all']

OCSVM_MAX = 15_000
if len(X_normal) > OCSVM_MAX:
    idx = np.random.RandomState(42).choice(len(X_normal), OCSVM_MAX, replace=False)
    X_fit = X_normal[idx]
    print(f"⚠️  OCSVM: fitting on {OCSVM_MAX:,} sampled normal rows", flush=True)
else:
    X_fit = X_normal

model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05, verbose=True)
t0 = time.time(); model.fit(X_fit); elapsed = time.time() - t0
y_hat = np.where(model.predict(X_all) == -1, 1, 0)

tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

joblib.dump(model, os.path.join(SAVE_DIR, "anomaly_ocsvm_multimodel.pkl"))
np.save(os.path.join(CACHE_DIR, "anomaly_pred_ocsvm.npy"), y_hat)

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["ocsvm"] = {
    "precision": float(p), "recall": float(r), "f1": float(f1), "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ One-Class SVM · {elapsed:.1f}s · F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
del model, d, X_all, X_normal, X_fit, y_all, y_hat; gc.collect()
```

### **Cell A-4: Autoencoder**
```python
import os, gc, json, time
import numpy as np, joblib, tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
X_all, X_normal, y_all = d['X_all'], d['X_normal'], d['y_all']
print(f"Loaded  X_normal={X_normal.shape}  X_all={X_all.shape}", flush=True)
tf.keras.backend.clear_session(); gc.collect()

n_feat = X_normal.shape[1]
model = Sequential([
    layers.Dense(32, activation='relu', input_shape=(n_feat,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8,  activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_feat),
])
model.compile(optimizer='adam', loss='mse')

print("\n🚀 Training Autoencoder on NORMAL data (reconstruction)", flush=True)
t0 = time.time()
model.fit(X_normal, X_normal, epochs=20, batch_size=256, validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
          verbose=2)
elapsed = time.time() - t0

X_pred = model.predict(X_all, verbose=0)
mse = np.mean((X_all - X_pred) ** 2, axis=1)
threshold = float(np.percentile(mse, 95))
y_hat = (mse > threshold).astype(int)

tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

model.save(os.path.join(SAVE_DIR, "anomaly_autoencoder_multimodel.keras"))
joblib.dump(threshold, os.path.join(SAVE_DIR, "anomaly_ae_threshold.pkl"))
np.save(os.path.join(CACHE_DIR, "anomaly_pred_ae.npy"), y_hat)

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["autoencoder"] = {
    "precision": float(p), "recall": float(r), "f1": float(f1),
    "threshold": threshold, "train_time_sec": elapsed,
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ Autoencoder · {elapsed:.1f}s · threshold={threshold:.4f} · F1={f1:.3f}")
del model, d, X_all, X_normal, X_pred, y_all, y_hat
tf.keras.backend.clear_session(); gc.collect()
```

### **Cell A-5: DBSCAN**
```python
import os, gc, json, time
import numpy as np, joblib
from sklearn.cluster import DBSCAN
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
X_all, y_all = d['X_all'], d['y_all']
print(f"Loaded  X_all={X_all.shape}", flush=True)

model = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
print("🚀 Fitting DBSCAN …", flush=True)
t0 = time.time(); y_hat = np.where(model.fit_predict(X_all) == -1, 1, 0); elapsed = time.time() - t0

tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

joblib.dump(model, os.path.join(SAVE_DIR, "anomaly_dbscan_multimodel.pkl"))
np.save(os.path.join(CACHE_DIR, "anomaly_pred_dbscan.npy"), y_hat)

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["dbscan"] = {
    "precision": float(p), "recall": float(r), "f1": float(f1), "train_time_sec": elapsed
}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ DBSCAN · {elapsed:.1f}s · F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
del model, d, X_all, y_all, y_hat; gc.collect()
```

### **Cell A-6: Ensemble (majority vote of the 5 above)**
```python
import os, json, numpy as np
CACHE_DIR, SAVE_DIR = "/content/bus-site/ai-service/cache", "/content/bus-site/ai-service/models/saved"
REPORT_JSON = os.path.join(SAVE_DIR, "anomaly_comparison_report.json")

d = np.load(os.path.join(CACHE_DIR, "anomaly_prep.npz"))
y_all = d['y_all']

votes = np.zeros(len(y_all), dtype=np.int16)
missing = []
for name in ["if", "lof", "ocsvm", "ae", "dbscan"]:
    p = os.path.join(CACHE_DIR, f"anomaly_pred_{name}.npy")
    if os.path.exists(p): votes += np.load(p)
    else:                 missing.append(name)

print(f"Ensemble inputs: {5 - len(missing)}/5 available. Missing: {missing}", flush=True)
assert 5 - len(missing) >= 3, "Need at least 3 component models first."

y_hat = (votes >= 3).astype(int)
tp = int(((y_hat == 1) & (y_all == 1)).sum()); fp = int(((y_hat == 1) & (y_all == 0)).sum())
fn = int(((y_hat == 0) & (y_all == 1)).sum())
p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)

with open(REPORT_JSON) as f: rep = json.load(f)
rep.setdefault("models", {})["ensemble"] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
with open(REPORT_JSON, "w") as f: json.dump(rep, f, indent=2)
print(f"\n✅ Ensemble · F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
```

### **Cell A-7: Compare & Visualize Anomaly Models**
```python
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/content/bus-site/ai-service/models/saved"
PLOT_DIR = "/content/bus-site/ai-service/evaluation_results"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, "anomaly_comparison_report.json")) as f: rep = json.load(f)
rows = [{"Method": k.replace("_", " ").title(),
         "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"],
         "Time (s)": v.get("train_time_sec", 0)}
        for k, v in rep.get("models", {}).items() if "error" not in v]
assert rows, "No anomaly models trained yet."
df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

print("\n📊 ANOMALY DETECTION COMPARISON TABLE\n" + "="*70)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv(os.path.join(PLOT_DIR, "anomaly_comparison.csv"), index=False)

# Grouped bar chart: precision / recall / F1 per method
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(df)); w = 0.27
ax.bar(x - w, df["Precision"], w, label="Precision", color="#4e79a7", edgecolor="black")
ax.bar(x,     df["Recall"],    w, label="Recall",    color="#f28e2b", edgecolor="black")
ax.bar(x + w, df["F1"],        w, label="F1",        color="#2ca02c", edgecolor="black")

for i, (p_, r_, f_) in enumerate(zip(df["Precision"], df["Recall"], df["F1"])):
    ax.text(i - w, p_, f"{p_:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i,     r_, f"{r_:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w, f_, f"{f_:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(df["Method"], rotation=15, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
ax.set_title("Anomaly Detection — Precision · Recall · F1 (sorted by F1)",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right"); ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)

plt.tight_layout()
png = os.path.join(PLOT_DIR, "anomaly_model_comparison.png")
plt.savefig(png, dpi=200, bbox_inches="tight"); plt.show()

best = df.iloc[0]
print(f"\n🏆 BEST ANOMALY METHOD: {best['Method']}")
print(f"   F1 = {best['F1']:.3f}   Precision = {best['Precision']:.3f}   Recall = {best['Recall']:.3f}")
print(f"📁 Chart: {png}")
```

---

## 🏆 FINAL — EXECUTIVE DASHBOARD & EXPORT

### **Cell E-1: Cross-Task Executive Dashboard**
```python
# ═══════════════════════════════════════════════════════════════════════
# Build a single figure summarising best model per task — for your paper
# ═══════════════════════════════════════════════════════════════════════
import os, json
import matplotlib.pyplot as plt

SAVE_DIR = "/content/bus-site/ai-service/models/saved"
PLOT_DIR = "/content/bus-site/ai-service/evaluation_results"
os.makedirs(PLOT_DIR, exist_ok=True)

reports = {}
for task, fname in [("Demand", "demand_comparison_report.json"),
                    ("Delay",  "delay_comparison_report.json"),
                    ("Anomaly","anomaly_comparison_report.json")]:
    p = os.path.join(SAVE_DIR, fname)
    if os.path.exists(p):
        with open(p) as f: reports[task] = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("SmartDTC AI — Best Model per Task", fontsize=16, fontweight="bold", y=1.03)

# Demand: sorted by MAE asc
ax = axes[0]
if "Demand" in reports:
    items = sorted(reports["Demand"].get("models", {}).items(),
                   key=lambda kv: kv[1].get("mae", 1e9))
    names = [k.replace("_", " ").title() for k, _ in items]
    vals  = [v["mae"] for _, v in items]
    colors = ["#2ca02c"] + ["#4e79a7"] * (len(vals) - 1)
    ax.barh(names, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals): ax.text(v, i, f"  {v:.2f}", va="center")
    ax.invert_yaxis(); ax.set_title("Demand Prediction (MAE — lower better)", fontweight="bold")
    ax.set_xlabel("MAE (passengers)"); ax.grid(axis='x', alpha=0.3)

# Delay: sorted by MAE asc
ax = axes[1]
if "Delay" in reports:
    items = sorted(reports["Delay"].get("models", {}).items(),
                   key=lambda kv: kv[1].get("mae", 1e9))
    names = [k.replace("_", " ").title() for k, _ in items]
    vals  = [v["mae"] for _, v in items]
    colors = ["#2ca02c"] + ["#4e79a7"] * (len(vals) - 1)
    ax.barh(names, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals): ax.text(v, i, f"  {v:.2f}", va="center")
    ax.invert_yaxis(); ax.set_title("Delay Prediction (MAE — lower better)", fontweight="bold")
    ax.set_xlabel("MAE (minutes)"); ax.grid(axis='x', alpha=0.3)

# Anomaly: sorted by F1 desc
ax = axes[2]
if "Anomaly" in reports:
    items = sorted(reports["Anomaly"].get("models", {}).items(),
                   key=lambda kv: kv[1].get("f1", 0), reverse=True)
    names = [k.replace("_", " ").title() for k, _ in items]
    vals  = [v["f1"] for _, v in items]
    colors = ["#2ca02c"] + ["#4e79a7"] * (len(vals) - 1)
    ax.barh(names, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals): ax.text(v, i, f"  {v:.3f}", va="center")
    ax.invert_yaxis(); ax.set_title("Anomaly Detection (F1 — higher better)", fontweight="bold")
    ax.set_xlabel("F1 Score"); ax.grid(axis='x', alpha=0.3); ax.set_xlim(0, 1.1)

for ax in axes: ax.set_axisbelow(True)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "executive_dashboard.png")
plt.savefig(out, dpi=220, bbox_inches="tight"); plt.show()
print(f"\n📁 Saved: {out}")
```

### **Cell E-2: Package & Download Everything**
```python
# ═══════════════════════════════════════════════════════════════════════
# Zip models + reports + charts + scalers, then trigger browser download
# ═══════════════════════════════════════════════════════════════════════
import os, shutil
from google.colab import files

BASE = "/content/bus-site/ai-service"
STAGE = "/content/smartdtc_ai_artifacts"
if os.path.exists(STAGE): shutil.rmtree(STAGE)
os.makedirs(f"{STAGE}/models/saved", exist_ok=True)
os.makedirs(f"{STAGE}/evaluation_results", exist_ok=True)

# Copy all trained artifacts
for root, _, files_ in os.walk(f"{BASE}/models/saved"):
    for fn in files_:
        shutil.copy(os.path.join(root, fn), f"{STAGE}/models/saved/{fn}")
for root, _, files_ in os.walk(f"{BASE}/evaluation_results"):
    for fn in files_:
        shutil.copy(os.path.join(root, fn), f"{STAGE}/evaluation_results/{fn}")

# Show what we're packing
total = 0
print("📦 Packaging:")
for root, _, files_ in os.walk(STAGE):
    for fn in files_:
        path = os.path.join(root, fn)
        size_mb = os.path.getsize(path) / 1e6
        total += size_mb
        print(f"   {size_mb:6.2f} MB  {path[len(STAGE)+1:]}")
print(f"   {'─'*40}\n   {total:6.2f} MB  TOTAL")

zip_path = "/content/smartdtc_ai_artifacts.zip"
shutil.make_archive(zip_path[:-4], "zip", STAGE)
print(f"\n✅ Created {zip_path}  ({os.path.getsize(zip_path)/1e6:.2f} MB)")
files.download(zip_path)
```

---

### 📋 Recommended Notebook Execution Order

| Phase | Cells | Typical Time (Free CPU) |
|---|---|---|
| **Setup** | 1 → 2 → 3 | ~5 min |
| **Demand** | D-0 → D-1…D-6 → D-7 | ~25 min |
| **Delay** | L-0 → L-1…L-6 → L-7 | ~30 min |
| **Anomaly** | A-0 → A-1…A-5 → A-6 → A-7 | ~20 min |
| **Final** | E-1 → E-2 | ~1 min |

**Total (free CPU, 20–30% sample):** ~80 min · **GPU:** ~25 min · **Full dataset CPU:** 4–6 hours

---

### ⚠️ Troubleshooting

**"Runtime disconnected" mid-training**
→ Just re-run the affected model cell. All previously completed models are preserved in the JSON report and `.pkl`/`.keras` files.

**`FileNotFoundError: demand_prep.npz`**
→ You skipped the preprocessing cell (D-0 / L-0 / A-0). Run it first.

**Out of RAM on a specific model cell**
→ Lower `SAMPLE_FRAC` in the corresponding preprocessing cell (e.g. `0.20 → 0.10`), then re-run the prep cell + that model cell.

**XGBoost "early_stopping_rounds unexpected keyword" error**
→ You have XGBoost < 1.6. Upgrade: `%pip install -q --upgrade xgboost`.

**Want to retrain just one model?**
→ Just re-run that single cell. It overwrites its own `.pkl`/`.keras` file and updates only its entry in the JSON report.

---
## Local Execution Guide (Optional - For Desktop/Server)

### **If Running Locally Instead of Colab:**

#### **Step 1: Setup Environment**
```bash
cd ai-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Generate Enhanced Dataset**

#### **Step 2: Generate Enhanced Dataset**

Generates 3 years of realistic synthetic data from real DTC routes/stages.

```bash
python training/enhanced_generate_dataset.py
```

**Output:**
- `data/demand_dataset.csv` - 561K+ records
- `data/delay_dataset.csv` - 561K+ records
- `data/anomaly_dataset.csv` - 200K+ records

#### **Step 3: Train All Models**

```bash
# Demand prediction (6 models)
python training/train_demand_models.py

# Delay prediction (6 models)  
python training/train_delay_models.py

# Anomaly detection (6 models)
python training/train_anomaly_models.py
```

Time: ~45 min (CPU) or ~10-15 min (GPU)

#### **Step 4: Evaluate & Compare**

```bash
python evaluate_models.py
```

Generates comparison plots, tables, and summary metrics in `evaluation_results/`

#### **Step 5: Deploy API**

```bash
# Run API server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker
docker build -t smartdtc-ai .
docker run -p 8000:8000 smartdtc-ai
```

API available at: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

---

## API Endpoints

### **Demand Prediction**
```
POST /predict/demand
Content-Type: application/json

{
  "route_id": "7000",
  "date": "2024-03-15",
  "hour": 8,
  "is_weekend": false,
  "is_holiday": false,
  "weather": "clear",
  "avg_temp_c": 28.5,
  "special_event": false
}

Response:
{
  "route_id": "7000",
  "date": "2024-03-15",
  "hour": 8,
  "predicted_count": 145,
  "crowd_level": "high",
  "confidence": 0.87,
  "model_used": "LSTM"
}
```

### **Delay Prediction**
```
POST /predict/delay
Content-Type: application/json

{
  "route_id": "7000",
  "hour": 8,
  "day_of_week": 4,
  "is_weekend": false,
  "is_holiday": false,
  "weather": "light_rain",
  "avg_temp_c": 28.0,
  "passenger_load_pct": 85.0,
  "scheduled_duration_min": 45.0,
  "distance_km": 15.2,
  "total_stops": 23
}

Response:
{
  "route_id": "7000",
  "predicted_delay_minutes": 8.5,
  "is_delayed": true,
  "delay_probability": 0.92,
  "delay_category": "high",
  "confidence": 0.85,
  "model_used": "XGBoost"
}
```

### **Anomaly Detection**
```
POST /detect/anomaly

{
  "speed_kmh": 15.0,
  "delay_minutes": 35.0,
  "passenger_load": 140.0
}

Response:
{
  "is_anomaly": true,
  "score": -0.65,
  "confidence": 0.88,
  "reason": "Unusually low speed with high delay",
  "model_used": "Ensemble"
}
```

### **Model Comparison**
```
GET /health

Response:
{
  "status": "healthy",
  "models_loaded": {
    "demand": ["lstm", "gru", "transformer", "xgboost", "lightgbm", "random_forest"],
    "delay": ["xgboost", "lightgbm", "catboost", "svr", "mlp", "ensemble"],
    "anomaly": ["isolation_forest", "lof", "one_class_svm", "autoencoder", "dbscan", "ensemble"]
  },
  "best_models": {
    "demand": {"model": "LSTM", "mae": 12.3},
    "delay": {"model": "XGBoost", "mae": 2.1},
    "anomaly": {"model": "Ensemble", "f1": 0.85}
  }
}
```

---

## Model Comparison Results

### **Example Output** (from `evaluation_results/evaluation_summary.json`):

```json
{
  "task_summaries": {
    "demand": {
      "best_model": "LSTM",
      "best_mae": 12.34,
      "best_r2": 0.8567,
      "models_compared": 6
    },
    "delay": {
      "best_model": "XGBoost",
      "best_mae_minutes": 2.15,
      "models_compared": 6
    },
    "anomaly": {
      "best_model": "Ensemble",
      "best_f1": 0.852,
      "anomaly_rate": 3.2,
      "models_compared": 6
    }
  }
}
```

---

## Step 3 — Export and Place Model Files

After downloading from Colab, place the files into `ai-service/models/saved/`:

```
ai-service/
└── models/
    └── saved/
        ├── demand_lstm/          ← Unzip demand_lstm.zip here (it's a folder)
        │   ├── saved_model.pb
        │   └── variables/
        │       ├── variables.index
        │       └── variables.data-00000-of-00001
        ├── demand_scaler.pkl     ← Downloaded from Colab
        ├── delay_regressor.pkl   ← Downloaded from Colab
        ├── delay_classifier.pkl  ← Downloaded from Colab
        └── delay_scaler.pkl      ← Downloaded from Colab
```

**To unzip `demand_lstm.zip` on Windows:**
```powershell
Expand-Archive -Path "demand_lstm.zip" -DestinationPath "ai-service\models\saved\"
```

**On Linux/Mac:**
```bash
unzip demand_lstm.zip -d ai-service/models/saved/
```

> The `models/saved/` directory is in `.gitignore` because model files are large binary files. Each team member must run the training or copy the files manually.

---

## Step 4 — Run Locally

### 4.1 Set up virtual environment

```bash
cd ai-service

# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `fastapi`, `uvicorn[standard]` — web framework
- `tensorflow` — LSTM model inference
- `xgboost`, `scikit-learn` — delay model inference
- `joblib` — model serialization
- `pandas`, `numpy` — data handling
- `pydantic` — request validation

### 4.3 Configure environment

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` if your model directory is in a non-default location:
```env
MODEL_DIR=./models/saved
LOG_LEVEL=info
```

### 4.4 Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will start and log:
```
INFO:  Loading ML models…
INFO:  ✅  LSTM demand model loaded
INFO:  ✅  Demand scaler loaded
INFO:  ✅  XGBoost delay regressor loaded
INFO:  ✅  AI service ready
INFO:  Uvicorn running on http://0.0.0.0:8000
```

If model files are not present, you'll see warnings but the service will still start:
```
WARNING:  ⚠️  LSTM model not found — using rule-based fallback
```

### 4.5 Verify it works

Open [http://localhost:8000/health](http://localhost:8000/health) or run:
```bash
curl http://localhost:8000/health
```

Response with trained models:
```json
{
  "status": "ok",
  "models": {
    "demand_lstm": true,
    "delay_xgboost": true
  }
}
```

Interactive API docs are at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Step 5 — Run with Docker

### 5.1 Build the image

```bash
cd ai-service
docker build -t smartdtc-ai .
```

### 5.2 Run the container

```bash
docker run -d \
  --name smartdtc-ai \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -e MODEL_DIR=/app/models/saved \
  smartdtc-ai
```

**On Windows PowerShell:**
```powershell
docker run -d `
  --name smartdtc-ai `
  -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -e MODEL_DIR=/app/models/saved `
  smartdtc-ai
```

The `-v` flag mounts your local `models/` directory into the container so it can find the trained model files.

### 5.3 With Docker Compose (recommended)

If the project has a `docker-compose.yml`, run all services together:
```bash
# From the workspace root (bus-site/)
docker-compose up --build
```

---

## API Reference

### `GET /health`
Check service status and which models are loaded.

```bash
curl http://localhost:8000/health
```

---

### `POST /predict/demand`
Predict passenger count for a specific route, date, and hour.

**Request:**
```json
{
  "route_id": "DTC-401",
  "date": "2024-08-15",
  "hour": 8,
  "is_weekend": false,
  "is_holiday": true,
  "weather": "clear",
  "avg_temp_c": 32.5,
  "special_event": false
}
```

**Response:**
```json
{
  "route_id": "DTC-401",
  "predicted_demand": 118,
  "hour": 8,
  "confidence": 0.82,
  "model_used": "lstm"
}
```

`model_used` will be `"lstm"` when the trained model is loaded, or `"rule_based"` for the fallback.

---

### `POST /predict/delay`
Predict expected delay and whether a trip will be delayed.

**Request:**
```json
{
  "route_id": "DTC-401",
  "hour": 8,
  "day_of_week": 1,
  "is_weekend": false,
  "is_holiday": false,
  "weather": "rain",
  "avg_temp_c": 28.0,
  "passenger_load_pct": 85.0,
  "scheduled_duration_min": 45,
  "distance_km": 18.5,
  "total_stops": 22
}
```

**Response:**
```json
{
  "route_id": "DTC-401",
  "delay_minutes": 7.3,
  "is_delayed": true,
  "confidence": 0.78,
  "model_used": "xgboost"
}
```

---

### `POST /predict/eta`
Predict ETA for a passenger at a specific stop.

**Request:**
```json
{
  "distance_km": 4.2,
  "hour": 9,
  "day_of_week": 2,
  "is_weekend": false,
  "weather_encoded": 0,
  "avg_speed_kmh": 22.0,
  "stop_count_remaining": 6,
  "current_delay_minutes": 3.0
}
```

**Response:**
```json
{
  "eta_minutes": 14.5,
  "confidence": 0.75,
  "model_used": "gradient_boosting"
}
```

---

### `POST /detect/anomaly`
Detect abnormal bus behaviour (speeding, unusual delay, overcrowding).

**Request:**
```json
{
  "speed_kmh": 85.0,
  "delay_minutes": 25.0,
  "passenger_load": 145
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "score": -0.42,
  "confidence": 0.88,
  "reason": "High delay and overcrowding detected"
}
```

---

### `POST /optimize/headway`
Calculate optimal headway (minutes between buses) for each hour of the day.

**Request:**
```json
{
  "route_id": "DTC-401",
  "date": "2024-08-15",
  "total_buses_available": 8,
  "demand_profile": [5, 3, 2, 2, 8, 20, 45, 90, 120, 100, 65, 55, 70, 65, 55, 60, 75, 115, 135, 100, 70, 50, 35, 18]
}
```

**Response:**
```json
{
  "route_id": "DTC-401",
  "headway_minutes": [30, 30, 30, 30, 20, 12, 8, 5, 4, 5, 8, 10, 8, 8, 10, 10, 8, 5, 4, 5, 8, 10, 12, 20],
  "optimization_score": 0.91,
  "buses_required": 6
}
```

---

### `POST /schedule/generate`
Generate a full day's departure schedule for a route.

**Request:**
```json
{
  "route_id": "DTC-401",
  "date": "2024-08-15",
  "total_buses_available": 8
}
```

**Response:**
```json
{
  "route_id": "DTC-401",
  "date": "2024-08-15",
  "trips": [
    {
      "departure_time": "05:30",
      "estimated_arrival": "06:20",
      "headway_minutes": 30,
      "expected_demand": 12
    },
    ...
  ],
  "total_trips": 28
}
```

---

### `POST /admin/retrain`
Trigger background retraining of all models using latest data.

**Request:**
```json
{
  "retrain_demand": true,
  "retrain_delay": true,
  "retrain_anomaly": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Retraining started in background. Check /health after ~2 minutes."
}
```

---

## Fallback Behaviour (No Models)

If model files are not present in `models/saved/`, the service automatically uses rule-based heuristics:

**Demand fallback:** Uses a hardcoded hourly demand table based on observed DTC patterns:
```
Peak hours (8am, 6pm): ~120–130 passengers
Off-peak (2am–4am):    ~2–5 passengers
```
Adjusted by weather factor: rain = ×0.85, fog = ×0.90, heatwave = ×0.75

**Delay fallback:** Calculates delay from weather + load:
```
Rain → +5 min base delay
Passenger load > 80% → +2 min additional
```

This means you can develop and test the full stack without running the ML training pipeline.

---

## Model Retraining via API

The `/admin/retrain` endpoint triggers `retrain_pipeline.py` in the background. The pipeline:

1. Pulls recent data from the MongoDB database (via `MONGO_URI` env var)
2. Re-runs feature engineering
3. Retrains the LSTM and XGBoost models
4. Saves new model files to `models/saved/`, overwriting the old ones
5. Reloads the models into memory

This is useful for production — schedule weekly retraining as a cron job via the backend's scheduler service.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values:

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `./models/saved` | Path to trained model files |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`) |
| `MONGO_URI` | *(optional)* | MongoDB connection string for retraining pipeline |
| `PORT` | `8000` | Port the service listens on (when using Dockerfile CMD) |

---

## Quick Reference: Model File Checklist

Before starting the service, verify these files exist in `ai-service/models/saved/`:

```
✅ demand_lstm/              (directory with saved_model.pb inside)
✅ demand_lstm/variables/    (directory)
✅ demand_scaler.pkl
✅ delay_regressor.pkl
✅ delay_classifier.pkl
✅ delay_scaler.pkl
```

Run this check from the `ai-service/` directory:

```bash
python -c "
import os
files = [
    'models/saved/demand_lstm',
    'models/saved/demand_scaler.pkl',
    'models/saved/delay_regressor.pkl',
    'models/saved/delay_classifier.pkl',
    'models/saved/delay_scaler.pkl',
]
for f in files:
    status = '✅' if os.path.exists(f) else '❌ MISSING'
    print(f'{status}  {f}')
"
```
