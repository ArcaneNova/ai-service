"""
train_demand_lstm.py
Trains an LSTM model to predict passenger demand per route-hour.

Usage: python training/train_demand_lstm.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Config ─────────────────────────────────────────────────────────────────

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/demand_dataset.csv")
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "../models/saved")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_holiday",
    "weather_encoded", "avg_temp_c", "special_event", "month",
]
TARGET   = "passenger_count"
TIMESTEPS = 1    # single-step prediction (expand to 24 for sequential)
BATCH_SIZE = 256
EPOCHS     = 100
SEED       = 42
tf.random.set_seed(SEED)

# ── Load & Preprocess ──────────────────────────────────────────────────────

print("📂  Loading dataset…")
df = pd.read_csv(DATA_PATH)
print(f"    {len(df):,} rows loaded")

# Encode weather
weather_map = {"clear": 0, "rain": 1, "fog": 2, "heatwave": 3}
df["weather_encoded"] = df["weather"].map(weather_map).fillna(0)

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# Train / val / test split — NO data leakage (split by index not shuffle)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, shuffle=False)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)

print(f"    Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Fit scaler on TRAIN only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Reshape for LSTM: (samples, timesteps, features)
X_train_s = X_train_s.reshape(-1, TIMESTEPS, len(FEATURES))
X_val_s   = X_val_s.reshape(-1, TIMESTEPS, len(FEATURES))
X_test_s  = X_test_s.reshape(-1, TIMESTEPS, len(FEATURES))

# ── Build LSTM Model ───────────────────────────────────────────────────────

model = Sequential([
    LSTM(128, input_shape=(TIMESTEPS, len(FEATURES)), return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation="relu"),
    Dense(1, activation="relu"),  # demand cannot be negative
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)
model.summary()

# ── Train ──────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(os.path.join(SAVE_DIR, "demand_lstm_best.h5"), save_best_only=True),
]

print("\n🏋️  Training LSTM…")
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────────────────

y_pred   = model.predict(X_test_s).flatten()
mae      = np.mean(np.abs(y_pred - y_test))
mape     = np.mean(np.abs((y_pred - y_test) / (y_test + 1e-8))) * 100
rmse     = np.sqrt(np.mean((y_pred - y_test) ** 2))

print(f"\n📊  Test Results:")
print(f"    MAE  : {mae:.2f} passengers")
print(f"    MAPE : {mape:.2f}%")
print(f"    RMSE : {rmse:.2f}")

# ── Save ───────────────────────────────────────────────────────────────────

model.export(os.path.join(SAVE_DIR, "demand_lstm"))   # SavedModel format
joblib.dump(scaler, os.path.join(SAVE_DIR, "demand_scaler.pkl"))

print(f"\n✅  Model saved → {SAVE_DIR}/demand_lstm")
print(f"✅  Scaler saved → {SAVE_DIR}/demand_scaler.pkl")
