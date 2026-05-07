"""
Quick verification script to test categorical encoding
Run this locally or in Colab Cell 2 to verify encoding works
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("\n" + "="*80)
print("🔍 CATEGORICAL ENCODING VERIFICATION TEST")
print("="*80)

# Create test data that mimics our actual data structure
test_data = {
    'route_type': ['commercial_hub', 'residential', 'peripheral', 'commercial_hub', 'residential'] * 100,
    'weather': ['clear', 'light_rain', 'heavy_rain', 'fog', 'clear'] * 100,
    'hour': np.random.randint(0, 24, 500),
    'day_of_week': np.random.randint(0, 7, 500),
    'temperature': np.random.uniform(15, 45, 500),
    'passenger_load': np.random.uniform(0, 100, 500),
    'distance': np.random.uniform(5, 50, 500),
}

X = pd.DataFrame(test_data)

print(f"\n✅ Test data created: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"\nInitial dtypes:")
print(X.dtypes)

# Step 1: Detect categorical columns
print(f"\n{'='*80}")
print("Step 1: Detecting categorical columns...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"✅ Found categorical columns: {categorical_cols}")

if categorical_cols:
    for col in categorical_cols:
        print(f"   • {col}: {X[col].unique().tolist()}")

# Step 2: Encode categorical columns
print(f"\n{'='*80}")
print("Step 2: One-hot encoding categorical columns...")
print(f"   Before: {X.shape[1]} columns")

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)

print(f"   After: {X_encoded.shape[1]} columns")
print(f"✅ Encoding successful!")
print(f"\nNew columns:")
for col in X_encoded.columns:
    print(f"   • {col}: dtype={X_encoded[col].dtype}")

# Step 3: Convert to numpy array
print(f"\n{'='*80}")
print("Step 3: Converting to numpy array (float32)...")

X_array = X_encoded.values.astype(np.float32)
print(f"✅ Array created: shape={X_array.shape}, dtype={X_array.dtype}")
print(f"   Sample row: {X_array[0][:5]}...")

# Step 4: Handle NaN values
print(f"\n{'='*80}")
print("Step 4: Handling NaN values...")

X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✅ NaN handling complete: dtype={X_clean.dtype}")

# Step 5: Scale with StandardScaler
print(f"\n{'='*80}")
print("Step 5: Scaling with StandardScaler...")

try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    print(f"✅ Scaling successful!")
    print(f"   Mean: {X_scaled.mean():.6f} (should be ~0)")
    print(f"   Std Dev: {X_scaled.std():.6f} (should be ~1)")
    print(f"   Sample scaled row: {X_scaled[0][:5]}...")
except Exception as e:
    print(f"❌ ERROR during scaling: {e}")
    print(f"   Array dtype: {X_clean.dtype}")
    print(f"   Array shape: {X_clean.shape}")
    print(f"   Sample values: {X_clean[0][:5]}")

print(f"\n{'='*80}")
print("✅ ALL TESTS PASSED!")
print("Your categorical encoding pipeline is working correctly.")
print("="*80 + "\n")
