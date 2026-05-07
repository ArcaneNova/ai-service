"""
anomaly_detector.py — SmartDTC Multi-Model Anomaly Detection
Supports: Autoencoder (best F1), Isolation Forest, LOF, One-Class SVM, DBSCAN, Ensemble.
"""

import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models/saved")


def load_anomaly_model():
    """
    Loads are now handled by model_loader.load_models().
    This function exists for backward-compat lifespan hook.
    """
    import model_loader
    if not model_loader.anomaly_models:
        logger.warning("⚠️  No anomaly models in registry — building synthetic baseline")
        _build_synthetic_baseline()


def _build_synthetic_baseline():
    """Emergency fallback: train a tiny Isolation Forest on synthetic normals."""
    import model_loader
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n = 2000
    X = np.column_stack([
        np.random.uniform(20, 80, n),     # speed km/h
        np.random.exponential(3, n).clip(0, 10),  # delay min
        np.random.uniform(30, 90, n),     # passenger load
    ])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    mdl    = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    mdl.fit(X_sc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(mdl,    os.path.join(MODEL_DIR, "anomaly_isolation_forest_multimodel.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "anomaly_scaler_multimodel.pkl"))
    model_loader.anomaly_models["isolation_forest"] = mdl
    model_loader.anomaly_scaler = scaler
    model_loader.anomaly_best_model = "isolation_forest"
    logger.info("✅  Synthetic Isolation Forest baseline built")


# ── Feature builder ─────────────────────────────────────────────────────────────

def _build_features(speed_kmh: float, delay_minutes: float, passenger_load: float) -> np.ndarray:
    """Return a 3-feature vector (same as training data for the scaler)."""
    return np.array([[speed_kmh, delay_minutes, passenger_load]], dtype=np.float32)


# ── Per-model scorers ───────────────────────────────────────────────────────────

def _score_isolation_forest(model, X_scaled: np.ndarray) -> tuple[bool, float]:
    raw   = float(model.score_samples(X_scaled)[0])
    pred  = int(model.predict(X_scaled)[0])
    is_an = pred == -1
    conf  = float(np.clip((-raw - 0.1) / 0.4, 0, 1))
    return is_an, conf


def _score_lof(model, X_scaled: np.ndarray) -> tuple[bool, float]:
    # LOF predict: -1 = anomaly, 1 = normal
    pred  = int(model.predict(X_scaled)[0])
    is_an = pred == -1
    # Negative factor: more negative = more anomalous
    try:
        factor = float(model.negative_outlier_factor_[0]) if hasattr(model, "negative_outlier_factor_") else -1.5
    except Exception:
        factor = -1.5
    conf = float(np.clip((-factor - 1) / 2, 0, 1))
    return is_an, conf


def _score_ocsvm(model, X_scaled: np.ndarray) -> tuple[bool, float]:
    pred  = int(model.predict(X_scaled)[0])
    is_an = pred == -1
    dist  = float(model.decision_function(X_scaled)[0])
    conf  = float(np.clip(-dist / 2.0, 0, 1))
    return is_an, conf


def _score_autoencoder(model, X_scaled: np.ndarray, threshold: float) -> tuple[bool, float]:
    recon = model(X_scaled, training=False)
    if hasattr(recon, "numpy"):
        recon = recon.numpy()
    mse   = float(np.mean((recon - X_scaled) ** 2))
    is_an = mse > threshold
    conf  = float(np.clip((mse - threshold * 0.5) / threshold, 0, 1))
    return is_an, conf


def _score_dbscan(model, X_scaled: np.ndarray) -> tuple[bool, float]:
    # DBSCAN labels -1 as noise (anomaly). We re-predict by nearest cluster distance.
    try:
        from sklearn.metrics import pairwise_distances_argmin_min
        if not hasattr(model, "components_") or model.components_ is None or len(model.components_) == 0:
            return False, 0.0
        _, min_dists = pairwise_distances_argmin_min(X_scaled, model.components_)
        dist  = float(min_dists[0])
        # Threshold at eps (default ~0.5)
        eps   = getattr(model, "eps", 0.5)
        is_an = dist > eps
        conf  = float(np.clip((dist - eps) / (eps * 2), 0, 1))
        return is_an, conf
    except Exception:
        return False, 0.0


# ── Ensemble scorer ─────────────────────────────────────────────────────────────

def _ensemble_vote(votes: list[bool], confs: list[float]) -> tuple[bool, float]:
    """Majority voting — anomaly if > half of reliable models agree."""
    if not votes:
        return False, 0.0
    anomaly_votes = sum(votes)
    is_an  = anomaly_votes > len(votes) / 2
    conf   = float(np.mean(confs)) if confs else 0.0
    return is_an, conf


# ── Main detection function ─────────────────────────────────────────────────────

def detect_anomaly(
    speed_kmh:      float,
    delay_minutes:  float,
    passenger_load: float = 60.0,
    model_key:      str   = "auto",
) -> dict:
    """
    Detect anomalous bus behaviour.

    model_key: 'auto' uses best model (autoencoder if loaded),
               'ensemble' runs all models and majority-votes,
               or any of: 'isolation_forest','lof','ocsvm','autoencoder','dbscan'.
    """
    import model_loader

    if not model_loader.anomaly_models:
        load_anomaly_model()

    models   = model_loader.anomaly_models
    scaler   = model_loader.anomaly_scaler
    best     = model_loader.anomaly_best_model
    ae_thr   = model_loader.anomaly_ae_threshold

    X_raw = _build_features(speed_kmh, delay_minutes, passenger_load)

    # Scale features (scaler trained on 3 features)
    try:
        X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw
    except Exception:
        X_scaled = X_raw

    # Build human-readable reason
    reason_parts = []
    if speed_kmh < 5:          reason_parts.append("bus appears stationary")
    elif speed_kmh > 90:       reason_parts.append(f"high speed ({speed_kmh:.0f} km/h)")
    if delay_minutes > 20:     reason_parts.append(f"severe delay ({delay_minutes:.0f} min)")
    if passenger_load > 120:   reason_parts.append("overcrowded")

    all_results = {}

    def _run(key, mdl):
        try:
            if key == "isolation_forest":
                return _score_isolation_forest(mdl, X_scaled)
            elif key == "lof":
                return _score_lof(mdl, X_scaled)
            elif key == "ocsvm":
                return _score_ocsvm(mdl, X_scaled)
            elif key == "autoencoder":
                return _score_autoencoder(mdl, X_scaled, ae_thr)
            elif key == "dbscan":
                return _score_dbscan(mdl, X_scaled)
        except Exception as e:
            logger.warning(f"Anomaly scorer {key} failed: {e}")
        return None, None

    # Decide which models to run
    target = best if model_key == "auto" else model_key

    if target == "ensemble":
        votes, confs = [], []
        for k, m in models.items():
            is_an, conf = _run(k, m)
            if is_an is not None:
                all_results[k] = {"is_anomaly": is_an, "confidence": round(conf, 3)}
                votes.append(is_an)
                confs.append(conf)
        is_anomaly, confidence = _ensemble_vote(votes, confs)
        used_model = "ensemble"
    elif target in models:
        is_anomaly, confidence = _run(target, models[target])
        if is_anomaly is None:
            is_anomaly, confidence = False, 0.0
        used_model = target
        all_results[target] = {"is_anomaly": is_anomaly, "confidence": round(confidence or 0.0, 3)}
    else:
        # Rule-based fallback
        is_anomaly = speed_kmh > 100 or delay_minutes > 30 or speed_kmh < 0
        confidence = 0.9 if is_anomaly else 0.1
        used_model = "rule_based"

    confidence = confidence or 0.0
    reason = "; ".join(reason_parts) if reason_parts else (
        "anomaly pattern detected" if is_anomaly else "within normal parameters"
    )

    # Comparison metrics from report
    comp = (model_loader.anomaly_comparison.get("models") or {}).get(used_model, {})

    return {
        "is_anomaly":    bool(is_anomaly),
        "score":         round(float(confidence), 4),
        "confidence":    round(float(confidence), 3),
        "reason":        reason,
        "model":         used_model,
        "is_best_model": used_model == best or used_model == "ensemble",
        "metrics": {
            "precision": round(comp.get("precision", 0), 4),
            "recall":    round(comp.get("recall",    0), 4),
            "f1":        round(comp.get("f1",        0), 4),
        },
        "all_model_results": all_results,
    }
