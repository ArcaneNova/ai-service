"""
download_models.py — Run at Render build time to fetch models from GitHub Release assets.

Usage (in render.yaml buildCommand):
  pip install -r requirements.txt && python download_models.py

Set the MODELS_RELEASE_URL env var on Render to the base URL of your GitHub Release,
e.g. https://github.com/ArcaneNova/bus-site/releases/download/v1.0-models

If models already exist locally (committed to repo), this script does nothing.
"""

import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models/saved")
MODELS_ZIP_URL = os.getenv("MODELS_ZIP_URL", "")  # Set this on Render dashboard

SENTINEL = os.path.join(MODEL_DIR, "demand_xgboost_multimodel.pkl")


def download_models():
    if os.path.exists(SENTINEL):
        logger.info("✅ Models already present at %s — skipping download", MODEL_DIR)
        return

    if not MODELS_ZIP_URL:
        logger.warning("⚠️  MODELS_ZIP_URL not set and models not found locally. "
                       "Predictions will use fallback/mock responses.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = "/tmp/models.zip"

    logger.info("📥 Downloading models from %s", MODELS_ZIP_URL)
    urllib.request.urlretrieve(MODELS_ZIP_URL, zip_path)

    logger.info("📦 Extracting to %s", MODEL_DIR)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(MODEL_DIR)

    os.remove(zip_path)
    logger.info("✅ Models downloaded and extracted successfully")


if __name__ == "__main__":
    download_models()
