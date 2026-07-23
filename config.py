"""Application configuration for the Movie Recommendation System."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _runtime_root() -> str:
    """Use a writable directory in serverless hosts (/var/task is read-only)."""
    if custom := os.environ.get("MODEL_DIR"):
        return custom
    if os.access(BASE_DIR, os.W_OK):
        return BASE_DIR
    return "/tmp/movie-recommender"


RUNTIME_ROOT = _runtime_root()

# Flask
HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", "5002"))
DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() in {"1", "true", "yes"}
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-movie-recommender-key")

# Bundled paths (created during Vercel/Render build — readable on /var/task)
BUNDLED_DATA_DIR = os.path.join(BASE_DIR, "data")
BUNDLED_RAW_DATA_DIR = os.path.join(BUNDLED_DATA_DIR, "raw")
BUNDLED_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
BUNDLED_MODEL_PATH = os.path.join(BUNDLED_ARTIFACTS_DIR, "recommender.joblib")

# Writable runtime paths (local dev or serverless fallback)
DATA_DIR = os.path.join(RUNTIME_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
ARTIFACTS_DIR = os.path.join(RUNTIME_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "recommender.joblib")


def get_model_path() -> str:
    """Prefer bundled model from deploy build; fall back to writable runtime path."""
    if os.path.exists(BUNDLED_MODEL_PATH):
        return BUNDLED_MODEL_PATH
    return MODEL_PATH


def get_save_path() -> str:
    """Save into the project bundle when the filesystem is writable (Vercel build)."""
    if os.access(BASE_DIR, os.W_OK):
        os.makedirs(BUNDLED_ARTIFACTS_DIR, exist_ok=True)
        return BUNDLED_MODEL_PATH
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    return MODEL_PATH

# MovieLens download
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP = os.path.join(RAW_DATA_DIR, "ml-latest-small.zip")

# Recommender settings
TOP_N_RECOMMENDATIONS = 10
FUZZY_MATCH_THRESHOLD = 70  # rapidfuzz score (0-100)
MIN_RATINGS_FOR_POPULAR = 50
