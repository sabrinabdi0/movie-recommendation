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

# Data & model paths
DATA_DIR = os.path.join(RUNTIME_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
ARTIFACTS_DIR = os.path.join(RUNTIME_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "recommender.joblib")

# MovieLens download
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP = os.path.join(RAW_DATA_DIR, "ml-latest-small.zip")

# Recommender settings
TOP_N_RECOMMENDATIONS = 10
FUZZY_MATCH_THRESHOLD = 70  # rapidfuzz score (0-100)
MIN_RATINGS_FOR_POPULAR = 50
