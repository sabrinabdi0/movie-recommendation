"""Flask web application for movie recommendations."""

from __future__ import annotations

import logging
import os
import threading

from flask import Flask, jsonify, redirect, render_template, request, url_for
from whitenoise import WhiteNoise

import config
from recommender import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = config.SECRET_KEY

# Serve CSS/JS in production (Gunicorn/Render/Railway do not serve static files by default)
app.wsgi_app = WhiteNoise(
    app.wsgi_app,
    root=os.path.join(app.root_path, "static"),
    prefix="static/",
    autorefresh=True,
)

recommender = MovieRecommender()
_model_lock = threading.Lock()


def ensure_model_loaded() -> None:
    """Load saved model, or build it automatically if missing (e.g. after deploy)."""
    if recommender.movies is not None:
        return

    with _model_lock:
        if recommender.movies is not None:
            return

        if os.path.exists(config.get_model_path()):
            model_path = config.get_model_path()
            logger.info("Loading model from %s", model_path)
            recommender.load(model_path)
            return

        logger.info("Model not found — building automatically at %s", config.MODEL_PATH)
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        recommender.train()


@app.before_request
def load_model_once() -> None:
    if recommender.movies is None:
        ensure_model_loaded()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["GET", "POST"])
def recommend_page():
    title = request.values.get("title", "").strip()
    if not title:
        return redirect(url_for("index"))

    try:
        result = recommender.recommend(title)
        return render_template("results.html", result=result)
    except ValueError as exc:
        return render_template("index.html", error=str(exc), title=title), 400


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required query parameter: title"}), 400

    try:
        result = recommender.recommend(title)
        return jsonify(
            {
                "query": result.query,
                "matched_title": result.matched_title,
                "match_score": result.match_score,
                "note": result.note,
                "recommendations": result.recommendations,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


if __name__ == "__main__":
    ensure_model_loaded()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
