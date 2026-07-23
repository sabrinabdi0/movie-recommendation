"""Flask web application for movie recommendations."""

from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from whitenoise import WhiteNoise

import config
from recommender import MovieRecommender

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


def ensure_model_loaded() -> None:
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run `python build_model.py` first."
        )
    recommender.load(config.MODEL_PATH)


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


@app.errorhandler(FileNotFoundError)
def missing_model(error):
    message = str(error)
    return render_template("index.html", error=message), 503


if __name__ == "__main__":
    ensure_model_loaded()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
