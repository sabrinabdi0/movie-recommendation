# Movie Recommendation System

A content-based movie recommender built for a final project presentation. Users enter a movie title on a Flask web page and receive similar movies based on genres, tags, ratings, and TF-IDF cosine similarity over the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) dataset.

## Features

- **Web UI** — search form, polished results page, demo title shortcuts
- **Content-based filtering** — TF-IDF + cosine similarity on genres, tags, year, and rating signals
- **Fuzzy title matching** — handles typos and partial titles (`rapidfuzz`)
- **Graceful demo support** — supplemental curated entries for blockbuster titles like *Avatar*
- **JSON API** — optional `/api/recommend?title=Avatar` endpoint

## Tech Stack

- Python 3
- Pandas
- Scikit-learn
- Flask
- MovieLens dataset (no API key required)

## Project Structure

```
movie-recommender/
├── app.py              # Flask web application
├── build_model.py      # Download data and train/save model
├── config.py           # Host, port, paths, recommender settings
├── recommender.py      # Core recommender logic
├── requirements.txt
├── templates/          # Jinja HTML templates
├── static/css/         # Stylesheet
├── data/raw/           # Downloaded MovieLens files (gitignored)
└── artifacts/          # Saved model (gitignored)
```

## Quick start (automatic)

Open this folder in Cursor/VS Code — the server **starts automatically** in a terminal when the project opens.

Or run manually from any terminal:

```bash
cd ~/Projects/movie-recommender
chmod +x start.sh   # first time only
./start.sh
```

Then open **http://127.0.0.1:5002** in your browser.

## Setup (manual)

```bash
cd ~/Projects/movie-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

Downloads MovieLens (~1 MB zip), prepares features, and saves artifacts:

```bash
python build_model.py
```

Expected output:

```
Downloading MovieLens dataset from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip ...
Model trained on 9xxx movies.
Artifacts saved to .../artifacts/recommender.joblib
```

## Run the Web App

```bash
python app.py
```

Open [http://127.0.0.1:5002](http://127.0.0.1:5002)

### Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `HOST` | `127.0.0.1` | Flask bind address |
| `PORT` | `5002` | Flask port (avoids common 5000/5001 conflicts) |
| `DEBUG` | `true` | Flask debug mode |
| `MODEL_PATH` | `artifacts/recommender.joblib` | Saved recommender |
| `TOP_N_RECOMMENDATIONS` | `10` | Number of suggestions returned |
| `FUZZY_MATCH_THRESHOLD` | `70` | Minimum fuzzy match score (0–100) |

Override at runtime:

```bash
FLASK_PORT=5002 FLASK_DEBUG=false python app.py
```

## Demo Flow

1. Start the server (`python app.py`)
2. Visit the homepage
3. Enter **Avatar** and click **Get Recommendations**
4. Review similar titles such as **Titanic**, **Interstellar**, **Gravity**, and **The Martian**

For presentation queries like *Avatar*, curated demo ordering ensures blockbuster neighbors appear first while still showing real cosine-similarity scores from the model.

The app uses fuzzy matching if the exact title differs slightly from MovieLens naming (for example, year suffixes).

### API Example

```bash
curl "http://127.0.0.1:5002/api/recommend?title=Avatar"
```

Sample response:

```json
{
  "query": "Avatar",
  "matched_title": "Avatar",
  "match_score": 100.0,
  "note": null,
  "recommendations": [
    {
      "title": "Interstellar",
      "genres": "Adventure|Drama|Sci-Fi",
      "similarity": 42.5
    }
  ]
}
```

## How It Works

1. **Data** — MovieLens movies, tags, and ratings are merged into one catalog.
2. **Features** — Each movie becomes a text profile: title + genres + user tags + year + rating bucket.
3. **Vectorization** — `TfidfVectorizer` converts profiles into sparse vectors.
4. **Similarity** — Pairwise cosine similarity is precomputed and cached in memory at startup.
5. **Query** — User input is fuzzy-matched to the nearest catalog title, then top similar movies are returned.
6. **Demo polish** — Known presentation titles (e.g. Avatar) get curated neighbor ordering on top of model scores.

## Presentation Tips

- Show the homepage and explain content-based vs collaborative filtering.
- Demo **Avatar** live, then click **Explore** on a recommendation to chain suggestions.
- Show the JSON API for technical audiences.
- Mention supplemental titles ensure demo-friendly blockbuster coverage.

## License

MovieLens data is provided by GroupLens under their dataset terms. This project is for educational use.
