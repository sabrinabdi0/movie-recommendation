#!/usr/bin/env bash
# Auto-setup and start the Movie Recommendation System.
# Usage: ./start.sh   (or open this project in Cursor — it runs automatically)

set -e
cd "$(dirname "$0")"

echo "=== Movie Recommendation System ==="

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import flask" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt -q
fi

if [ ! -f "artifacts/recommender.joblib" ]; then
  echo "Building model (first run only — downloads MovieLens data)..."
  python build_model.py
fi

echo ""
echo "Server starting at http://127.0.0.1:5002"
echo "Press Ctrl+C to stop."
echo ""

python app.py
