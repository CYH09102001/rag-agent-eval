#!/usr/bin/env bash
# One-command setup + launch for Mac/Linux:   ./run.sh
# (first time: chmod +x run.sh)
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then echo "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1; fi

if [ ! -d venv ]; then
  echo ">> Creating virtual environment"
  "$PY" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo ">> Installing packages (first run takes a few minutes)"
pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Created .env. Open it and paste your ANTHROPIC_API_KEY (optional: the app also runs in search-only mode)."
fi

[ -d data ] && ls data/*.txt >/dev/null 2>&1 || python generate_sample_data.py

echo ">> Starting the app at http://localhost:8501  (Ctrl+C to stop)"
streamlit run app.py
