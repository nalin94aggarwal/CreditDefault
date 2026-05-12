#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f "artifacts/default_xgboost_artifacts.joblib" ]; then
  echo "Model artifacts not found. Training the model first..."
  python train_model.py --data ../credit.csv
fi

streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
