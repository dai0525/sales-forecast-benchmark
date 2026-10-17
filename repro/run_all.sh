#!/usr/bin/env bash
# Full reproduction pipeline for the M4-daily sales-forecast benchmark.
# All outputs land in data/ (metrics + preds) and web/data/series.json (frontend).
#
#   ./repro/run_all.sh [N_SERIES]        # default 1000 series
#
# Requires: data/Daily-train.csv, data/Daily-test.csv (official M4-methods CSVs).
# Fetch them with repro/fetch_m4.py (resumable, parallel byte ranges).
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
N=${1:-1000}

echo "== data prep =="
python data/prep_data.py

echo "== statistical baselines (N=$N) =="
python repro/baselines.py --series "$N" --arima-series 300 --jobs 6 --out bench.json

echo "== LightGBM global model (N=$N) =="
python repro/lgbm_global.py --series "$N" --threads 5 --max-origins 700 --out lgbm.json

echo "== frontend export =="
python repro/export_web.py --series "$N"

echo "== done =="
python - <<'EOF'
import json, pathlib
for f in ("data/bench.json", "data/lgbm.json"):
    d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
    print(f, "->", list(d.get("models", d.get("test", {})).keys()))
EOF
