#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Requires prior audit artifacts; run repro/run_audit.sh if they are absent.
python repro/communication_positive_control.py
python repro/communication_mpa_experiment.py
python repro/communication_capacity_control.py
python repro/communication_group_control.py
pytest -q
