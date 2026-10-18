#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python repro/audit_saved.py
python repro/fg_mp.py --val
python repro/bsts52.py --jobs 4
python repro/bench_base.py
STEPS=260 python repro/bench_deep.py
python repro/audit_experiment.py
python repro/audit_rolling.py
python repro/message_ablation.py
python repro/message_iterations.py
python repro/build_audit_report.py
pytest -q
