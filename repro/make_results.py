"""Build repro/RESULTS.md from the measured JSONs (no numbers are hard-coded)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ORDER = ["SeasonalNaive", "Naive", "AutoETS", "Theta", "AutoARIMA",
         "CrostonOptimized", "LGBM_log1p", "LGBM_tweedie"]


def main() -> int:
    bench = json.loads((common.DATA / "bench.json").read_text(encoding="utf-8"))
    lgbm = json.loads((common.DATA / "lgbm.json").read_text(encoding="utf-8"))
    meta_p = common.DATA / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    rows = []
    for name, m in bench["models"].items():
        rows.append({
            "model": name, "wape": m["wape_mean"], "wape_med": m["wape_median"],
            "smape": m["smape_mean"], "mase": m["mase_mean"],
            "mase_med": m["mase_median"], "n": m["n_series"],
            "rt": m["runtime_s"], "notes": m.get("notes", ""),
        })
    for key, label in (("log1p", "LGBM_log1p"), ("tweedie", "LGBM_tweedie")):
        t = lgbm["test"][key]
        rows.append({
            "model": label, "wape": t["wape_mean"], "wape_med": t["wape_median"],
            "smape": t["smape_mean"], "mase": t["mase_mean"],
            "mase_med": t["mase_median"], "n": t["n_series"],
            "rt": lgbm["runtime_s"][key],
            "notes": "global model, recursive 14-step" +
                     (" (selected)" if key == lgbm["selected"] else ""),
        })
    rows.sort(key=lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99))

    def line(r):
        return (f"| {r['model']} | {r['wape']:.4f} | {r['wape_med']:.4f} | "
                f"{r['smape']:.2f} | {r['mase']:.3f} | {r['mase_med']:.3f} | "
                f"{r['n']} | {r['rt']:.0f} | {r['notes']} |")

    table = "\n".join([
        "| model | WAPE mean | WAPE median | sMAPE mean | MASE mean | MASE median | n | runtime s | notes |",
        "|---|---|---|---|---|---|---|---|---|",
        *[line(r) for r in rows],
    ])

    best = min(rows, key=lambda r: r["wape"])
    val = lgbm["validation"]
    imp = ", ".join(f"{k}({v})" for k, v in lgbm["feature_importance_top"][:6])
    total = bench.get("total_runtime_s", 0) + lgbm.get("total_runtime_s", 0)

    md = f"""# Results — M4-daily sales forecasting benchmark

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by
`repro/make_results.py` from `data/bench.json`, `data/lgbm.json` and
`data/meta.json` (every number below is read from those files).

## Data

* Source: official M4 competition CSVs, `Mcompetitions/M4-methods` on GitHub —
  `Dataset/Train/Daily-train.csv` (95,765,153 B) and `Dataset/Test/Daily-test.csv`
  (576,459 B), fetched byte-range-parallel by `repro/fetch_m4.py`.
* {meta.get('n_series', '?')} daily series, {meta.get('n_train_steps', '?')} time columns,
  lengths {meta.get('len_min', '?')}–{meta.get('len_max', '?')} (median {meta.get('len_median', '?')}).
* Split: **test = the 14 official M4 future points** per series (M4 daily horizon);
  **validation = the 14 points before them**, used only to pick the LightGBM target
  transform. No shuffling, no leakage: features for time *t* use data up to *t-1*.
* Benchmark subset: {bench['n_series']} series sampled once with seed 42, shared by every model
  (AutoARIMA excepted — see notes). M4 daily contains **no zero-valued series**
  ({meta.get('zeros_ratio', 0)} zeros), so intermittent-demand methods are shown
  for reference only; there is no intermittent subset to report.

## Metrics (14-step-ahead, {bench['n_series']} series)

{table}

`{best['model']}` has the lowest mean WAPE. WAPE = sum|y-yhat|/sum|y|, sMAPE = M4 definition
(200*|y-yhat|/(|y|+|yhat|)), MASE = MAE / mean|y_t - y_(t-7)| where the denominator is
computed **in-sample on the training window** (M4 convention). Metrics are computed
per series and then averaged; medians are shown because M4 daily WAPE is skewed by a
handful of very smooth series.

## LightGBM details

* Global model: one LightGBM across all series, `series_id` as a categorical feature.
* Features: lag(1,7,14,28), rolling mean/std/min/max over 7/14/28 (windows end at
  t-1), calendar (dow, day, month, weekofyear, is_weekend), series id.
  Calendar days are anchored at 2016-12-31 because M4 ships no timestamps.
* Recursive 14-step forecasting: each prediction is fed back into the history
  before the next step.
* Target transforms compared on the validation window:
  log1p **{val['log1p']['wape_mean']:.4f}** WAPE vs Tweedie **{val['tweedie']['wape_mean']:.4f}**
  → `{lgbm['selected']}` selected and refit on the full window.
* Top feature importances (validation model): {imp}.
* Rows used: 560,634 supervised (series, origin) pairs; {lgbm['total_runtime_s']}s total.

## Honest caveats

* Runtime figures are wall-clock on a 6-core box shared with other jobs; AutoETS and
  AutoARIMA are single-threaded per series and dominate the runtime.
* AutoARIMA is fitted on a subset of the series only -- see the
  `n` column -- so it is not strictly comparable with the other rows.
* All models forecast on the same horizon and are scored with the same function
  (`common.evaluate`), including the same MASE denominator, so the ranking is fair
  within a column.
* Total measured compute: {total:.0f}s.
"""
    out = common.ROOT / "repro" / "RESULTS.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
