"""Export the benchmark for the frontend.

Writes
  web/data/series.json  -- 8 representative series (2 seasonal, 2 trend,
                           2 intermittent, 2 noisy) with the last 120 training
                           points, the 14-day truth and every model's forecast,
                           all as non-negative integers
  data/m4_daily.parquet -- long (series_id, ds, y) table of the training data
                           for reproducibility / ad-hoc analysis

Usage: python repro/export_web.py [--series 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TAIL = 120


def classify(y: np.ndarray) -> tuple[str, dict]:
    """Coarse archetype + the stats that justify it."""
    zero_ratio = float(np.mean(y == 0))
    mean, std = float(np.mean(y)), float(np.std(y))
    cv = std / mean if mean > 0 else 0.0
    t = np.arange(len(y))
    trend = float(np.corrcoef(t, y)[0, 1]) if len(y) > 3 and np.std(y) > 0 else 0.0
    v = y - y.mean()
    denom = float((v * v).sum())
    acf1 = float((v[1:] * v[:-1]).sum() / denom) if denom > 0 else 0.0
    acf7 = float((v[7:] * v[:-7]).sum() / denom) if denom > 0 and len(y) > 14 else 0.0
    # seasonality strength vs a 7-day centred moving average
    stats = {"zero_ratio": round(zero_ratio, 3), "cv": round(cv, 3),
             "trend_corr": round(trend, 3), "acf1": round(acf1, 3),
             "acf7": round(acf7, 3), "mean": round(mean, 1)}
    if zero_ratio > 0.20:
        return "intermittent", stats
    if abs(trend) > 0.5 and cv < 0.6:
        return "trend", stats
    if acf7 > 0.35:
        return "seasonal", stats
    if cv < 0.25 and abs(trend) < 0.35:
        return "stable", stats
    return "noisy", stats


def to_int(a: np.ndarray) -> list[int]:
    return [int(v) for v in np.clip(np.round(np.asarray(a, float)), 0, None)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    data = common.load_m4_daily(n_series=args.series, seed=args.seed)
    ids = data.ids
    N = data.n

    bench = np.load(common.DATA / "bench_preds.npz", allow_pickle=False)
    lgbm = np.load(common.DATA / "lgbm_preds.npz", allow_pickle=False)
    assert list(lgbm["ids"]) == ids, "LightGBM IDs differ"
    assert list(bench["ids"]) == ids, "bench preds were made for a different series set"

    preds: dict[str, np.ndarray] = {}
    for k in ("Naive", "SeasonalNaive", "AutoETS", "Theta", "CrostonOptimized"):
        if k in bench:
            preds[k] = bench[k]
    for k in ("log1p", "tweedie"):
        key = f"{k}_test"
        if key in lgbm:
            preds["LGBM_" + k] = lgbm[key]

    bench_meta = json.loads((common.DATA / "bench.json").read_text(encoding="utf-8"))
    lgbm_meta = json.loads((common.DATA / "lgbm.json").read_text(encoding="utf-8"))

    # per-series metrics for whatever is available
    per_series = {}
    for name, p in preds.items():
        res = common.evaluate(np.clip(p, 0, None), data.test, data.train, ids)
        per_series[name] = {r["id"]: r for r in res["per_series"]}

    # ---- choose representatives: strongest example of each archetype
    tagged = []
    for i in range(N):
        y = data.series(i)
        if len(y) < 200 or np.mean(y == 0) > 0.95:
            continue
        cat, st = classify(y)
        score = {"intermittent": st["zero_ratio"], "trend": abs(st["trend_corr"]),
                 "seasonal": st["acf7"], "stable": -st["cv"],
                 "noisy": st["cv"]}[cat]
        tagged.append((cat, score, i, st))

    picked, used = [], set()
    for cat in ("seasonal", "trend", "noisy", "stable", "intermittent"):
        cands = sorted([t for t in tagged if t[0] == cat], key=lambda t: -t[1])
        took = 0
        for c, score, i, st in cands:
            if took >= args.per_class:
                break
            if i in used:
                continue
            used.add(i)
            picked.append((cat, i, st))
            took += 1
        if took < args.per_class:
            print(f"[warn] only {took} series for class {cat}", flush=True)

    # top up to per_class*4 with the most volatile leftovers so the demo always
    # has a full set even when an archetype is rare in the data
    need = args.per_class * 4 - len(picked)
    if need > 0:
        rest = sorted([t for t in tagged if t[2] not in used],
                      key=lambda t: -t[3]["cv"])
        for c, score, i, st in rest[:need]:
            used.add(i)
            picked.append((c, i, st))
        print(f"[warn] topped up {min(need, len(rest))} series from leftovers", flush=True)

    out_series = []
    for cat, i, st in picked:
        L = int(np.sum(~np.isnan(data.train[i])))
        ytr = data.train[i, -L:]
        rec = {
            "id": ids[i],
            "category": cat,
            "stats": st,
            "train": to_int(ytr[-TAIL:]),
            "test": to_int(data.test[i]),
            "pred": {name: to_int(p[i]) for name, p in preds.items()},
            "metrics": {name: {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in per_series[name][ids[i]].items()
                               if k in ("wape", "smape", "mase", "mae")}
                        for name in preds},
        }
        out_series.append(rec)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "M4-daily (official M4-methods CSVs, GitHub)",
        "n_series_bench": N,
        "horizon": common.H,
        "season_length": common.SEASON,
        "tail_points": TAIL,
        "models": list(preds),
        "metrics": {
            **{k: {kk: v[kk] for kk in ("wape_mean", "wape_median", "smape_mean",
                                        "mase_mean", "mase_median", "n_series",
                                        "runtime_s")}
               for k, v in bench_meta["models"].items()},
            "LGBM_log1p": {**lgbm_meta["test"]["log1p"],
                           "runtime_s": lgbm_meta["runtime_s"]["log1p"]},
            "LGBM_tweedie": {**lgbm_meta["test"]["tweedie"],
                             "runtime_s": lgbm_meta["runtime_s"]["tweedie"]},
        },
        "series": out_series,
    }
    web = common.ROOT / "web" / "data" / "daily" / "series.json"
    common.write_json(web, payload)
    print(f"[web  ] {web.relative_to(common.ROOT)}: {len(out_series)} series "
          f"({', '.join(s['id'] + ':' + s['category'] for s in out_series)})", flush=True)

    # ---- long parquet for the whole training set
    pq = common.DATA / "m4_daily.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pqlib
        full = data if data.n == 4227 else common.load_m4_daily()
        writer = None
        for i in range(full.n):
            L = int(np.sum(~np.isnan(full.train[i])))
            y = full.train[i, -L:].astype(np.float32)
            tbl = pa.table({
                "series_id": pa.array([full.ids[i]] * L, pa.string()),
                "ds": pa.array(np.arange(L, dtype=np.int32)),
                "y": pa.array(y, pa.float32()),
            })
            writer = writer or pqlib.ParquetWriter(pq, tbl.schema, compression="zstd")
            writer.write_table(tbl)
        writer.close()
        rows = int(np.sum(full.lengths()))
        print(f"[data ] {pq.name}: {pq.stat().st_size/1e6:.1f} MB, {full.n} series, "
              f"{rows:,} rows", flush=True)
    except Exception as exc:
        print(f"[warn] parquet export failed: {exc}", flush=True)

    print(json.dumps({s["id"]: s["category"] for s in out_series}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
