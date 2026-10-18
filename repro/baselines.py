"""Statistical baselines for the M4 daily benchmark, via statsforecast.

Models: Naive, SeasonalNaive(7), AutoETS, Theta, CrostonOptimized (intermittent)
and AutoARIMA on a bounded subset (it is an order of magnitude more expensive).
Every model is fit on the full training window and scored on the official 14-day
M4 test horizon, on exactly the same series subset as the LightGBM run.

Practical notes that cost real time to discover:
  * statsforecast with a single 2.4M-row frame and n_jobs=6 thrashes swap on this
    box (6 cores, ~3GB free).  Series are therefore processed in chunks of
    --chunk and each model's predictions are cached in data/preds/, so a slow or
    interrupted model never costs the earlier ones.
  * `ds` is an integer step index (freq=1): M4 ships no timestamps and none of
    these models uses the calendar, only the season_length.

Usage: python repro/baselines.py [--series 1000] [--arima-series 250] [--jobs 4]
                                 [--chunk 200] [--force] [--finalize-only]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from statsforecast import StatsForecast  # noqa: E402
from statsforecast.models import (  # noqa: E402
    AutoETS, AutoARIMA, CrostonOptimized, Naive, SeasonalNaive, Theta,
)

warnings.filterwarnings("ignore")
CACHE = common.DATA / "preds"


def to_long(data: common.DailyData) -> pd.DataFrame:
    frames = []
    for i, sid in enumerate(data.ids):
        y = data.series(i)
        if len(y) == 0:
            continue
        frames.append(pd.DataFrame({"unique_id": sid,
                                    "ds": np.arange(len(y), dtype=np.int32),
                                    "y": y.astype(np.float64)}))
    return pd.concat(frames, ignore_index=True)


def predict_chunked(model, df: pd.DataFrame, ids: list[str], h: int,
                    jobs: int, chunk: int, name: str) -> np.ndarray:
    """Forecast h steps for every id; NaN where a model returned nothing."""
    preds = np.full((len(ids), h), np.nan)
    by_id: dict[str, np.ndarray] = {}
    t0 = time.time()
    for s in range(0, len(ids), chunk):
        sub = ids[s:s + chunk]
        part = df[df["unique_id"].isin(set(sub))]
        out = StatsForecast(models=[model], freq=1, n_jobs=jobs).forecast(df=part, h=h)
        col = out.columns[-1]
        for sid, grp in out.groupby("unique_id", sort=False):
            by_id[sid] = grp.sort_values("ds")[col].to_numpy(dtype=np.float64)[-h:]
        print(f"       {name}: {min(s+chunk, len(ids))}/{len(ids)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for i, sid in enumerate(ids):
        if sid in by_id:
            v = by_id[sid]
            preds[i, -len(v):] = v
    return preds


def cache_path(name: str) -> Path:
    return CACHE / f"{name}.npz"


def load_timings() -> dict:
    p = CACHE / "timings.json"
    if p.exists():
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_timings(t: dict) -> None:
    import json
    (CACHE / "timings.json").write_text(json.dumps(t, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=1000)
    ap.add_argument("--arima-series", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", default="",
                    help="comma separated subset, e.g. Theta,CrostonOptimized")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()
    jobs = min(args.jobs, 8)
    CACHE.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    data = common.load_m4_daily(n_series=args.series, seed=args.seed)
    N, ids = data.n, data.ids
    fingerprint = hashlib.sha256(data.train.tobytes() + "|".join(ids).encode()).hexdigest()
    def valid_cache(cp, expected):
        if not cp.exists(): return False
        z = np.load(cp)
        return ("ids" in z and list(z["ids"]) == expected and
                "fingerprint" in z and str(z["fingerprint"]) == fingerprint)
    print(f"[data] {N} series, horizon {common.H}, median length "
          f"{int(np.median(data.lengths()))}", flush=True)

    specs = [
        ("Naive", lambda: Naive(), "last value carried forward"),
        ("SeasonalNaive", lambda: SeasonalNaive(season_length=common.SEASON),
         "season_length=7"),
        ("AutoETS", lambda: AutoETS(season_length=common.SEASON),
         "season_length=7, model=ZZZ (auto selection)"),
        ("Theta", lambda: Theta(season_length=common.SEASON), "season_length=7"),
        ("CrostonOptimized", lambda: CrostonOptimized(),
         "intermittent-demand method; M4-daily has no zero-inflated series"),
    ]
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    if wanted:
        specs = [s for s in specs if s[0] in wanted]

    if not args.finalize_only:
        df = to_long(data)
        print(f"[data] long frame {len(df):,} rows", flush=True)
        timings = load_timings()
        for name, mk, _note in specs:
            cp = cache_path(name)
            if valid_cache(cp, ids) and not args.force:
                print(f"[skip] {name} cached", flush=True)
                continue
            t0 = time.time()
            p = predict_chunked(mk(), df, ids, common.H, jobs, args.chunk, name)
            np.savez_compressed(cp, preds=p.astype(np.float32), ids=np.array(ids), fingerprint=fingerprint)
            timings[name] = round(time.time() - t0, 1)
            save_timings(timings)
            print(f"[stat] {name:17s} computed in {time.time()-t0:.0f}s "
                  f"(nan rows {int(np.isnan(p).any(axis=1).sum())})", flush=True)

        # ---- AutoARIMA: expensive, bounded subset, reported with its own n_series
        if args.arima_series and args.arima_series > 0:
            cp = cache_path("AutoARIMA")
            if not valid_cache(cp, ids[:min(args.arima_series, N)]) or args.force:
                sub_ids = ids[: min(args.arima_series, N)]
                t0 = time.time()
                print(f"[stat] AutoARIMA on {len(sub_ids)} series ...", flush=True)
                p = predict_chunked(AutoARIMA(season_length=common.SEASON), df,
                                    sub_ids, common.H, jobs, 50, "AutoARIMA")
                np.savez_compressed(cp, preds=p.astype(np.float32),
                                    ids=np.array(sub_ids), fingerprint=fingerprint)
                print(f"[stat] AutoARIMA computed in {time.time()-t0:.0f}s", flush=True)

    # ---------------- scoring
    results, rows = {}, []
    timings = load_timings()
    zeros = np.array([np.mean(data.series(i) == 0) for i in range(N)])
    inter = zeros > 0.30
    for name, _mk, note in specs:
        cp = cache_path(name)
        if not cp.exists():
            continue
        if not valid_cache(cp, ids):
            raise ValueError(f"Unverified cache {cp}; rerun without --finalize-only")
        p = np.load(cp)["preds"].astype(np.float64)
        if p.shape[0] != N:
            continue
        nbad = int(np.isnan(p).any(axis=1).sum())
        if nbad:
            sn = np.load(cache_path("SeasonalNaive"))["preds"].astype(np.float64)
            p = np.where(np.isnan(p), sn, p)
            note += f"; {nbad} series fell back to seasonal naive"
        res = common.evaluate(np.clip(p, 0, None), data.test, data.train, ids)
        results[name] = common.summarise(name, res, timings.get(name, 0.0), note)
        print(f"[stat] {name:17s} WAPE {res['agg']['wape_mean']:.4f} "
              f"sMAPE {res['agg']['smape_mean']:.2f} MASE {res['agg']['mase_mean']:.3f} "
              f"({timings.get(name, 0):.0f}s)", flush=True)

    cp = cache_path("AutoARIMA")
    if cp.exists():
        z = np.load(cp)
        p, sub_ids = z["preds"].astype(np.float64), [str(s) for s in z["ids"]]
        pos = [ids.index(s) for s in sub_ids]
        res = common.evaluate(np.clip(p, 0, None), data.test[pos],
                              data.train[pos], sub_ids)
        results["AutoARIMA"] = common.summarise(
            "AutoARIMA", res, timings.get("AutoARIMA", 0.0),
            f"subset of the first {len(sub_ids)} series only; season_length=7; "
            f"not comparable 1:1 with the {N}-series models")
        print(f"[stat] {'AutoARIMA':17s} WAPE {res['agg']['wape_mean']:.4f} "
              f"sMAPE {res['agg']['smape_mean']:.2f} MASE {res['agg']['mase_mean']:.3f} "
              f"(n={len(sub_ids)}, {timings.get('AutoARIMA', 0):.0f}s)", flush=True)

    if inter.sum() >= 20:
        print(f"[stat] intermittent subset n={int(inter.sum())} (no model run: "
              f"M4-daily has no zero-inflated series)", flush=True)

    payload = {
        "dataset": "M4-daily",
        "n_series": N,
        "horizon": common.H,
        "ids": ids,
        "season_length": common.SEASON,
        "split": {"test": "the 14 official M4 test points per series",
                  "validation": "the 14 points before that (used to pick the "
                                "LightGBM target transform)"},
        "models": results,
        "intermittent_subset": {"threshold_zero_ratio": 0.3,
                               "n_series": int(inter.sum())},
        "total_runtime_s": round(time.time() - t_all, 1),
        "notes": "WAPE/sMAPE/MASE are computed per series and then averaged; "
                 "MASE scale = mean|y_t - y_{t-7}| over the training window; "
                 "series subset = random sample with seed 42.",
    }
    common.write_json(common.DATA / Path(args.out).name, payload)

    save = {"ids": np.array(ids), "test_truth": data.test.astype(np.float32)}
    for name in results:
        cp = cache_path(name)
        if cp.exists() and name != "AutoARIMA":
            save[name] = np.load(cp)["preds"]
    np.savez_compressed(common.DATA / "bench_preds.npz", **save)
    print(f"[done] {time.time()-t_all:.0f}s -> data/{Path(args.out).name}, "
          f"data/bench_preds.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
