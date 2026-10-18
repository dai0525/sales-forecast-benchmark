"""Global LightGBM model for the M4 daily benchmark (recursive multi-step).

One model is shared by every series; the series identity enters as a categorical
feature.  Features: lag(1,7,14,28), rolling mean/std/min/max over 7/14/28
(excluding the point being predicted), calendar (dow, day, month, weekofyear,
is_weekend) and the series id.

Multi-step is *recursive*: the model forecasts one step, the prediction is
written back into the history matrix, features are recomputed, and the next step
is forecast.  Two target formulations are trained and compared on the last 14
training points (the validation window):
  * log1p + L2        (skewed positive sales, multiplicative errors)
  * Tweedie (p=1.3)   (zero-inflated positive target with a log link)
The winner by validation WAPE is refit on the full training window and scored on
the official M4 test horizon.

Usage: python repro/lgbm_global.py [--series 1000] [--seed 42] [--threads 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import lightgbm as lgb  # noqa: E402

LAGS = list(common.LAGS)
ROLLS = list(common.ROLLS)
LAG_MIN = common.LAG_MAX        # history required before an origin is usable


def build_feats() -> list[str]:
    return (
        [f"lag{k}" for k in LAGS]
        + [f"roll{w}_{s}" for w in ROLLS for s in ("mean", "std", "min", "max")]
        + ["dow", "day", "month", "weekofyear", "is_weekend", "series_id"]
    )


FEATS = build_feats()


def _cal(cal: dict, cols: np.ndarray) -> dict:
    return {
        "dow": cal["dow"][cols].astype(np.int16),
        "day": cal["day"][cols].astype(np.int16),
        "month": cal["month"][cols].astype(np.int16),
        "weekofyear": cal["weekofyear"][cols].astype(np.int16),
        "is_weekend": (cal["dow"][cols] >= 5).astype(np.int8),
    }


def features(Y: np.ndarray, idx: np.ndarray, cols: np.ndarray, cal: dict) -> np.ndarray:
    """Feature matrix for many (series, origin) pairs.

    Y is (N, T) right aligned; `cols` is the index of the last *observed* point
    for each row, so the row's target is Y[idx, cols+1].
    """
    m = len(idx)
    X = np.empty((m, len(FEATS)), dtype=np.float32)
    j = 0
    for k in LAGS:
        X[:, j] = Y[idx, cols - k + 1]
        j += 1
    for w in ROLLS:
        win = Y[idx[:, None], (cols - w + 1)[:, None] + np.arange(w)[None, :]]
        X[:, j] = np.nanmean(win, axis=1)
        X[:, j + 1] = np.nanstd(win, axis=1)
        X[:, j + 2] = np.nanmin(np.where(np.isnan(win), np.inf, win), axis=1)
        X[:, j + 3] = np.nanmax(np.where(np.isnan(win), -np.inf, win), axis=1)
        j += 4
    c = _cal(cal, cols + 1)
    for name in ("dow", "day", "month", "weekofyear", "is_weekend"):
        X[:, j] = c[name]
        j += 1
    X[:, j] = idx
    # nanmin/nanmax leave +-inf for all-NaN windows -> make them missing
    X[~np.isfinite(X)] = np.nan
    return X


def training_rows(Y: np.ndarray, max_origins: int, rng: np.random.Generator):
    """(series idx, origin col) pairs with enough history and a real target."""
    N, T = Y.shape
    idx_l, col_l = [], []
    for i in range(N):
        L = int(np.sum(~np.isnan(Y[i])))
        lo = T - L + max(max(LAGS), max(ROLLS)) - 1             # first column with enough history behind it
        hi = T - 2                        # need cols+1 valid
        if hi < lo:
            continue
        cols = np.arange(lo, hi + 1)
        if len(cols) > max_origins:
            # keep the recent half exactly, sample the rest (recency matters)
            recent = cols[-max_origins // 2:]
            rest = cols[:-max_origins // 2]
            cols = np.concatenate([rng.choice(rest, max_origins - len(recent), replace=False),
                                   recent])
            cols.sort()
        idx_l.append(np.full(len(cols), i, dtype=np.int32))
        col_l.append(cols.astype(np.int32))
    return np.concatenate(idx_l), np.concatenate(col_l)


def fit_model(X, y, target: str, params: dict, threads: int) -> lgb.LGBMRegressor:
    p = dict(params)
    p["n_jobs"] = threads
    p["verbose"] = -1
    if target == "log1p":
        p.update(objective="regression", metric="l2")
        yy = np.log1p(y)
    else:
        p.update(objective="tweedie", tweedie_variance_power=1.3, metric="tweedie")
        yy = y
    m = lgb.LGBMRegressor(**p)
    m.fit(X, yy, categorical_feature=["series_id"], feature_name=FEATS)
    return m


def predict_recursive(model, Y_hist: np.ndarray, h: int, cal: dict,
                      target: str, cal_anchor_col: int) -> np.ndarray:
    """Forecast h steps for every series, writing each prediction back into Y."""
    N, T = Y_hist.shape
    A = np.concatenate([Y_hist, np.full((N, h), np.nan, dtype=np.float32)], axis=1)
    # calendar must cover columns 0..T+h (features read cols+1 for the last step)
    cal_ext = common.calendar_matrix(T + h + 1, cal_anchor_col)
    out = np.empty((N, h), dtype=np.float64)
    idx = np.arange(N, dtype=np.int32)
    for k in range(h):
        c = T - 1 + k
        X = features(A, idx, np.full(N, c, dtype=np.int32), cal_ext)
        p = model.predict(X)
        p = np.expm1(p) if target == "log1p" else p
        p = np.clip(p, 0.0, None)
        A[:, c + 1] = p
        out[:, k] = p
    return out


BASE = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=7,
    max_bin=127,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--max-origins", type=int, default=700)
    ap.add_argument("--out", default="data/lgbm.json")
    ap.add_argument("--lags", default="1,7,14,28")
    ap.add_argument("--rolls", default="7,14,28")
    ap.add_argument("--preds", default="lgbm_preds.npz")
    args = ap.parse_args()

    global LAGS, ROLLS, FEATS
    LAGS = [int(x) for x in args.lags.split(",") if x.strip()]
    ROLLS = [int(x) for x in args.rolls.split(",") if x.strip()]
    FEATS = build_feats()

    t_all = time.time()
    data = common.load_m4_daily(n_series=args.series, seed=args.seed)
    N, T = data.train.shape
    print(f"[data] {N} series, train {data.train.shape}, test {data.test.shape}, "
          f"len med {int(np.median(data.lengths()))}", flush=True)

    cal = common.calendar_matrix(T, T - 1)          # last train col = anchor
    rng = np.random.default_rng(args.seed)

    # ---------------- validation stage: fit on train[:-14], score on train[-14:]
    Yv = data.train[:, :-common.H]
    cal_v = common.calendar_matrix(Yv.shape[1], Yv.shape[1] - 1)
    iv, cv = training_rows(Yv, args.max_origins, rng)
    Xv = features(Yv, iv, cv, cal_v)
    yv = Yv[iv, cv + 1]
    print(f"[val ] {len(yv)} training rows, {Xv.shape[1]} features", flush=True)

    val_preds, val_metrics, fitted = {}, {}, {}
    truth_v = data.train[:, -common.H:]
    for target in ("log1p", "tweedie"):
        t0 = time.time()
        m = fit_model(Xv, yv, target, BASE, args.threads)
        p = predict_recursive(m, Yv, common.H, cal_v, target, Yv.shape[1] - 1)
        res = common.evaluate(p, truth_v, Yv, data.ids)
        val_preds[target] = p
        val_metrics[target] = res["agg"]
        fitted[target] = m
        print(f"[val ] {target:8s} WAPE {res['agg']['wape_mean']:.4f} "
              f"sMAPE {res['agg']['smape_mean']:.2f} MASE {res['agg']['mase_mean']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    best = min(val_metrics, key=lambda k: val_metrics[k]["wape_mean"])
    print(f"[pick] {best} wins validation WAPE", flush=True)

    # ---------------- final stage: refit on the full training window
    i_f, c_f = training_rows(data.train, args.max_origins, rng)
    X = features(data.train, i_f, c_f, cal)
    y = data.train[i_f, c_f + 1]
    print(f"[fit ] {len(y)} training rows (full window)", flush=True)

    final_preds, final_metrics, runtime = {}, {}, {}
    for target in ("log1p", "tweedie"):
        t0 = time.time()
        m = fit_model(X, y, target, BASE, args.threads)
        p = predict_recursive(m, data.train, common.H, cal, target, T - 1)
        res = common.evaluate(p, data.test, data.train, data.ids)
        final_preds[target] = p
        final_metrics[target] = res
        runtime[target] = time.time() - t0
        print(f"[test] {target:8s} WAPE {res['agg']['wape_mean']:.4f} "
              f"sMAPE {res['agg']['smape_mean']:.2f} MASE {res['agg']['mase_mean']:.3f} "
              f"({runtime[target]:.0f}s)", flush=True)

    imp = dict(zip(FEATS, fitted[best].feature_importances_.tolist()))
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:12]

    payload = {
        "model": "LightGBM-global",
        "n_series": N,
        "horizon": common.H,
        "strategy": "recursive",
        "selected": best,
        "features": FEATS,
        "params": {k: v for k, v in BASE.items()},
        "validation": val_metrics,
        "test": {t: final_metrics[t]["agg"] for t in final_metrics},
        "runtime_s": {t: round(runtime[t], 1) for t in runtime},
        "total_runtime_s": round(time.time() - t_all, 1),
        "feature_importance_top": top,
        "notes": ("M4 daily; single global model, series id as categorical; "
                  "log1p+tweedie compared on the last 14 training points; "
                  "metrics on the official 14-day test horizon."),
    }
    common.write_json(common.DATA / Path(args.out).name, payload)

    np.savez_compressed(common.DATA / args.preds,
                        ids=np.array(data.ids),
                        **{f"{t}_{s}": arr for t in final_preds
                           for s, arr in (("test", final_preds[t]),
                                          ("val", val_preds[t]))},
                        test_truth=data.test.astype(np.float32))
    print(f"[done] {time.time()-t_all:.0f}s -> {args.out}, data/{args.preds}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
