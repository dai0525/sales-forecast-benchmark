"""Probe: raw-level LightGBM vs multiplicative ratio LightGBM on M4 Weekly.

Direct multi-horizon, one model with h as a feature.  No test data is touched
for fitting; each origin only sees history < origin.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
import fg_mp

H = 13
SEASON = 52


def feats(yh: np.ndarray, h: int, sid: np.ndarray):
    """Feature frame for one origin; yh (N,o) history only."""
    o = yh.shape[1]
    d = {}
    for lag in (1, 2, 3, 4, 8, 13, 26, 52, 53, 104):
        d[f"lag{lag}"] = yh[:, -lag]
    for w in (4, 8, 13, 26, 52):
        seg = yh[:, -w:]
        d[f"mean{w}"] = seg.mean(1)
        d[f"std{w}"] = seg.std(1)
        d[f"max{w}"] = seg.max(1)
    d["last_ratio"] = yh[:, -1] / (yh[:, -52] + 1e-6)
    d["trend"] = yh[:, -4:].mean(1) / (yh[:, -52:-48].mean(1) + 1e-6)
    d["h"] = np.full(yh.shape[0], h)
    d["phase"] = np.full(yh.shape[0], (o + h - 1) % SEASON)
    d["sid"] = sid
    return pd.DataFrame(d)


def baseline_seasonal(yh, h, mode="gmean3"):
    """Baseline for target at origin+h using only history."""
    N, o = yh.shape
    idx = o + h - 1
    if mode == "snaive":
        return yh[:, idx - SEASON].copy()
    vals = []
    for k in (1, 2, 3):
        j = idx - SEASON * k
        vals.append(yh[:, max(j, 0)])
    V = np.vstack(vals).T  # N,3
    if mode == "gmean3":
        return np.exp(np.log(np.maximum(V, 1e-6)).mean(1))
    return np.median(V, axis=1)


def build(yh, sid, hmax, mode, c_frac=0.01):
    """Return X, target, baseline matrix B (N,hmax), log-target."""
    N, o = yh.shape
    Xs, ys, Bs = [], [], np.zeros((N, hmax))
    for h in range(1, hmax + 1):
        b = baseline_seasonal(yh, h, mode)
        b = np.maximum(b, 1e-6)
        Bs[:, h - 1] = b
        tgt = yh[:, o + h - 1] if o + h - 1 < yh.shape[1] else None
        Xs.append(feats(yh, h, sid))
        ys.append(tgt)
    return Xs, ys, Bs


def fit_predict(y_hist, sid, origin, mode, params, ratio=True, c_frac=0.01):
    """Train on all target days < origin, predict h=1..13 at origin."""
    N, _ = y_hist.shape
    Xtr, ytr, btr = [], [], []
    # training origins: every week from 120 to origin-1 (direct targets inside history)
    for ot in range(120, origin):
        for h in range(1, H + 1):
            t = ot + h - 1
            if t >= origin:
                continue
            b = baseline_seasonal(y_hist[:, :ot], h, mode)
            b = np.maximum(b, 1e-6)
            row = feats(y_hist[:, :ot], h, sid)
            Xtr.append(row)
            ytr.append(y_hist[:, t])
            btr.append(b)
    Xtr = pd.concat(Xtr, ignore_index=True)
    Xtr["sid"] = Xtr["sid"].astype("category")
    ytr = np.concatenate(ytr)
    btr = np.concatenate(btr)
    if ratio:
        # symmetric log-ratio around baseline; c stabilises zeros
        c = c_frac * max(np.median(np.abs(ytr)), 1.0)
        tgt = np.log((ytr + c) / (btr + c))
    else:
        tgt = np.log1p(ytr)
    m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1, **params)
    m.fit(Xtr, tgt, categorical_feature=["sid"])
    out, B = np.zeros((N, H)), np.zeros((N, H))
    for h in range(1, H + 1):
        b = np.maximum(baseline_seasonal(y_hist, h, mode), 1e-6)
        row = feats(y_hist, h, sid)
        row["sid"] = pd.Categorical(row["sid"], categories=Xtr["sid"].cat.categories)
        pr = m.predict(row)
        if ratio:
            c = c_frac * max(np.median(np.abs(ytr)), 1.0)
            out[:, h - 1] = (b + c) * np.exp(pr) - c
        else:
            out[:, h - 1] = np.expm1(pr)
        B[:, h - 1] = b
    return np.clip(out, 0, None), B, m


def wape(p, t):
    e = np.abs(p - t)
    return float((e.sum(1) / np.abs(t).sum(1)).mean())


if __name__ == "__main__":
    y, truth, ids = fg_mp.load()
    y = y.astype(np.float64)
    print("shape", y.shape, "zero share", float((y == 0).mean()), "test zero", float((truth == 0).mean()))
    sid = np.arange(y.shape[0])
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0)
    res = {}
    for origin in (148, 161, 174, 187):
        truth_o = y[:, origin:origin + H] if origin + H <= y.shape[1] else truth
        for name, ratio, mode in [("raw_snaive", False, "snaive"),
                                  ("raw_gm", False, "gmean3"),
                                  ("ratio_snaive", True, "snaive"),
                                  ("ratio_gm", True, "gmean3"),
                                  ("ratio_med", True, "median3")]:
            t0 = time.time()
            p, B, m = fit_predict(y[:, :origin], sid, origin, mode, params, ratio=ratio)
            print(origin, name, round(wape(p, truth_o), 5), f"{time.time()-t0:.1f}s", flush=True)
            res[f"{origin}/{name}"] = wape(p, truth_o)
    print(json.dumps(res, indent=1))
