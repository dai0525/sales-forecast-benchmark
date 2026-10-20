"""Probe 3: log-target objective + WAPE-optimal multiplicative calibration."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
import fg_mp
from probe_weekly import feats, wape

H = 13


def train_predict(y_hist, sid, origin, objective="l2", n=600):
    X, Y = [], []
    for ot in range(120, origin):
        for h in range(1, H + 1):
            t = ot + h - 1
            if t >= origin:
                continue
            X.append(feats(y_hist[:, :ot], h, sid))
            Y.append(y_hist[:, t])
    X = pd.concat(X, ignore_index=True)
    X["sid"] = X["sid"].astype("category")
    m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1, objective=objective,
                          n_estimators=n, learning_rate=0.05, num_leaves=63,
                          min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
                          reg_lambda=1.0)
    m.fit(X, np.log1p(np.concatenate(Y)), categorical_feature=["sid"])
    out = np.zeros((y_hist.shape[0], H))
    for h in range(1, H + 1):
        row = feats(y_hist, h, sid)
        row["sid"] = pd.Categorical(row["sid"], categories=X["sid"].cat.categories)
        out[:, h - 1] = np.expm1(m.predict(row[X.columns]))
    return np.clip(out, 0, None), m


def calibration(y_hist, sid, origin, objective, k=8.0):
    """Fit on < origin-H, score the last H weeks, return shrunk per-series factor."""
    ocal = origin - H
    p, _ = train_predict(y_hist[:, :ocal], sid, ocal, objective)
    y = y_hist[:, ocal:origin]
    r = np.log((y + 1.0) / (p + 1.0))
    med = np.median(r, axis=1)
    # shrink toward 0 (no correction); each series contributes H observations
    med = med * (H / (H + k))
    return np.exp(med)


if __name__ == "__main__":
    y, truth, ids = fg_mp.load()
    y = y.astype(np.float64)
    sid = np.arange(y.shape[0])
    for origin in (161, 174, 187):
        tgt = y[:, origin:origin + H]
        for obj in ("l2", "regression_l1"):
            t0 = time.time()
            p, _ = train_predict(y[:, :origin], sid, origin, obj)
            print(origin, obj, "base", round(wape(p, tgt), 5), f"{time.time()-t0:.1f}s", flush=True)
            a = calibration(y[:, :origin], sid, origin, obj)
            print(origin, obj, "cal ", round(wape(p * a[:, None], tgt), 5),
                  "factor p10/50/90", np.round(np.percentile(a, [10, 50, 90]), 3), flush=True)
