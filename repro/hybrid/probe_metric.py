"""Probe 2: metric-aligned training (weighted L1 == mean WAPE objective)."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
import fg_mp
from probe_weekly import feats, baseline_seasonal, wape

H = 13


def build_train(y_hist, sid, origin, hmax=H):
    X, Y, W = [], [], []
    for ot in range(120, origin):
        den = np.abs(y_hist[:, ot:ot + hmax]).sum(1)
        den = np.maximum(den, 1.0)
        for h in range(1, hmax + 1):
            t = ot + h - 1
            if t >= origin:
                continue
            X.append(feats(y_hist[:, :ot], h, sid))
            Y.append(y_hist[:, t])
            # each row carries its series/origin denominator -> exact mean-WAPE weights
            W.append(1.0 / den)
    return pd.concat(X, ignore_index=True), np.concatenate(Y), np.concatenate(W)


def fit(y_hist, sid, origin, objective="l2", weight=False, params=None):
    X, Y, W = build_train(y_hist, sid, origin)
    X["sid"] = X["sid"].astype("category")
    m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1, objective=objective,
                          **(params or {}))
    m.fit(X, Y, sample_weight=W if weight else None, categorical_feature=["sid"])
    out = np.zeros((y_hist.shape[0], H))
    for h in range(1, H + 1):
        row = feats(y_hist, h, sid)
        row["sid"] = pd.Categorical(row["sid"], categories=X["sid"].cat.categories)
        out[:, h - 1] = m.predict(row[X.columns])
    return np.clip(out, 0, None), m


if __name__ == "__main__":
    y, truth, ids = fg_mp.load()
    y = y.astype(np.float64)
    sid = np.arange(y.shape[0])
    variants = {
        "l2_plain": dict(objective="l2", weight=False, params=dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)),
        "l2_weight": dict(objective="l2", weight=True, params=dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)),
        "l1_weight": dict(objective="regression_l1", weight=True, params=dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)),
        "l1_plain": dict(objective="regression_l1", weight=False, params=dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)),
    }
    for origin in (161, 174, 187):
        tgt = y[:, origin:origin + H]
        for name, cfg in variants.items():
            t0 = time.time()
            p, _ = fit(y[:, :origin], sid, origin, **cfg)
            print(origin, name, round(wape(p, tgt), 5), f"{time.time()-t0:.1f}s", flush=True)
