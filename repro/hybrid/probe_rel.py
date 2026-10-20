"""Probe 4: horizon-aware target-relative features for direct multi-step GBM.

For target t = origin + h - 1, y[t-k] is observed whenever k >= h.  Using these
phase-aligned lags (instead of only origin-anchored lags) lets one global model
see the right seasonal history for every horizon.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
sys.path.insert(0, str(ROOT / "repro" / "alt"))
import fg_mp
from bench_rossmann import Panel, PANEL

H = 13


def wape(p, t):
    return float((np.abs(p - t).sum(1) / np.abs(t).sum(1)).mean())


def row(yh, o, h, season, sid, exog=None, cal=None, static=None):
    t = o + h - 1
    d = {}
    for k in (season, season + 1, 2 * season, 3 * season):
        j = t - k
        d[f"slag{k}"] = yh[:, j] if 0 <= j < o else np.full(yh.shape[0], np.nan)
    for k in (1, 2, 3, 7, 14, 28):
        j = t - k
        d[f"tlag{k}"] = yh[:, j] if 0 <= j < o else np.full(yh.shape[0], np.nan)
    for k in (1, 7, 14):
        j = o - k
        d[f"lag{k}"] = yh[:, j] if j >= 0 else np.full(yh.shape[0], np.nan)
    for w in (season // 2, season, 2 * season):
        w = min(w, yh.shape[1])
        seg = yh[:, -w:]
        d[f"m{w}"] = seg.mean(1)
        d[f"s{w}"] = seg.std(1)
    d["lvl_ratio"] = d[f"m{season}"] / np.maximum(d[f"m{2*season}"], 1.0)
    d["phase"] = np.full(yh.shape[0], t % season)
    d["h"] = np.full(yh.shape[0], h)
    d["sid"] = sid
    if cal is not None:
        d["dow"] = cal["dow"][t]
        d["month"] = cal["month"][t]
        d["dom"] = cal["day"][t]
        d["woy"] = cal["woy"][t]
    if static:
        d.update(static)
    if exog:
        for k, v in exog.items():
            d[k] = v
    return pd.DataFrame(d)


def build_weekly(origin):
    y, truth, ids = fg_mp.load()
    y = y.astype(np.float64)
    return y, y[:, origin:origin + H] if origin + H <= y.shape[1] else truth, np.arange(y.shape[0])


def fit_weekly(origin, objective="l2", n=600):
    y, tgt, sid = build_weekly(origin)
    X, Y = [], []
    for ot in range(120, origin):
        for h in range(1, H + 1):
            T = ot + h - 1
            if T >= origin:
                continue
            X.append(row(y[:, :ot], ot, h, 52, sid))
            Y.append(y[:, T])
    X = pd.concat(X, ignore_index=True)
    X["sid"] = X["sid"].astype("category")
    m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1, objective=objective,
                          n_estimators=n, learning_rate=0.05, num_leaves=63,
                          min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
    m.fit(X, np.log1p(np.concatenate(Y)), categorical_feature=["sid"])
    out = np.zeros((len(sid), H))
    for h in range(1, H + 1):
        r = row(y[:, :origin], origin, h, 52, sid)
        r["sid"] = pd.Categorical(r["sid"], categories=X["sid"].cat.categories)
        out[:, h - 1] = np.expm1(m.predict(r[X.columns]))
    return np.clip(out, 0, None)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    if which == "weekly":
        for o in (161, 174, 187):
            t0 = time.time()
            p = fit_weekly(o)
            y, tgt, _ = build_weekly(o)
            print(o, round(wape(p, tgt), 5), f"{time.time()-t0:.1f}s", flush=True)
    else:
        p = Panel(PANEL)
        o = int(sys.argv[2]) if len(sys.argv) > 2 else 774
        t0 = time.time()
        sid = p.stores
        X, Y = [], []
        for ot in range(56, o - 42 + 1, 7):
            for h in range(1, 43):
                t = ot + h - 1
                if t >= o:
                    continue
                cal = {"dow": p.cal["dow"], "month": p.cal["month"],
                       "day": p.cal["day"], "woy": p.cal["woy"]}
                exog = {"promo": p.promo[:, t], "open": p.open_[:, t],
                        "school": p.school[:, t], "state": p.state[:, t]}
                static = {"store_type": p.store_type, "assortment": p.assortment,
                          "promo2": p.promo2, "comp": np.log1p(p.comp_dist)}
                X.append(row(p.y[:, :ot].astype(np.float64), ot, h, 7, sid,
                             exog=exog, cal=cal, static=static))
                Y.append(p.y[:, t])
        X = pd.concat(X, ignore_index=True)
        X["sid"] = X["sid"].astype("category")
        m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1,
                              n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=50, subsample=0.9,
                              colsample_bytree=0.9, reg_lambda=1.0)
        m.fit(X, np.log1p(np.concatenate(Y)), categorical_feature=["sid"])
        out = np.zeros((p.n, 42))
        for h in range(1, 43):
            t = o + h - 1
            cal = {"dow": p.cal["dow"], "month": p.cal["month"],
                   "day": p.cal["day"], "woy": p.cal["woy"]}
            exog = {"promo": p.promo[:, t], "open": p.open_[:, t],
                    "school": p.school[:, t], "state": p.state[:, t]}
            static = {"store_type": p.store_type, "assortment": p.assortment,
                      "promo2": p.promo2, "comp": np.log1p(p.comp_dist)}
            r = row(p.y[:, :o].astype(np.float64), o, h, 7, p.stores,
                    exog=exog, cal=cal, static=static)
            r["sid"] = pd.Categorical(r["sid"], categories=X["sid"].cat.categories)
            out[:, h - 1] = np.expm1(m.predict(r[X.columns]))
        out = np.where(p.open_[:, o:o + 42] == 1, out, 0.0)
        y = p.y[:, o:o + 42].astype(np.float64)
        print("rossmann_rel", o, round(wape(np.clip(out, 0, None), y), 5), f"{time.time()-t0:.1f}s")

