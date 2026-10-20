"""Probe: Rossmann LightGBM variants (raw level vs multiplicative ratio)."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
sys.path.insert(0, str(ROOT / "repro" / "alt"))
from bench_rossmann import Panel, PANEL, H, SEASON

MIN_HIST = 56
STEP = 7


def base_same_dow(yh, openh, o, h, weeks=4):
    """Median of open same-weekday values over the last `weeks` weeks."""
    N = yh.shape[0]
    j = o + h - 1
    vals = np.full((N, weeks), np.nan)
    for k in range(1, weeks + 1):
        t = j - 7 * k
        if t >= 0:
            v = np.where(openh[:, t] == 1, yh[:, t], np.nan)
            vals[:, k - 1] = v
    med = np.nanmedian(vals, axis=1)
    # fallback: recent open days median, then overall open mean
    recent = np.where(openh[:, max(0, o - 28):o] == 1, yh[:, max(0, o - 28):o], np.nan)
    fb = np.nanmedian(recent, axis=1)
    med = np.where(np.isfinite(med), med, fb)
    med = np.where(np.isfinite(med), med, np.nanmean(np.where(openh[:, :o] == 1, yh[:, :o], np.nan), axis=1))
    return np.maximum(med, 1.0)


def features(p, o, h):
    yh = p.y[:, :o].astype(np.float64)
    oh = p.open_[:, :o]
    t = o + h - 1
    d = {}
    for lag in (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42):
        d[f"lag{lag}"] = yh[:, -lag]
    for w in (7, 14, 28, 56):
        seg = yh[:, -w:]
        d[f"mean{w}"] = seg.mean(1)
        d[f"std{w}"] = seg.std(1)
        d[f"max{w}"] = seg.max(1)
    for k in (1, 2, 3, 4):
        idx = [o - 7 * k + (t - o) % 7]
        d[f"same_dow_{k}"] = yh[:, max(0, idx[0])]
    d["open_rate28"] = oh[:, -28:].mean(1)
    d["promo_rate28"] = p.promo[:, o - 28:o].mean(1)
    d["log_level"] = np.log1p(np.maximum(yh[:, -28:].mean(1), 0))
    d["ratio_7_28"] = d["mean7"] / np.maximum(d["mean28"], 1.0)
    d["ratio_28_56"] = d["mean28"] / np.maximum(d["mean56"], 1.0)
    d["h"] = np.full(p.n, h)
    d["dow"] = np.full(p.n, p.cal["dow"][t])
    d["day"] = np.full(p.n, p.cal["day"][t])
    d["month"] = np.full(p.n, p.cal["month"][t])
    d["woy"] = np.full(p.n, p.cal["woy"][t])
    d["promo"] = p.promo[:, t]
    d["school"] = p.school[:, t]
    d["state"] = p.state[:, t]
    d["open"] = p.open_[:, t]
    d["store_type"] = p.store_type
    d["assortment"] = p.assortment
    d["promo2"] = p.promo2
    d["comp_dist"] = np.log1p(p.comp_dist)
    d["store_id"] = p.stores
    return pd.DataFrame(d)


def fit_predict(p, o, mode="ratio", params=None, use_store=True):
    Xtr, ytr, btr = [], [], []
    for ot in range(MIN_HIST, o - H + 1, STEP):
        for h in range(1, H + 1):
            t = ot + h - 1
            if t >= o:
                continue
            Xtr.append(features(p, ot, h))
            ytr.append(p.y[:, t].astype(np.float64))
            if mode == "ratio":
                btr.append(base_same_dow(p.y, p.open_, ot, h))
    Xtr = pd.concat(Xtr, ignore_index=True)
    ytr = np.concatenate(ytr)
    if use_store:
        Xtr["store_id"] = Xtr["store_id"].astype("category")
    else:
        Xtr = Xtr.drop(columns=["store_id"])
    if mode == "ratio":
        btr = np.concatenate(btr)
        c = 0.01 * max(np.median(ytr[ytr > 0]), 1.0)
        tgt = np.log((ytr + c) / (btr + c))
    else:
        c = 0.01 * max(np.median(ytr[ytr > 0]), 1.0)
        tgt = np.log1p(ytr)
    m = lgb.LGBMRegressor(random_state=42, n_jobs=6, verbose=-1, **(params or {}))
    m.fit(Xtr, tgt, categorical_feature=["store_id"] if use_store else [])
    out = np.zeros((p.n, H))
    for h in range(1, H + 1):
        row = features(p, o, h)
        if use_store:
            row["store_id"] = pd.Categorical(row["store_id"], categories=Xtr["store_id"].cat.categories)
        else:
            row = row.drop(columns=["store_id"])
        pr = m.predict(row[Xtr.columns])
        if mode == "ratio":
            b = base_same_dow(p.y, p.open_, o, h)
            out[:, h - 1] = (b + c) * np.exp(pr) - c
        else:
            out[:, h - 1] = np.expm1(pr)
    # known closed days are exactly zero
    out = np.where(p.open_[:, o:o + H] == 1, out, 0.0)
    return np.clip(out, 0, None), m


def wape_mean(p, y):
    den = np.abs(y).sum(1)
    return float(np.mean(np.abs(p - y).sum(1) / den))


if __name__ == "__main__":
    panel = Panel(PANEL)
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                  min_child_samples=50, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "ratio"
    origins = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [774]
    for o in origins:
        t0 = time.time()
        pr, m = fit_predict(panel, o, mode=mode, params=params)
        y = panel.y[:, o:o + H].astype(np.float64)
        print(mode, o, round(wape_mean(pr, y), 5), f"{time.time()-t0:.1f}s", flush=True)
