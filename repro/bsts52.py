"""BSTS 结构模型（周度，season=52）全量跑分。

- 模型：局部水平 + 随机季节(52) + 观测噪声，Kalman 滤波 + 极大似然
- 指标：逐序列 WAPE 取均值（与报告口径一致）与总量加权 WAPE（两种都报）
- 基线：SeasonalNaive(52) 与 SeasonalNaive(13) 在同一批序列上重算
用法：python repro/bsts52.py --jobs 3
输出：repro/metrics_bsts52.json、data/bsts_preds.npz
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
H = 13
SEASON = 52


def load(n: int | None = None, tail: int = 200):
    z = np.load(ROOT / "data" / "weekly.npz", allow_pickle=True)
    tr, te, ids = z["train"].T, z["test"].T, [str(s) for s in z["ids"]]
    keep = []
    for i in range(tr.shape[0]):
        v = tr[i]
        fin = np.flatnonzero(~np.isnan(v))
        v = v[fin[0]:fin[-1] + 1] if len(fin) else np.zeros(tail)
        if len(v) < tail:
            v = np.concatenate([np.repeat(v[0], tail - len(v)), v])
        keep.append(v[-tail:])
    Y = np.vstack(keep)
    T = np.nan_to_num(te, nan=0.0)
    if n:
        return Y[:n], T[:n], ids[:n]
    return Y, T, ids


def fit_one(y):
    try:
        m = UnobservedComponents(y, level="local level", seasonal=SEASON,
                                 stochastic_seasonal=True)
        r = m.fit(method="lbfgs", maxiter=150, disp=False)
        p = np.asarray(r.forecast(H), float)
        if not np.all(np.isfinite(p)):
            return None
        return np.clip(p, 0, None)
    except Exception:
        return None


def sn(y, season):
    return np.array([y[-season + (i % season)] for i in range(H)])


def wape_vec(p, t):
    d = np.abs(t).sum()
    return float(np.abs(p - t).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    Y, T, ids = load(args.n or None)
    N = len(ids)
    print(f"[data] {N} series, season {SEASON}, horizon {H}", flush=True)

    import time
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, backend="loky")(delayed(fit_one)(Y[i]) for i in range(N))
    fit_s = time.time() - t0

    preds = np.zeros((N, H))
    failed = 0
    for i in range(N):
        if res[i] is None:
            preds[i] = sn(Y[i], SEASON)
            failed += 1
        else:
            preds[i] = res[i]

    err = np.abs(preds - T)
    tot = np.abs(T)
    out = {
        "model": "BSTS-structural (UCM, MLE), season=52",
        "n_series": N,
        "horizon": H,
        "season": SEASON,
        "fit_runtime_s": round(fit_s, 1),
        "wape_mean_perseries": float(np.mean(err.sum(1) / np.maximum(tot.sum(1), 1e-9))),
        "wape_aggregate": float(err.sum() / tot.sum()),
        "n_failed": failed,
        "baseline_sn52_mean_perseries": float(np.mean(
            [wape_vec(sn(Y[i], 52), T[i]) for i in range(N)])),
        "baseline_sn52_aggregate": float(
            np.abs(np.vstack([sn(Y[i], 52) for i in range(N)]) - T).sum() / tot.sum()),
        "baseline_sn13_mean_perseries": float(np.mean(
            [wape_vec(sn(Y[i], 13), T[i]) for i in range(N)])),
    }
    (ROOT / "repro" / "metrics_bsts52.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "bsts_preds.npz", preds=preds, ids=np.array(ids))
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
