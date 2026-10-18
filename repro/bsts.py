"""BSTS 的极大似然版：结构时序模型（局部水平 + 随机季节 + 观测噪声）。

BSTS (Bayesian Structural Time Series) = 同一个状态空间模型 + 先验 + MCMC。
statsmodels 的 UnobservedComponents 是同一模型的 Kalman 滤波 + 极大似然估计，
所以这里的数字说明"模型结构值多少"，而不包含贝叶斯先验带来的差异。

用法：python repro/bsts.py --n 120
输出：repro/metrics_bsts.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
H = 13          # M4-Weekly 官方预测步长
SEASON = 13     # 季度长度假设；年周期应为 52，不能与季度数混淆


def load(n: int | None, tail: int = 200):
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
    if n and n < len(ids):
        idx = np.arange(n)
        return Y[idx], T[idx], [ids[i] for i in idx]
    return Y, T, ids


def fit_one(y: np.ndarray):
    """返回 (forecast, ok)；失败返回 None。"""
    try:
        mod = UnobservedComponents(
            y, level="local level", seasonal=SEASON,
            stochastic_level=True, stochastic_seasonal=True,
        )
        res = mod.fit(method="lbfgs", maxiter=150, disp=False)
        fc = res.forecast(H)
        fc = np.asarray(fc, dtype=float)
        if not np.all(np.isfinite(fc)):
            return None
        return np.clip(fc, 0, None)
    except Exception:
        return None


def seasonal_naive(y: np.ndarray) -> np.ndarray:
    return np.asarray(y[-SEASON:], dtype=float)


def wape(p, t):
    d = np.abs(t).sum()
    return float(np.abs(p - t).sum() / d) if d > 0 else np.nan


def mase(p, t, y):
    scale = np.mean(np.abs(y[SEASON:] - y[:-SEASON])) if len(y) > SEASON else np.nan
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="只跑前 n 条序列（0 = 全部）")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="metrics_bsts.json")
    args = ap.parse_args()

    Y, T, ids = load(args.n or None)
    N = len(ids)
    print(f"[data] {N} series, horizon {H}, season {SEASON}, tail {Y.shape[1]}", flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(fit_one)(Y[i]) for i in range(N)
    )
    fit_s = time.time() - t0

    w_b, w_s, m_b = [], [], []
    failed = []
    preds = np.zeros((N, H))
    for i in range(N):
        p = res[i]
        if p is None:
            p = seasonal_naive(Y[i])
            failed.append(ids[i])
        preds[i] = p
        w_b.append(wape(p, T[i]))
        w_s.append(wape(seasonal_naive(Y[i]), T[i]))
        m_b.append(mase(p, T[i], Y[i]))

    out = {
        "model": "BSTS-structural (UCM, MLE)",
        "note": "局部水平 + 随机季节(13) + 观测噪声；Kalman 滤波 + 极大似然。"
                "BSTS 的贝叶斯版是同一模型加先验 + MCMC。",
        "n_series": N,
        "horizon": H,
        "fit_runtime_s": round(fit_s, 1),
        "wape": float(np.nanmean(w_b)),
        "wape_median": float(np.nanmedian(w_b)),
        "mase": float(np.nanmean(m_b)),
        "seasonal_naive_wape_same_subset": float(np.nanmean(w_s)),
        "n_failed": len(failed),
        "failed_ids": failed[:10],
    }
    (ROOT / "repro" / args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
