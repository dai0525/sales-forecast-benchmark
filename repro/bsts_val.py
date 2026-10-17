"""在训练段内部切出验证窗口（最后 13 周），为 BSTS 生成验证期预测。

用途：为"逐序列模型选择/精度加权"这类元策略提供**只依赖训练段**的证据，
避免用测试集挑模型（那是泄漏）。
输出：data/bsts_val_preds.npz
"""
from __future__ import annotations

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


def load(tail: int = 200):
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
    return np.vstack(keep), np.nan_to_num(te, nan=0.0), ids


def fit_one(y):
    try:
        m = UnobservedComponents(y, level="local level", seasonal=SEASON,
                                 stochastic_seasonal=True)
        r = m.fit(method="lbfgs", maxiter=150, disp=False)
        p = np.asarray(r.forecast(H), float)
        return np.clip(p, 0, None) if np.all(np.isfinite(p)) else None
    except Exception:
        return None


def main():
    Y, T, ids = load()
    Ytr, Yva = Y[:, :-H], Y[:, -H:]
    res = Parallel(n_jobs=3, backend="loky")(delayed(fit_one)(Ytr[i]) for i in range(len(ids)))
    preds = np.zeros((len(ids), H))
    for i, r in enumerate(res):
        preds[i] = r if r is not None else Yva[i - 0].mean()
    w = float(np.mean(np.abs(preds - Yva).sum(1) / np.maximum(np.abs(Yva).sum(1), 1e-9)))
    print(json.dumps({"n": len(ids), "val_wape": round(w, 5), "failed": sum(r is None for r in res)},
                     ensure_ascii=False))
    np.savez_compressed(ROOT / "data" / "bsts_val_preds.npz", preds=preds, ids=np.array(ids))


if __name__ == "__main__":
    main()
