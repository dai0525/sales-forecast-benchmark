"""融合实验：FG-MP × BSTS × SeasonalNaive

因子图视角下，两个模型对同一未来点的预测各是一条高斯消息 N(μ_j, σ_j²)。
消息相乘（= 后验）得到精度加权平均，权重 ∝ 1/σ²。这里先用三种免调参的融合：
  1) 算术平均（等权，M4 竞赛里最稳的做法）
  2) 几何平均（对量级更稳健）
  3) 中位数（抗单点崩）
并给出"每个序列都选最好的那个"的 oracle 上界，用来判断还有多少空间。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def wape_mean(p, t):
    return float(np.mean(np.abs(p - t).sum(1) / np.maximum(np.abs(t).sum(1), 1e-9)))


def main():
    z = np.load(ROOT / "data" / "weekly.npz", allow_pickle=True)
    tr, te = z["train"].T, z["test"].T
    T = np.nan_to_num(te, nan=0.0)

    fg = np.load(ROOT / "data" / "fg_preds.npz")["preds"]
    bs = np.load(ROOT / "data" / "bsts_preds.npz")["preds"]
    sn = np.vstack([np.array([v[-52 + (i % 52)] for i in range(T.shape[1])])
                    for v in [r[np.isfinite(r)] for r in tr]])

    models = {"FG-MP": fg, "BSTS": bs, "SeasonalNaive": sn}
    individual = {k: wape_mean(v, T) for k, v in models.items()}

    combos = {
        "FG-MP + BSTS（算术平均）": (fg + bs) / 2,
        "FG-MP + BSTS（几何平均）": np.sqrt(np.clip(fg, 1e-6, None) * np.clip(bs, 1e-6, None)),
        "FG-MP + BSTS（中位数）": np.median(np.stack([fg, bs]), axis=0),
        "FG + BSTS + SN（算术平均）": (fg + bs + sn) / 3,
        "FG×2 + BSTS（偏 FG）": (2 * fg + bs) / 3,
        "FG + BSTS×2（偏 BSTS）": (fg + 2 * bs) / 3,
    }
    res = {k: wape_mean(v, T) for k, v in combos.items()}

    oracle = np.minimum(np.minimum(fg, bs), sn)
    # oracle 需要按序列选模型，这里只在"整条序列选最优模型"意义上成立
    per = np.stack([np.abs(fg - T).sum(1) / np.abs(T).sum(1),
                    np.abs(bs - T).sum(1) / np.abs(T).sum(1),
                    np.abs(sn - T).sum(1) / np.abs(T).sum(1)])
    oracle_series = float(per.min(0).mean())

    print("== 单模型 ==")
    for k, v in sorted(individual.items(), key=lambda x: x[1]):
        print(f"  {k:<14} {v * 100:5.2f}%")
    print("\n== 融合 ==")
    for k, v in sorted(res.items(), key=lambda x: x[1]):
        print(f"  {k:<26} {v * 100:5.2f}%")
    print(f"\n  oracle（逐序列选最优）  {oracle_series * 100:5.2f}%")
    best_name = min(res, key=res.get)
    print(f"\n最佳：{best_name} = {res[best_name] * 100:.2f}%  "
          f"（相对 BSTS 改善 {(individual['BSTS'] - res[best_name]) * 100:.2f} pp）")

    out = {"individual": {k: round(v, 5) for k, v in individual.items()},
           "combos": {k: round(v, 5) for k, v in res.items()},
           "oracle_series": round(oracle_series, 5),
           "best": best_name}
    (ROOT / "repro" / "metrics_ensemble.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "ensemble_preds.npz", preds=combos[best_name])


if __name__ == "__main__":
    main()
