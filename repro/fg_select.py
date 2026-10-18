"""元策略实验：用「训练段内部验证窗口」的证据，逐序列决定信 FG-MP 还是信 BSTS。

这不是在测试集上调参——每个序列的选择只用它自己在 Ytr 末尾 13 周的表现，
然后到真正的测试段去打分。对应因子图的说法：
    两条模型消息 N(μ_fg, σ_fg²)、N(μ_bs, σ_bs²) 相乘 → 权重 ∝ 1/σ²
这里 σ 由各序列自己的验证误差估计，因此权重是**逐序列**的，而不是全局一个系数。

另外输出：
  - 两个模型的误差相关系数（判断它们的错误是否同源）
  - 按序列特征（变异系数）分组的表现，看 FG 在哪些序列上更强
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H = 13

spec = importlib.util.spec_from_file_location("fg", ROOT / "repro" / "fg_mp.py")
fg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fg)


def wape_row(p, t):
    return np.abs(p - t).sum(1) / np.maximum(np.abs(t).sum(1), 1e-9)


def main():
    Y, T, ids = fg.load()
    Ytr, Yva = Y[:, :-H], Y[:, -H:]
    hyper = json.loads((ROOT / "repro" / "metrics_fg.json").read_text(encoding="utf-8"))["hyper"]

    for file in ['bsts_val_preds.npz', 'bsts_preds.npz', 'fg_preds.npz']:
        with np.load(ROOT / 'data' / file) as cached:
            if list(cached['ids']) != ids: raise ValueError(f'ID mismatch: {file}')

    # ---- FG-MP：训练段拟合 → 验证窗口预测
    m_val = fg.fit(Ytr, 52, 5, qr=hyper["qr"], ridge=hyper["ridge"], kappa=hyper["kappa"])
    val_fg = fg.forecast(m_val, damp=hyper["damp"])
    val_bs = np.load(ROOT / "data" / "bsts_val_preds.npz")["preds"]

    # ---- 测试段预测
    test_fg = np.load(ROOT / "data" / "fg_preds.npz")["preds"]
    test_bs = np.load(ROOT / "data" / "bsts_preds.npz")["preds"]

    e_val_fg, e_val_bs = wape_row(val_fg, Yva), wape_row(val_bs, Yva)
    print(f"验证窗口：FG {e_val_fg.mean() * 100:.2f}%  BSTS {e_val_bs.mean() * 100:.2f}%")

    # 1) 硬选择：验证窗口更好的那个模型
    pick_fg = e_val_fg <= e_val_bs
    sel = np.where(pick_fg[:, None], test_fg, test_bs)
    w_sel = wape_row(sel, T).mean()

    # 2) 精度加权（高斯消息乘积）：w = σ_bs²/(σ_fg²+σ_bs²)
    s_fg = np.maximum(e_val_fg, 1e-6) ** 2
    s_bs = np.maximum(e_val_bs, 1e-6) ** 2
    wgt = (s_bs / (s_fg + s_bs))[:, None]
    fused = wgt * test_fg + (1 - wgt) * test_bs
    w_fuse = wape_row(fused, T).mean()

    # 3) λ=0 等权，λ=1 验证权重；只在验证段选 λ（仍有同窗口复用偏差）
    best_shr, best_w = None, None
    for lam in np.arange(0, 1.01, 0.1):
        p = (lam * wgt + (1 - lam) * 0.5) * val_fg + (1 - (lam * wgt + (1 - lam) * 0.5)) * val_bs
        w = wape_row(p, Yva).mean()
        if best_w is None or w < best_w:
            best_shr, best_w = lam, w

    selected_weight = best_shr * wgt + (1 - best_shr) * 0.5
    best_w = wape_row(selected_weight * test_fg + (1-selected_weight)*test_bs, T).mean()

    # 4) 误差相关性（同期、逐序列）
    ef, eb = test_fg - T, test_bs - T
    corr = float(np.corrcoef(ef.ravel(), eb.ravel())[0, 1])

    # 5) 按波动性分组
    cv = np.std(Ytr, axis=1) / np.maximum(np.mean(Ytr, axis=1), 1e-9)
    q = np.quantile(cv, [0.33, 0.66])
    groups = {"低波动": cv <= q[0], "中波动": (cv > q[0]) & (cv <= q[1]), "高波动": cv > q[1]}

    print("\n== 元策略（测试段打分） ==")
    print(f"  FG-MP 单独            {wape_row(test_fg, T).mean() * 100:5.2f}%")
    print(f"  BSTS 单独             {wape_row(test_bs, T).mean() * 100:5.2f}%")
    print(f"  验证窗口硬选择         {w_sel * 100:5.2f}%   (选 FG {pick_fg.mean() * 100:.0f}% 的序列)")
    print(f"  精度加权融合           {w_fuse * 100:5.2f}%")
    print(f"  收缩后融合 (λ={best_shr:.1f})    {best_w * 100:5.2f}%")
    print(f"\n  误差相关系数           {corr:.3f}")
    print("\n== 按波动性分组（WAPE） ==")
    for g, m in groups.items():
        print(f"  {g:<6} n={m.sum():3d}  FG {wape_row(test_fg[m], T[m]).mean() * 100:5.2f}%  "
              f"BSTS {wape_row(test_bs[m], T[m]).mean() * 100:5.2f}%")

    out = {
        "val_wape": {"fg": float(e_val_fg.mean()), "bsts": float(e_val_bs.mean())},
        "test_wape": {
            "fg": float(wape_row(test_fg, T).mean()),
            "bsts": float(wape_row(test_bs, T).mean()),
            "hard_select": float(w_sel),
            "precision_fusion": float(w_fuse),
            "shrunk_fusion": float(best_w),
            "shrink_lambda": float(best_shr),
        },
        "error_corr": corr,
        "pick_fg_share": float(pick_fg.mean()),
        "groups": {g: {"n": int(m.sum()),
                       "fg": float(wape_row(test_fg[m], T[m]).mean()),
                       "bsts": float(wape_row(test_bs[m], T[m]).mean())}
                   for g, m in groups.items()},
    }
    (ROOT / "repro" / "metrics_select.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
