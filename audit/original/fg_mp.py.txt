"""FG-MP: 因子图 + 高斯信念传播（Message Passing）做销量预测

把「总销量 = 多个因子叠加」写成一个因子图，用 message passing 求各因子贡献：

    观测因子  ψ_t : y_t = level_t + λ_i·S_{φ(t)} + e_t
    水平变量  level_t   —— 随机游走（局部水平）
    共享季节  S_k (k=1..52) —— 跨所有序列共享，是"多用户叠加在同一个资源上"的类比
    载荷      λ_i    —— 每条序列对共享季节的"信道增益"

图结构是二部图：变量节点 {level_t, S_k} ←→ 观测因子 {y_t}。
在线性-高斯假设下，sum-product 消息 = 均值/方差，因此：
  - 变量→因子 消息：高斯 N(μ, σ²)
  - 因子→变量 消息：线性组合的高斯
  - 迭代到不动点 = EM / 交替最小二乘（backfitting）

关键区别（相对于 SCMA 的 MPA）：
  SCMA 的码本是离散有限集，消息是 M 维向量；这里因子是连续高斯，消息退化成
  (mean, precision) 两个标量 → 复杂度从 O(M^df) 降到 O(1)，但结构完全同构。

用法：
    python repro/fg_mp.py --period 52 --iters 5 --qr 0.15
输出：
    repro/metrics_fg.json、data/fg_preds.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H = 13
PERIOD = 52


# ---------------------------------------------------------------- 数据
def load(tail: int = 200, n: int | None = None):
    z = np.load(ROOT / "data" / "weekly.npz", allow_pickle=True)
    tr, te, ids = z["train"].T, z["test"].T, [str(s) for s in z["ids"]]
    keep = []
    for i in range(tr.shape[0]):
        v = tr[i]
        fin = np.flatnonzero(~np.isnan(v))
        v = v[fin[0]:fin[-1] + 1] if len(fin) else np.zeros(tail)
        if len(v) < tail:                      # 短的左边用首值填充
            v = np.concatenate([np.repeat(v[0], tail - len(v)), v])
        keep.append(v[-tail:])
    Y = np.vstack(keep)
    T = np.nan_to_num(te, nan=0.0)
    if n:
        return Y[:n], T[:n], ids[:n]
    return Y, T, ids


# ---------------------------------------------------------------- 消息：Kalman
def kalman_level(z: np.ndarray, qr: float = 0.15):
    """局部水平模型的高斯消息传递（前向滤波 + RTS 后向平滑）。

    z: (N, T)。返回平滑后的水平 L (N, T) 与其方差 P (N, T)。
    单条序列的递推：
        predict:  m⁻ = m,        P⁻ = P + q          # 因子→变量 消息
        update:   K  = P⁻/(P⁻+r), m = m⁻ + K(z−m⁻)   # 变量→因子 消息
                  P  = (1−K)P⁻
    这里把 r 固定为 1，q = qr，等价于调信噪比。
    """
    N, T = z.shape
    q, r = qr, 1.0
    m = np.zeros((N, T))
    P = np.zeros((N, T))
    mi = np.median(z[:, :8], axis=1)           # 初始水平：前 8 点的中位数（抗异常）
    Pi = np.full(N, np.var(z[:, :8], axis=1) + 1e-6)
    for t in range(T):
        if t == 0:
            mp, Pp = mi, Pi
        else:
            mp, Pp = m[:, t - 1], P[:, t - 1] + q
        K = Pp / (Pp + r)
        m[:, t] = mp + K * (z[:, t] - mp)
        P[:, t] = (1 - K) * Pp
    # RTS 平滑
    L = m.copy()
    Ps = P.copy()
    for t in range(T - 2, -1, -1):
        Pp = P[:, t] + q
        C = P[:, t] / Pp
        L[:, t] = m[:, t] + C * (L[:, t + 1] - m[:, t])
        Ps[:, t] = P[:, t] + C ** 2 * (Ps[:, t + 1] - Pp)
    return L, Ps


# ---------------------------------------------------------------- 主模型
def fit(Y: np.ndarray, period: int = PERIOD, iters: int = 5,
        qr: float = 0.15, ridge: float = 5.0, kappa: float = 10.0,
        verbose: bool = False):
    """交替消息传递：level ⇄ λ ⇄ S。

    每次迭代等价于在因子图上把三类消息各传一轮：
      1) 用当前 (λ, S) 解出水平路径  —— 观测因子 → level 变量
      2) 用当前 level 估计载荷 λ     —— 观测因子 → λ 变量（线性回归消息）
      3) 用当前 level, λ 汇总共享季节 —— 全部序列 → S 变量（"多用户共识"消息）
    """
    N, T = Y.shape
    # —— 每条序列先做稳健尺度标准化（p10–p90 跨度），使 λ 变成无量纲的 O(1) 量，跨序列可比
    p10 = np.percentile(Y, 10, axis=1)
    p90 = np.percentile(Y, 90, axis=1)
    scale = (p90 - p10) / 2.563 + 1e-9
    # 近常数序列的跨度会趋近 0，用均值量级兜底，避免标准化后爆掉
    scale = np.maximum(scale, 0.02 * np.abs(Y).mean(axis=1) + 1e-6)
    Y = Y / scale[:, None]
    phase = np.arange(T) % period
    fut_phase = (T + np.arange(H)) % period

    # —— 初始化：把各序列按中位数归一化后去水平，得到共享季节的初始消息
    s0 = np.median(Y, axis=1, keepdims=True) + 1e-9
    Zn = Y / s0
    trend = np.apply_along_axis(lambda r: np.convolve(r, np.ones(period) / period, "same"), 1, Zn)
    detr = Zn - trend
    S = np.zeros(period)
    for k in range(period):
        S[k] = detr[:, phase == k].mean()
    S -= S.mean()
    s_norm = S.std() + 1e-9
    S /= s_norm
    lam = np.ones(N)

    hist = []
    for it in range(iters):
        Z = Y - lam[:, None] * S[phase]
        L, P = kalman_level(Z, qr=qr)

        R = Y - L
        # λ：加权最小二乘（ridge 抑制 λ 爆炸），并裁剪到物理合理区间
        num = (R * S[phase]).sum(1)
        den = (S[phase] ** 2).sum() + ridge
        lam = np.clip(num / den, -3.0, 5.0)

        # S：跨序列汇总，按 λ² 加权（精度加权 = 方差倒数加权的高斯消息）
        num_s = np.zeros(period)
        den_s = np.zeros(period)
        for k in range(period):
            msk = phase == k
            if not msk.any():
                continue
            num_s[k] = (lam[:, None] * (Y[:, msk] - L[:, msk])).sum()
            den_s[k] = (lam ** 2).sum() * msk.sum() + 1e-9
        S = num_s / den_s
        S -= S.mean()
        s_norm = S.std() + 1e-9
        S /= s_norm
        lam = lam * s_norm                    # 归一化后把尺度还给 λ

        # 训练段一步预测误差（诊断用）
        pred1 = L + lam[:, None] * S[phase]
        hist.append(float(np.abs(pred1 - Y).sum() / np.abs(Y).sum()))
        if verbose:
            print(f"  iter {it + 1}: 拟合 WAPE {hist[-1] * 100:.2f}%", flush=True)

    # —— 分层收缩：每条序列自己的季节偏差，按样本数向全局共识收缩
    #    s_i,k = r_i,k · n_k/(n_k+κ)：n_k 大（样本多）就更信自己，κ 是共识的先验强度
    R = Y - L - lam[:, None] * S[phase]
    S_ind = np.zeros((N, period))
    cnt = np.array([(phase == k).sum() for k in range(period)])
    for k in range(period):
        msk = phase == k
        if not msk.any():
            continue
        w = cnt[k] / (cnt[k] + kappa)
        S_ind[:, k] = R[:, msk].mean(axis=1) * w
    S_ind -= S_ind.mean(axis=1, keepdims=True)      # 每条序列的季节偏差零均值

    return {"S": S, "lam": lam, "L": L, "P": P, "phase": phase,
            "fut_phase": fut_phase, "hist": hist, "qr": qr, "iters": iters,
            "S_ind": S_ind, "kappa": kappa, "scale": scale}


def forecast(model, h: int = H, damp: float = 1.0, drift_win: int = 26):
    """预测：水平外推（阻尼漂移）+ 共享季节载荷。

    level_{T+j} = level_T + d · Σ_{k=1..j} damp^k
    d 取训练末端 drift_win 周的稳健斜率（中位数差分），damp<1 表示趋势逐步衰减。
    """
    L = model["L"]
    S, lam = model["S"], model["lam"]
    level_T = L[:, -1]
    diffs = np.diff(L[:, -drift_win - 1:], axis=1)
    d = np.median(diffs, axis=1)                      # 稳健斜率，抗单点噪声
    out = np.zeros((L.shape[0], h))
    acc = 0.0
    for i in range(h):
        acc += damp ** (i + 1)
        out[:, i] = (level_T + d * acc + lam * S[model["fut_phase"][i]]
                     + model["S_ind"][:, model["fut_phase"][i]])
    return np.clip(out, 0, None) * model["scale"][:, None]


def wape_mean(pred, true):
    return float(np.mean(np.abs(pred - true).sum(1) / np.maximum(np.abs(true).sum(1), 1e-9)))


def wape_agg(pred, true):
    return float(np.abs(pred - true).sum() / np.abs(true).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=PERIOD)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--qr", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--val", action="store_true",
                    help="在训练段内部切 13 周做验证来选超参（推荐），再 refit 全训练段预测测试")
    ap.add_argument("--sweep", action="store_true", help="直接在测试集上扫（会过拟合，仅诊断用）")
    args = ap.parse_args()

    Y, T, ids = load(n=args.n or None)
    print(f"[data] {len(ids)} series × {Y.shape[1]} 周，预测 {H} 周", flush=True)

    t0 = time.time()
    chosen = {"qr": args.qr, "damp": 1.0, "ridge": args.ridge}

    if args.val:
        # ---- 只用训练段：前 T-H 周训练，最后 H 周验证
        Ytr, Yva = Y[:, :-H], Y[:, -H:]
        best = None
        for qr in (0.02, 0.05, 0.1, 0.2, 0.4):
            for ridge in (5.0, 20.0):
                for kappa in (2.0, 5.0, 10.0, 30.0):
                    m = fit(Ytr, args.period, args.iters, qr=qr, ridge=ridge, kappa=kappa)
                    for damp in (0.5, 0.7, 0.85, 1.0):
                        w = wape_mean(forecast(m, damp=damp), Yva)
                        if best is None or w < best[0]:
                            best = (w, qr, ridge, kappa, damp)
        wv, qr, ridge, kappa, damp = best
        chosen = {"qr": qr, "damp": damp, "ridge": ridge, "kappa": kappa}
        print(f"[val] 内部验证选出 qr={qr} ridge={ridge} kappa={kappa} damp={damp} "
              f"→ 验证 WAPE {wv * 100:.2f}%", flush=True)
        model = fit(Y, args.period, args.iters, qr=qr, ridge=ridge, kappa=kappa, verbose=True)
        pred = forecast(model, damp=damp)
        w = wape_mean(pred, T)
    elif args.sweep:
        print("[sweep] 直接在测试集扫参，仅用于诊断（结果偏乐观）")
        best = None
        for qr in (0.05, 0.15, 0.4):
            m = fit(Y, args.period, args.iters, qr=qr, ridge=args.ridge)
            for damp in (0.7, 0.85, 1.0):
                p = forecast(m, damp=damp)
                ww = wape_mean(p, T)
                print(f"  qr={qr:<5} damp={damp:<4} → WAPE {ww * 100:.2f}%", flush=True)
                if best is None or ww < best[0]:
                    best = (ww, qr, damp, m, p)
        w, qr, damp, model, pred = best
        chosen = {"qr": qr, "damp": damp, "ridge": args.ridge}
        print(f"[best] qr={qr} damp={damp} WAPE {w * 100:.2f}%")
    else:
        model = fit(Y, args.period, args.iters, qr=args.qr, ridge=args.ridge, verbose=True)
        pred = forecast(model, damp=chosen["damp"])
        w = wape_mean(pred, T)

    out = {
        "model": "FG-MP (factor graph + Gaussian message passing)",
        "n_series": len(ids),
        "horizon": H,
        "period": args.period,
        "iters": args.iters,
        "hyper": chosen,
        "wape_mean_perseries": w,
        "wape_aggregate": wape_agg(pred, T),
        "fit_runtime_s": round(time.time() - t0, 1),
        "fit_history": [round(x, 4) for x in model["hist"]],
        "lambda_stats": {
            "min": float(np.min(model["lam"])), "max": float(np.max(model["lam"])),
            "median": float(np.median(model["lam"])),
            "n_zero_load": int((np.abs(model["lam"]) < 1e-3).sum()),
        },
    }
    (ROOT / "repro" / "metrics_fg.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "fg_preds.npz",
                        preds=pred, ids=np.array(ids), S=model["S"], lam=model["lam"])
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
