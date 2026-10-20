"""Rossmann final run: generic feature families + L1/L2 GBM ensemble.

Uses the repository's leakage-audited Rossmann feature builder (history lags,
rolling stats, calendar, store attributes, known-future promo/holiday/open),
trains two global GBMs on log1p targets (L2 and L1), gates out dominated
components on the inner validation origin, and fits stage-wise simplex weights.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "alt"))
sys.path.insert(0, str(ROOT / "repro"))
import bench_rossmann as b
from fg_state import forecast as state_forecast

OUT = ROOT / "audit" / "results" / "ase"
H = 42
VAL, OUTER = 774, [816, 858, 900]
PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
              reg_lambda=1.0, n_jobs=6, verbose=-1)


def wape(p, y):
    return float(np.mean(np.abs(p - y).sum(1) / np.abs(y).sum(1)))


def fit(p, o, objective):
    X, y = b.build_training(p, o, True)
    X["store_id"] = X["store_id"].astype("category")
    m = lgb.LGBMRegressor(objective=objective, random_state=42, **PARAMS)
    m.fit(X, np.log1p(y), categorical_feature=["store_id"])
    hist = b.history_features(p.y[:, :o])
    out = np.empty((p.n, H))
    for h in range(1, H + 1):
        row = pd.DataFrame(b.build_matrix(p, o, h, True, hist=hist))
        row["store_id"] = pd.Categorical(row["store_id"], categories=X["store_id"].cat.categories)
        out[:, h - 1] = np.expm1(m.predict(row[X.columns]))
    out = np.where(p.open_[:, o:o + H] == 1, out, 0.0)
    return np.clip(out, 0, None), m, list(X.columns), X['store_id'].cat.categories


def stages(H):
    a, c = H // 3, 2 * H // 3
    return {"short": list(range(0, a)), "mid": list(range(a, c)), "long": list(range(c, H))}


def combine(preds, w):
    names = list(w)
    acc = sum(w[k] * np.log1p(np.clip(preds[k], 0, None)) for k in names)
    return np.clip(np.expm1(acc), 0, None)


def fit_weights(preds, truth, gate=1.5, shrink=1 / 3, step=0.05):
    ok = np.abs(truth).sum(1) > 0
    sc = {k: wape(preds[k][ok], truth[ok]) for k in preds}
    best = min(sc.values())
    keep = [k for k in preds if sc[k] <= gate * best]
    n = int(round(1 / step))
    best_w, best_s = None, np.inf
    def rec(rem, k, acc):
        nonlocal best_w, best_s
        if k == len(keep) - 1:
            w = np.array(acc + [rem / n])
            s = wape(combine({x: preds[x][ok] for x in keep}, dict(zip(keep, w))), truth[ok])
            if s < best_s:
                best_w, best_s = w, s
            return
        for i in range(rem + 1):
            rec(rem - i, k + 1, acc + [i / n])
    rec(n, 0, [])
    best_w = (1 - shrink) * best_w + shrink / len(keep)
    return {k: (float(best_w[keep.index(k)]) if k in keep else 0.0) for k in preds}, sc, keep


def main():
    t0 = time.time()
    p = b.Panel(b.PANEL)
    st = stages(H)
    val_pred = {}
    for name, obj in (("l2", "l2"), ("l1", "regression_l1")):
        pr, m, cols, cats = fit(p, VAL, obj)
        val_pred[name] = pr
        print(f"[val] {name} {wape(pr, p.y[:, VAL:VAL+H]):.4f}", flush=True)
    state_val = state_forecast(p.y[:, :VAL], h=H, period=7, harmonics=3, qr=0.02,
                               trend_q=0.0, damp=0.98)
    val_pred["state"] = state_val
    print(f"[val] state {wape(state_val, p.y[:, VAL:VAL+H]):.4f}", flush=True)

    weights = {}
    for s, idx in st.items():
        w, sc, keep = fit_weights({k: v[:, idx] for k, v in val_pred.items()},
                                  p.y[:, VAL:VAL+H][:, idx])
        weights[s] = w
        print(f"[weights] {s} {w} keep={keep} scores={ {k: round(v,4) for k,v in sc.items()} }", flush=True)

    preds, metrics = {}, {}
    for o in OUTER:
        for name, obj in (("l2", "l2"), ("l1", "regression_l1")):
            pr, m, cols, cats = fit(p, o, obj)
            preds[(o, name)] = pr
            print(f"[test] {o} {name} {wape(pr, p.y[:, o:o+H]):.4f}", flush=True)
        stp = state_forecast(p.y[:, :o], h=H, period=7, harmonics=3, qr=0.02,
                             trend_q=0.0, damp=0.98)
        preds[(o, "state")] = stp
        ens = np.zeros((p.n, H))
        for s, idx in st.items():
            ens[:, idx] = combine({k: preds[(o, k)][:, idx] for k in weights[s]}, weights[s])
        preds[(o, "ase")] = ens
        truth = p.y[:, o:o+H]
        metrics[str(o)] = {k: wape(preds[(o, k)], truth) for k in ("l2", "l1", "state", "ase")}
        print(f"[test] {o} ASE {metrics[str(o)]['ase']:.4f}", flush=True)

    # --- attribution: TreeSHAP of the dominant component (L1) at the last origin
    o = OUTER[-1]
    pr, m, cols, cats = fit(p, o, "regression_l1")
    hist = b.history_features(p.y[:, :o])
    group_of = {}
    for c in cols:
        if c in ("promo", "promo_rate28"):
            g = "促销"
        elif c in ("school", "state"):
            g = "假日/学校"
        elif c in ("dow", "day", "month", "woy", "is_weekend", "h"):
            g = "日历"
        elif c in ("open", "open_rate28"):
            g = "营业状态"
        elif c.startswith(("lag", "mean", "std", "max", "same_dow", "ratio")):
            g = "历史/水平"
        else:
            g = "门店属性"
        group_of[c] = g
    gnames = sorted(set(group_of.values()))
    contrib = {g: np.zeros((p.n, H)) for g in gnames}
    base_sum = 0.0
    for h in range(1, H + 1):
        row = pd.DataFrame(b.build_matrix(p, o, h, True, hist=hist))
        row["store_id"] = pd.Categorical(row["store_id"], categories=cats)
        raw = m.predict(row[cols], pred_contrib=True)
        phi = raw[:, :-1]
        base_sum += float(raw[:, -1].mean())
        for g in gnames:
            j = [i for i, c in enumerate(cols) if group_of[c] == g]
            contrib[g][:, h - 1] = phi[:, j].sum(1)
    total = sum(np.abs(v).mean() for v in contrib.values())
    attr = sorted([{"group": g, "share": float(np.abs(v).mean() / total),
                    "mean_abs_log": float(np.abs(v).mean()),
                    "mean_signed_log": float(v.mean())} for g, v in contrib.items()],
                  key=lambda d: -d["mean_abs_log"])
    np.savez_compressed(OUT / "rossmann_final_attrib.npz",
                        **{f"g_{g}": v for g, v in contrib.items()})
    (OUT / "rossmann_final_attrib.json").write_text(
        json.dumps({"origin": o, "groups": attr, "mean_base_log": base_sum / H,
                    "note": "log1p space; exp(sum) is a multiplicative factor on (1+y)"},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    out = {"dataset": "Rossmann", "H": H, "val_origin": VAL, "origins": OUTER,
           "weights": weights, "metrics": metrics, "epoch_s": time.time() - t0}
    (OUT / "rossmann_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    np.savez_compressed(OUT / "rossmann_final_preds.npz",
                        **{f"{o}_{k}": v for (o, k), v in preds.items()},
                        **{f"{o}_truth": p.y[:, o:o+H] for o in OUTER})
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
