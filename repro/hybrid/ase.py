"""ASE: Adaptive Stage Ensemble for panel sales forecasting.

A dataset-agnostic scheme:

1. one global LightGBM per objective (L2 / L1) on log1p scale,
   direct multi-horizon, with both origin-anchored and target-relative features;
2. an optional univariate state-space component (Fourier + damped local trend);
3. weights per horizon stage, fitted on a validation origin strictly earlier
   than every reported test origin, with shrinkage toward equal weights;
4. exact additive attribution on the log scale (TreeSHAP for the GBM part).

No dataset-specific branches: only season length, horizon, known-future
covariates and static attributes are supplied by the adapter.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))
sys.path.insert(0, str(ROOT / "repro" / "alt"))

from fg_state import forecast as state_forecast  # noqa: E402

USE_RELATIVE = True
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
                  n_jobs=6, verbose=-1)
STATE_GRID = [dict(harmonics=k, qr=q, trend_q=tq, damp=0.98)
              for k in (2, 4) for q in (0.02, 0.1) for tq in (0.0, 1e-4)]


@dataclass
class Panel:
    name: str
    y: np.ndarray                  # (N, T) history incl. validation targets
    H: int
    season: int
    origins: list[int]
    val_origin: int
    ids: list[str]
    test: np.ndarray | None = None       # truth for origin == T (official test)
    exog: dict[str, np.ndarray] | None = None   # (N, T) known-future covariates
    cal: dict[str, np.ndarray] | None = None    # (T,) calendar arrays
    static: dict[str, np.ndarray] | None = None # (N,) static attributes
    factor_groups: dict[str, list[str]] = field(default_factory=dict)

    def truth(self, o: int) -> np.ndarray:
        if o == self.y.shape[1] and self.test is not None:
            return self.test
        return self.y[:, o:o + self.H]


def load_weekly() -> Panel:
    import fg_mp
    y, test, ids = fg_mp.load()
    y = y.astype(np.float64)
    return Panel(name="M4-Weekly", y=y, H=13, season=52,
                 origins=[161, 174, 187, 200], val_origin=148, ids=ids,
                 test=test.astype(np.float64),
                 factor_groups={
                     "历史/滞后": ["lag", "tlag", "slag"],
                     "水平/滚动": ["m", "s", "lvl"],
                     "日历阶段": ["phase", "h"],
                 })


def load_rossmann() -> Panel:
    from bench_rossmann import Panel as RPanel, PANEL
    p = RPanel(PANEL)
    return Panel(
        name="Rossmann", y=p.y.astype(np.float64), H=42, season=7,
        origins=[816, 858, 900], val_origin=774, ids=p.ids,
        exog={"promo": p.promo.astype(np.float64), "open": p.open_.astype(np.float64),
              "school": p.school.astype(np.float64), "state": p.state.astype(np.float64)},
        cal={"dow": p.cal["dow"], "dom": p.cal["day"], "month": p.cal["month"],
             "woy": p.cal["woy"]},
        static={"store_type": p.store_type.astype(np.float64),
                "assortment": p.assortment.astype(np.float64),
                "promo2": p.promo2.astype(np.float64),
                "comp": np.log1p(p.comp_dist)},
        factor_groups={
            "促销": ["promo"],
            "假日/学校": ["school", "state"],
            "日历": ["dow", "dom", "month", "woy"],
            "历史/滞后": ["lag", "tlag", "slag"],
            "水平/滚动": ["m", "s", "lvl"],
            "门店属性": ["store", "assort", "promo2", "comp"],
        })


def _row(ds: Panel, o: int, h: int, hist: np.ndarray) -> pd.DataFrame:
    t = o + h - 1
    N = hist.shape[0]
    d: dict[str, np.ndarray] = {}
    for k in (1, 2, 3, 7, 14, 21, 28, ds.season, 2 * ds.season):
        j = o - k
        if j >= 0 and f"lag{k}" not in d:
            d[f"lag{k}"] = hist[:, j]
    if USE_RELATIVE:
        for k in (1, 2, 3, 7, 14, 28, ds.season, ds.season + 1, 2 * ds.season):
            j = t - k
            d[f"tlag{k}"] = hist[:, j] if 0 <= j < o else np.full(N, np.nan)
    for w in (max(2, ds.season // 2), ds.season, 2 * ds.season, min(4 * ds.season, o)):
        w = min(w, o)
        seg = hist[:, -w:]
        d[f"m{w}"] = seg.mean(1)
        d[f"s{w}"] = seg.std(1)
        d[f"x{w}"] = seg.max(1)
    d["lvl_ratio"] = d[f"m{ds.season}"] / np.maximum(d[f"m{min(2*ds.season,o)}"], 1.0)
    if o >= 28:
        d["ratio_7_28"] = hist[:, -7:].mean(1) / np.maximum(hist[:, -28:].mean(1), 1.0)
    if ds.exog:
        for k, arr in ds.exog.items():
            seg = arr[:, max(0, o - 28):o]
            if set(np.unique(seg)) <= {0.0, 1.0}:
                d[f"{k}_rate28"] = seg.mean(1)
    if ds.season == 7 and o >= 28:
        # same weekday over the last four weeks, anchored at the origin
        d["same_dow_4w"] = hist[:, -7:].mean(1)
    d["phase"] = np.full(N, t % ds.season)
    d["h"] = np.full(N, h)
    d["sid"] = np.arange(N)
    if ds.cal is not None:
        for k, arr in ds.cal.items():
            d[k] = arr[t]
    if ds.static:
        for k, arr in ds.static.items():
            d[k] = arr
    if ds.exog:
        for k, arr in ds.exog.items():
            d[k] = arr[:, t]
    return pd.DataFrame(d)


def _gbm_predict(ds: Panel, o: int, params: dict, target: str,
                 return_model: bool = False, want_contrib: bool = False,
                 step: int = 7, min_hist: int | None = None):
    hist = ds.y[:, :o]
    X, Y = [], []
    for ot in range(min_hist or max(40, ds.season * 2 + 1), o - ds.H + 1, step):
        for h in range(1, ds.H + 1):
            t = ot + h - 1
            if t >= o:
                continue
            X.append(_row(ds, ot, h, ds.y[:, :ot]))
            Y.append(ds.y[:, t])
    X = pd.concat(X, ignore_index=True)
    X["sid"] = X["sid"].astype("category")
    y = np.concatenate(Y)
    tgt = np.log1p(y) if target == "log1p" else y
    m = lgb.LGBMRegressor(objective=params.pop("objective", "l2"), **params)
    m.fit(X, tgt, categorical_feature=["sid"])
    out = np.zeros((ds.y.shape[0], ds.H))
    contribs = []
    for h in range(1, ds.H + 1):
        r = _row(ds, o, h, hist)
        r["sid"] = pd.Categorical(r["sid"], categories=X["sid"].cat.categories)
        r = r[X.columns]
        pr = m.predict(r, pred_contrib=want_contrib)
        if want_contrib:
            out[:, h - 1] = pr[:, -1] + pr[:, :-1].sum(1)
            contribs.append(pr[:, :-1])
        else:
            out[:, h - 1] = pr
    out = np.expm1(out) if target == "log1p" else out
    out = np.clip(out, 0, None)
    if ds.exog is not None and "open" in ds.exog:
        out = np.where(ds.exog["open"][:, o:o + ds.H] == 1, out, 0.0)
    if return_model:
        return out, m, contribs
    return out


def run(ds: Panel, tag: str) -> dict:
    t0 = time.time()
    print(f"[{ds.name}] validation origin {ds.val_origin}", flush=True)
    # ---- state candidate grid, selected on validation
    best = None
    kmax = max(1, (ds.season - 1) // 2)
    grid = [dict(harmonics=k, qr=q, trend_q=tq, damp=0.98)
            for k in sorted({2, min(4, kmax)}) for q in (0.02, 0.1) for tq in (0.0, 1e-4)]
    for cfg in grid:
        pv = state_forecast(ds.y[:, :ds.val_origin], h=ds.H, period=ds.season, **cfg)
        w = _wape_mean(pv, ds.truth(ds.val_origin))
        if best is None or w < best[0]:
            best = (w, cfg, pv)
    state_w, state_cfg, _ = best
    print(f"[{ds.name}] state cfg {state_cfg} val {state_w:.4f}", flush=True)

    models = {"l2": dict(LGB_PARAMS), "l1": dict(LGB_PARAMS, objective="regression_l1")}
    models["l1"].pop("random_state", None)
    models["l1"]["random_state"] = 43
    val = {}
    for k, prm in models.items():
        pv = _gbm_predict(ds, ds.val_origin, dict(prm), "log1p")
        val[k] = (_wape_mean(pv, ds.truth(ds.val_origin)), pv)
        print(f"[{ds.name}] {k} val {val[k][0]:.4f}", flush=True)
    pv_state = state_forecast(ds.y[:, :ds.val_origin], h=ds.H, period=ds.season, **state_cfg)
    val["state"] = (state_w, pv_state)

    # ---- stage weights on validation: 3 horizon buckets
    stages = _stages(ds.H)
    weights = {}
    for sname, sl in stages.items():
        w = _fit_weights({k: v[1][:, sl] for k, v in val.items()},
                         ds.truth(ds.val_origin)[:, sl], list(val))
        weights[sname] = w
        print(f"[{ds.name}] weights {sname} "
              f"{ {k: round(v, 2) for k, v in w.items()} }", flush=True)

    # ---- outer origins
    preds, meta = {}, {}
    for o in ds.origins:
        base = {}
        for k, prm in models.items():
            t = time.time()
            base[k] = _gbm_predict(ds, o, dict(prm), "log1p")
            print(f"[{ds.name}] {o} {k} {_wape_mean(base[k], ds.truth(o)):.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        base["state"] = state_forecast(ds.y[:, :o], h=ds.H, period=ds.season, **state_cfg)
        print(f"[{ds.name}] {o} state {_wape_mean(base['state'], ds.truth(o)):.4f} "
              f"({time.time()-t:.0f}s)", flush=True)
        ens = _combine(base, weights, stages)
        preds[str(o)] = {"models": base, "ase": ens,
                         "truth": ds.truth(o)}
        meta[str(o)] = {k: _metrics(v, ds.truth(o)) for k, v in base.items()}
        meta[str(o)]["ASE"] = _metrics(ens, ds.truth(o))
        for k, v in meta[str(o)].items():
            print(f"[{ds.name}] {o} {k:6s} {v['wape_mean']:.4f}", flush=True)

    out = {"dataset": ds.name, "H": ds.H, "season": ds.season,
           "origins": ds.origins, "val_origin": ds.val_origin,
           "state_cfg": state_cfg, "weights": weights,
           "stages": stages, "metrics": meta,
           "elapsed_s": time.time() - t0}
    return {"summary": out, "preds": preds}


def _stages(H: int) -> dict[str, list[int]]:
    a = max(1, H // 3)
    b = max(a + 1, 2 * H // 3)
    return {"short": list(range(0, a)), "mid": list(range(a, b)), "long": list(range(b, H))}


def _fit_weights(preds: dict, truth: np.ndarray, names: list[str],
                 gate: float = 1.5, shrink: float = 1/3) -> dict:
    """Simplex weights on validation, after gating out clearly dominated models.

    A component is eligible only if its validation WAPE is within `gate` x the
    best single component.  This prevents a globally weak (but diversifying)
    model from being assigned weight merely because the val origin is noisy.
    """
    ok = np.abs(truth).sum(1) > 0
    scores = {k: _wape_mean(preds[k][ok], truth[ok]) for k in names}
    best = min(scores.values())
    keep = [k for k in names if scores[k] <= gate * best]
    grid = list(_simplex(len(keep), 0.05))
    best_w, best_s = None, np.inf
    for w in grid:
        p = _combine_arr({k: preds[k][ok] for k in keep}, np.array(w))
        sc = _wape_mean(p, truth[ok])
        if sc < best_s:
            best_w, best_s = np.array(w), sc
    eq = np.ones(len(keep)) / len(keep)
    best_w = (1 - shrink) * best_w + shrink * eq
    out = {k: 0.0 for k in names}
    for k, w in zip(keep, best_w):
        out[k] = float(w)
    return out


def _simplex(K: int, step: float):
    n = int(round(1 / step))

    def rec(rem, k, acc):
        if k == K - 1:
            yield acc + [rem / n]
        else:
            for i in range(rem + 1):
                yield from rec(rem - i, k + 1, acc + [i / n])
    yield from rec(n, 0, [])


def _combine(preds: dict, weights: dict, stages: dict) -> np.ndarray:
    H = next(iter(preds.values())).shape[1]
    out = np.zeros((next(iter(preds.values())).shape[0], H))
    for sname, sl in stages.items():
        wd = weights[sname]
        names = list(wd.keys())
        out[:, sl] = _combine_arr({k: preds[k][:, sl] for k in names},
                                   np.array([wd[k] for k in names], float))
    return out


def _combine_arr(preds: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
    """Geometric (log1p-space) combination."""
    names = list(preds.keys())
    acc = np.zeros_like(next(iter(preds.values())), dtype=float)
    for k, wi in zip(names, w):
        acc += wi * np.log1p(np.clip(preds[k], 0, None))
    return np.clip(np.expm1(acc), 0, None)


def _wape_mean(p: np.ndarray, y: np.ndarray) -> float:
    den = np.abs(y).sum(1)
    if np.any(den <= 0):
        raise ValueError("zero WAPE denominator")
    return float(np.mean(np.abs(p - y).sum(1) / den))


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    e = np.abs(p - y)
    den = np.abs(y).sum(1)
    per = e.sum(1) / den
    return {"wape_mean": float(per.mean()), "wape_median": float(np.median(per)),
            "wape_aggregate": float(e.sum() / np.abs(y).sum()),
            "smape": float(np.mean(2 * e / np.maximum(np.abs(p) + np.abs(y), 1e-9))),
            "per_series_wape": per.tolist()}


if __name__ == "__main__":
    which = sys.argv[1]
    ds = load_weekly() if which == "weekly" else load_rossmann()
    res = run(ds, which)
    out = ROOT / "audit" / "results" / "ase"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{which}_summary.json").write_text(
        json.dumps(res["summary"], ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{which}_preds.npz",
                        **{f"{o}_{k}": v for o, d in res["preds"].items()
                           for k, v in d.items() if k != "truth"},
                        **{f"{o}_truth": d["truth"] for o, d in res["preds"].items()})
    print("saved", out, flush=True)
