"""Second-dataset benchmark: Rossmann Store Sales (daily, 400 stores, H=42).

Design (see audit/results/rossmann/protocol.json for the machine-readable copy)
------------------------------------------------------------------------------
* panel            : 400 stores x 942 days (2013-01-01 .. 2015-07-31), from
                     repro/alt/prep_rossmann.py; closed days kept as Sales=0.
* horizon          : H = 42 days (the Rossmann competition's 6-week test span).
* outer origins    : three, 42 days apart, the latest ending on the last day of
                     the data:  o = 816, 858, 900  (history = y[:, :o],
                     truth = y[:, o:o+42]).
* hyperparameters  : chosen ONLY on an inner validation origin o = 774
                     (truth 774..815), which is strictly earlier than every
                     outer test window.  The outer windows are scored once.
* metric           : per-series WAPE, averaged over series (primary);
                     aggregate WAPE = sum|y-yhat| / sum|y| (secondary).
                     A zero denominator raises instead of being dropped.
* clipping         : negative point forecasts are clipped to 0 (repo convention,
                     same as repro/common.evaluate); counts are recorded.

Models
------
SeasonalNaive7, MovingAvg, UCM(7) [statsmodels], StateFourier(7) [fg_state.py],
LightGBM-direct (known-future covariates), LightGBM-cal (calendar only, the
ablation), plus one diagnostic row SeasonalNaive7+Open.

Usage
-----
  python repro/alt/bench_rossmann.py                # full run
  python repro/alt/bench_rossmann.py --quick        # smoke test (80 series, 1 origin)
  python repro/alt/bench_rossmann.py --models lgbm  # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro"))

PANEL = ROOT / "data" / "alt" / "rossmann" / "panel.npz"
OUTDIR = ROOT / "audit" / "results" / "rossmann"
PREDDIR = OUTDIR / "preds"

H = 42
SEASON = 7
STEP = 21                      # stride of training origins for the global GBDT
MIN_HIST = 56
VAL_ORIGIN = 774               # inner validation: truth 774..815
OUTER_ORIGINS = (816, 858, 900)
N_JOBS = 4

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ data
class Panel:
    def __init__(self, path: Path, n_series: int | None = None):
        z = np.load(path, allow_pickle=False)
        self.y = z["y"].astype(np.float32)                 # (N, T)
        self.ids = [str(s) for s in z["ids"]]
        self.stores = z["stores"]
        self.days = z["dates"].astype("datetime64[D]")
        self.promo = z["promo"].astype(np.int8)
        self.open_ = z["open_"].astype(np.int8)
        self.school = z["school"].astype(np.int8)
        self.state = z["state"].astype(np.int8)
        self.store_type = z["store_type"].astype(np.int8)
        self.assortment = z["assortment"].astype(np.int8)
        self.promo2 = z["promo2"].astype(np.int8)
        self.comp_dist = z["comp_dist"].astype(np.float32)
        if n_series:
            self.y = self.y[:n_series]
            self.ids = self.ids[:n_series]
            for a in ("promo", "open_", "school", "state"):
                setattr(self, a, getattr(self, a)[:n_series])
            for a in ("store_type", "assortment", "promo2", "comp_dist", "stores"):
                setattr(self, a, getattr(self, a)[:n_series])
        self.n, self.T = self.y.shape
        self._cal = self._calendar()

    def _calendar(self) -> dict[str, np.ndarray]:
        d = self.days
        doy = (d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
        dow = ((d.astype(int) + 3) % 7).astype(np.int8)     # 0=Mon
        return {
            "dow": dow,
            "day": ((d - d.astype("datetime64[M]")).astype(int) + 1).astype(np.int8),
            "month": (d.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int8),
            "woy": (doy // 7 + 1).astype(np.int8),
            "is_weekend": (dow >= 5).astype(np.int8),
        }

    @property
    def cal(self):
        return self._cal


# ------------------------------------------------------------------ metrics
def series_wape(y: np.ndarray, p: np.ndarray, sid: str) -> float:
    den = float(np.abs(y).sum())
    if den <= 0:
        raise ValueError(f"{sid}: zero WAPE denominator -- refusing to drop the series")
    return float(np.abs(y - p).sum() / den)


def score(y: np.ndarray, p: np.ndarray, ids: list[str], hist: np.ndarray,
          season: int = SEASON) -> dict:
    """y, p: (N, H).  hist: (N, T) history used to scale MASE."""
    raw_negative = int((p < 0).sum())
    p = np.clip(p.astype(np.float64), 0.0, None)
    y = y.astype(np.float64)
    w, sm, ma = [], [], []
    for i, sid in enumerate(ids):
        w.append(series_wape(y[i], p[i], sid))
        den = np.abs(y[i]) + np.abs(p[i])
        sm.append(float(np.mean(np.where(den == 0, 0.0, 200 * np.abs(y[i] - p[i]) / np.maximum(den, 1e-9)))))
        h = hist[i]
        h = h[np.isfinite(h)]
        if len(h) > season:
            scale = float(np.mean(np.abs(h[season:] - h[:-season])))
        elif len(h) > 1:
            scale = float(np.mean(np.abs(np.diff(h))))
        else:
            scale = float("nan")
        ma.append(float(np.mean(np.abs(y[i] - p[i])) / scale) if scale and scale > 0 else float("nan"))
    tot = float(np.abs(y).sum())
    return {
        "wape_mean": float(np.mean(w)),
        "wape_median": float(np.median(w)),
        "wape_aggregate": float(np.abs(y - p).sum() / tot) if tot > 0 else float("nan"),
        "smape_mean": float(np.mean(sm)),
        "mase_mean": float(np.nanmean(ma)) if np.isfinite(ma).any() else float("nan"),
        "n_series": len(ids),
        "n_negative_clipped": raw_negative,
        "n_wape_zero_denominator": 0,
        "_per_series_wape": w,
    }


# ------------------------------------------------------------------ baselines
def seasonal_naive(yh: np.ndarray, h: int, season: int = SEASON) -> np.ndarray:
    if yh.shape[1] >= season:
        tail = yh[:, -season:]
        return np.tile(tail, (1, h // season + 1))[:, :h]
    return np.repeat(yh[:, -1:], h, axis=1)


def moving_average(yh: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.repeat(yh[:, -w:].mean(axis=1, keepdims=True), h, axis=1)


def ucm_fit_one(y: np.ndarray, h: int, stochastic: bool, season: int) -> np.ndarray | None:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    try:
        m = UnobservedComponents(np.asarray(y, float), level="local level",
                                 seasonal=season, stochastic_seasonal=stochastic)
        r = m.fit(method="lbfgs", maxiter=200, disp=False)
        p = np.asarray(r.forecast(h), float)
        return p if np.isfinite(p).all() else None
    except Exception:
        return None


def ucm_forecast(yh: np.ndarray, h: int, stochastic: bool, season: int = SEASON):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(ucm_fit_one)(yh[i], h, stochastic, season) for i in range(yh.shape[0]))
    p = np.empty((yh.shape[0], h), float)
    failed = 0
    fallback = seasonal_naive(yh, h, season)
    for i, r in enumerate(res):
        if r is None:
            p[i] = fallback[i]
            failed += 1
        else:
            p[i] = r
    return p, failed


def state_fourier(yh: np.ndarray, h: int, harmonics: int, qr: float,
                  damp: float, trend_q: float, period: int = SEASON) -> np.ndarray:
    from fg_state import forecast as fg_forecast
    return fg_forecast(yh.astype(np.float64), h=h, harmonics=harmonics, qr=qr,
                       trend_q=trend_q, damp=damp, period=period)


# ------------------------------------------------------------------ global GBDT
def history_features(yh: np.ndarray) -> dict[str, np.ndarray]:
    """Features computable at the forecast origin (no future target access)."""
    f = {}
    for lag in (1, 7, 14, 21, 28):
        f[f"lag{lag}"] = yh[:, -lag].astype(np.float32)
    for w in (7, 14, 28):
        seg = yh[:, -w:]
        f[f"mean{w}"] = seg.mean(axis=1).astype(np.float32)
        f[f"std{w}"] = seg.std(axis=1).astype(np.float32)
        f[f"max{w}"] = seg.max(axis=1).astype(np.float32)
    f["same_dow_4w"] = yh[:, -28:].reshape(yh.shape[0], 4, 7).mean(axis=1).mean(axis=1).astype(np.float32)
    f["ratio_7_28"] = (f["mean7"] / np.maximum(f["mean28"], 1.0)).astype(np.float32)
    return f


def _exog_for_target(p: Panel, t: np.ndarray | int, full: bool) -> dict[str, np.ndarray]:
    """Known-future covariates for target day index t (scalar or (N,) array)."""
    if not full:
        return {}
    if np.isscalar(t):
        return {
            "promo": np.full(p.n, p.promo[:, t], dtype=np.float32),
            "open": np.full(p.n, p.open_[:, t], dtype=np.float32),
            "school": np.full(p.n, p.school[:, t], dtype=np.float32),
            "state": np.full(p.n, p.state[:, t], dtype=np.float32),
        }
    idx = np.arange(p.n)
    return {
        "promo": p.promo[idx, t].astype(np.float32),
        "open": p.open_[idx, t].astype(np.float32),
        "school": p.school[idx, t].astype(np.float32),
        "state": p.state[idx, t].astype(np.float32),
    }


def build_matrix(p: Panel, o: int, h: int, full: bool, hist: dict | None = None):
    """Rows = series, for one (origin o, horizon h)."""
    yh = p.y[:, :o]
    hist = hist if hist is not None else history_features(yh)
    t = o + h - 1
    cols = {k: v for k, v in hist.items()}
    cols["h"] = np.full(p.n, h, dtype=np.float32)
    cols["dow"] = np.full(p.n, p.cal["dow"][t], dtype=np.float32)
    cols["day"] = np.full(p.n, p.cal["day"][t], dtype=np.float32)
    cols["month"] = np.full(p.n, p.cal["month"][t], dtype=np.float32)
    cols["woy"] = np.full(p.n, p.cal["woy"][t], dtype=np.float32)
    cols["is_weekend"] = np.full(p.n, p.cal["is_weekend"][t], dtype=np.float32)
    cols["store_id"] = p.stores.astype(np.float32)
    if full:
        for k, v in _exog_for_target(p, t, True).items():
            cols[k] = v
        cols["store_type"] = p.store_type.astype(np.float32)
        cols["assortment"] = p.assortment.astype(np.float32)
        cols["promo2"] = p.promo2.astype(np.float32)
        cols["comp_dist"] = np.log1p(p.comp_dist).astype(np.float32)
        # realised operating pattern in the recent history (known at origin)
        cols["open_rate28"] = p.open_[:, o - 28:o].mean(axis=1).astype(np.float32)
        cols["promo_rate28"] = p.promo[:, o - 28:o].mean(axis=1).astype(np.float32)
    return cols


def build_training(p: Panel, o: int, full: bool):
    import pandas as pd
    X, Y = [], []
    for ot in range(MIN_HIST, o - H + 1, STEP):
        hist = history_features(p.y[:, :ot])
        for h in range(1, min(H, o - ot) + 1):
            cols = build_matrix(p, ot, h, full, hist=hist)
            X.append(pd_frame(cols))
            Y.append(p.y[:, ot + h - 1])
    return pd.concat(X, ignore_index=True), np.concatenate(Y)


def pd_frame(cols: dict) -> "object":
    import pandas as pd
    return pd.DataFrame({k: v for k, v in cols.items()})


def lgbm_forecast(p: Panel, o: int, full: bool, params: dict) -> np.ndarray:
    import lightgbm as lgb
    import pandas as pd
    Xtr, ytr = build_training(p, o, full)
    Xtr["store_id"] = Xtr["store_id"].astype("category")
    model = lgb.LGBMRegressor(random_state=42, n_jobs=N_JOBS, verbose=-1, **params)
    model.fit(Xtr, ytr, categorical_feature=["store_id"])
    hist = history_features(p.y[:, :o])
    out = np.empty((p.n, H), float)
    for h in range(1, H + 1):
        cols = build_matrix(p, o, h, full, hist=hist)
        row = pd.DataFrame(cols)
        row["store_id"] = pd.Categorical(row["store_id"], categories=Xtr["store_id"].cat.categories)
        out[:, h - 1] = model.predict(row[Xtr.columns])
    return out


# ------------------------------------------------------------------ runner
MODEL_NAMES = ["SeasonalNaive7", "MovingAvg", "UCM7", "StateFourier7",
               "LGBM-direct", "LGBM-cal", "SeasonalNaive7+Open"]


def run_model(name: str, p: Panel, o: int, cfg: dict):
    t0 = time.time()
    yh = p.y[:, :o].astype(np.float64)
    extra = {}
    if name == "SeasonalNaive7":
        pr = seasonal_naive(p.y[:, :o], H)
    elif name == "SeasonalNaive7+Open":
        pr = seasonal_naive(p.y[:, :o], H)
        fut_open = p.open_[:, o:o + H]
        pr = pr * np.where(fut_open == 1, 1.0, 0.0)
    elif name == "MovingAvg":
        pr = moving_average(p.y[:, :o], H, cfg["window"])
    elif name == "UCM7":
        pr, failed = ucm_forecast(yh, H, cfg["stochastic"])
        extra["n_failed_fallback"] = failed
    elif name == "StateFourier7":
        pr = state_fourier(yh, H, cfg["harmonics"], cfg["qr"], cfg["damp"], cfg["trend_q"])
    elif name in ("LGBM-direct", "LGBM-cal"):
        pr = lgbm_forecast(p, o, full=(name == "LGBM-direct"), params=cfg["params"])
    else:
        raise ValueError(name)
    return pr, time.time() - t0, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--n-series", type=int, default=0)
    args = ap.parse_args()

    p = Panel(PANEL, n_series=(args.n_series or (80 if args.quick else None)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PREDDIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] {p.n} series x {p.T} days, zero share {(p.y == 0).mean():.3f}", flush=True)

    models = [m for m in MODEL_NAMES if not args.models or m in args.models.split(",")]
    if args.quick:
        models = [m for m in models if m != "UCM7"]
    origins = (VAL_ORIGIN,) if args.quick else (VAL_ORIGIN,) + OUTER_ORIGINS

    # ---------------- stage 1: hyperparameters, inner validation only
    grids = {
        "MovingAvg": [{"window": w} for w in (14, 28, 56)],
        "UCM7": [{"stochastic": s} for s in (True, False)],
        "StateFourier7": [{"harmonics": k, "qr": q, "damp": d, "trend_q": tq}
                          for k in (2, 3) for q in (0.02, 0.05) for d in (0.98,) for tq in (0.0,)],
        "LGBM-direct": [{"params": dict(n_estimators=n, learning_rate=0.05, num_leaves=63,
                                        min_child_samples=50, subsample=0.9,
                                        colsample_bytree=0.9, reg_lambda=1.0)}
                        for n in (300, 600)],
        "LGBM-cal": [{"params": dict(n_estimators=n, learning_rate=0.05, num_leaves=63,
                                     min_child_samples=50, subsample=0.9,
                                     colsample_bytree=0.9, reg_lambda=1.0)}
                     for n in (300, 600)],
        "SeasonalNaive7": [{}], "SeasonalNaive7+Open": [{}],
    }

    sel_rows, chosen = [], {}
    truth_val = p.y[:, VAL_ORIGIN:VAL_ORIGIN + H]
    for name in models:
        cands = grids[name][:1] if args.quick else grids[name]
        best, best_wape = None, np.inf
        for cfg in cands:
            pr, secs, extra = run_model(name, p, VAL_ORIGIN, cfg)
            m = score(truth_val, pr, p.ids, p.y[:, :VAL_ORIGIN])
            sel_rows.append({"model": name, "config": cfg, "wape_mean": m["wape_mean"],
                             "wape_aggregate": m["wape_aggregate"], "runtime_s": round(secs, 2),
                             **extra})
            print(f"[val ] {name:22s} {json.dumps(cfg)[:60]:60s} "
                  f"wape={m['wape_mean']:.4f} ({secs:.1f}s)", flush=True)
            if m["wape_mean"] < best_wape:
                best, best_wape = cfg, m["wape_mean"]
        chosen[name] = {"config": best, "val_wape_mean": best_wape}
        print(f"[pick] {name:22s} -> {json.dumps(best)[:70]}  val={best_wape:.4f}", flush=True)

    if args.quick:
        (OUTDIR / "smoke.json").write_text(json.dumps(
            {"selection": sel_rows, "chosen": chosen}, indent=2), encoding="utf-8")
        print("[quick] done")
        return

    # ---------------- stage 2: outer rolling origins, scored once each
    results = {"origins": {}}
    for o in OUTER_ORIGINS:
        key = f"o{o}"
        truth = p.y[:, o:o + H]
        dates = [str(d) for d in p.days[o:o + H]]
        results["origins"][key] = {
            "origin": o, "train_end": str(p.days[o - 1]),
            "test_start": dates[0], "test_end": dates[-1], "models": {}, "pred_files": {},
        }
        np.savez_compressed(PREDDIR / f"truth_{key}.npz",
                            truth=truth, preds=truth, ids=np.array(p.ids),
                            dates=np.array(dates), stores=p.stores)
        for name in models:
            pr, secs, extra = run_model(name, p, o, chosen[name]["config"])
            m = score(truth, pr, p.ids, p.y[:, :o])
            per = m.pop("_per_series_wape")
            fname = f"preds_{name.replace('+', 'plus')}_{key}.npz"
            np.savez_compressed(PREDDIR / fname,
                                preds=np.clip(pr, 0, None).astype(np.float32),
                                ids=np.array(p.ids), dates=np.array(dates),
                                stores=p.stores, truth=truth.astype(np.float32),
                                per_series_wape=np.array(per, dtype=np.float64))
            m.update({"runtime_s": round(secs, 2), **extra})
            results["origins"][key]["models"][name] = m
            results["origins"][key]["pred_files"][name] = str(
                (PREDDIR / fname).relative_to(ROOT))
            print(f"[test] {key} {name:22s} wape_mean={m['wape_mean']:.4f} "
                  f"agg={m['wape_aggregate']:.4f} ({secs:.1f}s)", flush=True)

    # ---------------- aggregate over origins
    summary = {}
    for name in models:
        wm = [results["origins"][f"o{o}"]["models"][name]["wape_mean"] for o in OUTER_ORIGINS]
        wa = [results["origins"][f"o{o}"]["models"][name]["wape_aggregate"] for o in OUTER_ORIGINS]
        rt = [results["origins"][f"o{o}"]["models"][name]["runtime_s"] for o in OUTER_ORIGINS]
        summary[name] = {
            "wape_mean": float(np.mean(wm)), "wape_mean_by_origin": [round(v, 6) for v in wm],
            "wape_aggregate": float(np.mean(wa)),
            "wape_aggregate_by_origin": [round(v, 6) for v in wa],
            "runtime_s_mean": float(np.mean(rt)),
        }
    results["summary"] = summary
    results["selection"] = {"origin": VAL_ORIGIN, "rows": sel_rows, "chosen": chosen,
                            "val_window": [str(p.days[VAL_ORIGIN]),
                                           str(p.days[VAL_ORIGIN + H - 1])]}
    results["protocol"] = {
        "dataset": "Rossmann Store Sales", "freq": "D", "horizon": H, "season": SEASON,
        "n_series": p.n, "outer_origins": list(OUTER_ORIGINS),
        "val_origin": VAL_ORIGIN,
        "sampling": "400 of 934 complete stores, numpy default_rng(42), sorted",
        "metric_primary": "mean over series of sum|y-yhat|/sum|y| (per-series WAPE)",
        "metric_secondary": "sum|y-yhat|/sum|y| over all series and steps",
        "zero_denominator": "raises ValueError, never dropped",
        "negative_forecasts": "clipped to 0, count recorded per model/origin",
        "covariates": "promo/open/school/state/store meta treated as known at forecast time",
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("[ok] metrics ->", OUTDIR / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
