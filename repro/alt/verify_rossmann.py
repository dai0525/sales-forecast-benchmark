"""Independent re-computation of the Rossmann leaderboard from the saved npz files.

Nothing here imports bench_rossmann: the point is to recompute the headline
numbers from `preds` + `ids` + `truth` alone (the same discipline the repo uses
in repro/audit_saved.py) and to fail loudly if they disagree with metrics.json.

Also runs a paired bootstrap over series for the two comparisons that matter:
  StateFourier7 vs UCM7   (does the univariate state model ranking hold?)
  LGBM-direct  vs LGBM-cal (how much do the exogenous covariates buy?)

Output: audit/results/rossmann/verification.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "audit" / "results" / "rossmann"
PRED = RES / "preds"
ORIGINS = ("o816", "o858", "o900")
TOL = 1e-6


def wape(y, p):
    d = float(np.abs(y).sum())
    if d <= 0:
        raise ValueError("zero denominator")
    return float(np.abs(y - p).sum() / d)


def load(key):
    t = np.load(PRED / f"truth_{key}.npz", allow_pickle=False)
    return t["truth"], [str(s) for s in t["ids"]]


def main() -> None:
    metrics = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
    files = sorted(p.name for p in PRED.glob("preds_*_o*.npz"))
    models = sorted({f[len("preds_"):].rsplit("_o", 1)[0] for f in files})
    # file names cannot carry "+"; map back to the model names used in metrics.json
    name_of = {m: (m if m in metrics["summary"] else m.replace("plus", "+")) for m in models}
    models = sorted(name_of.values())
    file_stem = {v: k for k, v in name_of.items()}
    orig_by_name = {m: {k: f"preds_{file_stem[m]}_{k}.npz" for k in ORIGINS} for m in models}

    out = {"n_files": len(files) + len(ORIGINS), "models": models, "checks": [],
           "recomputed": {}, "bootstrap": {}}
    per_series = {}
    for m in models:
        wm, wa = [], []
        for key in ORIGINS:
            truth, ids = load(key)
            z = np.load(PRED / orig_by_name[m][key], allow_pickle=False)
            if [str(s) for s in z["ids"]] != ids:
                raise SystemExit(f"{m}/{key}: id mismatch")
            p = np.clip(z["preds"].astype(float), 0, None)
            if p.shape != truth.shape:
                raise SystemExit(f"{m}/{key}: shape {p.shape} vs {truth.shape}")
            if not np.isfinite(p).all():
                raise SystemExit(f"{m}/{key}: non-finite predictions")
            w = [wape(truth[i], p[i]) for i in range(len(ids))]
            wm.append(float(np.mean(w)))
            wa.append(wape(truth, p))
            per_series[(m, key)] = np.array(w)
            ref = metrics["origins"][key]["models"][m]
            for got, want, tag in ((wm[-1], ref["wape_mean"], "wape_mean"),
                                   (wa[-1], ref["wape_aggregate"], "wape_aggregate")):
                if abs(got - want) > TOL:
                    raise SystemExit(f"{m}/{key}: {tag} {got} != stored {want}")
            out["checks"].append({"model": m, "origin": key,
                                  "wape_mean": round(wm[-1], 6),
                                  "wape_aggregate": round(wa[-1], 6), "match": True})
        out["recomputed"][m] = {"wape_mean": float(np.mean(wm)),
                                "wape_aggregate": float(np.mean(wa)),
                                "by_origin": [round(v, 6) for v in wm]}

    # sanity: stored summary agrees with the recomputation
    for m in models:
        s = metrics["summary"][m]
        if abs(s["wape_mean"] - out["recomputed"][m]["wape_mean"]) > TOL:
            raise SystemExit(f"summary mismatch for {m}")

    # ---- paired bootstrap over series (mean WAPE difference), 3 origins pooled
    rng = np.random.default_rng(7)
    for a, b in (("StateFourier7", "UCM7"), ("LGBM-direct", "LGBM-cal"),
                 ("LGBM-direct", "StateFourier7")):
        diffs = np.concatenate([per_series[(a, k)] - per_series[(b, k)] for k in ORIGINS])
        n = len(diffs)
        boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["bootstrap"][f"{a} - {b}"] = {
            "mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
            "n_series_origins": int(n),
            "share_series_a_better": float((diffs < 0).mean()),
        }

    (RES / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"recomputed": {k: round(v["wape_mean"], 4) for k, v in
                                     out["recomputed"].items()},
                      "bootstrap": out["bootstrap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
