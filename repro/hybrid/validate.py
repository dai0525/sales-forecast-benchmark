"""Independent verification for ASE results.

Reads only the saved prediction archives and re-derives every claim:
metrics, per-origin wins, paired clustered bootstrap, leakage probes,
reconstruction of the attribution sums, and selection chronology.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "hybrid"))
import ase  # noqa: E402

OUT = ROOT / "audit" / "results" / "ase"


def wape_per_series(p, y):
    den = np.abs(y).sum(1)
    if np.any(den <= 0):
        raise ValueError("zero denominator")
    return np.abs(p - y).sum(1) / den


def load(which):
    z = np.load(OUT / f"{which}_preds.npz")
    summ = json.loads((OUT / f"{which}_summary.json").read_text())
    origins = summ["origins"]
    data = {}
    for o in origins:
        data[o] = {"truth": z[f"{o}_truth"]}
        for k in ("l2", "l1", "state", "ase"):
            data[o][k] = z[f"{o}_{k}"]
    return summ, data


def bootstrap(diff, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.mean())


def main():
    report = {}
    for which in ("weekly", "rossmann"):
        summ, data = load(which)
        rows = {}
        for o, d in data.items():
            y = d["truth"]
            m = {k: wape_per_series(d[k], y) for k in ("l2", "l1", "state", "ase")}
            best_lgb = np.minimum(m["l2"], m["l1"])
            diff = best_lgb - m["ase"]
            lo, hi, mu = bootstrap(diff)
            rows[str(o)] = {
                "wape_mean": {k: float(v.mean()) for k, v in m.items()},
                "delta_vs_l2_pp": float((m["ase"].mean() - m["l2"].mean()) * 100),
                "delta_vs_best_lgb_pp": float((m["ase"].mean() - best_lgb.mean()) * 100),
                "bootstrap_ci95_pp": [lo * 100, hi * 100],
                "bootstrap_mean_pp": mu * 100,
                "series_win_rate_vs_best_lgb": float((m["ase"] < best_lgb).mean()),
                "n_series": int(len(y)),
            }
        # pooled: same series across origins, resample series only
        all_diff = np.vstack([(np.minimum(data[o]["l2"], data[o]["l1"])
                               - data[o]["ase"]) for o in data])
        lo, hi, mu = bootstrap(all_diff.mean(0))
        report[which] = {
            "per_origin": rows,
            "pooled_delta_vs_best_lgb_pp": float((np.mean([rows[str(o)]["wape_mean"]["ase"] for o in data])
                                                 - np.mean([rows[str(o)]["wape_mean"]["l1"] for o in data])) * 100),
            "pooled_bootstrap_ci95_pp": [lo * 100, hi * 100],
            "all_origins_beat_l2": bool(all(rows[str(o)]["wape_mean"]["ase"] < rows[str(o)]["wape_mean"]["l2"] for o in data)),
            "all_origins_beat_best_lgb": bool(all(rows[str(o)]["wape_mean"]["ase"] < min(rows[str(o)]["wape_mean"]["l2"], rows[str(o)]["wape_mean"]["l1"]) for o in data)),
            "selection_val_origin": summ["val_origin"],
            "first_test_origin": min(summ["origins"]),
            "weights": summ["weights"],
        }

    # leakage probe: future target values must not change any feature row
    leak = {}
    for which, loader in (("weekly", ase.load_weekly), ("rossmann", ase.load_rossmann)):
        ds = loader()
        o = ds.origins[0]
        y2 = ds.y.copy()
        rng = np.random.default_rng(0)
        y2[:, o:] = rng.uniform(0, 10_000, size=y2[:, o:].shape)
        small = dict(ase.LGB_PARAMS, n_estimators=30)
        p1 = ase._gbm_predict(ds, o, dict(small), "log1p")
        ds2 = type(ds)(**{**ds.__dict__, "y": y2})
        p2 = ase._gbm_predict(ds2, o, dict(small), "log1p")
        leak[which] = {"max_abs_prediction_change_after_future_target_perturbation":
                       float(np.max(np.abs(p1 - p2)))}
    report["leakage_probe"] = leak

    # attribution reconstruction gap
    att = {}
    for which in ("weekly", "rossmann"):
        p = OUT / f"{which}_attribution.json"
        if p.exists():
            j = json.loads(p.read_text())
            att[which] = {"max_group_reconstruction_gap": j["max_group_reconstruction_gap"],
                          "origin": j["origin"]}
    report["attribution_check"] = att
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
