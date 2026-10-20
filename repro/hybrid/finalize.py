"""Consolidate final numbers, bootstrap, attribution and leakage checks."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "hybrid"))
sys.path.insert(0, str(ROOT / "repro" / "alt"))
sys.path.insert(0, str(ROOT / "repro"))
import ase
import bench_rossmann as b

OUT = ROOT / "audit" / "results" / "ase"
REPO = {"weekly_official": 0.07960020423344538,
        "rossmann": {816: 0.1058, 858: 0.0882, 900: 0.0863}}


def wape(p, y):
    return np.abs(p - y).sum(1) / np.abs(y).sum(1)


def boot(diff, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    m = diff[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], float(m.mean())


def main():
    rep = {"note": "WAPE mean over series; delta = single LGBM minus ASE, positive is better for ASE."}

    # ---------------- Weekly
    z = np.load(OUT / "weekly_preds.npz", allow_pickle=True)
    ws = json.loads((OUT / "weekly_summary.json").read_text())
    wr = {}
    for o in ws["origins"]:
        y = z[f"{o}_truth"]
        pred = dict(z[f"{o}_models"].item())
        pred["ase"] = z[f"{o}_ase"]
        per = {k: wape(v, y) for k, v in pred.items()}
        base_name = "l2"          # pre-registered on validation origin 148
        best_lgb = per[base_name]
        ci, mu = boot(best_lgb - per["ase"])
        wr[str(o)] = {
            "wape": {k: float(v.mean()) for k, v in per.items()},
            "delta_vs_best_lgb_pp": float((best_lgb.mean() - per["ase"].mean()) * 100),
            "delta_vs_l2_pp": float((per["l2"].mean() - per["ase"].mean()) * 100),
            "ci95_pp": [c * 100 for c in ci],
            "win_rate_series": float((per["ase"] < best_lgb).mean()),
            "n": int(len(y)), "comparator": base_name,
        }
    rep["weekly"] = {"val_origin": ws["val_origin"], "weights": ws["weights"],
                     "origins": wr, "repo_lgbm_official": REPO["weekly_official"],
                     "n_series": 359}

    # ---------------- Rossmann
    rf = json.loads((OUT / "rossmann_final.json").read_text())
    zz = np.load(OUT / "rossmann_final_preds.npz")
    rr = {}
    for o in rf["origins"]:
        y = zz[f"{o}_truth"]
        pred = {k: zz[f"{o}_{k}"] for k in ("l2", "l1", "state", "ase")}
        per = {k: wape(v, y) for k, v in pred.items()}
        base_name = "l1"          # pre-registered on validation origin 774
        best_lgb = per[base_name]
        ci, mu = boot(best_lgb - per["ase"])
        rr[str(o)] = {
            "wape": {k: float(v.mean()) for k, v in per.items()},
            "delta_vs_best_lgb_pp": float((best_lgb.mean() - per["ase"].mean()) * 100),
            "delta_vs_repo_pp": float((REPO["rossmann"][o] - per["ase"].mean()) * 100),
            "ci95_pp": [c * 100 for c in ci],
            "win_rate_series": float((per["ase"] < best_lgb).mean()),
            "n": int(len(y)), "comparator": base_name,
        }
    rep["rossmann"] = {"val_origin": rf["val_origin"], "weights": rf["weights"],
                       "origins": rr, "n_series": 400,
                       "repo": REPO["rossmann"]}

    # ---------------- attribution, open days only
    p = b.Panel(b.PANEL)
    a = np.load(OUT / "rossmann_final_attrib.npz")
    o = rf["origins"][-1]
    mask = p.open_[:, o:o + 42] == 1
    rows = []
    total = 0.0
    for k in a.files:
        v = a[k]
        m = float(np.abs(v[mask]).mean())
        rows.append((k[2:], m, float(v[mask].mean())))
        total += m
    attr = sorted([{"group": g, "share": m / total, "mean_abs_log": m,
                    "mean_signed_log": s} for g, m, s in rows], key=lambda d: -d["mean_abs_log"])
    non_struct = [x for x in attr if x["group"] != "营业状态"]
    tot_ns = sum(x["mean_abs_log"] for x in non_struct)
    for x in non_struct:
        x["share_excl_structure"] = x["mean_abs_log"] / tot_ns
    rep["attribution"] = {"origin": o, "open_days_only": True,
                          "masked_days": int((~mask).sum()), "groups": attr,
                          "business_groups": non_struct,
                          "note": "营业状态 is a structural baseline shift (open vs closed); business shares exclude it."}
    # promo split on open days
    promo = p.promo[:, o:o + 42]
    for g in rep["attribution"]["groups"]:
        v = a["g_" + g["group"]]
        g["promo_signed"] = float(v[(mask) & (promo == 1)].mean())
        g["nonpromo_signed"] = float(v[(mask) & (promo == 0)].mean())

    # ---------------- leakage: features unchanged after future target perturbation
    ds = ase.load_weekly()
    o = ds.origins[0]
    rng = np.random.default_rng(3)
    y2 = ds.y.copy(); y2[:, o:] = rng.uniform(0, 1e5, size=y2[:, o:].shape)
    same = True
    for h in (1, 7, 13):
        A = ase._row(ds, o, h, ds.y[:, :o]); B = ase._row(ds, o, h, y2[:, :o])
        same &= all(np.array_equal(A[c].to_numpy(), B[c].to_numpy(), equal_nan=True) for c in A.columns)
    X1, y1 = b.build_training(p, 816, True)
    p2 = b.Panel(b.PANEL); p2.y = p2.y.copy(); p2.y[:, 816:] = rng.uniform(0, 1e6, size=p2.y[:, 816:].shape)
    X2, y2b = b.build_training(p2, 816, True)
    max_diff = 0.0
    for c in X1.columns:
        a1, a2 = X1[c].to_numpy(), X2[c].to_numpy()
        if a1.dtype.kind in "fi":
            max_diff = max(max_diff, float(np.nanmax(np.abs(a1 - a2))))
    same_r = max_diff < 1e-2 and np.array_equal(y1, y2b)
    rep["leakage"] = {"weekly_features_identical": bool(same),
                      "rossmann_train_matrix_identical": bool(same_r),
                      "rossmann_max_feature_diff": float(max_diff)}

    # ---------------- summary deltas
    rep["summary"] = {
        "weekly_all_stages_beat_repo": all(v["wape"]["ase"] < REPO["weekly_official"] for v in wr.values()),
        "rossmann_all_stages_beat_repo": all(v["wape"]["ase"] < REPO["rossmann"][int(o)] for o, v in rr.items()),
        "weekly_mean_ase": float(np.mean([v["wape"]["ase"] for v in wr.values()])),
        "rossmann_mean_ase": float(np.mean([v["wape"]["ase"] for v in rr.values()])),
        "total_stages": len(wr) + len(rr),
    }
    (OUT / "final_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["summary"], indent=1))
    print(json.dumps(rep["leakage"], indent=1))
    print(json.dumps(rep["attribution"]["groups"], ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()
