"""M4 CSV -> unified numpy dataset.

Note on the CSV layout: the official M4 files are *wide* -- one row per series,
first cell = series id, remaining cells = observations, header = V1..Vn naming
the time columns.  Shorter series are not padded, so `float("")` shows up if you
parse naively; `repro/common.py` handles that and right-aligns every series.

Outputs
  data/m4daily.npz   train (T, N) float32 right-aligned + NaN padding,
                     test (14, N), ids (N,), plus metadata
  data/demo_series.json   8 demo series for the frontend fallback
  data/meta.json          dataset summary
  data/.ready             marker that downstream agents wait for
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent / "repro"))
import common  # noqa: E402

H = 14


def main():
    data = common.load_m4_daily()
    tr, te, ids = data.train, data.test, data.ids
    np.savez_compressed(
        D / "m4daily.npz",
        ids=np.array(ids),
        train=np.ascontiguousarray(tr.T),      # (T, N) as documented
        test=np.ascontiguousarray(te.T),       # (14, N)
        horizon=H,
        n_series=len(ids),
        n_train_steps=int(tr.shape[1]),
    )

    # 8 demo series: stable, well-populated, mid-magnitude, strong weekly pattern
    y = tr
    L = data.lengths()
    nnz = np.array([np.mean(y[i, -L[i]:] > 0) for i in range(len(ids))])
    mean = np.array([np.mean(y[i, -L[i]:]) for i in range(len(ids))])
    std = np.array([np.std(y[i, -L[i]:]) for i in range(len(ids))])
    cv = std / np.maximum(mean, 1e-9)

    def acf7(i):
        v = y[i, -L[i]:] - mean[i]
        d = float((v * v).sum())
        return float((v[7:] * v[:-7]).sum() / d) if d > 0 else 0.0

    cand = [i for i in range(len(ids)) if nnz[i] > 0.98 and 2 < mean[i] < 500 and 0.15 < cv[i] < 0.9]
    cand.sort(key=acf7, reverse=True)
    pick = cand[:8] if len(cand) >= 8 else cand
    demo = [{
        "id": ids[i],
        "train": [int(v) for v in y[i, -L[i]:][-140:]],
        "test": [int(v) for v in te[i]],
        "mean": round(float(mean[i]), 1),
        "acf7": round(acf7(i), 3),
    } for i in pick]
    (D / "demo_series.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")

    meta = {
        "n_series": len(ids), "n_train_steps": int(tr.shape[1]), "horizon": H,
        "zeros_ratio": round(float(np.nanmean(tr == 0)), 4),
        "len_min": int(L.min()), "len_median": int(np.median(L)), "len_max": int(L.max()),
        "demo_ids": [d["id"] for d in demo],
    }
    (D / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    (D / ".ready").write_text("ok")


if __name__ == "__main__":
    sys.exit(main())
