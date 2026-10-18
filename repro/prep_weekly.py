"""M4-Weekly -> npz + demo series（CSV 带引号、列优先）"""
import csv
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "data"
H = 13


def _is_id(v):
    try:
        float(v)
        return False
    except ValueError:
        return True


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.reader(f, delimiter=","))
    header = r[0]
    body = [row for row in r[1:] if row]
    if body and _is_id(body[0][0]):
        # 行优先：每行一条序列，首列是 id，其余是时间步
        ids = [row[0].strip().strip('"') for row in body]
        width = max(len(row)-1 for row in body)
        mat = np.full((len(body), width), np.nan)
        for i, row in enumerate(body):
            mat[i, :len(row)-1] = [_num(x) for x in row[1:]]
        return ids, mat  # (N, T)
    ids = [h.strip().strip('"') for h in header]
    mat = np.array([[_num(x) for x in row] for row in body], dtype=float).T
    return ids, mat


def _num(v):
    v = v.strip()
    if v in ("", "NA", "nan", "NaN"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def main():
    ids, tr = load(D / "Weekly-train.csv")
    ids2, te = load(D / "Weekly-test.csv")
    print("train", tr.shape, "test", te.shape, "ids", len(ids))
    assert ids == ids2, (ids[:3], ids2[:3])
    assert te.shape[1] == H, te.shape
    if not np.isfinite(te).all():
        raise ValueError("Incomplete Weekly test data")
    np.savez_compressed(D / "weekly.npz", train=tr.T, test=te.T, ids=np.array(ids))
    # 统计与演示序列用去掉尾部空值补齐后的真实段
    clean = []
    for i in range(tr.shape[0]):
        fin = np.flatnonzero(~np.isnan(tr[i]))
        clean.append(tr[i][fin[0]:fin[-1] + 1] if len(fin) else np.zeros(10))
    y = [np.asarray(v) for v in clean]
    mean = np.array([v.mean() for v in y])
    cv = np.array([v.std() / max(v.mean(), 1e-9) for v in y])
    nnz = np.array([(v > 0).mean() for v in y])

    def acf(i, lag):
        v = y[i] - y[i].mean()
        d = (v * v).sum()
        return float((v[lag:] * v[:-lag]).sum() / d) if d > 0 else 0.0

    cand = [i for i in range(len(ids)) if mean[i] > 10 and 0.05 < cv[i] < 2.0]
    cand.sort(key=lambda i: abs(acf(i, 52)), reverse=True)
    pick = cand[:8]
    demo = [{
        "id": ids[i],
        "train": [round(float(v), 2) for v in y[i][-130:]],
        "test": [round(float(v), 2) for v in te[i]],
        "mean": round(float(mean[i]), 1),
        "cv": round(float(cv[i]), 3),
        "acf52": round(acf(i, 52), 3),
    } for i in pick]
    (D / "demo_series_weekly.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
    meta = {"dataset": "M4-Weekly", "n_series": int(len(ids)), "n_train_steps": int(tr.shape[1]),
            "horizon": H, "zeros_ratio": round(float(np.mean([(v == 0).mean() for v in y])), 4),
            "demo_ids": [d["id"] for d in demo], "source": "M4 competitions (Mcompetitions/M4-methods)"}
    (D / "meta_weekly.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (D / ".ready").write_text("weekly")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
