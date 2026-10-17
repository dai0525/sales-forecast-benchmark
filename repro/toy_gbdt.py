"""教学用：在 28 天玩具序列上真跑一个 LightGBM，看每加一棵树误差怎么变。
输出：repro/toy_gbdt.json
"""
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "repro" / "toy_gbdt.json"

W = [
    [100, 110, 105, 98, 120, 160, 150],
    [102, 115, 109, 101, 126, 166, 155],
    [104, 118, 112, 104, 131, 172, 161],
    [107, 121, 116, 108, 136, 178, 166],
]
Y = np.array([v for w in W for v in w], dtype=float)      # 28 天
TRUE = np.array([110, 125, 119, 111, 140, 183, 170], dtype=float)
DOW = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def features(hist):
    """把历史摊成一行特征：lag1 / lag7 / roll7_mean / dow。"""
    return [
        hist[-1],
        hist[-7],
        float(np.mean(hist[-7:])),
        len(hist) % 7,
    ]


def build_training(y):
    X, t = [], []
    for i in range(7, len(y)):
        hist = list(y[:i])
        X.append(features(hist))
        t.append(y[i])
    return np.array(X), np.array(t)


def recursive_forecast(model, y, h=7, iters=None):
    hist = list(y)
    out = []
    for _ in range(h):
        p = model.predict(np.array([features(hist)]), num_iteration=iters)[0]
        p = max(p, 0.0)
        out.append(p)
        hist.append(p)
    return np.array(out)


def wape(p, t):
    return float(np.abs(p - t).sum() / t.sum())


X, t = build_training(Y)
model = lgb.LGBMRegressor(
    n_estimators=60, learning_rate=0.3, num_leaves=4, min_child_samples=3,
    subsample=1.0, colsample_bytree=1.0, max_depth=2, random_state=7, verbose=-1,
)
model.fit(X, t)

curve = []
for k in [1, 2, 5, 10, 20, 40, 60]:
    p = recursive_forecast(model, Y, iters=k)
    curve.append({"trees": k, "wape": round(wape(p, TRUE), 4),
                  "pred": [round(x, 1) for x in p]})

p1 = recursive_forecast(model, Y, iters=1)
p60 = recursive_forecast(model, Y, iters=60)

# 第一棵树的叶子（用来讲"一棵树只会输出几个常数"）
df = model.booster_.trees_to_dataframe()
first = df[df["tree_index"] == 0]
leaves = first[first["split_feature"].isna()][["node_depth", "value", "count"]]

rows = []
for i in range(7):
    rows.append({
        "dow": DOW[i], "y": TRUE[i],
        "p1": round(p1[i], 1), "res1": round(TRUE[i] - p1[i], 1),
        "p60": round(p60[i], 1), "res60": round(TRUE[i] - p60[i], 1),
    })

print("=== 每加树的递归 WAPE ===")
for c in curve:
    print(f"  {c['trees']:>2} 棵  WAPE {c['wape']*100:5.2f}%   {c['pred']}")
print("=== 第一棵树的叶子 ===")
for _, r in leaves.iterrows():
    print(f"  depth {int(r['node_depth'])}  value {r['value']:.1f}  n={int(r['count'])}")
print("=== 逐日对照 ===")
for r in rows:
    print(f"  {r['dow']} y={r['y']:.0f}  1棵={r['p1']:6.1f} (残差 {r['res1']:6.1f})  "
          f"60棵={r['p60']:6.1f} (残差 {r['res60']:6.1f})")

OUT.write_text(json.dumps({
    "curve": curve, "rows": rows,
    "leaves": [{"depth": int(r["node_depth"]), "value": round(float(r["value"]), 1),
                "count": int(r["count"])} for _, r in leaves.iterrows()],
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("->", OUT)
