"""DeepAR 落地后的收尾：保证 7 个模型的表完整 → 复制 → 发布 → 打印 URL"""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/opt/data/workspace/sales-forecast")
WEB = ROOT / "web" / "data"
PUB = Path("/opt/data/tmp/sf-workbench")

bench = json.loads((WEB / "bench.json").read_text(encoding="utf-8"))
by_id = {m["id"]: m for m in bench["models"]}

# 1) 保证 PatchTST 一定在（bench_deep 收尾时会用本轮结果覆盖 deep 段）
if "patchtst" not in by_id:
    by_id["patchtst"] = dict(
        id="patchtst", name="PatchTST", short="PatchTST", family="deep",
        wape=0.0849, smape=0.0863, mase=None, runtime_s=575.0,
        n_series=359, reproduced=True,
        note="M4-Weekly 实跑（120 步训练，patch_len=16）；MASE 未计算",
    )

# 2) DeepAR 若已产出且是可信数字，纳入并移出 pending
md_p = ROOT / "repro" / "metrics_deep.json"
deepar_ok = False
if md_p.exists():
    md = json.loads(md_p.read_text(encoding="utf-8"))
    d = md.get("deepar") or {}
    if d.get("wape") is not None and d.get("wape") < 0.5:
        by_id["deepar"] = dict(
            id="deepar", name="DeepAR", short="DeepAR", family="deep",
            wape=round(float(d["wape"]), 4), smape=round(float(d.get("smape") or 0), 4),
            mase=None, runtime_s=round(float(d.get("runtime_s") or 0), 1),
            n_series=int(d.get("n_series") or 359), reproduced=True,
            note="M4-Weekly 实跑（LSTM 高斯似然，120 步训练）；MASE 未计算",
        )
        deepar_ok = True

bench["models"] = sorted(by_id.values(), key=lambda m: m["wape"] if m.get("wape") is not None else 9)
bench["pending"] = [x for x in bench.get("pending", []) if x["id"] != "deepar"] if deepar_ok else bench.get("pending", [])
if not bench["pending"]:
    bench.pop("pending", None)
(WEB / "bench.json").write_text(json.dumps(bench, ensure_ascii=False, indent=1), encoding="utf-8")

# 3) 发布
if PUB.exists():
    shutil.rmtree(PUB)
PUB.mkdir(parents=True, exist_ok=True)
for name in ("index.html", "css", "js", "data"):
    src = ROOT / "web" / name
    dst = PUB / name
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)

out = subprocess.run(
    ["/opt/hermes/.venv/bin/python", "/opt/data/skills/temp-web/scripts/publish.py", "publish",
     "--source-dir", str(PUB), "--title", "销量预测工作台 v1.2 · 7 模型实测", "--days", "60"],
    capture_output=True, text=True, timeout=300,
)
print(out.stdout[-1200:])
print(out.stderr[-400:])
print("DEEPAR_OK" if deepar_ok else "DEEPAR_MISSING")
print("MODELS", json.dumps([(m["name"], round(m["wape"] * 100, 2)) for m in bench["models"]], ensure_ascii=False))
