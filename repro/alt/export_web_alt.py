"""把 Rossmann 第二数据集的指标导出为 web/data/bench_alt.json。

只做汇总与改名，不重新计算指标：数值来自 audit/results/rossmann/metrics.json，
该文件由 bench_rossmann.py 写出、verify_rossmann.py 从逐序列 npz 独立重算校验。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "audit" / "results" / "rossmann" / "metrics.json"
OUT = ROOT / "web" / "data" / "bench_alt.json"

DISPLAY = {
    "LGBM-direct": ("LightGBM 直接多步（含促销/假日/门店）", "LightGBM+外生", "gbdt"),
    "LGBM-cal": ("LightGBM 历史+日历+店ID（消融）", "LightGBM 无外因", "gbdt"),
    "SeasonalNaive7+Open": ("SeasonalNaive7 + 营业开关", "SN7+Open", "baseline"),
    "StateFourier7": ("State-Fourier（周期 7）", "State-Fourier", "state"),
    "UCM7": ("UCM 局部水平+季节（周期 7）", "UCM7", "state"),
    "SeasonalNaive7": ("SeasonalNaive（周期 7）", "SeasonalNaive7", "baseline"),
    "MovingAvg": ("MovingAvg（窗口 56）", "MovingAvg56", "baseline"),
}


def main() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    origins = m["origins"]
    names = list(next(iter(origins.values()))["models"])
    models = []
    for name in names:
        rows = [o["models"][name] for o in origins.values() if name in o["models"]]
        if not rows:
            continue
        label, short, family = DISPLAY.get(name, (name, name, "other"))
        models.append({
            "id": name,
            "name": label,
            "short": short,
            "family": family,
            "wape_mean": round(sum(r["wape_mean"] for r in rows) / len(rows), 6),
            "wape_aggregate": round(sum(r["wape_aggregate"] for r in rows) / len(rows), 6),
            "runtime_s": round(sum(r.get("runtime_s", 0.0) for r in rows)/len(rows), 2),
            "n_series": rows[0]["n_series"],
            "n_origins": len(rows),
        })
    extra=ROOT/'audit/results/rossmann_multifactor/metrics.json'
    if extra.exists():
        runs=json.loads(extra.read_text())
        labels={'SF_full':('State-Fourier 六因子','SF 六因子','state'),
                'SF_open_only':('State-Fourier 仅营业状态','SF 营业状态','state'),
                'LGBM_six':('LightGBM 相同六类因子','LGBM 六因子','gbdt')}
        for key,(label,short,family) in labels.items():
            rr=[r for r in runs if r['model']==key]
            if len(rr)!=3:raise ValueError('Incomplete multifactor result')
            models.append({'id':key,'name':label,'short':short,'family':family,
                           'wape_mean':sum(r['wape_mean'] for r in rr)/3,
                           'wape_aggregate':sum(r['wape_aggregate'] for r in rr)/3,
                           'runtime_s':sum(r['elapsed_s'] for r in rr)/3,
                           'n_series':400,'n_origins':3})
    models.sort(key=lambda x: x["wape_mean"])
    for i, x in enumerate(models, 1):
        x["rank"] = i
        x["wape"] = x["wape_mean"]  # 前端统一字段

    payload = {
        "dataset": {
            "name": "Rossmann Store Sales（400 店抽样，审计）",
            "n_series": models[0]["n_series"] if models else 400,
            "horizon": 42,
            "freq": "D",
            "n_origins": len(origins),
            "factors": ["Open", "Promo2", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance"],
            "source": "https://www.kaggle.com/competitions/rossmann-store-sales/data",
            "note": "3 个扩张窗原点（o=816/858/900）分别打分后取平均；超参只在更早的内部验证原点 o=774 选择。",
        },
        "models": models,
        "note": "H=42 天；3 个扩张窗原点（o=816/858/900）各打分一次后取平均；"
                "超参只在更早的内部验证原点 o=774 选择，测试段未参与选参；负预测 clip 到 0 并逐模型记录。"
                "数据：Rossmann 400店日营业额；闭店日全样本占16.94%，三个测试窗占18.2%。"
                "LGBM-cal保留历史、日历、店ID，分别选参；此表State-Fourier仍未接外因。新增六因子行包含Promo2计划与竞争已开业状态；外因均按预测可用条件评估，细节见MULTIFACTOR-ROSSMANN.md。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(models)} models)")


if __name__ == "__main__":
    main()
