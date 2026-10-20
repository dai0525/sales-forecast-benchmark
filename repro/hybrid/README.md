# ASE experiments (hybrid)

> 2026-10-20 独立复核见 [ASE-REVIEW.md](../../ASE-REVIEW.md)。存档预测可复算，
> 但当前 Weekly 训练代码存在特征列不一致错误，不能按下述命令完整重现。
> 简单组合消融未支持分阶段权重的必要性；现有 Rossmann 归因仅对应 L1 组件。
> 原始实现和存档保留，独立复核结果在 `audit/results/ase_review/`。

最终方案：`ase.py`（Weekly 主实现）与 `run_rossmann_final.py`（Rossmann 最终回测）。
两者共用同一套规则：log1p 目标、L2/L1 双 GBM、可选状态空间分量、按 short/mid/long
阶段在验证原点拟合权重，并把验证 WAPE 超过最优分量 1.5 倍的分量权重置 0。

## 复现

```bash
source .venv/bin/activate
python repro/hybrid/ase.py weekly                 # -> audit/results/ase/weekly_*.{json,npz}
python repro/hybrid/run_rossmann_final.py         # -> audit/results/ase/rossmann_final_*.{json,npz}
python repro/hybrid/finalize.py                   # -> audit/results/ase/final_report.json
python repro/hybrid/build_page.py                 # -> /opt/data/tmp/ase-report/index.html
```

`finalize.py` 只读存档预测，重算 WAPE、配对 bootstrap、归因占比与泄漏探针。
`validate.py` / `attribution.py` 是更早的独立校验草稿，最终数字以 `finalize.py` 为准。

## 目录内其它脚本

`probe_*.py` 是否定试验（比率目标、WAPE 加权 L1、逐序列乘性校准、目标相对滞后），
保留用于说明哪些方向被实测否掉，不属于最终方案。

`*.log` 是运行日志，已被 `.gitignore` 排除。
