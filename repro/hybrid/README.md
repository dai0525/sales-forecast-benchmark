# ASE experiments (hybrid)

> 2026-10-20 独立复核见 [ASE-REVIEW.md](../../ASE-REVIEW.md)，针对历史提交 `d594f30`。
> Weekly 特征列不一致错误现已修复；新结果与历史存档分别保存。
> 原存档的 Rossmann 归因仅对应 L1 组件，不是最终集成的精确归因。
> 后续实验方案在 `audit/results/ase_iteration/protocol.json`，其主要方案在确认组评分前固定。

最终方案：`ase.py`（Weekly 主实现）与 `run_rossmann_final.py`（Rossmann 最终回测）。
两者共用同一套规则：log1p 目标、L2/L1 双 GBM、可选状态空间分量、按 short/mid/long
阶段在验证原点拟合权重，并把验证 WAPE 超过最优分量 1.5 倍的分量权重置 0。

## 复现

```bash
source .venv/bin/activate
python repro/hybrid/ase.py weekly                 # -> audit/results/ase_repaired/weekly_*.{json,npz}
python repro/hybrid/run_rossmann_final.py         # -> audit/results/ase/rossmann_final_*.{json,npz}
python repro/hybrid/finalize.py                   # -> audit/results/ase/final_report_recomputed.json
python repro/hybrid/build_page.py                 # -> audit/results/ase/report/index.html
```

`finalize.py` 只读存档预测，重算 WAPE、配对 bootstrap、归因占比与泄漏探针。
`validate.py` / `attribution.py` 是更早的独立校验草稿，最终数字以 `finalize.py` 为准。
该脚本现在输出 `final_report_recomputed.json`，保留原 `final_report.json` 作为历史记录。
`audit_ase.py` 的重跑结果同样写入 `archive_audit_current.json`，不覆盖历史审查记录。

## 目录内其它脚本

`probe_*.py` 是历史探索（比率目标、WAPE 加权 L1、逐序列乘性校准、目标相对滞后），
不属于最终方案。比率目标探针的未来销量读取现已修复；修复前的结果不能作为否定该方向的证据。

`*.log` 是运行日志，已被 `.gitignore` 排除。

## 冻结方案的后续研究

```bash
python repro/alt/fetch_rossmann.py
python repro/hybrid/research_iteration.py weekly
python repro/hybrid/research_iteration.py rossmann
python repro/hybrid/research_iteration.py confirmation
python repro/hybrid/confirm_six_factor.py
python repro/hybrid/report_iteration.py
```

`confirmation` 读取原 400 家门店上已冻结的权重，在其余 534 家完整记录门店的历史上
重训基模型，不能据确认组成绩重新选择方案。新门店仍来自相同日期和同一连锁企业，
不是全新时间段或独立外部数据集。所有实验预测和汇总另存于 `audit/results/ase_iteration/`。
组件预测文件存在时会复用；如修改训练配置或数据，应使用新的输出目录，不能混用缓存。
