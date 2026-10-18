# 销量预测 Benchmark 与算法实验台

在 **M4 公开数据集**（Weekly / Daily 子集）上，把统计基线、梯度提升、深度时序模型、
贝叶斯结构时序（BSTS）以及自研的**因子图 + 消息传递（FG-MP）**原型放在同一套口径下实测对比，
并附一个可交互的网页工作台（`web/`）解释每个算法的原理。

> 所有数字都是真实跑出来的，脚本可复现；**包含负结果**（见「已知结论 / 负结果」）。

---

## 1. 实测榜单（Weekly，359 条序列，预测 13 周）

指标：**逐序列 WAPE 取均值**（不是总量加权），详见 §4。

| 排名 | 模型 | WAPE | 总量加权 WAPE | 耗时（359 条） |
|---|---|---|---|---|
| 1 | BSTS（局部水平 + 随机季节 m=52，MLE） | **7.84%** | 5.56% | 268 s |
| 2 | LightGBM（全局模型，22 维特征） | 8.06% | — | 148 s |
| 3 | PatchTST | 8.49% | — | 575 s |
| 4 | **FG-MP（因子图 + 高斯消息传递，本仓库自研原型）** | **8.71%** | 5.98% | **0.3 s** |
| 5 | N-HiTS | 8.88% | — | — |
| 6 | TFT | 9.76% | — | — |
| 7 | N-BEATSx | 10.48% | — | — |
| 8 | MovingAvg8 | 10.84% | — | — |
| 9 | SeasonalNaive（52 周） | 15.55% | 13.25% | 0 s |

Daily（1000 条序列，统计基线 / 深度模型分别跑）：

| 类 | 模型 | WAPE | 备注 |
|---|---|---|---|
| 统计 | AutoARIMA | **2.36%** | 250 条子集，`stepwise=True`，约 94 个候选，1097 s |
| 统计 | AutoETS | 3.43% | 1000 条 |
| 统计 | Theta | 3.47% | |
| 统计 | Croston | 3.98% | |
| 统计 | SeasonalNaive | 4.44% | |
| 深度 | N-HiTS | 3.14% | 300 条 |
| 深度 | N-BEATSx | 3.31% | |
| 深度 | DeepAR | 5.21% | |

---

## 2. 仓库结构

```
.
├── repro/                       # 所有可运行的实验脚本（核心）
│   ├── common.py                # 数据加载、滑窗切分、WAPE/MASE/sMAPE
│   ├── fetch_m4.py              # 下载 M4 原始数据（Daily/Weekly CSV）
│   ├── prep_weekly.py           # 预处理：M4 Weekly → data/weekly.npz
│   ├── bench_base.py            # 统计基线：Naive/SeasonalNaive/MA/Theta/ETS/ARIMA/Croston
│   ├── baselines.py             # 统计基线（statsforecast 版，日度 1000 条）
│   ├── lgbm_global.py           # 全局 LightGBM（递归多步，22 维特征）
│   ├── deep_models.py           # 深度模型定义（PatchTST/N-HiTS/TFT/N-BEATSx/DeepAR）
│   ├── bench_deep.py            # 深度模型训练与评估
│   ├── bsts.py                  # BSTS 结构模型（statsmodels UCM，season=13）
│   ├── bsts52.py                # BSTS season=52 全量（榜单第 1 名）
│   ├── bsts_val.py              # BSTS 在训练段内部验证窗口的预测（元策略用）
│   ├── fg_mp.py                 # ★ FG-MP：因子图 + 高斯消息传递（本仓库的研究原型）
│   ├── ensemble_eval.py         # 融合实验：FG × BSTS × SeasonalNaive
│   ├── fg_select.py             # 元策略：验证窗口硬选择 / 精度加权融合
│   ├── toy_gbdt.py              # 教学：逐棵树看 GBDT 残差收敛
│   └── export_web.py            # 把结果导出成 web/data/*.json
├── data/                        # 小体积数据（大文件见 .gitignore，可重新下载）
│   ├── weekly.npz               # Weekly 359×200 train + 359×13 test（已对齐）
│   ├── fg_preds.npz             # FG-MP 测试段预测
│   ├── bsts_preds.npz           # BSTS 测试段预测
│   ├── bsts_val_preds.npz       # BSTS 验证段预测
│   └── lgbm_preds*.npz
├── web/                         # 可交互工作台（纯静态 HTML/CSS/JS）
│   ├── index.html               # 算法讲解 / 实测成绩 / 论文库 / 怎么用
│   ├── principles.html          # 从零开始（机制详解版，含 10 张 SVG 图）
│   └── data/bench.json          # 榜单数据（唯一真相源）
├── REPORT-MPA.md                # ★ MPA/因子图用于销量预测的调研 + 可行性 + 实测报告
├── CONTRACT.md                  # 指标口径与发布约定
├── MERGE-NOTE.md                # 工作台与原理页的分工约定
└── requirements.txt
```

---

## 3. 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 拿数据（M4 官方仓库）
python repro/fetch_m4.py            # 下载 Daily/Weekly CSV 到 data/
python repro/prep_weekly.py         # → data/weekly.npz

# 2) 统计基线
python repro/bench_base.py

# 3) 全局 LightGBM
python repro/lgbm_global.py

# 4) BSTS（榜单第一名）
python repro/bsts52.py --jobs 3

# 5) FG-MP 原型（超参只在训练段内部验证窗口选）
python repro/fg_mp.py --val

# 6) 融合与元策略
python repro/ensemble_eval.py
python repro/bsts_val.py && python repro/fg_select.py

# 7) 打开工作台（纯静态，任何静态服务器都行）
python -m http.server 8899 --directory web
```

---

## 4. 指标口径（**重要，跨仓库比较前必读**）

```python
WAPE = sum(|y - ŷ|) / sum(|y|)          # 单条序列内计算
榜单 WAPE = mean(逐序列 WAPE)            # ← 榜单用的是这个
总量加权 WAPE = sum(|y-ŷ|)/sum(|y|)      # 全序列汇总，通常更小
```

**三个容易踩的坑**（都真实发生过）：

1. **季节周期**：`SeasonalNaive` 与 BSTS 早期版本一个用 52 周、一个用 13 周，
   直接比较会得出"BSTS 8.79% 打败 SeasonalNaive 15.55%"的错误结论。
   统一到 52 后：BSTS **7.84%**、SeasonalNaive **15.55%**，才可比。
2. **子集 vs 全量**：60 条子集上 BSTS(m=52) 是 9.61%，全量 359 条是 7.84%。子集不可外推。
3. **逐序列均值 vs 总量加权**：同一模型可以差 1–2 个百分点（BSTS 逐序列 7.84% / 总量 5.56%）。

---

## 5. 已知结论 / 负结果（对分析者最有价值的部分）

1. **FG-MP 没有打赢 BSTS**（8.71% vs 7.84%）。原因是数学上的：
   线性高斯因子图上的 sum-product 消息传播**等价于卡尔曼滤波**，
   所以"用 MPA 做加性分解"与结构时序是同一个估计量，不可能指望它更好。
2. **FG-MP 与 BSTS 的误差相关系数 0.900**，融合后反而更差
   （算术平均 8.01% / 几何平均 7.98% / 中位数 8.01%）。
3. **逐序列选模型也失败**（8.33%）：13 点验证窗口太短，选择噪声大于信号；
   BSTS 在验证段 6.88% → 测试段 7.84%（衰减 0.96 pp），FG-MP 8.03% → 8.71%。
   而 oracle（每序列选真正最优）是 6.55%，说明**空间存在，只是选不出来**。
4. **FG-MP 唯一的确定收益是速度**：0.84 ms/序列 vs BSTS 0.75 s/序列（约 900×）。
   10 万 SKU：FG-MP 约 84 秒，BSTS 约 21 小时。
5. **加性分解的可辨识性来自约束，不来自算法**：尺度自由度（`λ·S` 缩放不变）、
   水平–季节混淆、因子旋转，三者必须靠零和约束 / 单位范数 / 外生变量钉死。
   详见 `REPORT-MPA.md` §2。

---

## 6. 报告

- [`REPORT-MPA.md`](REPORT-MPA.md) — MPA / 因子图 / Belief Propagation / GAM·EBM / BART / BSTS 调研，
  可辨识性证明，FG-MP 架构与消息更新式，全部实测数字，论文与开源项目链接。

---

## 7. 致谢与数据来源

- 数据：[M4 Competition](https://github.com/Mcompetitions/M4-methods)（公开数据集）
- 部分基线实现参考 `statsforecast`、`statsmodels`、`pytorch-lightning`
- 网页工作台的算法讲解由多个 AI 协作产出（见 `MERGE-NOTE.md`）

## 8. License

MIT
