# Rossmann 原实验：复核后的结果与边界

审计对象：远端 main 的 929519e 及其前序 Rossmann 实现。旧报告保存在 `audit/results/rossmann_review/original/`，问题清单见 [后续审计](DEEPSEEK-REVIEW.md)。

**保存预测重算吻合：原榜单点预测成绩没有发现算错。** 但原报告对算法族、误差归因及统计显著性的解释过强。此报告替换这些结论，不改变已有模型预测。

## 数据与协议

- Rossmann Sales 是门店日营业额，不是销售件数。400家完整日历门店，从934家完整店中固定抽样；按全时段完整性筛选有幸存者偏差，不代表全部1115家店或新店。
- 训练/验证/测试：验证原点774，测试816/858/900，每次42天，扩张训练窗；三个测试窗口均已被查看，后续研究是回顾性探索。
- 外生字段：Promo、Open、SchoolHoliday、StateHoliday，加StoreType、Assortment、Promo2、CompetitionDistance。Customers为事后字段，未使用。
- Promo/Open/假期按提前已知的计划处理，但文件只有历史记录，没有当时版本的计划；门店属性只有静态快照，不能证明历史每个原点均已知或从未变化。
- 原版State-Fourier和UCM仅使用销量；LGBM使用外部信息与跨店训练。不能用该差距判定状态空间模型族无法使用因子。State-Fourier的Rossmann网格固定trend_q=0，并未包含有趋势的全部变体。
- LGBM-cal保留历史销量、日历和门店ID，并不是只输入日历。它与完整模型分别选出300/600棵树，原消融不是完全固定超参数的因子对照。
- CSV来自GitHub镜像，结构/日期/行数检查及镜像SHA256一致；未获取官方文件进行独立逐字节比对，不能宣称已证明与Kaggle原件字节一致。

## 原榜单重算

| 模型 | 平均 WAPE | 总量 WAPE（逐窗口平均） | 逐原点平均 WAPE |
|---|---:|---:|---|
| LGBM-direct | 9.3444% | 9.1960% | 10.58% / 8.82% / 8.63% |
| LGBM-cal | 19.8554% | 19.6218% | 25.25% / 22.58% / 11.74% |
| SeasonalNaive7+Open | 23.2250% | 22.9326% | 28.24% / 18.99% / 22.44% |
| StateFourier7 | 25.6436% | 25.3128% | 31.21% / 24.66% / 21.06% |
| UCM7 | 25.9652% | 25.6255% | 31.91% / 25.27% / 20.71% |
| SeasonalNaive7 | 28.7171% | 28.3074% | 35.51% / 27.98% / 22.66% |
| MovingAvg | 43.0138% | 42.3672% | 49.17% / 43.42% / 36.45% |

平均WAPE是先逐店再逐窗口等权平均；表中总量WAPE是各窗口的总量WAPE再平均，不等于三个窗口合并后的总量WAPE。合并口径另存 diagnostics.json。耗时原记录包括拟合和预测，环境不同不能直接推断速度倍数。

## 统一口径下的闭店误差分解

全部采用逐店逐窗口的同一分母：总绝对误差/该店该窗口营业额 = 闭店日绝对误差贡献 + 营业日绝对误差贡献。

| 模型 | 闭店贡献 | 营业日贡献 | 合计 WAPE |
|---|---:|---:|---:|
| StateFourier7 | 6.573pp | 19.070pp | 25.644% |
| LGBM-direct | 0.164pp | 9.180pp | 9.344% |

两模型总差距为 16.299pp，其中闭店误差差距为 6.409pp，占 39.3%。这是误差按日期分组的算术分解，不是促销/假期的因果贡献；不能把其余比例归因给某些业务因素。

## 配对不确定性

按门店聚类重采样，每次保留该门店三个原点；不再把1200个门店×窗口视为独立序列。仍未覆盖跨店共享节日冲击或新预测日期的不确定性，区间仅对已选三个窗口有条件成立。

| 比较（前者减后者） | 平均差 | 门店聚类bootstrap 95%区间 |
|---|---:|---|
| StateFourier7 - UCM7 | -0.322pp | [-0.530, -0.122]pp |
| LGBM-direct - LGBM-cal | -10.511pp | [-10.732, -10.311]pp |
| LGBM-direct - StateFourier7 | -16.299pp | [-16.717, -15.918]pp |

结论：这些外部信息在原LightGBM实验中有明显预测价值；State-Fourier与UCM的旧单变量分数接近。不能推出“多因子一定比任何换算法更重要”或“有外因时状态空间必然落后”。多因子State-Fourier与匹配信息集的比较见 [六因子实验](MULTIFACTOR-ROSSMANN.md)。

## 复现

```bash
python repro/alt/fetch_rossmann.py
python repro/alt/prep_rossmann.py
python repro/alt/verify_rossmann.py
python repro/alt/report_rossmann.py
```

逐预测结果：`audit/results/rossmann/preds/`；点预测原始训练协议与选参记录：`audit/results/rossmann/metrics.json`。重新训练使用 `repro/alt/bench_rossmann.py`，单独指定输出目录以保留原始证据。
