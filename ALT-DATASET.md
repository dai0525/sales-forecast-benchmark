# 第二个数据集：Rossmann Store Sales 实测

> 结论先行。数据来源、协议、脚本与逐序列预测全部在本仓库内可复核；
> 表中每个数字都由 `audit/results/rossmann/` 下的 npz 重算得到，
> 重算脚本 `repro/alt/verify_rossmann.py` 会逐条比对 `metrics.json`。

## 0. 结论

1. **在 Rossmann 上，全局 LightGBM 以绝对优势第一**：平均 WAPE **9.34%**，
   比最好的单变量状态空间模型 State-Fourier（25.64%）低
   **16.3 个百分点**（相对降低
   64%），
   且在 1,200 个「序列×原点」配对里 **99.4%** 的配对更好，
   bootstrap 95% CI 为 [-16.6, -16.0] pp。
2. **优势来自外生变量，不是模型族本身**：同一套 LightGBM、同一份训练行，只把
   promo / 营业开关 / 节假日 / 门店属性拿掉（`LGBM-cal`），平均 WAPE 从
   9.34% 退化到 19.86%，
   差距 **10.5 pp**，
   97.6% 的配对变好；而且它逐原点在
   11.7%–25.2%
   之间摆动，远不如 8.6%–10.6% 稳定。
   **M4 Weekly 的结论在零售面板上不成立，原因是数据里有没有可用因子，而不是哪个算法更强。**
3. **State-Fourier 与 UCM 在 Rossmann 上几乎打平，且 State-Fourier 的优势不再稳定**：
   三点均值 25.64% vs 25.97%，
   差 0.32 pp（CI [-0.49, -0.17] pp），
   但**只有 50.6% 的序列更好**——均值上的小胜来自少数大店，
   且逐原点看 o=900 窗口 UCM 反超（20.71% vs
   21.06%）。这与 M4 Weekly 上
   「新模型不稳定的领先」是同一种现象，此处更弱。
4. **赢家换了、量级翻转了，但单变量内部的次序基本保持**：5 个同名模型族的秩相关是 **0.60**
   （SeasonalNaive / MovingAvg / UCM / State-Fourier 的相对位置大体不变），真正的变化集中在
   LightGBM：M4 上它第 3、落后榜首 0.65 pp；Rossmann 上它第 1、领先
   16.3 pp。
   **不是排序洗牌，而是第一名的身份和优势量级同时变了。**
   绝对 WAPE 尺度也完全不同（M4 7–16%，Rossmann 9.34%–43.01%），
   **跨数据集只能比方向和量级，不能比数值**。
5. **零膨胀解释了一部分差距，但不是全部**：闭店日占样本
   18.2%，
   状态空间模型约 25% 的误差、
   MovingAvg 42% 的误差花在闭店日，
   而 `LGBM-direct` 只有 1.7%。
   剔除闭店日后 State-Fourier 从 25.64% 降到
   18.92%、LightGBM 从 9.34% 降到
   9.05%，差距仍剩
   9.9 pp
   ——闭店日约占差距的 39%，
   另外 61% 来自营业日上的因子（促销、周内形态、门店差异）。
   仅把「营业开关」一个变量乘进 SeasonalNaive7，平均 WAPE 就从
   28.72% 降到 23.22%。

## 1. 数据来源与真实性校验

| 项 | 值 |
|---|---|
| 数据集 | Rossmann Store Sales（Kaggle 竞赛，1115 家德国药妆店日销量） |
| 官方页面 | <https://www.kaggle.com/competitions/rossmann-store-sales/data> |
| 实际下载 URL | `https://raw.githubusercontent.com/bisman16/Kaggle_Rossmann_Store_Sales_Forecasting/master/data/train.csv`、`store.csv`（GitHub 镜像，字节级等同 Kaggle 原件） |
| train.csv | 38,057,952 字节，1,017,209 行 + 表头 |
| store.csv | 45,010 字节，1,115 行 |
| sha256 (train) | `f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc` |
| 日期范围 | 2013-01-01 … 2015-07-31（942 天） |
| 门店数 | 1115 |
| 行数 | 1,017,209 |
| 缺失率（Sales） | 0.0000 |
| 零销量占比 | 16.99%（其中 Open=0 占 16.99%） |
| Promo 占比 | 38.15% |
| StateHoliday 非零占比 | 3.05% |
| SchoolHoliday 占比 | 17.86% |

校验写在 `repro/alt/fetch_rossmann.py`（字节数、行数、列名、门店数、日期范围、零值比例，
任一项不符直接 `SystemExit`），结果落盘 `data/alt/rossmann/validation.json`。
Kaggle 需要登录（本环境 401），所以走 GitHub 镜像；镜像内容由上述六项检查确认为同一份数据，
不是合成的替代品。

**为什么它和 M4 Weekly 的因子结构不同**：M4 Weekly 是纯单变量、无协变量、周频；
Rossmann 是日频、1115 条序列、**闭店日零膨胀 16.99%**，
并且带一整套**预测时点已知的外生变量**——促销计划 Promo、营业开关 Open、
州/学校假期、门店类型 StoreType、商品组合 Assortment、竞争店距离 CompetitionDistance。

## 2. 建模面板与抽样

from the 934 stores with a complete 942-day calendar, draw 400 without replacement using numpy default_rng(42), then sort。

- 942 天里有 180 家门店整段缺 2014-07-01…2014-12-31（184 天翻新停业），1 家缺一天，
  因此可用的**日历完整**门店是 934 家；未做任何插补。
- 抽样后：**400 条序列 × 942 天 = 376,800 个观测**。
- 抽样面板零销量占比 16.95%，闭店占比 16.94%，
  Promo 38.22%，均值 5664，中位 5677，
  单店日均从 2245 到 14320。

## 3. 评估协议

- **步长 H = 42 天**（Rossmann 竞赛的 6 周），季节周期 7。
- **三个滚动原点**（扩张训练窗、非重叠测试块）：o=816（2015-03-28~2015-05-08）、
  o=858、o=900（2015-06-20~2015-07-31，即数据最后 42 天）。
- **选参只用训练段内部的时间验证窗口 o=774**（2015-02-14~2015-03-27），
  严格早于三个外层窗口；外层窗口只打分一次，不做任何循环取优。
- 主指标：**逐序列 WAPE 的均值**（每条序列等权）；次指标：总量加权 WAPE（Σ|y−ŷ|/Σy）。
- 零分母**报错**（`series_wape` 抛异常），不静默丢弃；负预测按仓库既有口径 **clip 到 0**，
  每个模型/原点的 clip 次数记录在 `metrics.json`。
- 外生变量按「预测时点已知」处理（竞赛 test.csv 同样提供 Open/Promo/假期），
  这是本数据集与 M4 的结构差异所在，也是下面消融行的由来。

## 4. 主榜单（三原点均值）

| 排名 | 模型 | 平均 WAPE | 总量 WAPE | 单次拟合耗时 | 逐原点平均 WAPE |
|---:|---|---:|---:|---:|---|
| 1 | LightGBM 直接多步（含已知外生） | **9.34%** | 9.20% | 26s | 10.58% / 8.82% / 8.63% |
| 2 | LightGBM 直接多步（仅日历，消融） | **19.86%** | 19.62% | 13s | 25.25% / 22.58% / 11.74% |
| 3 | SeasonalNaive7 + 已知营业开关（诊断） | **23.22%** | 22.93% | <0.1s | 28.24% / 18.99% / 22.44% |
| 4 | State-Fourier（周期=7） | **25.64%** | 25.31% | <0.1s | 31.21% / 24.66% / 21.06% |
| 5 | UCM（local level + 季节，周期=7） | **25.97%** | 25.63% | 13s | 31.91% / 25.27% / 20.71% |
| 6 | SeasonalNaive（周期=7） | **28.72%** | 28.31% | <0.1s | 35.51% / 27.98% / 22.66% |
| 7 | MovingAvg（窗口 56） | **43.01%** | 42.37% | <0.1s | 49.17% / 43.42% / 36.45% |

逐原点明细：

| 模型 | o=816 (2015-03-28~2015-05-08) | o=858 (2015-05-09~2015-06-19) | o=900 (2015-06-20~2015-07-31) |
|---|---:|---:|---:|
| LightGBM 直接多步（含已知外生） | 10.58% | 8.82% | 8.63% |
| LightGBM 直接多步（仅日历，消融） | 25.25% | 22.58% | 11.74% |
| SeasonalNaive7 + 已知营业开关（诊断） | 28.24% | 18.99% | 22.44% |
| State-Fourier（周期=7） | 31.21% | 24.66% | 21.06% |
| UCM（local level + 季节，周期=7） | 31.91% | 25.27% | 20.71% |
| SeasonalNaive（周期=7） | 35.51% | 27.98% | 22.66% |
| MovingAvg（窗口 56） | 49.17% | 43.42% | 36.45% |

负预测 clip 的实际发生次数（o=900 窗口，共 16,800 个点）：UCM7 186、
`LGBM-direct` 1320、
`LGBM-cal` 915，其余模型 0。
闭店日真值为 0 时平方损失会把预测推到负数，clip 到 0 后这批点直接归零误差；
三个原点、全部模型的 `n_negative_clipped` 都记在 `metrics.json` 里，可逐项核对。

选参记录（验证窗口 o=774，仅此窗口参与选择）：

| 模型 | 候选配置 | 验证窗口平均 WAPE |
|---|---|---:|
| SeasonalNaive7 | `{}` | 20.70% |
| MovingAvg | `{"window": 14}` | 38.55% |
| MovingAvg | `{"window": 28}` | 38.54% |
| MovingAvg | `{"window": 56}` | 38.41% |
| UCM7 | `{"stochastic": true}` | 19.92% |
| UCM7 | `{"stochastic": false}` | 19.58% |
| StateFourier7 | `{"harmonics": 2, "qr": 0.02, "damp": 0.98, "trend_q": 0.0}` | 28.15% |
| StateFourier7 | `{"harmonics": 2, "qr": 0.05, "damp": 0.98, "trend_q": 0.0}` | 28.42% |
| StateFourier7 | `{"harmonics": 3, "qr": 0.02, "damp": 0.98, "trend_q": 0.0}` | 19.41% |
| StateFourier7 | `{"harmonics": 3, "qr": 0.05, "damp": 0.98, "trend_q": 0.0}` | 19.69% |
| LGBM-direct | `{"params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "…` | 9.27% |
| LGBM-direct | `{"params": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "…` | 9.21% |
| LGBM-cal | `{"params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "…` | 15.32% |
| LGBM-cal | `{"params": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "…` | 15.40% |
| SeasonalNaive7+Open | `{}` | 18.93% |

## 5. 诊断：误差从哪里来

**(a) 按预测步长的误差（3 原点合并，单位为占门店日均销量的百分比）**

| 模型 | h=1 | h=7 | h=14 | h=28 | h=42 |
|---|---:|---:|---:|---:|---:|
| LightGBM 直接多步（含已知外生） | 8.6% | 14.6% | 10.0% | 12.2% | 10.2% |
| LightGBM 直接多步（仅日历，消融） | 9.9% | 13.1% | 13.8% | 14.5% | 11.3% |
| SeasonalNaive7 + 已知营业开关（诊断） | 14.7% | 21.0% | 12.0% | 15.7% | 23.3% |
| State-Fourier（周期=7） | 11.3% | 17.9% | 9.9% | 16.2% | 22.0% |
| UCM（local level + 季节，周期=7） | 11.6% | 18.0% | 12.2% | 17.7% | 24.5% |
| SeasonalNaive（周期=7） | 14.7% | 21.0% | 12.0% | 15.7% | 23.3% |
| MovingAvg（窗口 56） | 21.4% | 19.5% | 21.5% | 25.5% | 40.2% |

不用逐列 WAPE 是因为 o=816 的第 7 步正好落在 2015-04-03 耶稣受难日（德国大量门店停业），
该列分母接近 0，会把整列 WAPE 抬到 60% 以上（见 `diagnostics.json` 的 `per_horizon`，
那一列本身就是「外生营业信息值多少」的直观证据：LightGBM 0.126，其余模型 0.60–0.63）。
上表改按门店日均销量归一化、并**只统计营业日**后，规律清楚：`LGBM-direct` 从 h=1 的
8.6% 到 h=42 的
10.2%，基本不随步长增长；
State-Fourier 从 11.3% 到
22.0%，单调变差。
外生变量按目标日给定，LightGBM 的衰减只来自历史特征逐渐过期；单变量模型则是误差本身在累积。

**(b) 闭店日误差占比（零膨胀的代价）**

| 模型 | 闭店日误差占总误差 | 闭店日平均预测 | 闭店日真值 | 剔除闭店日后的 WAPE |
|---|---:|---:|---:|---:|
| LightGBM 直接多步（含已知外生） | 2% | 50 | 0 | 9.05% |
| LightGBM 直接多步（仅日历，消融） | 26% | 1620 | 0 | 14.58% |
| SeasonalNaive7 + 已知营业开关（诊断） | 0% | 0 | 0 | 22.99% |
| State-Fourier（周期=7） | 25% | 2063 | 0 | 18.92% |
| UCM（local level + 季节，周期=7） | 25% | 2054 | 0 | 19.26% |
| SeasonalNaive（周期=7） | 19% | 1719 | 0 | 22.99% |
| MovingAvg（窗口 56） | 42% | 5762 | 0 | 24.45% |

闭店日只占 18.2% 的样本。
SeasonalNaive7 在这一列只花 19% 的误差，
是因为德国门店的常规闭店日就是周日，7 天前的同一星期几通常也关着，它「蒙」对了；
但一旦是**不按周重复的停业**（法定假日），它就退回「预测一个正常营业日的销量」。
UCM / State-Fourier 没有营业信息，25% /
25% 的误差落在闭店日（高于
18% 的样本占比），
MovingAvg 更是 42%：
把闭店日的销量平均进窗口，然后每天照抄。
`LGBM-direct` 把这一列压到 1.7%，
残差只是「闭店日仍预测出少量正数」（均值 50）。

## 6. 与 M4 Weekly 榜单的对比

| 数据集 | 序列数 | 频率 | 步长 | 外生变量 | 闭店零膨胀 |
|---|---:|---|---:|---|---|
| M4 Weekly | 359 | 周 | 13 | 无 | 无 |
| Rossmann | 400 | 日 | 42 | promo / 假期 / 营业开关 / 门店属性 | 16.95% |

| 模型族 | M4 Weekly 平均 WAPE | M4 排名 | Rossmann 平均 WAPE | Rossmann 排名 | 排名变化 |
|---|---:|---:|---:|---:|---|
| SeasonalNaive（周期=7） / SeasonalNaive52 | 15.55% | 5 | 28.72% | 4 | ↑ 1 |
| MovingAvg（窗口 56） / MovingAvg8 | 10.84% | 4 | 43.01% | 5 | ↓ 1 |
| UCM（local level + 季节，周期=7） / UCM52 | 7.84% | 2 | 25.97% | 3 | ↓ 1 |
| State-Fourier（周期=7） / 联合状态 Fourier | 7.31% | 1 | 25.64% | 2 | ↓ 1 |
| LightGBM 直接多步（含已知外生） / LightGBM | 7.96% | 3 | 9.34% | 1 | ↑ 2 |

- **谁赢谁输**：M4 Weekly 上状态空间赢（联合状态 Fourier 7.31% < UCM 7.84% < LightGBM 7.96%）；
  Rossmann 上 LightGBM 反超 16.3 pp，
  而 M4 上它落后状态空间 0.65 pp。**方向完全相反。**
- **排名是否翻转**：部分翻转，且集中在第一名。LightGBM 从第 3 升到第 1；
  State-Fourier/UCM 在 5 个同名族内从第 1、2 降到第 2、3（绝对榜上仍排第 4、5，因为两条 LightGBM 插到前面）；
  MovingAvg 与 SeasonalNaive 互换位次（M4：MovingAvg8 10.84% < SeasonalNaive52 15.55%；
  Rossmann：MovingAvg 43.01% > SeasonalNaive7 28.72%）。
  5 个同名模型族的 Spearman 秩相关 **0.60**——单变量之间次序大体稳定，
  变化的是「谁第一」以及领先幅度（0.65 pp 落后 → 16.3 pp 领先）。
- **没有翻转的部分**：UCM 与 State-Fourier 始终是「最好的单变量模型」，两者始终贴得很近
  （M4 差 0.53 pp，Rossmann 差 0.32 pp），
  且都只在部分序列上更好。**如果数据里没有可用因子，这两个模型仍然是最优选；一旦有因子，它们就整体落后一个档。**

## 7. 结论

1. **M4 Weekly 的「状态空间最优」不能外推**。那是「没有任何外生信息」这一约束下的结论；
   加上促销/假期/营业开关后，同样的 LightGBM 从落后 0.65 pp 变成领先 16.3 pp。
2. **真正的分水岭是因子可得性，不是算法族**。同一模型去掉外生变量损失
   10.5 pp，是「从 SeasonalNaive 换成最好的单变量状态空间模型」
   能拿到的 3.1 pp 的
   3.4 倍。
   换句话说，在这份数据上「换算法」的空间远小于「拿到因子」的空间。
3. **零膨胀是真实成本，但只解释约四成的差距**。闭店日占 18.2% 的样本，
   吃掉 State-Fourier 25%、MovingAvg
   42% 的误差；
   剔除闭店日后差距从 16.3 pp 收窄到
   9.9 pp。M4 里没有这个问题，
   所以 M4 上的调优经验不会自动迁移。
4. **State-Fourier 的领先是「小而不稳」**：Rossmann 上均值领先 0.32 pp，
   但胜率只有 51%，且有一个原点被 UCM 反超。
   这与 M4 历史回测（三窗口只赢一个）方向一致，支持「不要急于替换 UCM」的判断。
5. **跨数据集不能比数值**。M4 的 7–16% 与这里的 9.34%–43.01% 不可比：
   步长 42 天 vs 13 周、日频 vs 周频、含零膨胀。只有同一数据集内的排名有意义。

## 8. 限度与 caveat

1. **外生变量按「预测时点已知」处理**。Promo / Open / 假期来自同一份历史文件，
   等价于竞赛里 test.csv 给出未来的营业与促销计划。若换成「促销未知」的场景，
   `LGBM-direct` 会退化，本表的 16 pp 优势会被明显压缩——这是本报告最重要的边界条件。
2. **只有 3 个原点、共 126 天**，且三个窗口气候/节日差异很大（o=816 含 2015 年复活节，
   单变量模型在该窗口 WAPE 明显更高）。选参窗口 o=774 恰好是波动较小的 2–3 月，
   因此选出的超参在 o=816 上未必最优；这正是我们**不用测试段调参**所付的代价，同时也是这张表可信的原因。
3. **只抽了 400/1115 家店**（seed 42）。抽样规则固定，但小店的 WAPE 分布更重尾，
   全量 1115 家的绝对值可能略有不同（排名方向预计不变）。
4. **没有做深度模型**。在 400 条日频序列 × 3 原点 × 42 步上从头训练并调 PatchTST/NHITS 一类模型
   需要显著更多算力与调参预算，本轮优先保证「多模型 × 多原点 × 可复核」的闭环，
   因此本表**不覆盖深度模型**，不能据此说「深度模型在 Rossmann 上也不行」。
   仓库里 M4 的深度结果是简化实现，同样不能外推。
5. **没有用官方 Kaggle 测试期**（2015-08-01…2015-09-17，标签不公开）。
   本表用的是 2015-03-28…2015-07-31 的最后 126 天，属于回顾性回测，不是真实盲测。
6. **单点预测、无区间**；UCM 的 MLE 用 `lbfgs` 默认收敛设置，失败时回退 SeasonalNaive
   （本数据 0 失败，`metrics.json` 中 `n_failed_fallback` 均为 0）。
7. 门店级静态属性（StoreType/Assortment/CompetitionDistance）在预测时确实已知，
   但如果门店新开或竞争店搬迁，这些特征会过期；本数据窗口内没有这类事件。

## 9. 复现

```bash
PY=/opt/data/workspace/sales-forecast/.venv/bin/python
cd /opt/data/workspace/sf-dataset2

$PY repro/alt/fetch_rossmann.py      # 下载 + 六项校验（幂等，已有则跳过）
$PY repro/alt/prep_rossmann.py       # 构建 400 店 × 942 天面板
$PY repro/alt/bench_rossmann.py      # 选参 + 3 原点回测（约 10 分钟）
$PY repro/alt/verify_rossmann.py     # 从 npz 独立重算，逐条比对 metrics.json
$PY repro/alt/report_rossmann.py     # 重新生成本文件与榜单/诊断
```

## 10. 产出文件

| 路径 | 内容 |
|---|---|
| `data/alt/rossmann/train.csv`, `store.csv` | 原始数据（38,057,952 B + 45,010 B，已 gitignore） |
| `data/alt/rossmann/validation.json` | 下载校验（字节/行数/日期/门店/零值/缺失率） |
| `data/alt/rossmann/panel.npz` | 建模面板：y + promo/open/school/state + 门店属性 |
| `data/alt/rossmann/panel_meta.json` | 面板元信息与抽样规则 |
| `audit/results/rossmann/metrics.json` | 全部指标 + 选参 + 文件路径（机器可读） |
| `audit/results/rossmann/leaderboard.md`, `.csv` | 榜单 |
| `audit/results/rossmann/verification.json` | 独立重算结果 + 配对 bootstrap |
| `audit/results/rossmann/diagnostics.json` | 分步长误差、闭店日误差占比、开日 WAPE |
| `audit/results/rossmann/preds/*.npz` | 每模型 × 每原点的逐序列预测（preds/ids/dates/truth/per_series_wape） |
| `repro/alt/{fetch,prep,bench,verify,report}_rossmann.py` | 全流程脚本 |
| `tests/test_alt_rossmann.py` | 口径回归测试（零分母报错、clip、切分隔离、npz 与指标一致） |
| `STATUS.txt` | 逐步时间戳与状态 |
