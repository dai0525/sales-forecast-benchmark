# 销量预测 Benchmark 与算法实验台

在 **M4 Weekly / Daily** 与 **Rossmann Store Sales** 上比较统计、全局 LightGBM、简化深度模型和状态空间模型。M4 包含多个领域，并不全是零售销量；Rossmann 带促销/假日等外生因子。本仓库附算法教学网页。

本次审计以提交 `83f3947` 为起点，修正数据污染、验证泄漏、测试集选参、模型命名和导出问题。原报告与指标保存在 `audit/original/`；完整问题清单见 [AUDIT.md](AUDIT.md)，技术分析、论文与开源项目见 [REPORT-MPA.md](REPORT-MPA.md)。

## 核验后的 Weekly 结果

359 条序列，最近 200 点训练，官方测试 13 步；主指标是每条序列 WAPE 的平均值。原始训练长度不足 200 的 65 条沿用原协议首值左填充。下表由保存的全量预测与 ID 重算，来源为 [机器可读结果](audit/results/benchmark.json)。

| 模型 | 平均 WAPE | 总量 WAPE |
|---|---:|---:|
| **联合状态 Fourier（本次改进）** | **7.313%** | **5.476%** |
| UCM52（MLE，原称 BSTS） | 7.840% | 5.565% |
| LightGBM（修正训练样本） | 7.960% | 5.713% |
| BSTS MCMC†（原版 R 包） | 7.981% | 5.734% |
| PatchTST | 8.132% | 5.989% |
| 原 FG backfitting | 8.706% | 5.980% |
| Naive | 9.001% | 6.340% |
| NHITS-inspired pooled residual MLP | 9.360% | 6.713% |
| TFT-style 中位数模型 | 10.463% | 8.311% |
| MovingAvg8 | 10.836% | 7.840% |
| N-BEATS-style（无外生变量） | 11.864% | 8.335% |
| 直接多步 Gaussian LSTM | 13.600% | 10.422% |
| SeasonalNaive52 | 15.553% | 13.249% |

† BSTS 使用两条短链，359 条均完成，但没有序列的全部监控参数通过严格收敛阈值；这是有限预算的探索性成绩，不能作为 BSTS 能力上限。详见 [采样诊断与结果](BSTS-MCMC.md)。

原 FG 和 UCM 的主要数字从头复现一致。旧 LightGBM 和深度榜单缺完整预测，不能逐点证实；上表是修正后重新训练的结果。深度模型使用 260 步固定预算、时间验证，保留最佳检查点，没有用验证目标梯度重新拟合；它们多为简化实现，不代表同名论文的最佳水平。各模型调参预算也不相等。

**新模型在官方测试段有提升，但没有稳定胜过 UCM。** 三个嵌套历史窗口中只赢一个；平均 WAPE 为 7.864%，UCM 为 7.670%。详见 [历史回测](audit/results/rolling_summary.md)。公开测试已被原项目反复查看，本次是探索性验证，不能称全新盲测。

## 第二个数据集：Rossmann Store Sales（带外生因子）

M4 Weekly 是纯单变量、无协变量、季节性极强的数据；为了检验「这些算法是不是只对这份数据有效」，
增加了 **Rossmann**（Kaggle 竞赛，德国药妆店日销量）：400 家完整日历门店抽样，H=42 天，
3 个扩张窗原点（o=816/858/900），超参只在更早的内部验证原点 o=774 选择。
因子：`Promo`（38.2% 的天）、`StateHoliday`、`SchoolHoliday`、`StoreType`、`Assortment`、`CompetitionDistance`。
详见 [ALT-DATASET.md](ALT-DATASET.md)。

| 排名 | 模型 | 平均 WAPE | 总量 WAPE |
|---:|---|---:|---:|
| 1 | **LightGBM 直接多步（含促销/假日/门店）** | **9.34%** | 9.20% |
| 2 | LightGBM 仅日历（消融） | 19.86% | 19.62% |
| 3 | SeasonalNaive7 + 营业开关 | 23.22% | 22.93% |
| 4 | State-Fourier（周期 7） | 25.64% | 25.31% |
| 5 | UCM（local level + 季节，周期 7） | 25.97% | 25.63% |
| 6 | SeasonalNaive（周期 7） | 28.72% | 28.31% |
| 7 | MovingAvg（窗口 56） | 43.01% | 42.37% |

**结论（与 M4 对照）**：

1. **赢家换人、量级反转**：LightGBM 在 M4 排第 3（落后榜首 0.65 pp），在 Rossmann 排第 1
   （领先 16.3 pp，1,200 个配对中 99.4% 更好，bootstrap 95% CI [-16.6, -16.0] pp）。
2. **优势来自「有没有可用因子」，不是模型族**：同一套 LightGBM 去掉 promo/营业开关/节假日/门店属性后
   从 9.34% 退化到 19.86%（10.5 pp），逐原点在 11.7%–25.2% 摆动；这个损失是从
   SeasonalNaive 换成最好的状态空间模型所能拿到的 3.1 pp 的 3.4 倍。
3. **单变量族内部次序大体保持**（5 个同名模型 Spearman 秩相关 0.60）：UCM 与 State-Fourier
   仍是最好的单变量模型，且两者在 Rossmann 上只差 0.32 pp、只有 50.6% 的序列新模型更好，
   o=900 窗口 UCM 反超——**不要因为 M4 上的 0.53 pp 就替换 UCM**。
4. **零膨胀解释约四成**：闭店日占 18.2%，状态模型约 25% 的误差落在闭店日而 LightGBM 只有 1.7%；
   剔除闭店日后差距仍有 9.9 pp。**跨数据集只能比方向和量级，绝对 WAPE 不可横比**
   （M4 7–16%，Rossmann 9–43%）。

## 研究结论

原 FG 是条件 Kalman 平滑与启发式交替拟合，不是完整联合 BP 或严格 EM。BP 与 Kalman 的等价要求相同的链式线性高斯模型、参数和先验，不能据此推导不同预测模型“精度不可能提高”。

新实现使用独立季节系数、阻尼趋势及完整联合协方差更新，属于成熟状态空间与 Fourier 季节方法的工程改进。它证明原实现仍有改进空间，但不构成新的消息传播理论。仅凭总销量无法唯一识别真实促销、价格等因果贡献。下一阶段应在带日期、价格、库存和可预知促销的真实零售数据上，与 UCM 和全局 LightGBM 做多起点对照，再判断是否需要商品关系图。

## Daily 核验范围

已按官方测试 CSV 与 ID 重算已存预测。1000 条上 Naive 3.4056%、AutoETS 3.4324%、Theta 3.4658%、当前 LightGBM log1p 3.6959%。AutoARIMA 的 2.3579% 只涉及 250 条，同样 250 条的 Naive 为 2.3850%，不能直接与 1000 条榜单比较。

重新下载训练 CSV 后，抽样 ID 和 Naive / SeasonalNaive 预测吻合。其它 Daily 统计模型本次未全部重新训练；Daily 深度结果缺全量预测，仍属未独立复现的历史记录。详见 [原结果核验](audit/results/saved_metrics.json)。

## 复现

推荐独立 Python 3.12 环境；实际依赖版本见 `requirements-audit.txt` 与 `audit/results/environment.json`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-audit.txt

# 仓库已包含本次 Weekly 数据与保存预测；缺原始 CSV 时下载并预处理
python repro/fetch_m4.py --only Weekly
python repro/prep_weekly.py

# 快速复算保存成绩、运行新模型和检查
python repro/audit_saved.py
OPENBLAS_NUM_THREADS=1 python repro/audit_experiment.py
python repro/build_audit_report.py
pytest -q

# 从头重训 Weekly 基线、五个深度模型、新模型及嵌套回测（CPU 需数分钟以上）
bash repro/run_audit.sh

# 浏览工作台
python -m http.server 8899 --directory web
```

完整历史 Daily 模型依赖另见 `requirements.txt`。Daily 下载用 `python repro/fetch_m4.py --only Daily`；Daily 导出默认在 `web/data/daily/`，避免覆盖 Weekly。禁用 `postdeploy.py` 的旧硬编码发布流程。

## 文件入口与指标

- `repro/fg_state.py`：新联合状态模型；`audit_experiment.py`：训练内选参、消融和配对 bootstrap；`audit_rolling.py`：嵌套历史回测。
- `repro/fg_mp.py`：保留原算法作数值回归；`bsts52.py`：UCM52；`bench_base.py`：Weekly 基线与 LightGBM；`bench_deep.py`：Weekly 深度训练。
- `repro/build_audit_report.py`：从保存预测生成榜单，更新 `web/data/bench.json`；`audit/results/`：协议、预测、指标、环境、文献核验和浏览器截图。
- `repro/alt/`：第二数据集（Rossmann）的下载/校验、预处理、基准、独立重算与网页导出；报告见 [ALT-DATASET.md](ALT-DATASET.md)。
- `tests/test_audit.py`：高斯条件分布独立校验、数据隔离、时间切分等回归测试；`tests/test_alt_rossmann.py`：第二数据集口径测试。
- `web/`：静态教学工作台；滑块/注意力图不等同于重新训练或因果解释。教学页旧榜单已按审计重跑更新，深度模型标注为简化实现。

单条 WAPE = Σ|y−ŷ| / Σ|y|；平均 WAPE 对每条序列等权，总量 WAPE 将所有分子与分母先汇总。后者不是净总量偏差。零分母在新审计评估器中报错，不静默丢弃。不同数据子集、预测步长和指标口径不可直接排名。本表不使用 M4 官方 OWA，因此不能与官方 OWA 榜单直接比较。

数据来源：[M4 Competition](https://github.com/Mcompetitions/M4-methods)。License：MIT。

## 补充消融：消息传播到底有没有用？

已完成固定结构/参数的跨成分协方差消融，以及等价精度更新对照。完整更新 7.313%，切断跨成分协方差 11.185%，重新选参 9.319%；等价求解器结果一致。证据支持保留成分间后验依赖，但不支持“MPA 胜过 Kalman”。详见 [消融报告](MESSAGE-ABLATION.md)。

第二轮进一步固定同一概率模型，对比一次分离、反复反馈、两种更新顺序、精确批量求解和历史验证早停，见 [迭代消融报告](MESSAGE-ITERATIONS.md)。充分迭代改善一次分离，但收敛后与 Kalman 同解。

## 通信式 MPA 阶段性结项

已补做真正的有限码本 sum-product、稀疏通信正对照、同模型枚举、四窗口回测、6,561状态容量检查与因子分组诊断。当前销量迁移路线未显示稳定收益，建议暂停；不把结论外推为所有图模型无效。详见 [通信MPA结项报告](COMMUNICATION-MPA.md)。补充实验中的连续先验收缩结果属于探索性发现，原排行榜保持不变。

## 原创性核查与原版 BSTS

已核对 DHR、TBATS、BSTS 官方组件及直接相关论文。State-Fourier 可通过状态变换写成已有模型，现阶段应定位工程/实证贡献，不能宣称新模型族。详见 [先行研究与论文定位](PRIOR-ART-STATE-FOURIER.md)。

新增原作者 R `bsts` 包的真正 MCMC 基线，包含参数后验采样、链诊断、全量预测和固定子集的原论文风格季节结构。配置及有限采样预算的限制见 [BSTS-MCMC.md](BSTS-MCMC.md)；R 环境由 `environment-bsts.yml` 定义，入口为 `repro/bench_bsts_mcmc.py`。它和之前仅做 MLE 的 UCM52 是不同实验。

后续补充：[BSTS 收敛与先验敏感性实验](BSTS-SENSITIVITY.md)，在固定12条上比较采样预算和先验强度，结果与全量榜单分开保存。
