# 销量预测（Sales / Demand Forecasting）近十年文献调研

**范围与用法**：本调研覆盖 2016–2026 年销量/需求预测的代表性文献，共 42 篇，含综述、顶会/顶刊方法论文、竞赛总结与工业实证。文中编号 [n] 与 `papers.json` 中 `p01…p42` 一一对应（[1]=p01）。所有条目的标题、作者、年份、venue 均取自 Crossref / arXiv / OpenAlex / 会议官网，并逐条记录了链接可达性；对出版社反爬导致的 403，核验方式在 `verification_note` 中单独说明。

---

## 一、十年演进脉络（2016–2026）

### 1.1 2016–2017：统计基线仍然统治，工程化封装开始

这一时期的主流仍是 ETS、ARIMA、Theta 与 Croston 类方法。Theta 用"长期趋势线 + 短期波动"的分解在 M3 中击败大量复杂模型 [16]，说明结构恰当的简单模型极难被替代；Croston 及其 SBA 修正 [13][14] 则长期占据慢销品与备件预测的默认位置。2017 年 Prophet 把"分段趋势 + 多周期季节性 + 节假日 + 变点"封装成可调可解释的回归模型，并配套分析师在环的诊断流程 [12]，让业务团队可以自己维护上万条序列的预测，而不必逐条写模型。

这一阶段的共识是：**销量序列的可预测性主要来自日历、促销与趋势结构，而不是模型容量**。该共识在此后十年反复被竞赛验证。

### 1.2 2017–2019：GBDT 把预测改写成表格回归

XGBoost [21] 与 LightGBM [20] 提供了两件关键能力：对缺失值、类别特征与非线性交互的鲁棒处理，以及在千万级样本上的可训练性。于是销量预测的范式从"一条序列一个模型"转为"一张宽表一个全局模型"：每行是 (SKU, 门店, 日期)，特征是 lag、滚动统计、价格与促销标记、日历与静态属性，目标是销量或对数销量。

Kaggle 的 Rossmann、Favorita、Walmart 竞赛把这条路线推到极致：头部方案几乎都是树模型加大量手工特征，配合分门店/分品类建模与目标变换 [7]。2022 年的 M5 竞赛进一步确认，LightGBM 类方案占据排行榜前列 [10]。值得注意的是，同一时期 Makridakis 等人在 M3 的 1,045 条月度序列上发现"裸 ML"全面落后于统计方法 [11]——两者并不矛盾：**赢的不是某个算法，而是"全局学习 + 强特征 + 正确的验证协议"这一整套工程**。

### 1.3 2018–2021：概率深度预测，从逐序列走向全局分布学习

DeepAR 是这一转向的标志 [23]：在大量相关序列上训练一个自回归 RNN，输出负二项/高斯等参数化分布，从而同时获得概率预测、跨序列迁移与冷启动能力。随后出现三条互补的技术路线：

- **分位数路线**：MQ-RNN/MQ-CNN 用直接多步 + 分位数损失绕开分布假设 [24]；SQF-RNN 用单调样条参数化整个分位数函数 [25]。
- **状态空间与因子路线**：Deep State Space 用 RNN 参数化线性高斯状态空间、以 Kalman 滤波算似然 [29]；Deep Factors 用全局深度隐因子 + 局部经典模型兼顾数据效率与不确定性 [26]；低秩高斯 copula 过程则解决数千序列的时变相关问题 [27]。
- **工具化**：GluonTS 把上述模型与统一的回测评估流程打包 [30]，使"深度概率预测"从论文变成可复现的实验基础设施。

这一阶段真正改变实践的是三件事：全局建模替代逐序列建模、分布输出替代点预测、协变量（已知未来输入）成为一等公民。

### 1.4 2019–2023：Transformer、patch 与长跨度预测

TFT 是最贴合零售表结构的深度模型：静态属性、已知未来输入（促销计划、日历）与历史外生序列被分别编码，变量选择网络给出特征重要性，注意力给出时间解释，输出多 horizon 分位数 [31]。与之并行，Informer 用 ProbSparse 注意力把复杂度降到 O(L log L) [35]，Autoformer 把序列分解嵌入网络并用自相关机制替代点式注意力 [36]。

另一条被低估的路线是纯 MLP：N-BEATS 用残差堆叠 + 基函数展开，在 M3/M4/Tourism 上超过统计基准，并比 M4 冠军再好约 3% [32]；N-HiTS 用多速率采样与分层插值，把长跨度误差再降近 20%、计算时间降约 50 倍 [33]。2023 年的 PatchTST 则确立了两个实用默认值：把序列切成 patch 作为 token、通道独立共享权重 [34]。

需要提醒的是，"长期预测基准"（ETT、交通、电力）与零售销量并不相同：前者序列少、周期强、可看很长历史；后者序列极多、噪声大、受促销与缺货干扰。把这些基准上的排名直接迁移到销量预测，是常见的方法论错误。

### 1.5 2023–2026：基础模型与理性回归

TimeGPT [38]、Chronos [39]、TimesFM [40]、Moirai [41] 把"大规模预训练 + 零样本预测"带入时序领域：Chronos 把数值量化成 token 用 T5 训练，在 42 个数据集上零样本接近专用模型；Moirai 在 27B 观测的 LOTSA 上预训练，支持跨频率与任意变量数。对销量预测最直接的价值是新品与长尾品类的冷启动。

但 2024 年 NeurIPS 的消融研究给出了必要的冷水：在三个主流 LLM-based 预测方法中，去掉 LLM 组件或替换成简单注意力层后性能不降反升，预训练 LLM 并不优于从头训练 [42]。结合 M5 的经验，较稳妥的判断是：**基础模型是"低成本冷启动与兜底"的工具，不是替代特征工程与 GBDT 的理由**；在自有数据上仍须以滚动回测比较。

---

## 二、算法家族对比

| 家族 | 代表文献 | 核心优点 | 主要局限 | 适用数据形态 | 计算成本 |
|---|---|---|---|---|---|
| 统计模型（ETS/ARIMA/Theta/Prophet） | [2][12][16] | 可解释、样本效率高、无需 GPU、易做基线 | 难处理多序列交互与复杂促销；逐序列拟合成本随 SKU 数线性增长 | 单序列、历史较长、结构稳定 | 极低（秒级/序列） |
| 间歇需求（Croston/SBA/TSB/贝叶斯） | [5][13][14][15][28] | 专门处理零膨胀与慢销；TSB 可建模需求消亡 | 只看自身历史，难利用价格/促销等外生变量 | 备件、长尾 SKU、低频次销量 | 极低 |
| 层级调和（MinT/时间层级/学习式调和） | [4][17][18][19][22] | 保证聚合一致，通常同时提升各层级精度 | 需维护聚合结构与协方差估计；调和可能掩盖底层模型缺陷 | SKU-品类-门店-区域、日-周-月 | 低（矩阵运算） |
| GBDT（XGBoost/LightGBM） | [7][10][20][21] | 表格数据最强性价比；特征重要性可解释；训练快 | 需手工构造 lag/滚动特征；外推能力弱；对分布漂移敏感 | 多 SKU 宽表、含价格促销与日历 | 中（分钟–小时，CPU 可完成） |
| 深度概率（DeepAR/状态空间/因子/copula） | [23][24][25][26][27][29][30] | 原生分布输出；跨序列共享；支持协变量与冷启动 | 训练不稳、调参成本高；对数据量与序列相关性有要求 | 大量相关序列、需要 P50/P90 | 高（GPU 小时级） |
| 深度 Transformer（TFT/Informer/Autoformer/PatchTST） | [31][34][35][36] | 长依赖与多变量建模；TFT 自带可解释性 | 参数多、易过拟合；对缺失与促销脉冲不稳健 | 中长历史、多外生变量、层级较少 | 高 |
| 深度 MLP（N-BEATS/N-HiTS） | [32][33] | 结构简单、训练快；长跨度精度好 | 单变量为主，外生变量支持弱 | 单序列或通道独立的多序列 | 中高 |
| 基础模型（TimeGPT/Chronos/TimesFM/Moirai） | [38][39][40][41][42] | 零样本、无需训练、天然概率输出 | 领域偏差不可控；批量推理成本高；可解释性差 | 冷启动、长尾、快速试点 | 推理中高，训练不可行 |
| 混合/组合 | [9][22][32] | 通常最稳：统计保底 + ML 抓非线性 | pipeline 复杂、维护成本高 | 大多数真实业务 | 中 |

**成本量级**：单序列 ETS/Theta 为毫秒–秒；全量 LightGBM（百万行宽表）为分钟–小时（CPU）；DeepAR/TFT 为 GPU 小时级；基础模型无需训练但批量推理成本随 SKU 数线性上升。选型时应先问"预测频率 × SKU 数 × horizon"三件事，再谈模型。

---

## 三、M4 / M5 / Kaggle 给出的经验结论

**M4（2020，10 万条序列、61 种方法）** [9]：冠军是统计与神经网络的混合方案（ES-RNN 思路），纯机器学习单模型整体未占优；组合预测、多步策略与季节调整的收益大于换模型；并把预测区间纳入统一评估，说明"点预测准"并不等于"决策可用"。此前的 M3 复盘 [11] 进一步指出，ML 的劣势主要来自无法外推趋势与缺乏序列级结构，而非模型不够复杂。

**M5（2022，Walmart 42,840 条层级日销量）** [10]：这是最接近真实零售课题的公开竞赛。结论可以概括为四条：其一，排行榜前列几乎清一色是 LightGBM 类方案，模型结构的差异远小于特征与验证策略的差异；其二，价格、促销、日历（含节假日与 SNAP）与滞后/滚动特征决定上限；其三，层级结构必须显式处理，但是否做 reconciliation、在哪一层做，收益因方案而异；其四，WAPE 这类"按销量加权"的指标与业务损失更对齐，避免了低销量 SKU 主导平均误差。

**Kaggle 零售竞赛** [7]：Rossmann、Favorita、Walmart 的获胜方案共同点是——大量滞后与滚动特征、按门店/品类分组建模、目标变换（log 或 Tweedie）、对缺失与零销量的显式处理，以及谨慎的线下验证设计。Favorita 亚军则用膨胀因果卷积（WaveNet 风格）证明 CNN 也能进入第一梯队 [37]。

**五条可迁移结论**：
1. 没有"最强模型"，只有"最强基线 + 最强特征 + 最贴近业务的验证"。
2. 组合/混合可能提升，必须通过独立回测确认，但维护成本高。
3. 评估指标必须与决策对齐（补货成本、缺货惩罚），根据成本选择 WAPE/RMSSE 等指标；不存在普遍优于其它指标的排序。
4. 层级一致性是硬约束，不是精度技巧。
5. 竞赛中的外部数据与后处理往往不可复现，工业落地要打折看待。

---

## 四、对「销量预测课题」的可执行建议

### 4.1 数据与特征工程优先级

- **P0（没有它们就不要谈模型）**：日历特征（星期、月、节假日、发薪日、促销档期）、价格与实际成交价、促销类型与折扣深度、缺货/断货标记、lag（1/7/14/28/364）与对应滚动统计（mean/std/min/max，务必先 shift 再 rolling）。
- **P1**：层级聚合特征（品类/门店/区域的同期销量与占比）、静态属性（品类、品牌、包装、生命周期阶段）、天气与节假日邻近效应。
- **P2**：交叉特征（价格×促销、品类×星期）、竞品价格、库存与履约约束。
- **目标处理**：零膨胀明显时用 Tweedie 或 Poisson 目标；右偏严重时用 log1p，但要在原尺度评估。缺失与零销量必须区分"真零"与"未观测"。
- **泄漏检查**：所有滚动特征必须只使用预测时点之前的信息；促销计划若在预测时点未知，不能进入特征。

### 4.2 评估协议与指标

- **切分**：滚动原点回测（rolling origin），确保所有训练标签截止于预测原点；只有标签窗口重叠或业务信息延迟才需要额外 gap；holdout 至少覆盖一个完整年度周期与一次大促。
- **主指标**：WAPE（Σ|y-ŷ|/Σy），按销量加权；M5 accuracy 的官方指标是 WRMSSE，并非 WAPE；辅以 bias（系统性高估/低估）与 MASE（跨序列可比）。
- **概率指标**：pinball loss / WQL，用于评估 P50/P90 是否可信；补货场景应看分位数覆盖率而非仅看均值。
- **分桶报告**：按 ADI/CV² 四象限 [14] 区分快销/慢销/间歇/块状，分别报告；新品单列 cold-start 桶。平均值会掩盖慢销品的灾难性误差。

### 4.3 基线阶梯（建议按序建立，不要跳级）

1. Seasonal Naive（周同日/年同日）——最低门槛。
2. ETS/Theta（逐序列）[16] 与 Prophet [12]（含节假日）。
3. 慢销品：Croston/SBA/TSB [13][14][15]。
4. 全局 LightGBM 直接多步（含上述 P0 特征）[20][21]——大多数课题的真实强基线。
5. 深度模型：数据量足够时上 DeepAR/TFT [23][31]；长跨度可试 N-HiTS [33]；需要概率区间时优先分位数损失 [24]。
6. 基础模型零样本 [39][40] 仅作为冷启动对照，不替代 4。

### 4.4 层级与决策耦合

先确定"预测粒度 ≠ 考核粒度"的事实：日粒度建模、周/月粒度考核时，用时间层级调和 [17]；SKU-品类-门店存在聚合约束时，优先 MinT [19]，关系随促销变化时考虑学习式调和 [22]。库存决策应使用预测分布而非点预测，安全库存由分位数决定。

### 4.5 常见坑

- 用随机切分代替时间切分，导致滞后特征泄漏。
- 用 MAPE 评估长尾 SKU，误差被低销量放大。
- 只报告总体 WAPE，不报告分桶与新品。
- 把长期预测基准（ETT/交通）的 SOTA 直接当作销量预测的 SOTA。
- 忽略缺货造成的"假零销量"，把供给约束误学成需求信号。

---

## 五、参考文献

[1] Lim, B., Zohren, S. (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*, 379(2194). https://arxiv.org/abs/2004.13408
[2] Petropoulos, F., Apiletti, D., Assimakopoulos, V., et al. (2022). Forecasting: theory and practice. *International Journal of Forecasting*, 38(3), 705–871. https://arxiv.org/abs/2012.03854
[3] Fildes, R., Ma, S., Kolassa, S. (2022). Retail forecasting: Research and practice. *International Journal of Forecasting*, 38(4), 1283–1318. https://doi.org/10.1016/j.ijforecast.2019.06.004
[4] Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., Panagiotelis, A. (2024). Forecast reconciliation: A review. *International Journal of Forecasting*, 40(2), 430–456. https://doi.org/10.1016/j.ijforecast.2023.10.010
[5] Pinçe, Ç., Turrini, L., Meissner, J. (2021). Intermittent demand forecasting for spare parts: A critical review. *Omega*, 105, 102513. https://doi.org/10.1016/j.omega.2021.102513
[6] Giannopoulos, P. G., Dasaklis, T. K., Tsantilis, I., Patsakis, C. (2025). Machine learning algorithms in intermittent demand forecasting: a review. *International Journal of Production Research*. https://doi.org/10.1080/00207543.2025.2578701
[7] Bojer, C. S., Meldgaard, J. P. (2021). Kaggle forecasting competitions: An overlooked learning opportunity. *International Journal of Forecasting*, 37(2), 587–603. https://doi.org/10.1016/j.ijforecast.2020.07.007
[8] Januschowski, T., Gasthaus, J., Wang, Y., et al. (2020). Criteria for classifying forecasting methods. *International Journal of Forecasting*, 36(1), 167–177. https://doi.org/10.1016/j.ijforecast.2019.05.008
[9] Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014
[10] Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
[11] Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3), e0194889. https://doi.org/10.1371/journal.pone.0194889
[12] Taylor, S. J., Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37–45. https://doi.org/10.1080/00031305.2017.1380080
[13] Croston, J. D. (1972). Forecasting and Stock Control for Intermittent Demands. *Journal of the Operational Research Society*, 23(3), 289–303. https://doi.org/10.1057/jors.1972.50
[14] Syntetos, A. A., Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303–314. https://doi.org/10.1016/j.ijforecast.2004.10.001
[15] Teunter, R. H., Syntetos, A. A., Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 215(1), 101–112. https://doi.org/10.1016/j.ejor.2011.05.018
[16] Assimakopoulos, V., Nikolopoulos, K. (2000). The theta model: a decomposition approach to forecasting. *International Journal of Forecasting*, 16(4), 521–530. https://doi.org/10.1016/S0169-2070(00)00066-2
[17] Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., Petropoulos, F. (2017). Forecasting with temporal hierarchies. *European Journal of Operational Research*, 262(1), 60–74. https://doi.org/10.1016/j.ejor.2017.02.046
[18] Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., Shang, H. L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis*, 55(9), 2579–2589. https://doi.org/10.1016/j.csda.2011.03.006
[19] Wickramasuriya, S. L., Athanasopoulos, G., Hyndman, R. J. (2019). Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization. *Journal of the American Statistical Association*, 114(526), 804–819. https://doi.org/10.1080/01621459.2018.1448825
[20] Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*. https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
[21] Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785
[22] Spiliotis, E., Abolghasemi, M., Hyndman, R. J., Petropoulos, F., Assimakopoulos, V. (2021). Hierarchical forecast reconciliation with machine learning. *Applied Soft Computing*, 112, 107756. https://doi.org/10.1016/j.asoc.2021.107756
[23] Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191. https://arxiv.org/abs/1704.04110
[24] Wen, R., Torkkola, K., Narayanaswamy, B., Madeka, D. (2017). A Multi-Horizon Quantile Recurrent Forecaster. *NeurIPS 2017 Time Series Workshop*. https://arxiv.org/abs/1711.11053
[25] Gasthaus, J., Benidis, K., Wang, Y., et al. (2019). Probabilistic Forecasting with Spline Quantile Function RNNs. *AISTATS 2019, PMLR 89*. https://proceedings.mlr.press/v89/gasthaus19a.html
[26] Wang, Y., Smola, A., Maddix, D. C., Gasthaus, J., Foster, D., Januschowski, T. (2019). Deep Factors for Forecasting. *ICML 2019, PMLR 97*. https://arxiv.org/abs/1905.12417
[27] Salinas, D., Bohlke-Schneider, M., Callot, L., Medico, R., Gasthaus, J. (2019). High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes. *NeurIPS 2019*. https://arxiv.org/abs/1910.03002
[28] Seeger, M. W., Salinas, D., Flunkert, V. (2016). Bayesian Intermittent Demand Forecasting for Large Inventories. *NeurIPS 2016*. https://papers.nips.cc/paper_files/paper/2016/hash/03255088ed63354a54e0e5ed957e9008-Abstract.html
[29] Rangapuram, S. S., Seeger, M. W., Gasthaus, J., Stella, L., Wang, Y., Januschowski, T. (2018). Deep State Space Models for Time Series Forecasting. *NeurIPS 2018*. https://papers.nips.cc/paper_files/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
[30] Alexandrov, A., Benidis, K., Bohlke-Schneider, M., et al. (2020). GluonTS: Probabilistic Time Series Models in Python. *Journal of Machine Learning Research*, 21(116). https://arxiv.org/abs/1906.05264
[31] Lim, B., Arık, S. Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. https://arxiv.org/abs/1912.09363
[32] Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*. https://arxiv.org/abs/1905.10437
[33] Challu, C., Olivares, K. G., Oreshkin, B. N., et al. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI 2023*. https://arxiv.org/abs/2201.12886
[34] Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730
[35] Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021*. https://arxiv.org/abs/2012.07436
[36] Wu, H., Xu, J., Wang, J., Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS 2021*. https://arxiv.org/abs/2106.13008
[37] Kechyn, G., Yu, L., Zang, Y., Kechyn, S. (2018). Sales forecasting using WaveNet within the framework of the Kaggle competition. *arXiv:1803.04037*. https://arxiv.org/abs/1803.04037
[38] Garza, A., Challu, C., Mergenthaler-Canseco, M. (2023). TimeGPT-1. *arXiv:2310.03589*. https://arxiv.org/abs/2310.03589
[39] Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). Chronos: Learning the Language of Time Series. *TMLR*. https://arxiv.org/abs/2403.07815
[40] Das, A., Kong, W., Sen, R., Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688
[41] Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., Sahoo, D. (2024). Unified Training of Universal Time Series Forecasting Transformers. *ICML 2024*. https://arxiv.org/abs/2402.02592
[42] Tan, M., Merrill, M. A., Gupta, V., Althoff, T., Hartvigsen, T. (2024). Are Language Models Actually Useful for Time Series Forecasting? *NeurIPS 2024*. https://arxiv.org/abs/2406.16964
