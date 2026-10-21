# 扩充真实零售数据：来源核验与评测建议

日期：2026-10-21。此次完成数据调研、下载和结构审计，**没有训练新模型，也没有得到新准确率结果**。暂停动态专家组合研究，优先检验简单模型是否在不同业务上稳定有效。

## 为什么现有证据不足

M4 是跨领域预测数据，并不全是零售销量；Rossmann 来自同一连锁，目标是销售额。新增 534 家门店仍属于同一数据来源、同一评估日期，不能当作跨零售商或未来时间的独立验证。需要增加业务来源、销量粒度、预测周期和缺货情景，而不仅是序列数量。

## 推荐顺序

| 优先级 | 数据与官方来源 | 补足的能力 | 主要限制 | 本次核验状态 |
|---|---|---|---|---|
| 1 | [FreshRetailNet-50K，叮咚](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) | 中国生鲜、商品日销量、折扣/活动/天气/缺货，多因子 | 历史只有 90 天；销量经统一缩放；真实缺货期间潜在需求不可观测 | 两个原始 Parquet 已下载、固定版本并核验 SHA256，完成结构审计 |
| 1 | [VN1，Flieber / SupChains](https://supchains.com/vn1-forecasting-competition/) | 多商家、多仓库、周销量、13 周长预测 | 历史缺货未标记，评估样本筛除了缺货商品；解释因子较少 | 主办方压缩包已下载并核验 CSV 结构 |
| 2 | [Favorita Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) | 厄瓜多尔商超、多年日销量、促销/节假日/门店/油价 | 品类聚合粒度；未来真实客流、油价不能直接用 | 已核验来源与字段说明，尚未下载审计；Kaggle 下载可能需账号及接受规则 |
| 2 | [Walmart M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) | 长历史、SKU—门店、间歇销量及层级预测 | 没有完整库存/缺货或丰富促销机制记录 | 已核验来源，尚未下载审计 |
| 3 | [Dominick's 原始扫描数据](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) | 周销量、价格、促销代码、产品和门店属性 | 年代较早；官网限定学术研究用途 | 已核验官方字段与下载入口，尚未下载审计 |
| 3 | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 英国线上交易及批发客户，2009—2011 年，跨业务补充 | 退货/取消和缺失需处理，没有丰富促销或库存字段 | 已核验官方来源，尚未下载审计 |

M5 的背景与业务适用性见 [竞赛论文](https://doi.org/10.1016/j.ijforecast.2021.11.013) 和 [Walmart 适用性讨论](https://doi.org/10.1016/j.ijforecast.2021.06.002)。不把竞赛排名直接等同于业务收益。

## 已实际核验的两个数据集

### FreshRetailNet-50K

固定 Hugging Face revision：`c374e85975c82816c495acfa904cc55ab14a47d3`。原始下载约 115 MB，原文件不提交 Git；许可 CC BY 4.0，使用应引用 [原论文](https://arxiv.org/abs/2505.16319)，并参考 [作者基线](https://github.com/Dingdong-Inc/frn-50k-baseline)。

实际文件统计：

- 训练：4,500,000 行，2024-03-28 至 2024-06-25，每条序列 90 天。
- 评估：350,000 行，2024-06-26 至 2024-07-02，每条序列 7 天。
- 两集合各 50,000 个门店—商品组合、898 家门店、865 个商品、18 个城市。
- 无重复门店—商品—日期键；两集合键交集为零；已检查的标量列无空值。没有据此宣称数组内部、异常值或业务语义已全部审计。

`sale_amount` 的官方定义是统一缩放的销量，不是销售额，也不能直接视为原始件数。记录含折扣、活动标记、假日标记、降水、温度、湿度、风力和缺货状态，满足研究多个业务变量的需要；这些变量可能相关，不等于八个独立因果因素。

预测时可用性必须分开：日历已知；折扣与活动只有在预测起点已确定计划时可用；未来天气实测值只能用于明确标记的理想信息对照，实际应用用当时天气预报或历史值；未来实际缺货和小时销量不可作为同日销量预测输入。历史缺货可用于训练中的删失处理。真实缺货时丢失需求没有直接标签，不能用模型填补值验证模型自己的需求恢复精度。

复现：`python -m pip install pandas pyarrow`，然后 `python repro/audit_freshretail.py`。详见 [结构审计 JSON](audit/results/dataset_expansion/freshretail_structure.json)。

### VN1

使用主办方 [原始压缩包](https://supchains.com/downloads/vn1-forecasting-competition.zip)，下载约 8.5 MB，SHA256 固定在脚本。实际文件为 15,053 个 Client—Warehouse—Product 组合，46 个商家、328 个仓库、11,171 个商品。

Phase 0 为 2020-07-06 至 2023-10-02，共 170 周；Phase 1 为 2023-10-09 至 2024-01-01，共 13 周；Phase 2 为 2024-01-08 至 2024-04-01，共 13 周。五个销量/价格文件的 ID 和顺序一致、ID 无重复；销量单元格无空值。

历史价格是实际成交价格，未成交周为空，不是可提前知道的挂牌价。Phase 0 价格有 1,836,437 个空值；不可填成零价格。压缩包没有 Phase 2 价格或库存文件。主办方确认历史缺货未标记、评估期间筛除了缺货商品，因此应披露样本选择限制。其他镜像若声称含库存，必须另外核验，不能混用。

官方评价为 `(绝对误差之和 + 总误差的绝对值) / 总销量`，不同于仓库旧榜单的逐序列平均 WAPE。后续同时报告官方分数、总体 WAPE、Bias 和按商家分布，不直接跨指标比较数值大小。主办方要求使用时引用比赛；本次不重新分发原数据。

复现：`python -m pip install pandas`，然后 `python repro/audit_vn1.py`。详见 [结构审计](audit/results/dataset_expansion/vn1_structure.json) 和 [下载清单](audit/results/dataset_expansion/vn1_download.json)。未执行压缩包内代码或计算模型测试分数。

## 避免“数据集变多，但证据没有变强”

1. Favorita 的 SKU 版与品类版来自同一零售商；M5 Accuracy 与 Uncertainty 是同一底层数据；UCI Online Retail 与 II 有时间重叠。不能分别当作独立业务来源。
2. [dunnhumby 官方页](https://www.dunnhumby.com/source-files/) 对 Source Files 使用 representation / inspired 的描述；Complete Journey 是选定家庭的购买面板，不能当成完整门店销量；Let's Get Sort-of-Real 明确是模拟数据。适合辅助研究，不作为真实跨业务有效性的核心证据。
3. [Monash 的 Dominick 下载版本](https://zenodo.org/records/4004172) 描述目标为利润。研究销量应读取 Chicago Booth 原始 movement 文件的销量字段，不可只按数据集名称判断目标。

## 下一轮实验协议建议（尚未运行）

- 首轮使用 FreshRetailNet + VN1，随后加入 Favorita/M5。保留 Rossmann 做历史对照，M4 作为通用时序附录。
- 比较季节朴素预测、单一 LightGBM、State Fourier（无外因和带外因两版）。不继续增加动态专家复杂度；先验证基本模型在新业务上的表现。
- 因子消融按日历、价格/促销、天气、历史缺货分组；所有模型必须使用同一信息可用性规则和时间切分。因子归因属于预测关联，不直接声称因果效应。
- FreshRetailNet 在训练 90 天内部做时间验证，冻结参数后只评估官方 7 天一次；不足以证明年度季节性。VN1 使用 Phase 0 内部回测和 Phase 1 开发，冻结后评估 Phase 2。已经公开且被人看过的结果不能宣称完全未知的盲测。
- 不根据测试销量筛选“好预测”的 SKU。抽样若必要，只使用训练期信息和固定随机种子；报告抽样覆盖及与全量的差异。
- 同时报总体与逐序列/逐商家指标、偏差、零销量情景、训练/预测时间。零分母的 WAPE/MASE 明确处理并报告数量；不要默默丢掉。不同数据集不混成一个不透明平均分。
- 后续创新优先研究“缺货导致销量低估时，简单可解释模型能否改善预测”，但删失需求建模已有大量前人工作；此次数据调研并未证明方法创新或改进效果。

可发布的现阶段结论是：找到了两个已下载且可复现的新业务来源，能够显著扩展验证覆盖；现有模型是否更准确，仍需按上述协议实验后判断。
