# State-Fourier 先行研究核验与论文定位

核验日期：2026-10-18。对象为 `repro/fg_state.py`，不是后来另行测试的离散通信 MPA。

**结论：已有论文和官方实现覆盖了这一模型的核心结构及推断方法。State-Fourier 是成熟结构时序/动态谐波回归家族的一个受约束、批量化实现，不能因没有搜到相同名字就宣称原创算法。可以撰写并公开研究报告；若投稿论文，贡献应定位于可复现的经验研究、计算实现或新增的业务建模，而非“首次提出趋势＋Fourier＋消息传播”。目前证据不足以承诺方法论文的新颖性或录用。**

## 1. 检索范围与证据等级

实际进行了组合关键词检索，包括 `damped trend Fourier Kalman forecasting`、`Dynamic harmonic regression Damped Trend`、`Fourier semilocal forecasting`、共享协方差与多序列预测，以及沿 DHR、TBATS、BSTS 引用追踪。访问了作者论文、期刊/大学页面、CRAN 与 statsmodels 官方文档，并核对源代码链接。

主要来源、访问结果、内容哈希与核验范围见 `audit/results/prior_art_sources.json`。这是一轮针对模型结构的定向文献审查，不是系统综述，也不保证覆盖全部论文或未公开工作。DHR 1999 的作者页面及期刊摘要可核验；全文旧站访问失败，搜索索引可见其方程 (2c) 和 damped-trend 段落，报告不把这当作完整阅读全文。TBATS 作者工作论文的相关模型章节、BSTS 原始论文相关章节和官方组件公式已直接打开核对。

## 2. 最直接的先行工作

| 先行工作 | 已核验证据 | 与当前实现的关系与区别 |
|---|---|---|
| Young, Pedregal & Tych (1999), *Dynamic harmonic regression*, Journal of Forecasting 18(6):369–394 | [作者大学记录](https://research.lancaster-university.uk/en/publications/dynamic-harmonic-regression/)；[期刊 DOI](https://doi.org/10.1002/%28SICI%291099-131X%28199911%2918%3A6%3C369%3A%3AAID-FOR748%3E3.0.CO%3B2-K)。状态空间中的谐波分解、Kalman 滤波/平滑、噪声方差比选择已有明确先例；全文索引中广义随机游走包含阻尼趋势 | 直接模型家族先例。原文允许时变谐波系数，并采用频域准则选参数；我们用静态系数与历史预测误差选全局参数，不是逐行复刻其完整算法 |
| De Livera, Hyndman & Snyder (2011), *Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing*, JASA 106(496):1513–1527 | [正式 DOI](https://doi.org/10.1198/jasa.2011.tm09771)；[作者工作论文](https://robjhyndman.com/papers/ComplexSeasonality.pdf)，§2.2–2.3，方程 (3)–(4) | TBATS 已把阻尼趋势和 Fourier/三角季节放入状态空间。文中也说明季节更新参数为零时得到确定性季节。**但 TBATS 是单源 innovations 形式，我们使用独立观测/水平/斜率噪声，不能直接称二者完全相同** |
| Scott & Varian (2014), *Predicting the Present with Bayesian Structural Time Series* | [出版信息](https://doi.org/10.1504/IJMMNO.2014.059942)；[作者版全文](https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf)，§3.1、§4 | 模块化趋势/季节/回归状态，状态采样与参数后验学习。当前代码只在给定超参下精确更新状态，不积分噪声参数后验，不能称完整 BSTS |
| 原作者 `bsts` 的半局部趋势和三角组件 | [AddSemilocalLinearTrend](https://search.r-project.org/CRAN/refmans/bsts/html/add.semilocal.linear.trend.html) 的两条状态方程；[AddTrig](https://search.r-project.org/CRAN/refmans/bsts/html/add.trig.html) 的 Harmonic/Direct 两节 | **最直接的代数对应**：零长期斜率的半局部趋势，经状态缩放对应我们的趋势；三角系数演化噪声为零对应我们的静态 Fourier。默认 AddTrig 是随机季节、BSTS 默认还估计未知参数，所以默认拟合结果不相同 |
| statsmodels 的频域季节结构模型 | [UnobservedComponents 官方公式](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html)，Frequency-domain Seasonal | 可选谐波数和 `stochastic_freq_seasonal=False` 已直接提供静态谐波结构。默认 UCM 趋势不是我们这个阻尼趋势；不能把默认 UCM 和当前模型当作同一估计量 |

相邻先例也需要引用，但不作为“完全相同模型”的证据：

- [FPP3 §10.5 动态谐波回归](https://otexts.com/fpp3/dhr.html)：Fourier 回归加 ARMA 误差，直接适用于长季节周期；与我们的随机趋势定义不同。
- Taylor & Letham (2018), [Forecasting at Scale](https://doi.org/10.1080/00031305.2017.1380080)：趋势与带收缩的 Fourier 季节模块早已有应用；Prophet 的分段趋势不是我们的阻尼随机趋势。
- Montero-Manso & Hyndman (2021), [Principles and algorithms for forecasting groups of time series: Locality and globality](https://doi.org/10.1016/j.ijforecast.2021.03.004)，[作者预印本](https://arxiv.org/abs/2008.00444)：跨序列学习/全局与局部预测已有系统研究；不是我们特定的共享 Riccati 实现的逐项先例。
- Bujosa, García-Ferrer & Young (2007), [Linear Dynamic Harmonic Regression](https://doi.org/10.1016/j.csda.2007.07.008)，[大学项目与开源工具箱](https://www.ucm.es/fundamentos-analisis-economico2/linear-dynamic-harmonic-regression-ldhr)：DHR 的长期研究与实现脉络。

开源核验：[forecast/TBATS](https://github.com/robjhyndman/forecast/blob/master/R/tbats.R)、[CRAN bsts 源码](https://github.com/cran/bsts)、[statsmodels](https://github.com/statsmodels/statsmodels)。这些链接提供相关实现，不代表直接运行其默认设置即可复现我们的全部先验和超参选择。

## 3. 为什么不只是“几个零散组件相似”

当前模型（归一化量纲）的方程为：

```
l[t+1] = l[t] + phi*b[t] + eta_l[t]
b[t+1] = phi*b[t] + eta_b[t]
y[t] = l[t] + sum_k(a[k]*cos(w[k]*t) + c[k]*sin(w[k]*t)) + eps[t]
(a[k], c[k]) 是带 Gaussian 先验的静态未知系数。
```

### 3.1 趋势是半局部趋势的重参数化

对 phi>0，定义 `delta[t] = phi*b[t]`，得到：

```
l[t+1] = l[t] + delta[t] + eta_l[t]
delta[t+1] = phi*delta[t] + phi*eta_b[t]
```

这就是 `bsts` 官方半局部趋势方程令长期斜率 `D=0` 的形式。斜率过程方差与初始方差相应乘以 phi²，预测完全不变。phi=0 是无斜率贡献的退化边界。固定 phi 与方差不同于默认 BSTS 对它们进行贝叶斯估计，但不产生新的状态模型类别。

### 3.2 静态 Fourier 与确定性三角状态等价

对每个频率定义：

```
u[k,t] =  a[k]*cos(w[k]*t) + c[k]*sin(w[k]*t)
v[k,t] = -a[k]*sin(w[k]*t) + c[k]*cos(w[k]*t)
[u[k,t+1], v[k,t+1]]' = [[cos(w),sin(w)],[-sin(w),cos(w)]] * [u[k,t],v[k,t]]'
```

这是文献与官方软件使用的旋转状态形式，季节过程噪声为零。当前每对系数采用相同方差，旋转后初始季节协方差保持不变。将所有组件拼接并使用联合 Kalman 更新，就是标准结构时序模型的构造方式。

为避免仅靠术语判断，我们另写了旋转季节状态＋半局部趋势的独立实现，变换全部先验和过程方差，与 `fg_state.forecast` 对照。4 种配置（含无季节、无趋势及 phi=1），7 条合成序列、200 点历史、13 步预测：**原始预测均值最大绝对差 1.71×10⁻¹³，条件预测方差最大差 5.68×10⁻¹³**。脚本和数值保存在 `audit/results/prior_art_equivalence_check.py` / `.json`。这是坐标等价验证，不是新增准确率实验，也不是运行原版 BSTS 得到的结果。该验证要求先验、Q/R、滤波时序、观测尺度和设计矩阵一并对应；任一方另行估计参数、使用随机季节或不同先验，便不再是同一预测器。复现命令：`python audit/results/prior_art_equivalence_check.py`（从仓库根目录运行，只依赖 NumPy 与现有 fg_state 模块）。

## 4. 当前确实做了什么，哪些不能写成创新

| 本项目做法 | 合理贡献定位 | 不应宣称 |
|---|---|---|
| 小规模受约束状态模型、静态 Fourier 收缩、阻尼趋势 | 有效模型配置与工程实现；相对原 FG 原型的改进 | 首次提出这类模型 |
| 联合更新并保留交叉协方差 | 正确实现标准 Kalman 条件推断；消融证据 | 新的消息传播算法，或证明 MPA 胜过 BSTS |
| 多窗口全局选参、尺度归一化 | 预测目标驱动的全局超参选择流程，可研究其偏差/方差权衡 | 已实现全贝叶斯参数积分、分层贝叶斯模型 |
| 相同 F/H/Q/R/P0 的多序列共享 Riccati 递推 | 明确条件下的批量计算优化 | 普适新复杂度突破；任意缺失模式/长度/噪声下都能共享 |
| 原 FG、连续高斯及离散 MPA 的受控实验 | 可复现负结果和模型/求解器解耦研究 | 一个 M4 子集足以证明整个 MPA 方向无效 |

共享协方差可直接从 Kalman 公式推出：给定相同系统矩阵与观测可用模式，P 的递推不依赖观测值。此处无需假定各条序列真实状态相同。但当前代码计算完整均值状态矩阵仍随序列数线性增长，数据读取、选参以及变长/缺失处理均有成本。对速度的论文论证应实测端到端吞吐，不能把一次滤波与逐序列 MCMC/MLE 调参总时长直接当作同任务加速比。

## 5. 是否可以写论文

**可以写，不能以“没有搜到一模一样的名称/默认参数”为原创依据。** 学术论文不要求每个组件都新，但必须提出并验证明确的新知识；开源报告、预印本、软件或实证论文与宣称新推断理论是不同定位。

当前最诚实、最有希望的候选主题是：

> **共享超参数与批量推断的谐波结构时序基线：精度、成本及消息传播消融的可复现研究。**

可研究的问题：全局收缩与局部估计何时各自有优势；在有限预算下，一个简单批量状态模型是否形成更好的精度–成本折中；通信式离散化和有环近似为何在某些图结构中无收益。这些目前是研究问题，不是已经证实的创新结论。

正式投稿前至少补齐：

1. **直接相关对照**：真正 MCMC BSTS（准确标注组件）、TBATS、DHR/ARIMA-Fourier、相同结构的逐序列 MLE，以及共享与逐序列参数的对照。仅超过原自写 FG 不构成充分先行工作比较。
2. **外部验证**：至少另一类公开数据及真实零售数据，多预测原点，变长原始历史，新的未用于设计的留出测试。当前公开测试已反复查看，且三个历史窗口平均并未超过 UCM。
3. **隔离贡献**：固定结构，比较全局/局部超参、静态/随机季节、收缩强度，以及逐序列/批量计算；调参、初始化和预测预算分开计时。
4. **计算与不确定性**：固定线程和硬件、不同序列数/长度/缺失率的实际扩展曲线；若声称概率预测，需评估覆盖率与 CRPS/WIS，并明确固定超参条件方差忽略参数不确定性。
5. **完整报告负结果**：不从多个试验中挑测试最优配置，不把原型修 bug 的收益当作模型创新；公开协议、预测、种子和失败处理。

如果不打算投入这些验证，现阶段适合发布技术报告和开源 benchmark，而不建议以“原创算法论文”定位投稿。录用与否取决于完成后的贡献质量和目标期刊/会议，当前无法保证。

## 6. 对 BSTS 比较的精确定义

BSTS 是一套贝叶斯结构时序建模框架，不是唯一固定方程。Scott–Varian 原文 §3.1 的示例式 (3) 是局部线性趋势＋dummy 季节＋外生回归；§4 使用 MCMC，§3.1 说明状态采用 Durbin–Koopman 模拟平滑。没有外生数据时，不应虚构预测变量或为了凑名称加入无意义的 spike-and-slab。

使用原作者 R `bsts` 包并对状态/方差作后验采样，属于真正 BSTS。若用 `AddTrig` 代替原文示例的 dummy 季节，应准确命名 **BSTS-LL/LLT-Trig**，说明这是官方框架下的谐波季节配置，非原论文某个实验的严格复刻。AddTrig 默认允许季节随机变化；State-Fourier 将其变化方差固定为零。两者结果差异同时含结构、先验及估计方式，不能直接归因于“贝叶斯有没有用”。
