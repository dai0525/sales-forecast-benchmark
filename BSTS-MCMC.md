# 原版 BSTS 后验采样基线

本实验直接使用原作者维护的 CRAN `bsts` 0.9.11，R 4.5.3。噪声参数和隐状态通过 MCMC 后验采样，再生成未来观测的后验预测分布。它不再是把最大似然 UCM 改称 BSTS。

## 模型与可比范围

主实验对全部359条M4 Weekly序列使用相同的最近200点训练、13步官方测试、原有短序列左填充和ID顺序。结构是 `AddLocalLinearTrend` 加 `AddTrig(period=52, frequencies=1:4, method="harmonic")`，即局部线性趋势加年度随机谐波季节。没有可用外生变量，因此没有虚构回归项或无意义的spike-and-slab变量选择。

BSTS是模块化建模框架，不是唯一固定结构。原论文的示例结构更接近局部线性趋势加dummy季节和回归；本轮另外预先固定12个等间距索引，运行 `AddLocalLinearTrend + AddSeasonal(nseasons=52)` 核对这种季节表示。子集结果只和完全相同ID的其它模型比较，不能与359条榜单直接排名。

相对旧UCM52，本模型改变了趋势、季节表示、先验与参数推断；相对State-Fourier，也改变了随机季节、无阻尼趋势与参数学习。因此成绩差别不能单独归因于“贝叶斯有没有用”。

## 先验、预算与诊断

主实验保留原包默认先验。状态创新标准差的先验基于训练标准差，`SdPrior(0.01*sd(y), sample.size=0.01, upper.limit=sd(y))`；观测噪声先验与初始状态先验使用原包默认。每条序列的具体先验均保存在其JSON中。这些尺度来自历史数据，属于经验先验设置，不能称完全不依赖数据的先验。

每条序列2条独立种子链，各1000轮，前500轮弃置。主点预测为所有保留后验预测样本的均值，再按已有benchmark口径截断到非负；同时报告中位数、95%预测区间覆盖率、两条链各自点预测分数。中位数是辅助诊断，不按测试成绩选择预测统计量。

通过R `posterior` 包计算rank-normalized split Rhat、bulk/tail ESS和Monte Carlo标准误。报告阈值为Rhat≤1.01、bulk/tail ESS≥400，分别检查监控的噪声参数/末端状态，以及全部13个预测步长和13步均值。缺失诊断视为未通过，不丢弃不收敛的序列。

**这是一份有限预算的MCMC基线，不能假定所有参数都已收敛。** 初始训练数据诊断发现弱先验下季节方差混合缓慢；前两条延长到4000轮仍有参数Rhat偏高。正则化先验的训练侧pilot改善有限，未根据官方测试调整先验，因此主实验保留默认设置。预测分布可能比弱识别的方差参数更稳定，但不能用这一点掩盖参数诊断。

## pilot与审计记录

- `trend_trig_1000`：`method="direct"`的前两条训练侧pilot，出现很大的状态方差和过程SD贴近上限。未据此给该实现判定上游bug，也未用它排名；主实验使用官方默认的harmonic表示。
- `trend_harmonic_1000`：早期默认先验诊断与中断检查点；同样推断结果转存到正式默认目录续跑。重复文件保留用于追溯。
- `trend_harmonic_regularized_1000`：前两条训练侧先验强度检查（sample.size=32）。
- `trend_harmonic_default_4000`：前两条更长链检查。
- `trend_harmonic_default_1000`：全量主实验；不会删除未收敛序列或用其它模型替补。
- `trend_seasonal52_default_1000`：12条固定子集的原论文风格季节结构。

## 复现

```bash
# 已有Python审计环境另见README；R独立环境：
micromamba create -f environment-bsts.yml
micromamba activate sales-bsts

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repro/bench_bsts_mcmc.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repro/bench_bsts_mcmc.py --kind trend_seasonal52
```

若Rscript不在PATH，传入 `--rscript /path/to/Rscript`。已完成的逐序列JSON和RDS用作可恢复检查点；要重跑同配置，先指定一个独立工作副本或移走对应结果目录。`--iterations`可增加预算，结果写入不同目录。`--export-only`只计算诊断和分数，不重新拟合。

每条序列的RDS保存参数轨迹及两条链的全部后验预测样本；JSON保存种子协议、先验、预测和诊断。主输出为 `repro/metrics_bsts_mcmc.json` 与 `data/weekly_bsts_mcmc_preds.npz`，完整过程在 `audit/results/bsts_mcmc/`。

## 参考

- Scott & Varian (2014), [Predicting the Present with Bayesian Structural Time Series](https://doi.org/10.1504/IJMMNO.2014.059942)，[作者全文](https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf)。
- [CRAN bsts](https://cran.r-project.org/package=bsts)、[AddTrig](https://search.r-project.org/CRAN/refmans/bsts/html/add.trig.html)、[AddSeasonal](https://search.r-project.org/CRAN/refmans/bsts/html/add.seasonal.html)。
- [posterior诊断](https://mc-stan.org/posterior/reference/rhat.html)：参数后验收敛与有效样本量；有限预算和短历史的限制需与分数一起解读。

## 诊断与版本记录的补充说明

诊断监控的是噪声标准差及末端水平/斜率，并未覆盖所有末端谐波状态；因此“全部通过”只指已监控变量。后验预测样本还包含独立未来噪声，可能掩盖隐状态混合问题，预测诊断通过不能推翻参数诊断失败。

早期20条默认采样检查点只保存参数/末端状态诊断，后续版本增加了预测诊断字段；采样目标、随机种子和预测过程未因此改变。所有序列统一从RDS中两链原始预测样本重新计算独立预测诊断，避免依赖不同JSON版本。更改仅涉及诊断/复现记录，未依据测试成绩改先验或结构。

## 缓存完整性

采样入口现已检查训练CSV、ID文件、配置、R源码和R/bsts版本指纹。发生变化时拒绝静默复用；失败记录或缺失RDS也不会当作成功检查点。本次先前的检查点在逐值核对训练输入、ID和推断配置后显式迁移，协议保留legacy adoption说明；当前源码指纹不被冒充为原始诊断代码的运行时指纹。仅导出结果不会重写训练输入。

## 后续收敛与先验敏感性验证

固定12条子集增加到每链4000轮，并对比默认与正则化先验；详见 [补充实验](BSTS-SENSITIVITY.md)。这项试验单独保存，不替换359条主结果。导出器现在只允许原先冻结的完整默认协议更新主榜单，避免其它预算或先验的全量运行静默覆盖结果。

## 本次运行结果（由保存预测自动生成）

主实验：359条；R原版BSTS，LLT + 52周随机谐波（4阶）。主指标为逐序列WAPE均值。

| 方法 | 平均 WAPE | 总量 WAPE |
|---|---:|---:|
| BSTS MCMC† | 7.9813% | 5.7338% |
| State-Fourier | 7.3133% | 5.4755% |
| UCM52（MLE） | 7.8403% | 5.5647% |
| 原FG | 8.7060% | 5.9802% |

† 默认先验、2链×1000轮（各弃500轮）的有限预算结果。监控参数/末端状态全部通过阈值：**0/359**；13个预测步长及均值全部通过阈值：**198/359**。因此这不是已充分收敛的BSTS金标准，不能把分数当作该方法的准确率上限。

后验预测中位数作为点预测的辅助WAPE为 7.9681%；两条链各自均值预测的WAPE分别为 8.0398% / 7.9597%。主指标仍固定使用合并后验均值，没有依据测试择优。

95%后验预测区间在本测试段的经验覆盖率 94.58%，不能仅凭一次覆盖率认为区间已经校准；短链和模型假设仍影响区间。

BSTS−State-Fourier 的平均WAPE差为 +0.6681pp，按序列bootstrap的95%区间 [+0.2404, +1.1674]pp。该区间固定了已保存的MCMC点预测，不包含采样误差，也假设序列独立。

### 原论文风格结构的固定12条子集

| 方法（相同12条） | 平均 WAPE |
|---|---:|
| BSTS LLT + dummy seasonal52† | 8.6887% |
| State-Fourier | 6.4648% |
| UCM52（MLE） | 6.6829% |

子集参数/末端状态全部通过 0/12；预测诊断全部通过 6/12。此子集不外推到全量。

### 耗时的解释

全量逐序列拟合与预测的进程elapsed时间之和为 3413.9秒，包含并发争用；它不是整批wall time，不能直接除以359或线程数作为独立硬件性能。续跑的批次时间和缓存状态见run.json；此前pilot和环境安装不在其中。网页不提供一个容易误解的单一速度倍数。

### 本轮能够得出的结论

已将原作者包的真正参数/状态后验采样接入benchmark，并保存可复核的预测、区间和样本。有限预算下的点预测可以作为探索性参考；参数混合仍是主要限制，不能据此证明BSTS总体不如State-Fourier，也不能把现有模型与BSTS的差异单独归因于是否采用贝叶斯推断。若后续需要严格模型优劣结论，必须先改善采样/弱识别、验证先验敏感性，再按预先规定预算重跑。
