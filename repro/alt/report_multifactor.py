"""Verify predictions, report six-factor results and export weekly contributions."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bench_rossmann import ROOT,OUTER_ORIGINS
from rossmann_factors import FactorPanel,NAMES,GROUPS
from bench_multifactor import metric
OUT=ROOT/'audit/results/rossmann_multifactor'


def main():
    rows=json.loads((OUT/'metrics.json').read_text());p=FactorPanel();models=list(dict.fromkeys(r['model'] for r in rows));scores={};summary={}
    for name in models:
        per=[];preds=[];truths=[]
        for o in OUTER_ORIGINS:
            z=np.load(OUT/f'{name}_o{o}.npz')
            assert list(z['ids'])==p.ids and np.array_equal(z['dates'],p.days[o:o+42])
            assert np.array_equal(z['truth'],p.y[:,o:o+42]) and np.isfinite(z['preds']).all()
            rec=next(r for r in rows if r['model']==name and r['origin']==o)
            assert abs(metric(z['preds'],z['truth'])['wape_mean']-rec['wape_mean'])<1e-10
            if name=='SF_full':
                np.testing.assert_allclose(z['components'].sum(2)+z['clipping_adjustment'],z['preds'],atol=1e-8)
                assert z['min_covariance_eigenvalue']>=-1e-8
            per.append(abs(z['preds']-z['truth']).sum(1)/z['truth'].sum(1));preds.append(z['preds']);truths.append(z['truth'])
        scores[name]=np.array(per)
        summary[name]={'wape_mean':float(np.mean(per)),'by_origin':np.mean(per,axis=1).tolist(),
                       'pooled_aggregate_wape':metric(np.concatenate(preds,axis=1),np.concatenate(truths,axis=1))['wape_aggregate']}
    comparisons={};rng=np.random.default_rng(42)
    for name in models:
        diff=(scores[name]-scores['SF_full']).mean(0)
        boot=np.mean(diff[rng.integers(0,p.n,(3000,p.n))],axis=1)
        comparisons[name]={'delta_vs_full':float(diff.mean()),'store_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
                           'note':'Same stores retained across three dates; conditional on dates, not accounting for shared calendar shocks. Ablations refit with frozen full-model hyperparameters; correlated factor deltas do not add.'}
    supports={}
    for j,name in enumerate(NAMES):
        count=0
        for i in range(p.n):
            x=p.extra[i,:774,j][p.open_[i,:774]==1]
            count+=bool(len(x) and np.ptp(x)>0)
        supports[name]={'stores_with_variation_on_open_training_days':count}
    report={'summary':summary,'comparisons':comparisons,'open_day_support':supports,'validation':'All saved IDs/dates/truth/metrics checked; component sum and covariance eigenvalues checked'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # Export all stores, all windows, in seven-day forecast blocks (Sat-Fri).
    records=[];component_names=['level','season']+NAMES+['OpenAdjustment']
    for o in OUTER_ORIGINS:
        with np.load(OUT/f'SF_full_o{o}.npz') as archive:z={k:archive[k] for k in archive.files}
        with np.load(OUT/f'LGBM_six_o{o}.npz') as archive:g={k:archive[k] for k in archive.files}
        for i,sid in enumerate(p.ids):
            for week in range(6):
                sl=slice(week*7,week*7+7)
                r={'store':sid,'origin':o,'week_start':str(p.days[o+week*7]),'week_end':str(p.days[o+week*7+6]),
                   'actual_sales':float(z['truth'][i,sl].sum()),'state_forecast':float(z['preds'][i,sl].sum()),'lgbm_forecast':float(g['preds'][i,sl].sum()),
                   'clipping_adjustment':float(z['clipping_adjustment'][i,sl].sum())}
                r.update({name:float(z['components'][i,sl,k].sum()) for k,name in enumerate(component_names)})
                records.append(r)
    df=pd.DataFrame(records);df.to_csv(OUT/'weekly_contributions.csv',index=False)
    # Select display store using only training feature variation, never forecast error.
    eligible=np.flatnonzero((np.ptp(p.extra[:,:774,5],axis=1)>0)&(np.ptp(p.extra[:,:774,6],axis=1)>0))
    i=int(eligible[0]) if len(eligible) else 0;sid=p.ids[i];show=df[df.store==sid];time=pd.to_datetime(show.week_start)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for col,label in [('actual_sales','Actual'),('state_forecast','State-Fourier + 6 factors'),('lgbm_forecast','LightGBM + 6 factors')]:axes[0].plot(time,show[col],label=label,marker='.')
    axes[0].set_title(f'{sid}: weekly turnover forecasts (three separate 42-day origins)');axes[0].legend();axes[0].set_ylabel('Sales / turnover')
    for col in ['Promo','SchoolHoliday','StateHoliday_a','StateHoliday_b','StateHoliday_c','Promo2Active','CompetitionOpen','OpenAdjustment']:
        axes[1].plot(time,show[col],label=col)
    axes[1].axhline(0,color='grey',lw=.5);axes[1].legend(ncol=3,fontsize=8);axes[1].set_ylabel('Model contribution');axes[1].set_xlabel('7-day forecast block starts (Sat–Fri); model allocation, not causal effects')
    fig.savefig(OUT/'weekly_contributions.png',dpi=150);plt.close(fig)
    labels={'SF_univariate':'State-Fourier：无外因','SF_open_only':'State-Fourier：仅营业状态','SF_full':'State-Fourier：六因子','SF_without_open':'六因子去掉营业状态','LGBM_six':'LightGBM：相同六因子'}
    lines=['# 六因子真实零售实验：State-Fourier 与 LightGBM','',
           '**不是合成数据。** 使用Rossmann原始营业额及业务字段，保持旧实验的400家门店、验证原点774、测试816/858/900、42天步长。模型及网格在本次运行前写入 `audit/results/rossmann_review/protocol.json`。旧测试已被查看，因此是探索性回测，不是新的盲测。', '',
           '## 六类因子','',
           '| 因子组 | 来源与处理 |','|---|---|',
           '| 短期促销 | 原始 Promo；仅有二元标记，没有促销强度或满减/折扣类型 |',
           '| 营业状态 | 原始 Open；作为营业开关，闭店输出0 |',
           '| 学校假期 | 原始 SchoolHoliday |',
           '| 法定节假日 | 原始 StateHoliday 的 a/b/c 类型，三个哑变量只算一个因子组 |',
           '| 持续促销计划 | Promo2SinceYear/Week 与 PromoInterval 推导活动月份；只对预测原点已参加计划的门店生效，是计划代理变量 |',
           '| 竞争店开业状态 | CompetitionOpenSinceYear/Month 推导；未来42天冻结为预测原点的状态，不提前使用未来开店事件 |', '',
           '没有把同一日历字段拆成多个哑变量凑够因子数量，也没有使用未来Customers。竞争开店日期缺失时编码为0（未知/未确认已开业），不能解释为确定无竞争。静态门店类型、商品组合、竞争距离不放进逐店State-Fourier，以免与水平项混淆；新的LightGBM对照也不使用这些额外静态属性，只保留门店ID、历史、日历和相同六类外因。', '',
           '营业、短促和假期按已知计划评估，数据没有当时版本的计划；Promo2与竞争资料同样来自单次静态快照，历史可得性无法完全验证。这是有条件的回顾性效果，不是业务上线保证。', '',
           '## 模型与验证','',
           '模型：y_it = Open_it × [level_it + Fourier7_it + β_iᵀ x_it] + ε_it。按店保留完整联合协方差，Kalman更新水平、季节与回归系数；β为静态未知系数，先验零均值收缩。沿用原Rossmann模型无趋势的设定。营业状态通过观测设计参与训练，闭店记录不把营业日水平拉向零。它是已有状态空间回归的扩展，不是新的MPA算法。', '',
           '用验证窗口选择 qr∈{0.002,0.02,0.2}、β先验方差∈{0.1,1}，选中 qr=0.02、β方差=0.1；所有因子消融重训但固定此配置。LightGBM沿用原验证选出的600棵树，无外层择优。不同模型的调参空间、跨店共享方式与非线性能力不同，不能把差距只归因于推断器。', '',
           '## 主结果','', '| 模型 | 平均 WAPE | 三个原点 WAPE |','|---|---:|---|']
    for name in ['SF_univariate','SF_open_only','SF_full','LGBM_six']:
        r=summary[name];lines.append(f"| {labels[name]} | {100*r['wape_mean']:.4f}% | "+' / '.join(f'{100*v:.3f}%' for v in r['by_origin'])+' |')
    lines += ['', '平均WAPE按门店、窗口等权。未引入未来目标进行滤波更新，整个42天一次性预测。无外因版本与旧State-Fourier数值等价已通过测试；新结果文件单独保存，旧榜单未被替换。', '',
              '## 哪些因子有用：固定超参逐组去除','', '| 去除的因子 | 去除后 WAPE | 相对完整六因子的变化 | 门店聚类95%区间 |','|---|---:|---:|---|']
    for name in ['SF_without_open']+['SF_drop_'+g for g in GROUPS]:
        r=summary[name];c=comparisons[name];lines.append(f"| {name.removeprefix('SF_')} | {100*r['wape_mean']:.4f}% | {100*c['delta_vs_full']:+.4f}pp | [{100*c['store_cluster_ci95'][0]:+.4f}, {100*c['store_cluster_ci95'][1]:+.4f}]pp |")
    lines += ['', '主要收益来自短期促销和营业信息。学校/法定假期增量很小，竞争开业状态在当前配置下几乎无增量；去掉持续促销代理略好，但不能根据此测试结果事后删除后再称独立验证提升。因子相关、参数会重新拟合，消融差不能相加，也不是因果效果。区间按门店保留全部三个窗口，仍不能覆盖跨店共享节日冲击与未来新窗口的不确定性。', '',
              '## 贡献曲线与可辨识性','',
              f'保存全部400家店×18周的贡献表：[weekly_contributions.csv](audit/results/rossmann_multifactor/weekly_contributions.csv)。展示店 {sid} 是按训练期Promo2/竞争状态均有变化的首个ID选择，不按预测好坏挑选。每个窗口的6个块均为周六至周五；相邻窗口重新训练，曲线不是一次连续126天预测。', '',
              '![每周预测与业务项贡献](audit/results/rossmann_multifactor/weekly_contributions.png)', '',
              '完整加和：基础水平 + 周期 + 五类回归项 + 营业调整 + 非负截断调整 = 最终预测。营业调整在闭店时等于负的未关店预测，属于结构规则，不是实验估计的关店损失。CSV保留基础水平和周期，图下半部只展示业务贡献。', '',
              '六类输入不意味着每家店六类都可单独识别。在验证前训练期，Promo2在185家店变化、竞争状态只在52家变化；节假日常与闭店同时出现，对营业日节假日效应的数据支持更少。恒定因素与水平项重叠，收缩先验只能给出一种模型分配，不能创造因果识别。营业日上的变化覆盖见 summary.json。', '',
              '## 下一步','',
              '先在新的业务时间窗口、真实计划版本上检验；若研究改进，优先考虑促销强度/类型和促销×星期的交互或跨店系数共享。当前六因子State-Fourier比单变量明显改善，但在三个窗口均未达到同信息集LightGBM的误差水平，不能宣称更高预测精度。', '',
              '复现：', '', '```bash','python repro/alt/fetch_rossmann.py','OPENBLAS_NUM_THREADS=1 python repro/alt/bench_multifactor.py','python repro/alt/report_multifactor.py','```']
    (ROOT/'MULTIFACTOR-ROSSMANN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
