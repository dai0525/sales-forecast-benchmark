"""Recompute consistently weighted diagnostics and write the audited Rossmann report."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RES=ROOT/'audit/results/rossmann';PRED=RES/'preds'
ORIGINS=(816,858,900)


def main():
    m=json.loads((RES/'metrics.json').read_text());ver=json.loads((RES/'verification.json').read_text())
    panel=np.load(ROOT/'data/alt/rossmann/panel.npz');ids=list(panel['ids']);rows=[];diag={}
    for name,summary in sorted(m['summary'].items(),key=lambda kv:kv[1]['wape_mean']):
        total=[];opened=[];closed=[];scaled=[];abs_all=0.;truth_all=0.
        for o in ORIGINS:
            z=np.load(PRED/f"preds_{name.replace('+','plus')}_o{o}.npz")
            if list(z['ids'])!=ids:raise ValueError('ID mismatch')
            y=panel['y'][:,o:o+42].astype(float);pred=z['preds'].astype(float)
            if not np.array_equal(y,z['truth']):raise ValueError('Truth mismatch')
            error=abs(y-pred);den=y.sum(1);isclosed=panel['open_'][:,o:o+42]==0
            total.append(error.sum(1)/den);closed.append((error*isclosed).sum(1)/den);opened.append((error*~isclosed).sum(1)/den)
            # Horizon diagnostic scale uses only that origin's history.
            scale=panel['y'][:,:o].mean(1)
            scaled.append(np.where(isclosed,np.nan,error/scale[:,None]))
            abs_all+=error.sum();truth_all+=y.sum()
        d={'mean_wape':float(np.mean(total)),'closed_error_contribution_wape':float(np.mean(closed)),
           'open_error_contribution_wape':float(np.mean(opened)),
           'pooled_aggregate_wape':float(abs_all/truth_all),
           'per_horizon_open_scaled_train_mean':np.nanmean(np.stack(scaled),axis=(0,1)).tolist()}
        assert abs(d['mean_wape']-d['closed_error_contribution_wape']-d['open_error_contribution_wape'])<1e-10
        diag[name]=d
        rows.append({'model':name,**d,'origin_mean_aggregate_wape':summary['wape_aggregate'],'by_origin':summary['wape_mean_by_origin'],'fit_predict_elapsed_mean_s':summary['runtime_s_mean']})
    (RES/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    header=['| 模型 | 平均 WAPE | 总量 WAPE（逐窗口平均） | 逐原点平均 WAPE |','|---|---:|---:|---|']
    table=header+[f"| {r['model']} | {100*r['mean_wape']:.4f}% | {100*r['origin_mean_aggregate_wape']:.4f}% | "+' / '.join(f'{100*v:.2f}%' for v in r['by_origin'])+' |' for r in rows]
    (RES/'leaderboard.md').write_text('\n'.join(table)+'\n')
    (RES/'leaderboard.csv').write_text('model,wape_mean,wape_aggregate_origin_mean,wape_aggregate_pooled\n'+'\n'.join(f"{r['model']},{r['mean_wape']},{r['origin_mean_aggregate_wape']},{r['pooled_aggregate_wape']}" for r in rows)+'\n')
    gap=diag['StateFourier7']['mean_wape']-diag['LGBM-direct']['mean_wape']
    closegap=diag['StateFourier7']['closed_error_contribution_wape']-diag['LGBM-direct']['closed_error_contribution_wape']
    lines=['# Rossmann 原实验：复核后的结果与边界','',
           '审计对象：远端 main 的 929519e 及其前序 Rossmann 实现。旧报告保存在 `audit/results/rossmann_review/original/`，问题清单见 [后续审计](DEEPSEEK-REVIEW.md)。', '',
           '**保存预测重算吻合：原榜单点预测成绩没有发现算错。** 但原报告对算法族、误差归因及统计显著性的解释过强。此报告替换这些结论，不改变已有模型预测。', '',
           '## 数据与协议','',
           '- Rossmann Sales 是门店日营业额，不是销售件数。400家完整日历门店，从934家完整店中固定抽样；按全时段完整性筛选有幸存者偏差，不代表全部1115家店或新店。',
           '- 训练/验证/测试：验证原点774，测试816/858/900，每次42天，扩张训练窗；三个测试窗口均已被查看，后续研究是回顾性探索。',
           '- 外生字段：Promo、Open、SchoolHoliday、StateHoliday，加StoreType、Assortment、Promo2、CompetitionDistance。Customers为事后字段，未使用。',
           '- Promo/Open/假期按提前已知的计划处理，但文件只有历史记录，没有当时版本的计划；门店属性只有静态快照，不能证明历史每个原点均已知或从未变化。',
           '- 原版State-Fourier和UCM仅使用销量；LGBM使用外部信息与跨店训练。不能用该差距判定状态空间模型族无法使用因子。State-Fourier的Rossmann网格固定trend_q=0，并未包含有趋势的全部变体。',
           '- LGBM-cal保留历史销量、日历和门店ID，并不是只输入日历。它与完整模型分别选出300/600棵树，原消融不是完全固定超参数的因子对照。',
           '- CSV来自GitHub镜像，结构/日期/行数检查及镜像SHA256一致；未获取官方文件进行独立逐字节比对，不能宣称已证明与Kaggle原件字节一致。', '',
           '## 原榜单重算','']+table+['',
           '平均WAPE是先逐店再逐窗口等权平均；表中总量WAPE是各窗口的总量WAPE再平均，不等于三个窗口合并后的总量WAPE。合并口径另存 diagnostics.json。耗时原记录包括拟合和预测，环境不同不能直接推断速度倍数。', '',
           '## 统一口径下的闭店误差分解','',
           '全部采用逐店逐窗口的同一分母：总绝对误差/该店该窗口营业额 = 闭店日绝对误差贡献 + 营业日绝对误差贡献。', '',
           '| 模型 | 闭店贡献 | 营业日贡献 | 合计 WAPE |','|---|---:|---:|---:|']
    for name in ['StateFourier7','LGBM-direct']:
        d=diag[name];lines.append(f"| {name} | {100*d['closed_error_contribution_wape']:.3f}pp | {100*d['open_error_contribution_wape']:.3f}pp | {100*d['mean_wape']:.3f}% |")
    lines += ['',f'两模型总差距为 {100*gap:.3f}pp，其中闭店误差差距为 {100*closegap:.3f}pp，占 {100*closegap/gap:.1f}%。这是误差按日期分组的算术分解，不是促销/假期的因果贡献；不能把其余比例归因给某些业务因素。', '',
              '## 配对不确定性','',
              '按门店聚类重采样，每次保留该门店三个原点；不再把1200个门店×窗口视为独立序列。仍未覆盖跨店共享节日冲击或新预测日期的不确定性，区间仅对已选三个窗口有条件成立。', '',
              '| 比较（前者减后者） | 平均差 | 门店聚类bootstrap 95%区间 |','|---|---:|---|']
    for name,b in ver['bootstrap'].items():lines.append(f"| {name} | {100*b['mean_diff']:+.3f}pp | [{100*b['ci95'][0]:+.3f}, {100*b['ci95'][1]:+.3f}]pp |")
    lines += ['', '结论：这些外部信息在原LightGBM实验中有明显预测价值；State-Fourier与UCM的旧单变量分数接近。不能推出“多因子一定比任何换算法更重要”或“有外因时状态空间必然落后”。多因子State-Fourier与匹配信息集的比较见 [六因子实验](MULTIFACTOR-ROSSMANN.md)。', '',
              '## 复现','', '```bash','python repro/alt/fetch_rossmann.py','python repro/alt/prep_rossmann.py','python repro/alt/verify_rossmann.py','python repro/alt/report_rossmann.py','```', '',
              '逐预测结果：`audit/results/rossmann/preds/`；点预测原始训练协议与选参记录：`audit/results/rossmann/metrics.json`。重新训练使用 `repro/alt/bench_rossmann.py`，单独指定输出目录以保留原始证据。']
    (ROOT/'ALT-DATASET.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
