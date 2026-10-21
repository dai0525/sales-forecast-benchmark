"""Verify stored predictions and write the first cross-retailer results report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from retail_expansion import ROOT,OUT,metrics
from audit_freshretail import digest


def paired_ci(tag,o,a,b,cluster):
    za=np.load(OUT/tag/f'o{o}_{a}.npz');zb=np.load(OUT/tag/f'o{o}_{b}.npz')
    np.testing.assert_array_equal(za['truth'],zb['truth'])
    y=za['truth'];delta=np.abs(za['pred']-y).sum(1)-np.abs(zb['pred']-y).sum(1)
    s=pd.DataFrame({'group':cluster,'delta':delta,'den':y.sum(1)}).groupby('group').sum()
    rng=np.random.default_rng(20261021);samples=[]
    for _ in range(2000):
        ix=rng.integers(0,len(s),len(s));den=s.den.to_numpy()[ix].sum()
        if den>0:samples.append(s.delta.to_numpy()[ix].sum()/den)
    return {'candidate':a,'baseline':b,'delta_wape_pp':float(delta.sum()/y.sum()*100),'ci95_pp':(np.quantile(samples,[.025,.975])*100).tolist(),'clusters':len(s),'bootstrap_replicates':2000,'caveat':'paired cluster bootstrap conditional on these test dates; not future-date uncertainty'}


def main():
    lines=['# 新零售数据首轮实测','', '后续已补齐 [ASE 追加对照](RETAIL-ASE-RESULTS.md)；下文保留首轮未包含 ASE 的实验结果。','',
           '固定协议见 [protocol.json](audit/results/retail_expansion/protocol.json)。本轮不使用动态专家组合，没有根据最终测试结果调整参数。开发期前两个窗口选择单一模型，第三个窗口只做最终评估。',
           '', '## 本轮结论','',
           '叮咚上，条件性促销计划信息改善了预测；继续加入简单天气和缺货摘要没有优于计划组。VN1 上，开发期选中的单一 LightGBM 的 WAPE 优于简单基线和 State Fourier，但系统性低估较明显，按官方 WAPE + |Bias| 分数反而不如最近值基线。不能宣称全面优于基线或已经改善库存业务。',
           '', '下一步优先：在开发期研究偏差敏感损失或简单校准，冻结后用新的时间段/业务来源验证；叮咚扩展到全量，检查当前抽样结果是否保持；补充 Favorita/M5 长历史。扩充叮咚序列仍是同一时间段覆盖验证，不是新的时间盲测。不根据本轮最终集调参并重复声称独立验证。',
           '', '## 范围与信息条件','',
           '- 叮咚：按 ID 哈希选择 3,000/50,000 条序列，不按销量筛选；训练内两个 7 天开发窗口，最终为 2024-06-26 至 07-02。属于抽样初验，不能当成全量结果。',
           '- VN1：全部 15,053 条序列、46 家商家；开发窗口为 Phase 0 最后 13 周及 Phase 1，最终为 Phase 2（2024-01-08 至 04-01）。所有模型在最终预测前用相同历史重拟合。',
           '- LightGBM 为单一全局模型、直接多步、log1p 目标，250 棵树、固定参数，无最终集调参；不代表 LightGBM 可达到的最优精度。',
           '- State Fourier 使用仓库原有高斯状态空间实现，固定噪声参数；外因版是已有线性状态回归，不是新 MPA 算法。外因版统一跳过最初 7 天，纯状态版使用全部可用历史，故两者差异不能全部归因为外因。',
           '- 叮咚计划组包含折扣、活动、假日，假定未来计划提前确定。数据提供的是最终记录，未提供预测起点的计划版本，无法验证实际折扣/活动当时已知，因此这是条件信息实验，不能直接当作实盘可实现收益。纯历史组没有该假设。天气组仅用预测起点前天气（树模型取最近 7 天均值，状态回归在历史上拟合天气效应、未来使用最近 7 天均值）；缺货组仅为历史缺货时长。没有实际未来天气或缺货输入。',
           '- VN1 价格组仅用历史成交价，未来不提供实际价格；它不是计划报价。价格缺失保留为缺失（树模型），状态回归先历史前向填充、最初无价时以零填充后标准化，因此价格回归结果也受这种简化处理影响。',
           '- 本轮预测的是观测销量，不是缺货期间不可观测的潜在需求；因子贡献或消融差异不能直接解释为因果效应。','']
    verification=[];intervals={}
    for tag in ['freshretail','vn1']:
        out=OUT/tag;manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['protocol_sha256']==digest(OUT/'protocol.json')
        table=pd.read_csv(out/'metrics.csv');selection=json.loads((out/'selection.json').read_text());o=selection['final_origin']
        for r in table.to_dict('records'):
            z=np.load(out/f"o{r['origin']}_{r['model']}.npz")
            for k,v in metrics(z['pred'],z['truth']).items():assert np.isclose(v,r[k],rtol=1e-10,atol=1e-10),(tag,k)
            verification.append({'file':str(out.relative_to(ROOT)/f"o{r['origin']}_{r['model']}.npz"),'sha256':digest(out/f"o{r['origin']}_{r['model']}.npz")})
        dev=table[table.origin!=o]
        recomputed=dev.groupby('model')[['absolute_error','actual_total']].sum();scores=recomputed.absolute_error/recomputed.actual_total
        assert scores.idxmin()==selection['selected']
        for k,v in selection['development_wape'].items():assert np.isclose(v,scores[k])
        ids=pd.read_csv(out/'series_ids.csv')
        verification.append({'file':str(out.relative_to(ROOT)/'series_ids.csv'),'sha256':digest(out/'series_ids.csv')})
        groups=ids.store_id if tag=='freshretail' else ids.Client
        comparisons=[('lgb_plan','lgb_history'),('lgb_weather','lgb_plan'),('lgb_stock','lgb_weather'),('state_fourier','mean_recent')] if tag=='freshretail' else [('lgb_price','lgb_history'),('lgb_history','last_value'),('state_fourier','last_value'),('state_fourier','mean_recent')]
        intervals[tag]=[paired_ci(tag,o,a,b,groups) for a,b in comparisons]
        title='叮咚 3,000 条序列' if tag=='freshretail' else 'VN1 全部序列'
        lines.extend([f'## {title}','',f"开发期选择：**{selection['selected']}**，在最终评估前写入 selection.json。下表列出所有预先指定对照，不据最终排名重选模型。",'',
                      '| 模型 | 开发总体 WAPE | 最终总体 WAPE | 最终 Bias | 最终拟合+预测秒 |', '|---|---:|---:|---:|---:|'])
        final=table[table.origin==o]
        for r in final.to_dict('records'):
            lines.append(f"| {r['model']} | {scores[r['model']]*100:.3f}% | {r['wape']*100:.3f}% | {r['bias']*100:+.3f}% | {r['seconds']:.2f} |")
        lines.extend(['','耗时包含本次模型特征构造、拟合和预测，不含数据下载/适配和文件写盘；两数据任务并发运行，时间仅作本环境工程参考，不作为严格速度基准。',''])
        if tag=='vn1':
            for m,expected in [('last_value',.5070),('mean_recent',.8046)]:
                actual=float(final.loc[final.model==m,'vn1_score'].iloc[0])
                assert abs(actual-expected)<.00015,(m,actual,expected)
            lines.extend(['VN1 官方分数为 WAPE + |Bias|，本轮选择标准为 WAPE；它们的最优模型可能不同。','', '| 模型 | Phase 2 官方分数 |','|---|---:|'])
            for r in final.to_dict('records'):lines.append(f"| {r['model']} | {r['vn1_score']*100:.3f}% |")
            c=pd.read_csv(out/'client_metrics.csv');c=c[c.origin==o].pivot(index='client',columns='model',values='wape')
            a=selection['selected'];b='state_fourier'
            valid=c[[a,b]].dropna();wins=int((valid[a]<valid[b]).sum());ties=int((valid[a]==valid[b]).sum())
            lines.extend(['',f'开发期选中的 {a} 相对 state_fourier：{len(valid)} 家有定义的商家中 {wins} 胜、{ties} 平、{len(valid)-wins-ties} 负。商家零销量造成的未定义 WAPE 保留为空，不当成零误差。'])
        lines.extend(['','最终零分母序列数：'+str(int(final.iloc[0].zero_denominator_series))+'；它们仍计入总体误差，但不计入逐序列平均 WAPE。','',
                      '| 探索性配对对照（候选−基线） | 最终 WAPE 差（百分点） | 95% 配对簇自助区间 |', '|---|---:|---:|'])
        for r in intervals[tag]:lines.append(f"| {r['candidate']} − {r['baseline']} | {r['delta_wape_pp']:+.3f} | [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}] |")
        lines.extend(['','负值表示候选更好。叮咚按门店、VN1 按商家进行 2,000 次配对重采样；区间只反映这些日期下的簇差异，不证明跨未来时间的稳定性，也未进行多重比较校正。',''])
    (OUT/'verification.json').write_text(json.dumps({'status':'all_saved_metrics_recomputed','files':verification,'bootstrap':intervals},indent=2)+'\n')
    lines.extend(['## 复现与边界','',
                  '先运行 `repro/audit_freshretail.py` 与 `repro/audit_vn1.py` 获取固定校验和的数据，再运行 `python repro/retail_expansion.py freshretail`、`python repro/retail_expansion.py vn1`，最后 `python repro/report_retail_expansion.py`。实验脚本为防止混用拒绝覆盖已有预测；重跑应先将对应输出目录移到独立备份位置。',
                  '', '各窗口预测、ID 顺序、逐商家指标、模型参数和开发选择保存在 [结果目录](audit/results/retail_expansion/)。报告生成器独立从保存的预测重算每项分数并校验开发期选择。',
                  '', '局限：天气使用历史均值代理，不是当时真实天气预报；缺货特征只有简单历史摘要，不是需求恢复模型。此结果不能推论天气或库存信息本身没有价值。叮咚是单个零售商的短历史抽样；VN1 官方筛选了最终期无缺货的商品；只有一个最终时间块，且这些公开标签并非外部第三方保密盲测。需要更多时间窗口、叮咚全量和第三个零售来源验证后，才能声称一般优势。'])
    (ROOT/'RETAIL-EXPANSION-RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
