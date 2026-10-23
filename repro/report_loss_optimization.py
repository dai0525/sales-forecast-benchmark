"""Reconstruct chronological calibration and score saved optimization forecasts."""
import json
import numpy as np
import pandas as pd
from loss_optimization import ROOT,OUT,BASE,MODELS,choose,metrics
from audit_freshretail import digest


def interval(candidate,baseline,truth,groups,metric):
    df=pd.DataFrame({'group':groups,'ae_a':abs(candidate-truth).sum(1),'ae_b':abs(baseline-truth).sum(1),'e_a':(candidate-truth).sum(1),'e_b':(baseline-truth).sum(1),'den':truth.sum(1)}).groupby('group').sum()
    rng=np.random.default_rng(20261023);samples=[]
    for _ in range(2000):
        s=df.to_numpy()[rng.integers(0,len(df),len(df))].sum(0)
        if s[4]<=0:continue
        delta=s[0]-s[1]
        if metric=='vn1_score':delta+=abs(s[2])-abs(s[3])
        samples.append(delta/s[4]*100)
    return {'metric':metric,'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),'clusters':len(df),'replicates':2000,'scope':'exploratory cluster bootstrap conditional on observed dates; no multiple-comparison correction'}


def main():
    lines=['# 单模型损失函数与偏差校准探索','',
           '日期：2026-10-23。沿用已有公开最终窗口，本轮是探索，不是独立确认。实验前固定候选、时间顺序和选择指标；没有根据最终成绩改选主方案。代码只优化单一 LightGBM，没有增加动态专家或新组合算法。','',
           '## 为什么值得测试','',
           '原模型拟合 log(1+y)，再用 exp(z)-1 返回销量。在理想的平方损失拟合下，模型近似估计 E[log(1+Y)|X]；由 Jensen 不等式，exp(E[log(1+Y)|X])-1 ≤ E[Y|X]。因此简单逆变换一般不等于条件均值。但这不是对任何实际模型或时间窗都必然低估的保证：拟合误差和分布变化可能使偏差变号。',
           '', '直接在销量尺度用 L2 估计均值，或采用 Poisson/Tweedie 均值损失，可能减少反变换问题。Poisson/Tweedie 是成熟目标函数，不是新方法。叮咚销量已统一缩放，Poisson 在这里作为非负均值损失使用，不宣称数据服从整数 Poisson 分布。Tweedie 方差幂固定为 1.5，不进行最终集搜索。',
           '', '校准因子 c=clip(Σy/Σŷ,0.5,2)，输出 ŷ×[1+λ(c−1)]。λ∈{0,0.5,1}，λ=0 即保留原模型，防止默认认为修正必然有益。训练树数、叶子数、训练样本和特征沿用首轮小配置，改变的只是目标尺度和损失。','',
           '## 时间顺序与信息条件','',
           '- 第一开发窗口估计 c，第二开发窗口选择 λ 和模型；所有候选在同一个第二窗口比较。最终 c 用两个开发窗口的预测与真实值重新估计，最终标签不参与任何选择。',
           '- 叮咚按总体 WAPE 选择，VN1 按官方 WAPE + |Bias| 选择。后者体现业务评分目标不同，不能把所有改善归因于模型结构。',
           '- 叮咚仍是固定 3,000 条 ID 样本，纯历史/促销条件场景分开；未来折扣活动可知的假设未得到历史计划版本支持。VN1 使用全部 15,053 条序列，历史信息可用性与前轮一致。',
           '- 模型拟合观测销量，不是缺货潜在需求；校准不是需求恢复、因果识别或新的消息传递机制。','']
    verification=[];summaries={}
    for track in ['freshretail_history','freshretail_plan','vn1_history']:
        tag='vn1' if track.startswith('vn1') else 'freshretail';out=OUT/track
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['protocol_sha256']==digest(OUT/'protocol.json')
        table=pd.read_csv(out/'metrics.csv');o=manifest['final_origin'];dev=sorted(table.loc[table.origin!=o,'origin'].unique())
        pp={};yy={}
        for origin in dev:
            pp[origin]={}
            for model in MODELS:
                z=np.load(out/f'o{origin}_{model}.npz');pp[origin][model]=z['pred'];yy[origin]=z['truth']
        sel=json.loads((out/'selection.json').read_text());rebuilt=choose(pp[dev[0]],pp[dev[1]],yy[dev[0]],yy[dev[1]],manifest['selection_metric'])
        assert all(rebuilt[k]==sel[k] for k in rebuilt)
        for model in MODELS:
            z=np.load(out/f'o{o}_{model}.npz');opt=sel['options'][model]
            corrected=z['pred']*(1+opt['strength']*(opt['final_factor']-1))
            np.testing.assert_allclose(corrected,np.load(out/f'o{o}_{model}_calibrated.npz')['pred'],rtol=1e-12,atol=1e-12)
        for r in table.to_dict('records'):
            path=out/f"o{r['origin']}_{r['model']}.npz";z=np.load(path)
            for k,v in metrics(z['pred'],z['truth']).items():assert np.isclose(v,r[k],rtol=1e-10,atol=1e-10),(track,k)
            verification.append({'file':str(path.relative_to(ROOT)),'sha256':digest(path)})
        ids=pd.read_csv(out/'series_ids.csv');pd.testing.assert_frame_equal(ids,pd.read_csv(BASE/tag/'series_ids.csv'))
        final=table[table.origin==o].set_index('model');selected=manifest['selected_prediction'];m=sel['selected_model'];opt=sel['options'][m]
        metric=manifest['selection_metric'];a=np.load(out/f'o{o}_{selected}.npz');b=np.load(out/f'o{o}_log_l2.npz');np.testing.assert_array_equal(a['truth'],b['truth'])
        groups=ids.Client if tag=='vn1' else ids.store_id
        ci=interval(a['pred'],b['pred'],a['truth'],groups,metric)
        summaries[track]={'selected':selected,'metric':metric,'final_metrics':final.to_dict('index'),'interval':ci,'selection':sel}
        lines.extend([f'## {track}','',f"开发期选择 **{selected}**，λ={opt['strength']}，最终应用系数 {1+opt['strength']*(opt['final_factor']-1):.6f}。选择指标为 `{metric}`。",'',
                      '| 方法 | 总体 WAPE | Bias | WAPE + 绝对 Bias | 平均非零序列 WAPE |','|---|---:|---:|---:|---:|'])
        for name,r in final.iterrows():lines.append(f"| {name} | {r.wape*100:.3f}% | {r.bias*100:+.3f}% | {r.vn1_score*100:.3f}% | {r.mean_series_wape*100:.3f}% |")
        for label,folder,model in [('最近值基线',BASE/tag,'last_value'),('原 ASE',ROOT/'audit/results/retail_ase'/track,'ase_original')]:
            old=pd.read_csv(folder/'metrics.csv');r=old[(old.origin==o)&(old.model==model)].iloc[0]
            z=np.load(folder/f'o{o}_{model}.npz');np.testing.assert_array_equal(z['truth'],a['truth'])
            lines.append(f"| {label}（历史对照） | {r.wape*100:.3f}% | {r.bias*100:+.3f}% | {r.vn1_score*100:.3f}% | {r.mean_series_wape*100:.3f}% |")
        delta=(final.loc[selected,metric]-final.loc['log_l2',metric])*100
        lines.extend(['',f"所选方案相对原 log_l2 的主指标变化：{delta:+.3f} 个百分点；按{'商家' if tag=='vn1' else '门店'}重采样 2,000 次，探索性 95% 区间 [{ci['ci95_pp'][0]:+.3f}, {ci['ci95_pp'][1]:+.3f}]。负值表示更好；区间只反映当前日期下的簇差异，未校正多重比较，不证明未来稳定性。",'',
                      f"全零目标序列数 {int(final.iloc[0].zero_denominator_series)}，仍计入总体误差；只从逐序列 WAPE 均值中排除。`calibrated` 是各自基模型的输出乘一个全局系数，不是多个模型的集成。λ 为零时，该行与原模型相同。",'',
                      '| 基模型 | λ | 第一窗口系数 | 两窗口重估系数 | 第二窗口选择分数 |','|---|---:|---:|---:|---:|'])
        for model,opt in sel['options'].items():lines.append(f"| {model} | {opt['strength']} | {opt['first_origin_factor']:.4f} | {opt['final_factor']:.4f} | {opt['selection_score']*100:.3f}% |")
        if tag=='vn1':
            c=pd.read_csv(out/'client_metrics.csv').pivot(index='client',columns='model',values=metric)
            valid=c[[selected,'log_l2']].dropna();delta_client=valid[selected]-valid['log_l2'];wins=int((delta_client < -1e-12).sum());ties=int((abs(delta_client)<=1e-12).sum())
            lines.append(f'\n按商家官方指标，所选方案对原 log_l2：{wins} 胜、{ties} 平、{len(valid)-wins-ties} 负。该分布不等于独立商业部署验证。')
        lines.append('')
    from ase_objective_alignment import fit_weights,blend
    lines.extend(['## 追加探索：静态 ASE 的目标对齐','',
                  '候选与规则另存 ase_alignment_protocol.json，在该轮选权前固定。复用既有 500 棵树 L1/L2 和状态分量，仅改选权/门控的评价目标；保留 0.05 网格、1.5 倍门控、1/3 收缩。对数组合是目标对齐主对照，算术组合另改变了组合空间；空间按两个开发窗口的收缩后分数选择。没有新增模型训练，也没有动态专家切换。',''])
    for track in ['freshretail_history','freshretail_plan','vn1_history']:
        out=OUT/('aligned_'+track);source=ROOT/'audit/results/retail_ase'/track
        sel=json.loads((out/'selection.json').read_text());assert sel['protocol_sha256']==digest(OUT/'ase_alignment_protocol.json')
        o=sel['final_origin'];origins=[76,83] if track.startswith('freshretail') else [157,170];names=['l2','l1','state']
        p={k:np.concatenate([np.load(source/f'o{t}_{k}.npz')['pred'] for t in origins]) for k in names}
        y=np.concatenate([np.load(source/f'o{t}_l2.npz')['truth'] for t in origins])
        fp={k:np.load(source/f'o{o}_{k}.npz')['pred'] for k in names}
        rebuilt={space:fit_weights(p,y,sel['metric'],space) for space in ['log','arithmetic']}
        assert rebuilt==sel['fits'] and min(rebuilt,key=lambda k:rebuilt[k]['post_shrink_score'])==sel['selected_space']
        table=pd.read_csv(out/'metrics.csv')
        for space in ['log','arithmetic']:
            z=np.load(out/f'{space}.npz');np.testing.assert_allclose(z['pred'],blend(fp,rebuilt[space]['weights'],space),rtol=1e-12,atol=1e-12)
            row=table[table.model==space].iloc[0]
            for k,v in metrics(z['pred'],z['truth']).items():assert np.isclose(v,row[k])
            verification.append({'file':str((out/f'{space}.npz').relative_to(ROOT)),'sha256':digest(out/f'{space}.npz')})
        old=np.load(source/f'o{o}_ase_static.npz')
        for k,v in metrics(old['pred'],old['truth']).items():assert np.isclose(v,table[table.model=='old_static'].iloc[0][k])
        lines.extend([f"### {track}（开发期选择 {sel['selected_space']}）",'',
                      '| 方案 | WAPE | Bias | WAPE + 绝对 Bias |','|---|---:|---:|---:|'])
        for r in table.to_dict('records'):lines.append(f"| {r['model']} | {r['wape']*100:.3f}% | {r['bias']*100:+.3f}% | {r['vn1_score']*100:.3f}% |")
        lines.append('')
        summaries['aligned_'+track]={'selection':sel,'metrics':table.to_dict('records')}
    lines.extend(['## 结论与后续取舍','',
                  '此次没有找到可稳定替代现有配置的优化。直接预测销量减轻了部分低估，但也可能变成高估并增大 WAPE；全局比例校准受时间段变化影响，第二开发窗口没有支持将其用于本轮最终预测。不能因为最终某一候选的某项分数更好，就事后把它改成开发期选中的方案。',
                  '', 'VN1 业务目标对齐的静态算术组合，把官方分数从旧静态版的 53.900% 降到 51.939%，但 WAPE 从 46.148% 升到 50.720%，且仍不如最近值基线的官方分数 50.697%。这是指标之间的取舍，不是全面提升。',
                  '', '建议暂不替换现有主模型。后续优先增加跨季节滚动回测、检验时间衰减训练或分组校准是否能应对偏差变号；库存研究则需区分缺货、停售和真实零需求。这些是待检验方向，不是本轮已实现的提升。公开最终窗口已被反复研究，下一轮确认应增加新时间或新业务来源。',''])
    lines.extend(['## 复现与边界','',
                  '`python repro/loss_optimization.py freshretail history`、`python repro/loss_optimization.py freshretail plan`、`python repro/loss_optimization.py vn1 history`；完成后 `python repro/ase_objective_alignment.py` 生成追加对照，再运行 `python repro/report_loss_optimization.py`。已有输出默认不覆盖，中断恢复可用 `--resume`；基线沿用首轮已保存预测和原始计时。',
                  '', '单模型源码与协议在实验前提交；追加 ASE 对齐协议在该臂拟合前写入。所有候选、开发选择、原始预测、校准预测及校验记录保存在 audit/results/loss_optimization。若计划上线，应先冻结选中的配置，再在新增业务或新的时间留出集验证。旧最终窗口已被多轮研究看到，不能把本轮改进重新包装成盲测发现。'])
    (ROOT/'LOSS-OPTIMIZATION.md').write_text('\n'.join(lines)+'\n')
    (OUT/'verification.json').write_text(json.dumps({'status':'chronology_selection_calibration_and_metrics_reconstructed','files':verification,'runner_sha256':digest(ROOT/'repro/loss_optimization.py')},indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
