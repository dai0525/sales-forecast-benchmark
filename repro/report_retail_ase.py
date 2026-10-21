"""Reconstruct ASE selections and forecasts from stored components; report added comparisons."""
import json
import numpy as np
import pandas as pd
from retail_ase import ROOT,OUT,BASE,ase,metrics,select_weights
from audit_freshretail import digest


def main():
    lines=['# ASE 在新增零售数据上的追加对照','',
           '本次补齐前一轮遗漏的 ASE，属于看过首轮最终结果后的追加对照，不是新盲测。协议在本次拟合前固定；最终测试标签没有用于状态参数或权重选择。',
           '', '## 结论','',
           '原 ASE 在本轮三个场景的总体 WAPE 均略优于其同配置的 L1/L2 单模型，但这些小幅差异未做独立时间验证，不能声称稳定显著优势。静态改进版只在叮咚纯历史场景略好于原 ASE，另两个场景略差，未表现出稳定改进。',
           '', '简单等权对数组合在叮咚纯历史场景更好；VN1 的算术等权组合在官方评分上更好。这说明组合本身的收益不能直接归因于复杂选权。VN1 原 ASE 总体 WAPE 45.847%，仍略差于首轮 250 棵树单一 LightGBM 的 45.666%；原 ASE 官方分数 53.740% 也未胜过最近值基线的 50.697%。算术等权的官方分数 50.491% 只是在这个已公开时间块上的小幅改善。',
           '', '建议保留原 ASE 和简单等权作为对照，不把静态版宣布为全面升级。后续更值得研究选权目标与业务评价的一致性：原 ASE 优化非零序列平均 WAPE，VN1 官方同时惩罚总量偏差。任何新选权或校准方案都应在开发期确定，并用新的时间/业务来源验证。',
           '', '## 方法与可比性','',
           '- 原版 ASE：原有 L1/L2 LightGBM（500 棵树，原有参数）+ State Fourier；状态参数在最近开发窗口选定；沿用原有三阶段权重搜索、1.5 倍门控、0.05 网格和 1/3 等权收缩。',
           '- 改进静态版：状态参数在两个开发窗口汇总选定；同一组 L1/L2 模型加该状态模型，在两个开发窗口拟合一套覆盖所有预测步长的固定权重。',
           '- 保留原代码的选权目标：非零目标序列的平均 WAPE。这不同于总体 WAPE，也不同于 VN1 官方 WAPE + |Bias|。全零序列未参与选权，但其预测错误仍计入最终总体误差。',
           '- 复用本轮零售数据适配器、特征与训练起点。旧 M4/Rossmann 特征无法原样套用其他业务，因此是原算法的跨数据集移植；没有声称逐字节复现旧数据结果。',
           '- Rossmann 的 Open 门控与只在营业日训练的分量不适用这两套数据，不能把缺货当作关店。本次静态版移植的是多窗口统一权重机制，不伪造营业状态。',
           '- 树模型使用原 ASE 的 500 棵树配置，与首轮单模型 250 棵树不同。为区分基模型升级和组合收益，下表同时列出本次同配置 L1/L2 与开发期选择的单一模型。',
           '- 叮咚沿用固定 3,000 条 ID 样本；VN1 使用全部 15,053 条。叮咚纯历史与条件性促销计划场景分开，促销计划场景仍未验证计划在真实预测起点已经可知。','']
    verification=[];summary={}
    for track in ['freshretail_history','freshretail_plan','vn1_history']:
        out=OUT/track;tag='vn1' if track.startswith('vn1') else 'freshretail'
        sel=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['status']=='completed' and sel['protocol_sha256']==digest(OUT/'protocol.json')
        table=pd.read_csv(out/'metrics.csv');o=sel['final_origin'];H=7 if tag=='freshretail' else 13
        dev=sorted(table.loc[table.origin!=o,'origin'].unique());val={};truth={}
        for oo in dev:
            val[oo]={}
            for name,alias in [('l2','l2'),('l1','l1'),('original_state','state'),('state','pooled_state')]:
                z=np.load(out/f'o{oo}_{name}.npz');val[oo][alias]=z['pred'];truth[oo]=z['truth']
        rebuilt=select_weights(val,truth)
        for key in ['original','static','development_aggregate_wape','best_single']:assert rebuilt[key]==sel[key],(track,key)
        state_candidates=sel['state_search']
        assert min(state_candidates,key=lambda x:x['latest'])['cfg']==sel['original_state_cfg']
        assert min(state_candidates,key=lambda x:x['pooled'])['cfg']==sel['pooled_state_cfg']
        comp={k:np.load(out/f'o{o}_{k}.npz')['pred'] for k in ['l2','l1','original_state','state']}
        orig={'l2':comp['l2'],'l1':comp['l1'],'state':comp['original_state']};pooled={k:comp[k] for k in ['l2','l1','state']}
        checks={'ase_original':ase._combine(orig,sel['original'],ase._stages(H)),
                'ase_static':ase._combine(pooled,sel['static'],{'all':list(range(H))}),
                'equal_log':ase._combine_arr(orig,np.ones(3)/3),'equal_arithmetic':np.mean(list(orig.values()),axis=0),
                'selected_single':comp[sel['best_single']]}
        for name,pred in checks.items():np.testing.assert_allclose(pred,np.load(out/f'o{o}_{name}.npz')['pred'],rtol=1e-12,atol=1e-12)
        for r in table.to_dict('records'):
            path=out/f"o{r['origin']}_{r['model']}.npz";z=np.load(path)
            for k,v in metrics(z['pred'],z['truth']).items():assert np.isclose(v,r[k],rtol=1e-10,atol=1e-10),(track,k)
            verification.append({'file':str(path.relative_to(ROOT)),'sha256':digest(path)})
        final=table[table.origin==o].set_index('model');summary[track]=final.to_dict('index')
        lines.extend([f'## {track}','',f"开发期选择的单一模型：`{sel['best_single']}`。最终预测起点为 {o}。",'',
                      '| 模型 | 总体 WAPE | 平均非零序列 WAPE | Bias | WAPE + |Bias| |','|---|---:|---:|---:|---:|'])
        # Escape pipe in metric header for valid Markdown.
        lines[-2]='| 模型 | 总体 WAPE | 平均非零序列 WAPE | Bias | WAPE + 绝对 Bias |'
        for name,r in final.iterrows():lines.append(f"| {name} | {r.wape*100:.3f}% | {r.mean_series_wape*100:.3f}% | {r.bias*100:+.3f}% | {r.vn1_score*100:.3f}% |")
        naive=pd.read_csv(BASE/tag/'metrics.csv');naive=naive[(naive.origin==o)&(naive.model=='last_value')].iloc[0]
        lines.append(f"| 首轮最近值基线（相同测试目标） | {naive.wape*100:.3f}% | {naive.mean_series_wape*100:.3f}% | {naive.bias*100:+.3f}% | {naive.vn1_score*100:.3f}% |")
        ref=np.load(BASE/tag/f'o{o}_last_value.npz');new=np.load(out/f'o{o}_ase_original.npz');np.testing.assert_array_equal(ref['truth'],new['truth'])
        pd.testing.assert_frame_equal(pd.read_csv(BASE/tag/'series_ids.csv'),pd.read_csv(out/'series_ids.csv'))
        lines.extend(['',f"全零目标序列数：{int(final.iloc[0].zero_denominator_series)}；仍计入总体误差。最后一列在 VN1 是官方评价口径，在叮咚仅作偏差诊断。",'',
                      '原版阶段权重：`'+json.dumps(sel['original'])+'`。','',
                      '静态版统一权重：`'+json.dumps(sel['static']['all'])+'`。','',
                      '原版状态配置：`'+json.dumps(sel['original_state_cfg'])+'`；静态版状态配置：`'+json.dumps(sel['pooled_state_cfg'])+'`。',''])
        base=manifest['final_base_seconds'];orig_time=base['l2']+base['l1']+base['original_state']+float(final.loc['ase_original','seconds']);static_time=base['l2']+base['l1']+base['state']+float(final.loc['ase_static','seconds'])
        lines.append(f"最终重拟合并预测的分量耗时之和：原 ASE 约 {orig_time:.2f} 秒，静态版约 {static_time:.2f} 秒。开发状态搜索 {manifest['state_search_seconds']:.2f} 秒，两个版本权重搜索合计 {manifest['weight_selection_seconds']:.2f} 秒。两个版本共用树分量，不能把组合计算的毫秒耗时当成整个模型成本；各任务并发，非严格速度基准。")
        if tag=='vn1':
            c=pd.read_csv(out/'client_metrics.csv').pivot(index='client',columns='model',values='wape')
            for a,b in [('ase_original','selected_single'),('ase_static','selected_single'),('ase_static','ase_original')]:
                pair=c[[a,b]].dropna();wins=int(((pair[a]-pair[b]) < -1e-12).sum());ties=int(np.isclose(pair[a],pair[b],rtol=0,atol=1e-12).sum())
                lines.append(f"\n按商家总体 WAPE，{a} 对 {b}：{wins}/{len(pair)} 家胜出，{ties} 家数值相等（探索性分布）。")
        lines.append('')
    lines.extend(['## 复现','',
                  '`python repro/retail_ase.py freshretail history`、`python repro/retail_ase.py freshretail plan`、`python repro/retail_ase.py vn1 history`；完成后运行 `python repro/report_retail_ase.py`。为避免混用，实验脚本默认拒绝覆盖已有输出目录。仅对未完成且已有冻结 selection.json 的任务可加 `--resume`；恢复时校验目标、参数、特征和冻结选择，复用基模型原始训练耗时。本轮 VN1 因环境重启恢复过一次，见 resume.json；恢复校验的额外耗时未计入模型时间。',
                  '', '报告程序从保存的开发预测重建权重与单模型选择，从最终分量重建所有组合预测，并重算所有指标。它同时核对与首轮测试的 ID 顺序和真实目标一致。数据依赖、下载校验和与抽样规则沿用 [首轮报告](RETAIL-EXPANSION-RESULTS.md)。',
                  '', '本次补跑不是新方法，也没有验证潜在需求恢复或因果归因。任何组合优劣结论均限于本次移植、参数和测试时间段。'])
    (ROOT/'RETAIL-ASE-RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'verification.json').write_text(json.dumps({'status':'weights_predictions_metrics_and_ids_verified','files':verification,'initial_runner_commit':'fd161a4','verification_time_source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'repro/retail_ase.py',ROOT/'repro/retail_expansion.py',ROOT/'repro/hybrid/ase.py']}},indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
