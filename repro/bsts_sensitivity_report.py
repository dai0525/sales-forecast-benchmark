"""Compare a pre-specified budget/prior study without modifying the leaderboard."""
import json
from pathlib import Path
import numpy as np
from audit_experiment import ROOT, OUT, metrics
from bench_bsts_mcmc import read_result, diagnostic_summary
import fg_mp

DEST = OUT / 'bsts_sensitivity'
BASE = OUT / 'bsts_mcmc'


def finite_median(values):
    values = np.array([np.nan if v is None else v for v in values], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def summarize(folder, indices, ids, truth, training):
    protocol = json.loads((folder / 'protocol.json').read_text())
    rows = [read_result(folder / f'{i+1:03d}.json', ids[i]) for i in indices]
    ds = [diagnostic_summary(r) for r in rows]
    pred = np.maximum([r['mean'] for r in rows], 0)
    pred_diag = [json.loads((folder / f'{i+1:03d}_predictive.json').read_text()) for i in indices]
    def passes(d):
        return all(v.get('rhat') is not None and v.get('ess_bulk') is not None
                   and v.get('ess_tail') is not None and v['rhat'] <= 1.01
                   and v['ess_bulk'] >= 400 and v['ess_tail'] >= 400 for v in d)
    variable = {}
    for r in rows:
        d = r['diagnostics']
        for j, name in enumerate(d['variable']):
            if name.startswith('predictive.'):
                continue
            variable.setdefault(name, []).append({k: d[k][j] for k in ['rhat','ess_bulk','ess_tail']})
    variable = {name: {'passes': sum(passes([v]) for v in values),
                      'median_rhat': finite_median([v['rhat'] for v in values]),
                      'median_bulk_ess': finite_median([v['ess_bulk'] for v in values])}
                for name, values in variable.items()}
    chain = np.array([r['chain_means'] for r in rows])
    # The training mean scale never uses the official test outcomes.
    scale = np.maximum(np.mean(np.abs(training[indices]), axis=1), 1e-12)
    mcse = [next(v['mcse_mean'] for v in d if v['variable'] == 'mean13') for d in pred_diag]
    return {'protocol': protocol, 'ids': [ids[i] for i in indices],
            'n_series': len(indices), 'metrics': metrics(pred, truth[indices]),
            'monitored_pass': sum(d['passes_all'] for d in ds),
            'predictive_pass': sum(passes(d) for d in pred_diag),
            'median_max_rhat': finite_median([d['max_rhat'] for d in ds]),
            'median_min_bulk_ess': finite_median([d['min_bulk_ess'] for d in ds]),
            'variable_diagnostics': variable,
            'mean_between_chain_forecast_gap_over_train_scale': float(np.mean(np.mean(abs(chain[:,0]-chain[:,1]), axis=1)/scale)),
            'median_mean13_mcse_over_train_scale': finite_median([m/s if m is not None else None for m,s in zip(mcse,scale)]),
            'fit_predict_sum_series_s': sum(r['runtime_s'] for r in rows),
            'failed_ids': [ids[i] for i,d in zip(indices,ds) if not d['passes_all']]}, pred


def main():
    plan = json.loads((DEST/'protocol.json').read_text())
    indices = np.array(plan['indices_one_based']) - 1
    training, truth, ids = fg_mp.load()
    arms = [('默认先验 / 1000轮', BASE/'trend_harmonic_default_1000'),
            ('默认先验 / 4000轮', BASE/'trend_harmonic_spaced12_default_4000'),
            ('正则化先验 / 4000轮', BASE/'trend_harmonic_spaced12_regularized_4000')]
    results = {}; predictions = {}
    for name, folder in arms:
        results[name], predictions[name] = summarize(folder, indices, ids, truth, training)
    reference = predictions[arms[0][0]]
    scale = np.maximum(np.mean(np.abs(training[indices]),axis=1),1e-12)
    for name, pred in predictions.items():
        results[name]['mean_forecast_change_vs_short_default_over_train_scale'] = float(np.mean(np.mean(abs(pred-reference),axis=1)/scale))
    saved = np.load(OUT/'state_predictions.npz')
    if list(saved['ids']) != ids:
        raise ValueError('Baseline ID mismatch')
    baselines = {name: metrics(saved[name][indices],truth[indices]) for name in ['State-Fourier','UCM52-saved']}
    (DEST/'summary.json').write_text(json.dumps({'protocol':plan,'arms':results,'same_ids_baselines':baselines},ensure_ascii=False,indent=2)+'\n')
    lines = ['# BSTS 收敛与先验敏感性补充实验', '',
             '沿用此前固定的12个等间距索引；两个长链配置在运行前写入协议。使用原版R bsts、相同模型与种子，每链弃置前半样本，比较原1000轮与4000轮；两条独立链。正则化将状态创新先验的 sample.size 从0.01改为32，先验尺度仍仅来自训练数据。', '',
             '**这同时检验计算预算和先验敏感性。改变先验会改变后验目标，不能把正则化的效果单独归因于采样算法改进。** 两组均运行，不用测试成绩选择模型，不改全量榜单。', '',
             '| 配置 | 参数/末端水平斜率通过 | 预测诊断通过 | 每条最大 Rhat 的中位数 | 每条最小 bulk ESS 的中位数 | 平均 WAPE |',
             '|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['monitored_pass']}/12 | {r['predictive_pass']}/12 | {r['median_max_rhat']:.3f} | {r['median_min_bulk_ess']:.1f} | {100*r['metrics']['wape_mean']:.4f}% |")
    lines += ['', '阈值：rank-normalized split Rhat≤1.01，bulk/tail ESS≥400，必须每个监控变量都通过。没有监控全部末端谐波状态，因而即使通过也不等于验证了完整联合后验。预测样本含独立未来噪声，预测诊断通过不能替代参数诊断。', '',
              '## 哪些参数仍存在问题', '', '| 配置 | 变量 | 通过数 | Rhat中位数 | bulk ESS中位数 |','|---|---|---:|---:|---:|']
    for name,r in results.items():
        for variable,d in r['variable_diagnostics'].items():
            lines.append(f"| {name} | {variable} | {d['passes']}/12 | {d['median_rhat']:.3f} | {d['median_bulk_ess']:.1f} |")
    lines += ['', '## 预测稳定性', '', '| 配置 | 两链预测差 / 训练均值 | 相对短链的预测变化 / 训练均值 | 预测13步均值MCSE / 训练均值（中位数） |', '|---|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {100*r['mean_between_chain_forecast_gap_over_train_scale']:.3f}% | {100*r['mean_forecast_change_vs_short_default_over_train_scale']:.3f}% | {100*r['median_mean13_mcse_over_train_scale']:.3f}% |")
    lines += ['', '差值先逐序列归一化再等权平均。MCSE在混合不良时可能不可靠，不能作为可信误差保证。两种预算使用相同种子，因此不能把它们当作独立重复实验；两种预算均使用一半burn-in，保留样本区间不同。', '',
              '## 对照范围与结论', '',
              f"相同12条的 State-Fourier WAPE 为 {100*baselines['State-Fourier']['wape_mean']:.4f}%，UCM52 为 {100*baselines['UCM52-saved']['wape_mean']:.4f}%。这些是固定子集的探索性成绩，不能外推为全359条结果，不能从12条中选测试最优先验后宣称方法提升。", '']
    default = results[arms[1][0]]; regularized = results[arms[2][0]]
    lines += [f"延长默认先验采样后，监控诊断通过 {default['monitored_pass']}/12；增强正则化后通过 {regularized['monitored_pass']}/12。"]
    if default['monitored_pass'] < 12 or regularized['monitored_pass'] < 12:
        lines += ['本轮仍未得到在该子集全部通过监控诊断的配置。按预先记录的停止规则，不把未解决混合问题的配置扩展成新的全量榜单。当前可确认的是短历史下的弱识别/采样限制仍需处理，不能判定BSTS整体无效。']
    else:
        lines += ['两组监控诊断均通过；这支持继续独立验证，但尚未覆盖所有状态或更多序列，不能直接声称全量后验收敛。']
    lines += ['', '下一步若继续研究，应优先做训练侧先验预测检查、监控完整季节状态，并对同一后验目标尝试边缘化状态后的参数采样；固定季节方差或删除随机季节会改变模型，须另列结构消融。在具备日期/价格/库存的独立零售数据前，不再反复围绕已查看的M4测试集选模型。', '',
              '复现：', '', '```bash',
              'python repro/bench_bsts_mcmc.py --kind trend_harmonic_spaced12 --iterations 4000 --prior default',
              'python repro/bench_bsts_mcmc.py --kind trend_harmonic_spaced12 --iterations 4000 --prior regularized',
              'python repro/bsts_sensitivity_report.py', '```', '',
              '协议、逐变量诊断与机器可读结果：`audit/results/bsts_sensitivity/`。逐序列样本保存在相应 `audit/results/bsts_mcmc/trend_harmonic_spaced12_*_4000/`。']
    (ROOT/'BSTS-SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
