"""Build the BSTS report section from saved audited results."""
import json
from pathlib import Path
import numpy as np
from audit_experiment import ROOT,OUT


def main():
    full=json.loads((ROOT/'repro/metrics_bsts_mcmc.json').read_text())
    subset=json.loads((OUT/'bsts_mcmc/trend_seasonal52_default_1000/summary.json').read_text())
    lines=['## 本次运行结果（由保存预测自动生成）','','主实验：359条；R原版BSTS，LLT + 52周随机谐波（4阶）。主指标为逐序列WAPE均值。','', '| 方法 | 平均 WAPE | 总量 WAPE |','|---|---:|---:|']
    for name,m in [('BSTS MCMC†',full),('State-Fourier',full['same_ids_baselines']['State-Fourier']),('UCM52（MLE）',full['same_ids_baselines']['UCM52-saved']),('原FG',full['same_ids_baselines']['FG-original'])]:lines.append(f"| {name} | {m['wape_mean']*100:.4f}% | {m['wape_aggregate']*100:.4f}% |")
    d=full['diagnostics'];p=full['paired_vs_state_fourier']
    lines+=['',f"† 默认先验、2链×1000轮（各弃500轮）的有限预算结果。监控参数/末端状态全部通过阈值：**{d['all_monitored_variables_pass']}/359**；13个预测步长及均值全部通过阈值：**{d['predictive_all_horizons_pass']}/359**。因此这不是已充分收敛的BSTS金标准，不能把分数当作该方法的准确率上限。",'',f"后验预测中位数作为点预测的辅助WAPE为 {100*full['posterior_median_metrics']['wape_mean']:.4f}%；两条链各自均值预测的WAPE分别为 {100*full['chain_metrics'][0]['wape_mean']:.4f}% / {100*full['chain_metrics'][1]['wape_mean']:.4f}%。主指标仍固定使用合并后验均值，没有依据测试择优。",'',f"95%后验预测区间在本测试段的经验覆盖率 {100*full['interval95_empirical_coverage']:.2f}%，不能仅凭一次覆盖率认为区间已经校准；短链和模型假设仍影响区间。",'',f"BSTS−State-Fourier 的平均WAPE差为 {100*p['delta']:+.4f}pp，按序列bootstrap的95%区间 [{100*p['series_bootstrap_95'][0]:+.4f}, {100*p['series_bootstrap_95'][1]:+.4f}]pp。该区间固定了已保存的MCMC点预测，不包含采样误差，也假设序列独立。",'', '### 原论文风格结构的固定12条子集','', '| 方法（相同12条） | 平均 WAPE |','|---|---:|']
    for name,m in [('BSTS LLT + dummy seasonal52†',subset),('State-Fourier',subset['same_ids_baselines']['State-Fourier']),('UCM52（MLE）',subset['same_ids_baselines']['UCM52-saved'])]:lines.append(f"| {name} | {m['wape_mean']*100:.4f}% |")
    lines+=['',f"子集参数/末端状态全部通过 {subset['diagnostics']['all_monitored_variables_pass']}/12；预测诊断全部通过 {subset['diagnostics']['predictive_all_horizons_pass']}/12。此子集不外推到全量。",'', '### 耗时的解释','',f"全量逐序列拟合与预测的进程elapsed时间之和为 {full['fit_predict_sum_series_s']:.1f}秒，包含并发争用；它不是整批wall time，不能直接除以359或线程数作为独立硬件性能。续跑的批次时间和缓存状态见run.json；此前pilot和环境安装不在其中。网页不提供一个容易误解的单一速度倍数。",'', '### 本轮能够得出的结论','', '已将原作者包的真正参数/状态后验采样接入benchmark，并保存可复核的预测、区间和样本。有限预算下的点预测可以作为探索性参考；参数混合仍是主要限制，不能据此证明BSTS总体不如State-Fourier，也不能把现有模型与BSTS的差异单独归因于是否采用贝叶斯推断。若后续需要严格模型优劣结论，必须先改善采样/弱识别、验证先验敏感性，再按预先规定预算重跑。']
    path=ROOT/'BSTS-MCMC.md';text=path.read_text();marker='## 本次运行结果（由保存预测自动生成）';text=text.split(marker)[0].rstrip()+'\n\n'+'\n'.join(lines)+'\n';path.write_text(text)
    (OUT/'bsts_mcmc/score_table.md').write_text('\n'.join(lines[:10])+'\n')
    print('updated',path)
if __name__=='__main__':main()
