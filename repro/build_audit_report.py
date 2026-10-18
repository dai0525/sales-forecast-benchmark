"""Export a leaderboard from auditable prediction arrays, never hand-enter scores."""
import csv,json,sys,hashlib,subprocess,importlib.metadata as md
from pathlib import Path
import numpy as np
from audit_experiment import metrics,OUT,ROOT
import fg_mp

def main():
 y,t,ids=fg_mp.load(); state=json.loads((OUT/'state_experiment.json').read_text())
 z=np.load(OUT/'state_predictions.npz'); rows=[]
 assert list(z['ids'])==ids
 labels={'State-Fourier':'联合状态 Fourier','FG-original':'FG backfitting 原型','UCM52-saved':'UCM52（MLE）','Naive':'Naive','SeasonalNaive52':'SeasonalNaive52','MovingAvg8':'MovingAvg8'}
 for key,label in labels.items():
  r={'id':key,'name':label,'short':label,'family':'state' if key in ['State-Fourier','FG-original','UCM52-saved'] else 'statistical',**metrics(z[key],t),'reproduced':True}
  r['runtime_s']=state['results'][key].get('fit_predict_s')
  if key=='UCM52-saved':r['runtime_s']=json.loads((ROOT/'repro/metrics_bsts52.json').read_text())['fit_runtime_s']
  rows.append(r)
 deep=json.loads((ROOT/'repro/metrics_deep.json').read_text())
 files=[('lightgbm','LightGBM 全训练行','weekly_lgbm_preds.npz','gbdt')]+[(k,d['name'],f'weekly_{k}_preds.npz','deep') for k,d in deep.items()]
 for key,name,file,family in files:
  a=np.load(ROOT/'data'/file);assert list(a['ids'])==ids
  r={'id':key,'name':name,'short':name,'family':family,**metrics(np.maximum(a['preds'],0),t),'reproduced':True}
  r['runtime_s']=deep[key]['runtime_s'] if key in deep else json.loads((ROOT/'repro/metrics_base.json').read_text())['LightGBM']['runtime_s']
  rows.append(r)
 mcmc_path=ROOT/'repro/metrics_bsts_mcmc.json'
 if mcmc_path.exists():
  b=json.loads(mcmc_path.read_text());a=np.load(ROOT/'data/weekly_bsts_mcmc_preds.npz')
  assert list(a['ids'])==ids
  rows.append({'id':'bsts_mcmc','name':'BSTS MCMC†（有限预算，收敛受限）','short':'BSTS MCMC†','family':'state',**metrics(a['preds'],t),'reproduced':True,'runtime_s':None,'status':b['status'],'diagnostic_pass_series':b['diagnostics']['all_monitored_variables_pass']})
 for r in rows:
  r['wape']=r['wape_mean'];r['n_series']=len(ids);r['mase']=None
 rows.sort(key=lambda r:r['wape'])
 result={'dataset':{'name':'M4-Weekly（审计重跑）','n_series':359,'horizon':13,'input_size':200},'models':rows,
 'note':'BSTS MCMC†如出现，仅为有限预算试跑，参数收敛诊断未全部通过，不能视为已收敛金标准；其R运行时间见单独报告。逐序列 WAPE 均值；短于200点的65条沿用首值左补。深度模型是简化实现，时间验证、260步、单种子，最后13训练点未用于梯度更新。原始报告为历史记录；新模型官方测试领先不等于所有历史窗口领先。耗时含共享CPU负载，模型调参预算不同。'}
 (OUT/'benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 (ROOT/'web/data/bench.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (OUT/'benchmark.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['name','wape_mean','wape_aggregate','smape','bias_aggregate','runtime_s']);w.writeheader()
  for r in rows:w.writerow({k:r[k] for k in w.fieldnames})
 lines=['| 模型 | 平均 WAPE | 总量 WAPE |','|---|---:|---:|']+[f"| {r['name']} | {100*r['wape_mean']:.3f}% | {100*r['wape_aggregate']:.3f}% |" for r in rows]
 (OUT/'benchmark.md').write_text('\n'.join(lines)+'\n')
 if (OUT/'rolling.json').exists():
  roll=json.loads((OUT/'rolling.json').read_text());ls=['# 嵌套历史回测','\n每个外层窗口仅使用之前三个窗口选择新模型参数。','\n| 外层原点 | 联合状态 Fourier | UCM52 | 新−UCM (pp) |','|---|---:|---:|---:|']
  for k,r in roll.items():
   a,b=r['State-Fourier']['wape_mean'],r['UCM52']['wape_mean'];ls.append(f'| {k} | {a*100:.3f}% | {b*100:.3f}% | {(a-b)*100:+.3f} |')
  if len(roll)==3:
   a=np.mean([r['State-Fourier']['wape_mean'] for r in roll.values()]);b=np.mean([r['UCM52']['wape_mean'] for r in roll.values()]);ls.append(f'| 三窗口均值 | {a*100:.3f}% | {b*100:.3f}% | {(a-b)*100:+.3f} |')
   ls.append('\n结论：官方测试有提升，但历史窗口未显示稳定优于UCM；暂不以新模型替换全部UCM。')
  (OUT/'rolling_summary.md').write_text('\n'.join(ls)+'\n')
 packages=['numpy','pandas','scipy','lightgbm','statsmodels','joblib','torch','pytest']
 env={p:md.version(p) for p in packages}
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'packages':env},indent=2))
 (ROOT/'requirements-audit.txt').write_text('# Tested audit dependencies; install CPU torch first as described in README.\n'+'\n'.join(f'{p}=={v.split("+")[0]}' for p,v in env.items())+'\n')
 tracked=sorted(set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()) - {'audit/results/inventory.json'})
 inventory=[{'path':f,'bytes':(ROOT/f).stat().st_size,'sha256':hashlib.sha256((ROOT/f).read_bytes()).hexdigest()} for f in tracked if (ROOT/f).is_file()]
 (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()
