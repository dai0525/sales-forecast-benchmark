"""Prepare data, run original R bsts, and export an explicitly diagnosed baseline.
Requires R packages bsts, posterior, jsonlite (see environment-bsts.yml).
"""
import argparse,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import fg_mp
from audit_experiment import ROOT,OUT,metrics
FOLDER=OUT/'bsts_mcmc'


def read_result(path,expected_id):
    r=json.loads(Path(path).read_text())
    if not r.get('ok'):raise ValueError(f"BSTS failed for {expected_id}: {r.get('error')}")
    if r['id']!=expected_id:raise ValueError('BSTS ID mismatch')
    for name in ['mean','median','lower','upper']:
        if np.asarray(r[name]).shape!=(13,) or not np.isfinite(r[name]).all():raise ValueError('Invalid BSTS forecast')
    if np.any(np.asarray(r['lower'])>r['upper']):raise ValueError('Invalid posterior interval')
    return r


def diagnostic_summary(r):
    d=r['diagnostics']
    keep=[i for i,v in enumerate(d['variable']) if not v.startswith('predictive.')]
    if not keep:raise ValueError('Missing monitored parameter diagnostics')
    d={k:[v[i] for i in keep] for k,v in d.items()}
    rh=np.array([np.nan if x is None else x for x in d['rhat']])
    bulk=np.array([np.nan if x is None else x for x in d['ess_bulk']])
    tail=np.array([np.nan if x is None else x for x in d['ess_tail']])
    bad=~(np.isfinite(rh)&np.isfinite(bulk)&np.isfinite(tail)&(rh<=1.01)&(bulk>=400)&(tail>=400))
    def finite_stat(v,fn):
        return float(fn(v[np.isfinite(v)])) if np.isfinite(v).any() else None
    return {'max_rhat':finite_stat(rh,np.max),'min_bulk_ess':finite_stat(bulk,np.min),'min_tail_ess':finite_stat(tail,np.min), 'passes_all':bool(not bad.any()),'failed_variables':[v for v,b in zip(d['variable'],bad) if b]}


def is_main_benchmark(protocol, indices, total):
    """Only the frozen full experiment may replace the published baseline."""
    return (protocol.get('kind') == 'trend_harmonic'
            and protocol.get('priors') == 'default'
            and protocol.get('niter') == 1000
            and protocol.get('burn') == 500
            and protocol.get('chains') == 2
            and protocol.get('horizon') == 13
            and total == 359
            and np.array_equal(indices, np.arange(total)))


def export(folder):
    protocol=json.loads((folder/'protocol.json').read_text());train,truth,ids=fg_mp.load()
    indices=np.atleast_1d(protocol['indices']).astype(int)-1
    if not len(indices) or len(set(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(ids)):
        raise ValueError('Invalid or duplicate BSTS indices')
    if list(np.atleast_1d(protocol['ids']))!=[ids[i] for i in indices]:
        raise ValueError('BSTS protocol ID mismatch')
    rows=[read_result(folder/f'{i+1:03d}.json',ids[i]) for i in indices]
    if len(rows)!=len(indices):raise ValueError('Incomplete run')
    preds=np.maximum(np.array([r['mean'] for r in rows]),0)
    median=np.maximum(np.array([r['median'] for r in rows]),0)
    lower=np.array([r['lower'] for r in rows]);upper=np.array([r['upper'] for r in rows]);actual=truth[indices]
    ds=[diagnostic_summary(r) for r in rows]
    predictive=[]
    for i in indices:
        p=folder/f'{i+1:03d}_predictive.json'
        if not p.exists():raise ValueError('Missing posterior predictive diagnostics')
        d=json.loads(p.read_text());good=all(v['rhat'] is not None and v['ess_bulk'] is not None and v['ess_tail'] is not None and v['rhat']<=1.01 and v['ess_bulk']>=400 and v['ess_tail']>=400 for v in d)
        predictive.append({'id':ids[i],'passes_all_horizons':good,'max_rhat':max(v['rhat'] for v in d) if all(v['rhat'] is not None for v in d) else None,'mean13_mcse':next(v['mcse_mean'] for v in d if v['variable']=='mean13')})
    chain=np.array([r['chain_means'] for r in rows])
    # JSON matrix is two chains x 13 horizons per series.
    if chain.shape!=(len(rows),2,13):raise ValueError('Invalid chain mean shape')
    result={'name':'BSTS (R bsts MCMC, LLT + harmonic season K4)' if 'harmonic' in protocol['kind'] else 'BSTS (R bsts MCMC, LLT + dummy seasonal52), fixed subset',
            'protocol':protocol,'n_series':len(rows),'ids':[ids[i] for i in indices],**metrics(preds,actual),
            'posterior_median_metrics':metrics(median,actual),'point_forecast':'posterior predictive mean, clipped at zero after averaging',
            'raw_negative_point_forecasts':int((np.array([r['mean'] for r in rows])<0).sum()),
            'interval95_empirical_coverage':float(((actual>=lower)&(actual<=upper)).mean()),'interval95_mean_width':float((upper-lower).mean()),
            'diagnostics':{'thresholds':{'rhat_max':1.01,'bulk_ess_min':400,'tail_ess_min':400},'all_monitored_variables_pass':sum(d['passes_all'] for d in ds),'predictive_all_horizons_pass':sum(d['passes_all_horizons'] for d in predictive),'series':ds,'predictive':predictive},
            'chain_metrics':[metrics(np.maximum(chain[:,c,:],0),actual) for c in range(2)],'fit_predict_sum_series_s':float(sum(r['runtime_s'] for r in rows)),
            'run':json.loads((folder/'run.json').read_text()),'status':'exploratory: finite-budget MCMC; convergence diagnostics limit interpretation; not a converged gold standard'}
    baselines=np.load(OUT/'state_predictions.npz')
    if list(baselines['ids'])!=ids:raise ValueError('Baseline ID mismatch')
    result['same_ids_baselines']={k:metrics(baselines[k][indices],actual) for k in ['State-Fourier','UCM52-saved','FG-original','Naive']}
    rng=np.random.default_rng(42);diff=(abs(preds-actual).sum(1)-abs(baselines['State-Fourier'][indices]-actual).sum(1))/abs(actual).sum(1)
    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
    result['paired_vs_state_fourier']={'delta':float(diff.mean()),'series_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'caveat':'Treats saved MCMC point predictions as fixed; excludes Monte Carlo uncertainty and assumes independent series.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    np.savez_compressed(folder/'predictions.npz',ids=np.array(result['ids']),preds=preds,median=median,lower=lower,upper=upper,truth=actual,chain_means=chain)
    if is_main_benchmark(protocol, indices, len(ids)):
        np.savez_compressed(ROOT/'data/weekly_bsts_mcmc_preds.npz',ids=np.array(result['ids']),preds=preds)
        (ROOT/'repro/metrics_bsts_mcmc.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['protocol','ids','diagnostics','run']},indent=2))
    print('diagnostic pass:',result['diagnostics']['all_monitored_variables_pass'],'/',len(rows),'; predictive pass:',result['diagnostics']['predictive_all_horizons_pass'])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rscript',default=shutil.which('Rscript'));ap.add_argument('--iterations',type=int,default=1000);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--kind',choices=['trend_harmonic','trend_harmonic_spaced12','trend_seasonal52'],default='trend_harmonic');ap.add_argument('--prior',choices=['default','regularized'],default='default');ap.add_argument('--export-only',action='store_true');args=ap.parse_args()
    if not args.rscript:ap.error('Rscript not found. Activate environment-bsts.yml or use --rscript PATH.')
    FOLDER.mkdir(parents=True,exist_ok=True);y,t,ids=fg_mp.load()
    # Never silently overwrite a cached experiment's training inputs.
    if (FOLDER/'train.csv').exists():
        if not np.array_equal(np.loadtxt(FOLDER/'train.csv',delimiter=','),y):raise ValueError('Stored BSTS training inputs differ; use a separate experiment directory')
        if json.loads((FOLDER/'ids.json').read_text())!=ids:raise ValueError('Stored BSTS IDs differ')
    elif args.export_only:raise ValueError('Missing original BSTS training inputs')
    else:
        np.savetxt(FOLDER/'train.csv',y,delimiter=',');(FOLDER/'ids.json').write_text(json.dumps(ids))
    if not args.export_only and not (FOLDER/'test.csv').exists():np.savetxt(FOLDER/'test.csv',t,delimiter=',')
    folder=FOLDER/f'{args.kind}_{args.prior}_{args.iterations}'
    if not args.export_only:subprocess.run([args.rscript,str(ROOT/'repro/bench_bsts_mcmc.R'),str(ROOT),args.kind,str(args.iterations),str(args.workers),'0',args.prior],check=True)
    subprocess.run([args.rscript,str(ROOT/'repro/bsts_predictive_diagnostics.R'),str(folder)],check=True)
    export(folder)
if __name__=='__main__':main()
