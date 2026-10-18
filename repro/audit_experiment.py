"""Frozen train-only candidate grid, three rolling origins, one official test.
Run: OPENBLAS_NUM_THREADS=1 python repro/audit_experiment.py
"""
import itertools,json,time,hashlib,platform,sys
from pathlib import Path
import numpy as np
import fg_mp
from fg_state import forecast
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/results'

def metrics(p,t):
    if p.shape!=t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Invalid predictions')
    den=np.abs(t).sum(1)
    if (den==0).any(): raise ValueError('WAPE undefined for zero-total series')
    e=abs(p-t)
    return {'wape_mean':float((e.sum(1)/den).mean()),'wape_aggregate':float(e.sum()/den.sum()),
            'smape':float(np.mean(2*e/np.maximum(abs(p)+abs(t),1e-9))),
            'bias_aggregate':float((p-t).sum()/den.sum())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    y,truth,ids=fg_mp.load(); origins=[161,174,187]
    configs=[dict(harmonics=k,qr=q,trend_q=tq,damp=d) for k,q,tq,d in
             itertools.product((0,1,2,4,8),(.002,.02,.2),(0.,.0001,.001),(.9,.98))
             if tq or d==.98]
    # Written before scoring any test predictions; do not revise based on test.
    protocol={'origins':origins,'horizon':13,'tail':200,'grid':configs,
              'selection':'mean per-series WAPE across all three validation origins',
              'seed':42,'test_used_for_selection':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start=time.perf_counter(); scores=[]
    for i,c in enumerate(configs):
        v=[metrics(forecast(y[:,:o],**c),y[:,o:o+13])['wape_mean'] for o in origins]
        scores.append({'config':c,'folds':v,'mean':float(np.mean(v))})
        if i%15==0: print('grid',i,'/',len(configs),flush=True)
    best=min(scores,key=lambda r:r['mean'])
    no_season=min((r for r in scores if r['config']['harmonics']==0),key=lambda r:r['mean'])
    no_trend=min((r for r in scores if r['config']['trend_q']==0),key=lambda r:r['mean'])
    selection_s=time.perf_counter()-start
    predictions={}; results={}
    for name,r in [('State-Fourier',best),('State-no-season',no_season),('State-no-trend',no_trend)]:
        start=time.perf_counter(); p=forecast(y,**r['config']); seconds=time.perf_counter()-start
        predictions[name]=p; results[name]={**metrics(p,truth),'hyper':r['config'],'val_folds':r['folds'],'val_mean':r['mean'],'fit_predict_s':seconds}
    old=json.loads((ROOT/'audit/original/metrics_fg.json').read_text())['hyper']
    start=time.perf_counter(); m=fg_mp.fit(y,qr=old['qr'],ridge=old['ridge'],kappa=old['kappa'])
    predictions['FG-original']=fg_mp.forecast(m,damp=old['damp'])
    results['FG-original']={**metrics(predictions['FG-original'],truth),'fit_predict_s':time.perf_counter()-start}
    for name,p in [('Naive',np.repeat(y[:,-1:],13,axis=1)),('SeasonalNaive52',y[:,-52:-39]),('MovingAvg8',np.repeat(y[:,-8:].mean(1)[:,None],13,axis=1))]:
        predictions[name]=p; results[name]=metrics(p,truth)
    z=np.load(ROOT/'data/bsts_preds.npz'); assert list(z['ids'])==ids
    predictions['UCM52-saved']=z['preds']; results['UCM52-saved']=metrics(z['preds'],truth)
    rng=np.random.default_rng(42); comparisons={}
    for name in ['FG-original','UCM52-saved']:
        diff=(abs(predictions['State-Fourier']-truth).sum(1)-abs(predictions[name]-truth).sum(1))/abs(truth).sum(1)
        boot=diff[rng.integers(0,len(ids),(10000,len(ids)))].mean(1)
        comparisons[name]={'delta_wape':float(diff.mean()),'series_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'win_share':float((diff<0).mean())}
    report={'source_commit':'83f3947','data_sha256':hashlib.sha256((ROOT/'data/weekly.npz').read_bytes()).hexdigest(),
            'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
            'selection_s':selection_s,'results':results,'paired_comparisons':comparisons,'grid_scores':scores,
            'caveat':'Known public test, exploratory replication; bootstrap treats series as independent, not repeated future periods.'}
    (OUT/'state_experiment.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/'state_predictions.npz',ids=ids,truth=truth,**predictions)
    print(json.dumps({k:v for k,v in report.items() if k!='grid_scores'},indent=2),flush=True)
if __name__=='__main__': main()
