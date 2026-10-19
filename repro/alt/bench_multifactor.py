"""Pre-specified six-factor experiment, independent of the original leaderboard."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import bench_rossmann as old
from rossmann_factors import FactorPanel,GROUPS,NAMES
sys.path.insert(0,str(old.ROOT/'repro'))
from fg_state_exog import forecast
OUT=old.ROOT/'audit/results/rossmann_multifactor'


def metric(p,y):
    den=np.abs(y).sum(1)
    if np.any(den<=0) or not np.isfinite(p).all():raise ValueError('Invalid evaluation')
    return {'wape_mean':float(np.mean(np.abs(p-y).sum(1)/den)),
            'wape_aggregate':float(np.abs(p-y).sum()/np.abs(y).sum())}


def state_run(p,o,cfg,mode):
    xt,xf=p.at_origin(o,old.H);ot=p.open_[:,:o];of=p.open_[:,o:o+old.H]
    if mode=='univariate':xt=xt[:,:,:0];xf=xf[:,:,:0];ot=None;of=None
    elif mode=='open_only':xt=xt[:,:,:0];xf=xf[:,:,:0]
    elif mode=='without_open':ot=None;of=None
    elif mode.startswith('drop_'):
        keep=[i for i in range(7) if i not in GROUPS[mode[5:]]];xt=xt[:,:,keep];xf=xf[:,:,keep]
    return forecast(p.y[:,:o],xt,xf,ot,of,**cfg)


def save(name,o,pred,p,details=None):
    data={'preds':pred,'truth':p.y[:,o:o+old.H],'ids':np.array(p.ids),'dates':p.days[o:o+old.H]}
    if details:data.update(details)
    np.savez_compressed(OUT/f'{name}_o{o}.npz',**data)


def main():
    OUT.mkdir(exist_ok=True,parents=True);p=FactorPanel();rows=[];selection=[]
    for qr in [.002,.02,.2]:
        for beta in [.1,1.]:
            cfg={'qr':qr,'beta_var':beta};pred,_=state_run(p,774,cfg,'full');m=metric(pred,p.y[:,774:816])
            selection.append({'config':cfg,**m});print('selection',cfg,m,flush=True)
    cfg=min(selection,key=lambda r:r['wape_mean'])['config']
    (OUT/'selection.json').write_text(json.dumps({'selection':selection,'chosen':cfg},indent=2))
    modes=['univariate','open_only','full','without_open']+['drop_'+g for g in GROUPS]
    for o in old.OUTER_ORIGINS:
        for mode in modes:
            t=time.perf_counter();pr,d=state_run(p,o,cfg,mode)
            save('SF_'+mode,o,pr,p,d if mode=='full' else None)
            r={'model':'SF_'+mode,'origin':o,**metric(pr,p.y[:,o:o+42]),'elapsed_s':time.perf_counter()-t,
               'min_covariance_eigenvalue':d['min_covariance_eigenvalue']};rows.append(r);print(r,flush=True)
        (OUT/'state_metrics.json').write_text(json.dumps(rows,indent=2))
    # Training-origin feature coverage: dormant columns are not claimed identifiable.
    coverage={name:{'stores_with_train_variation':int((np.ptp(p.extra[:,:774,j],axis=1)>0).sum()),
                     'train_nonzero_share':float((p.extra[:,:774,j]!=0).mean())} for j,name in enumerate(NAMES)}
    (OUT/'factor_coverage.json').write_text(json.dumps(coverage,indent=2))
    # First independently reproduce the prior implementation, without changing its features.
    old.N_JOBS=2
    prior=json.loads((old.OUTDIR/'metrics.json').read_text())
    params=prior['selection']['chosen']['LGBM-direct']['config']['params']
    t=time.perf_counter();pr=old.lgbm_forecast(p,900,True,params)
    ref=np.load(old.PREDDIR/'preds_LGBM-direct_o900.npz')['preds']
    verify={'max_abs_diff':float(np.max(abs(np.maximum(pr,0)-ref))),
            'retrained':metric(np.maximum(pr,0),p.y[:,900:942]),'reference':metric(ref,p.y[:,900:942]),'elapsed_s':time.perf_counter()-t}
    (OUT/'lgbm_original_retrain.json').write_text(json.dumps(verify,indent=2));print('REPRODUCED',verify,flush=True)
    # Fixed 600-tree direct model with the same SIX dynamic groups and store identity.
    original_matrix=old.build_matrix
    def augmented(p,o,h,full,hist=None):
        cols=original_matrix(p,o,h,False,hist)
        target=o+h-1
        for j,name in enumerate(NAMES):
            value=p.extra[:,target,j].copy()
            if j==5:value*=(p.promo2_start<=p.days[o-1])
            if j==6:value=(p.competition_start<=p.days[o-1]).astype(float)
            cols[name]=value
        cols['Open']=p.open_[:,target].astype(float)
        return cols
    old.build_matrix=augmented
    for o in old.OUTER_ORIGINS:
        t=time.perf_counter();pr=np.maximum(old.lgbm_forecast(p,o,True,params),0)
        save('LGBM_six',o,pr,p)
        r={'model':'LGBM_six','origin':o,**metric(pr,p.y[:,o:o+42]),'elapsed_s':time.perf_counter()-t}
        rows.append(r);print(r,flush=True)
        (OUT/'metrics.json').write_text(json.dumps(rows,indent=2))
    # Record source and input fingerprints for this completed run.
    paths=['repro/fg_state_exog.py','repro/alt/rossmann_factors.py','repro/alt/bench_multifactor.py','repro/alt/bench_rossmann.py','data/alt/rossmann/panel.npz','data/alt/rossmann/store.csv']
    (OUT/'fingerprints.json').write_text(json.dumps({f:hashlib.sha256((old.ROOT/f).read_bytes()).hexdigest() for f in paths},indent=2))


if __name__=='__main__':main()
