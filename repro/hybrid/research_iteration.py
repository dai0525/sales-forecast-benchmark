"""Frozen ASE iteration; see audit/results/ase_iteration/protocol.json.
Usage: python repro/hybrid/research_iteration.py weekly|rossmann|confirmation
The confirmation command consumes frozen development weights, never reselects.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import ase
import run_rossmann_final as rf

ROOT=ase.ROOT
OUT=ROOT/'audit/results/ase_iteration'
PRIMARY='pooled_global_log'

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def metric(p,y):
    d=np.abs(y).sum(1)
    if not np.isfinite(p).all() or np.any(d<=0):raise ValueError('invalid prediction or denominator')
    return {'wape_mean':float(np.mean(np.abs(p-y).sum(1)/d)), 'wape_aggregate':float(np.abs(p-y).sum()/np.abs(y).sum())}

def fit_ross(p,o,name):
    X,y=rf.b.build_training(p,o,True)
    if name.startswith('open_'):
        keep=X['open'].to_numpy()==1
        X=X.loc[keep].copy();y=y[keep]
    X['store_id']=X['store_id'].astype('category')
    obj='regression_l1' if name.endswith('l1') else 'l2'
    model=lgb.LGBMRegressor(objective=obj,random_state=42,**dict(rf.PARAMS,n_jobs=2))
    model.fit(X,np.log1p(y),categorical_feature=['store_id'])
    hist=rf.b.history_features(p.y[:,:o]);pred=[]
    for h in range(1,43):
        row=pd.DataFrame(rf.b.build_matrix(p,o,h,True,hist=hist))
        row['store_id']=pd.Categorical(row['store_id'],categories=X['store_id'].cat.categories)
        pred.append(np.expm1(model.predict(row[X.columns])))
    return np.where(p.open_[:,o:o+42]==1,np.maximum(np.array(pred).T,0),0)

def component(tag,p,o,name,state_cfg=None):
    path=OUT/tag/f'components_o{o}_{name}.npz'
    if path.exists():return np.load(path)['pred']
    t=time.perf_counter();source='refit'
    if tag=='rossmann' and name in ['l2','l1'] and o in [774,816,858,900]:
        if o==774:
            z=np.load(ROOT/'audit/results/ase_review/rossmann_validation_preds.npz');pred=z[name]
            assert np.array_equal(z['truth'],p.y[:,774:816])
        else:
            z=np.load(ROOT/'audit/results/ase/rossmann_final_preds.npz');pred=z[f'{o}_{name}']
            assert np.array_equal(z[f'{o}_truth'],p.y[:,o:o+42])
        source='audited_archive'
    elif tag=='weekly':
        if 'state' in name:
            pred=ase.state_forecast(p.y[:,:o],h=13,period=52,**state_cfg)
        else:
            prm=dict(ase.LGB_PARAMS,n_jobs=2)
            if name=='l1':prm.update(objective='regression_l1',random_state=43)
            pred=ase._gbm_predict(p,o,prm,'log1p')
    else:pred=fit_ross(p,o,name)
    elapsed=time.perf_counter()-t
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,pred=pred,ids=np.array(p.ids))
    dump(path.with_suffix('.json'),{'origin':o,'model':name,'source':source,'elapsed_s':elapsed,'state_cfg':state_cfg})
    print(tag,o,name,source,round(elapsed,2),flush=True)
    return pred

def fit_mix(val,truth,names,H,original_weights=None):
    pooled={k:np.concatenate([v[k] for v in val.values()]) for k in names}
    y=np.concatenate(list(truth.values()))
    w=ase._fit_weights(pooled,y,names)
    configs={PRIMARY:{'all':w}}
    stage=ase._stages(H)
    configs['pooled_stage_log']={s:ase._fit_weights({k:v[:,idx] for k,v in pooled.items()},y[:,idx],names) for s,idx in stage.items()}
    last=max(val)
    v=dict(val[last]);orig_names=['l2','l1'] if H==42 else ['l2','l1','state']
    if H==13:v['state']=v['original_state']
    configs['original_stage']=original_weights or {s:ase._fit_weights({k:v[k][:,idx] for k in orig_names},truth[last][:,idx],orig_names) for s,idx in stage.items()}
    scores={k:metric(v,y)['wape_mean'] for k,v in pooled.items()}
    return {'configs':configs,'validation_wape':scores,'best_single':min(scores,key=scores.get),'component_names':names,'H':H}

def predict_methods(base,sel):
    H=sel['H'];methods={k:base[k] for k in sel['component_names']}
    for name,cfg in sel['configs'].items():
        b=dict(base)
        if name=='original_stage' and H==13:b['state']=b['original_state']
        stages={'all':list(range(H))} if 'all' in cfg else ase._stages(H)
        methods[name]=ase._combine(b,cfg,stages)
    orig=['l2','l1','state'] if H==13 else ['l2','l1']
    methods['equal_original_log']=np.expm1(np.mean([np.log1p(base[k]) for k in orig],axis=0))
    methods['equal_original_arithmetic']=np.mean([base[k] for k in orig],axis=0)
    methods['equal_all_arithmetic']=np.mean([base[k] for k in sel['component_names']],axis=0)
    if H==42:methods['equal_open_arithmetic']=(base['open_l2']+base['open_l1'])/2
    methods['validation_best_single']=base[sel['best_single']]
    return methods

def score_origins(tag,p,sel,origins,state_cfg=None,original_cfg=None):
    rows={}
    for o in origins:
        base={k:component(tag,p,o,k,state_cfg) for k in sel['component_names']}
        if tag=='weekly':base['original_state']=component(tag,p,o,'original_state',original_cfg)
        methods=predict_methods(base,sel)
        y=p.truth(o) if tag=='weekly' else p.y[:,o:o+42]
        rows[str(o)]={k:metric(v,y) for k,v in methods.items()}
        dates=np.arange(o,o+sel['H']) if tag=='weekly' else p.days[o:o+42]
        np.savez_compressed(OUT/tag/f'predictions_o{o}.npz',truth=y,ids=np.array(p.ids),dates=dates,**{'pred_'+k:v for k,v in methods.items()})
        dump(OUT/tag/'metrics.json',rows)
        print(tag,o,{k:round(v['wape_mean']*100,5) for k,v in rows[str(o)].items()},flush=True)
    return rows

def develop(tag):
    weekly=tag=='weekly';p=ase.load_weekly() if weekly else rf.b.Panel(rf.b.PANEL)
    vals=[135,148] if weekly else [732,774]
    names=['l2','l1','state'] if weekly else ['l2','l1','open_l2','open_l1']
    cfg=original_cfg=None
    if weekly:
        grid=[]
        for c in ase.STATE_GRID:
            scores=[metric(ase.state_forecast(p.y[:,:o],h=13,period=52,**c),p.truth(o))['wape_mean'] for o in vals]
            grid.append({'config':c,'scores':scores,'mean':float(np.mean(scores))})
        cfg=min(grid,key=lambda r:r['mean'])['config']
        original_cfg=min(grid,key=lambda r:r['scores'][-1])['config']
        dump(OUT/tag/'state_selection.json',{'grid':grid,'chosen':cfg,'original_chosen':original_cfg})
    val={};truth={}
    for o in vals:
        val[o]={k:component(tag,p,o,k,cfg) for k in names}
        if weekly:val[o]['original_state']=component(tag,p,o,'original_state',original_cfg)
        truth[o]=p.truth(o) if weekly else p.y[:,o:o+42]
    original=None
    if not weekly:
        old=json.loads((ROOT/'audit/results/ase/rossmann_final.json').read_text())['weights']
        original={s:{k:w[k] for k in ['l2','l1']} for s,w in old.items()}
    selection=fit_mix(val,truth,names,13 if weekly else 42,original)
    selection.update(state_config=cfg,original_state_config=original_cfg,validation_origins=vals,protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    dump(OUT/tag/'selection.json',selection)
    print('FROZEN',tag,selection['configs'][PRIMARY],selection['best_single'],flush=True)
    score_origins(tag,p,selection,[161,174,187,200] if weekly else [816,858,900],cfg,original_cfg)

def build_confirmation():
    path=OUT/'confirmation/panel.npz'
    if path.exists():return rf.b.Panel(path)
    src=ROOT/'data/alt/rossmann'
    import sys
    sys.path.insert(0,str(ROOT/'repro/alt'))
    from fetch_rossmann import EXPECTED_MIRROR_SHA256
    for name,expected in EXPECTED_MIRROR_SHA256.items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest()==expected
    tr=pd.read_csv(src/'train.csv',dtype={'StateHoliday':str},parse_dates=['Date'],low_memory=False)
    st=pd.read_csv(src/'store.csv');dates=pd.date_range('2013-01-01','2015-07-31')
    counts=tr.groupby('Store')['Date'].nunique();complete=set(counts[counts==942].index)
    original=rf.b.Panel(rf.b.PANEL);keep=np.array(sorted(complete-set(original.stores)))
    assert len(keep)==534 and not (set(keep)&set(original.stores))
    sub=tr[tr.Store.isin(keep)]
    assert not sub.duplicated(['Store','Date']).any()
    wide={c:sub.pivot(index='Store',columns='Date',values=c).reindex(index=keep,columns=dates) for c in ['Sales','Promo','Open','SchoolHoliday','StateHoliday']}
    assert all(not v.isna().any().any() for v in wide.values())
    meta=st.set_index('Store').reindex(keep)
    tm={c:i for i,c in enumerate(sorted(st.StoreType.unique()))};am={c:i for i,c in enumerate(sorted(st.Assortment.unique()))}
    state=wide['StateHoliday'].astype(str).replace({'0':0,'a':1,'b':2,'c':3}).to_numpy(np.int8)
    path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,y=wide['Sales'].to_numpy(np.float32),ids=np.array([f'S{s}' for s in keep]),stores=keep.astype(np.int32),dates=dates.to_numpy('datetime64[D]').astype('int64'),promo=wide['Promo'].to_numpy(np.int8),open_=wide['Open'].to_numpy(np.int8),school=wide['SchoolHoliday'].to_numpy(np.int8),state=state,store_type=meta.StoreType.map(tm).to_numpy(np.int8),assortment=meta.Assortment.map(am).to_numpy(np.int8),promo2=meta.Promo2.to_numpy(np.int8),comp_dist=meta.CompetitionDistance.to_numpy(np.float32))
    dump(path.with_suffix('.json'),{'n':len(keep),'stores':keep.tolist(),'overlap_with_development':0,'raw_sha256':EXPECTED_MIRROR_SHA256,'panel_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return rf.b.Panel(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['weekly','rossmann','confirmation']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=['repro/hybrid/ase.py','repro/hybrid/research_iteration.py','repro/hybrid/run_rossmann_final.py','repro/alt/bench_rossmann.py','repro/fg_state.py','audit/results/ase_iteration/protocol.json']
    dump(OUT/args.mode/'environment.json',{'python':platform.python_version(),'lightgbm':lgb.__version__,'numpy':np.__version__,'pandas':pd.__version__,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},'threads':2})
    if args.mode=='confirmation':
        selection=json.loads((OUT/'rossmann/selection.json').read_text())
        dump(OUT/'confirmation/frozen_selection.json',selection)
        p=build_confirmation();score_origins('confirmation',p,selection,[816,858,900])
    else:develop(args.mode)

if __name__=='__main__':main()
