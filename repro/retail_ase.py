"""Port existing ASE to the frozen retail adapters; explicitly an added comparison."""
import argparse,json,sys,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from retail_expansion import ROOT,OUT as BASE,load,features,metrics
from audit_freshretail import digest
sys.path.insert(0,str(ROOT/'repro/hybrid'))
import ase

OUT=ROOT/'audit/results/retail_ase'


def fit_tree(p,o,group,name):
    H=p['H']
    origins=sorted(set(list(range(28,o-H+1,7))+[o-H])) if H==7 else sorted([o-H-13*j for j in range(8) if o-H-13*j>=52])
    X=pd.concat([features(p,t,group) for t in origins],ignore_index=True)
    y=np.concatenate([p['y'][:,t:t+H].ravel() for t in origins])
    params=dict(ase.LGB_PARAMS,n_jobs=2)
    if name=='l1':params.update(objective='regression_l1',random_state=43)
    model=lgb.LGBMRegressor(**params).fit(X,np.log1p(y),categorical_feature=list(p['static']))
    xx=features(p,o,group);assert list(X)==list(xx)
    pred=np.maximum(np.expm1(model.predict(xx)),0).reshape(len(p['y']),H)
    return pred,{'params':model.get_params(),'training_origins':origins,'training_rows':len(X),'features':list(X)}


def state_grid(p):
    return [dict(harmonics=k,qr=q,trend_q=t,damp=.98) for k in sorted({2,min(4,(p['period']-1)//2)}) for q in (.02,.1) for t in (0.,1e-4)]


def positive_wape(pred,y):
    keep=y.sum(1)>0
    return ase._wape_mean(pred[keep],y[keep])


def select_weights(val,truth):
    """No test predictions/labels accepted here. Reuse original fitters unchanged."""
    names=['l2','l1','state'];origins=sorted(val);last=origins[-1]
    original={s:ase._fit_weights({k:val[last][k][:,ix] for k in names},truth[last][:,ix],names) for s,ix in ase._stages(truth[last].shape[1]).items()}
    pooled={k:np.concatenate([val[o]['pooled_state' if k=='state' else k] for o in origins]) for k in names}
    static={'all':ase._fit_weights(pooled,np.concatenate([truth[o] for o in origins]),names)}
    scores={k:metrics(p,np.concatenate([truth[o] for o in origins]))['wape'] for k,p in pooled.items()}
    # State using original single-window selection is also a declared single comparator.
    scores['original_state']=metrics(np.concatenate([val[o]['state'] for o in origins]),np.concatenate([truth[o] for o in origins]))['wape']
    return dict(original=original,static=static,development_aggregate_wape=scores,best_single=min(scores,key=scores.get))


def tree_component(p,o,group,name,out,resume):
    path=out/f'o{o}_{name}.npz';meta_path=path.with_suffix('.json')
    if resume and path.exists() and meta_path.exists():
        z=np.load(path);meta=json.loads(meta_path.read_text())
        np.testing.assert_array_equal(z['truth'],p['y'][:,o:o+p['H']])
        metrics(z['pred'],z['truth'])
        expected=dict(ase.LGB_PARAMS,n_jobs=2)
        if name=='l1':expected.update(objective='regression_l1',random_state=43)
        assert all(meta['params'][k]==v for k,v in expected.items())
        assert meta['features']==list(features(p,o,group))
        assert max(meta['training_origins'])+p['H']<=o
        return z['pred'],{k:v for k,v in meta.items() if k!='seconds'},meta['seconds']
    t=time.perf_counter();pred,meta=fit_tree(p,o,group,name)
    return pred,meta,time.perf_counter()-t


def run(tag,group,resume=False):
    key=tag+'_'+group;out=OUT/key
    if out.exists() and not resume:raise RuntimeError('Refusing overwrite: '+str(out))
    if resume:
        if (out/'manifest.json').exists():raise RuntimeError('Already completed; refusing resume')
        previous=json.loads((out/'selection.json').read_text())
        assert previous['protocol_sha256']==digest(OUT/'protocol.json')
        (out/'resume.json').write_text(json.dumps({'reason':'execution environment restarted during final L1 fit','prior_checkpoint_sha256':{f.name:digest(f) for f in out.glob('*.npz')},'preserve_cached_training_seconds':True},indent=2)+'\n')
    out.mkdir(parents=True,exist_ok=resume)
    p=load(tag);H=p['H'];dev=p['origins'][:-1];final=p['origins'][-1]
    ids=pd.read_csv(BASE/tag/'series_ids.csv');pd.testing.assert_frame_equal(ids,p['static'],check_dtype=False)
    p['static'].to_csv(out/'series_ids.csv',index=False)
    ph=digest(OUT/'protocol.json');rows=[];val={};truth={o:p['y'][:,o:o+H] for o in dev}
    grid=state_grid(p);states={};scores=[];search_start=time.perf_counter()
    for cfg in grid:
        pp={o:ase.state_forecast(p['y'][:,:o],h=H,period=p['period'],**cfg) for o in dev}
        scores.append({'cfg':cfg,'latest':positive_wape(pp[dev[-1]],truth[dev[-1]]),'pooled':positive_wape(np.concatenate(list(pp.values())),np.concatenate(list(truth.values())))})
        states[json.dumps(cfg)]=pp
    cfg_original=min(scores,key=lambda r:r['latest'])['cfg'];cfg_pooled=min(scores,key=lambda r:r['pooled'])['cfg']
    search_seconds=time.perf_counter()-search_start
    for o in dev:
        val[o]={'state':states[json.dumps(cfg_original)][o],'pooled_state':states[json.dumps(cfg_pooled)][o]}
        for name in ['l2','l1']:
            pred,meta,secs=tree_component(p,o,group,name,out,resume);val[o][name]=pred
            np.savez_compressed(out/f'o{o}_{name}.npz',pred=pred,truth=truth[o])
            (out/f'o{o}_{name}.json').write_text(json.dumps(dict(seconds=secs,**meta),indent=2)+'\n')
            rows.append(dict(origin=o,model=name,seconds=secs,**metrics(pred,truth[o])))
            print(key,o,name,round(rows[-1]['wape']*100,3),round(secs,2),flush=True)
        for name,pred in [('original_state',val[o]['state']),('state',val[o]['pooled_state'])]:
            np.savez_compressed(out/f'o{o}_{name}.npz',pred=pred,truth=truth[o]);rows.append(dict(origin=o,model=name,seconds=None,**metrics(pred,truth[o])))
    t=time.perf_counter();selection=select_weights(val,truth);weight_seconds=time.perf_counter()-t
    selection.update(protocol_sha256=ph,original_state_cfg=cfg_original,pooled_state_cfg=cfg_pooled,state_search=scores,state_search_seconds=search_seconds,weight_selection_seconds=weight_seconds,final_origin=final)
    if resume:
        for k in ['original','static','best_single','original_state_cfg','pooled_state_cfg','development_aggregate_wape']:
            assert selection[k]==previous[k],('Resume changed frozen selection',k)
        selection=previous
    else:
        (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    # Selection is persisted before final inference or final scoring.
    final_components={};base_seconds={};yy=p['y'][:,final:final+H]
    for name in ['l2','l1','original_state','state']:
        t=time.perf_counter();meta={}
        if name in ['l2','l1']:pred,meta,secs=tree_component(p,final,group,name,out,resume)
        else:
            pred=ase.state_forecast(p['y'][:,:final],h=H,period=p['period'],**(cfg_original if name=='original_state' else cfg_pooled))
            secs=time.perf_counter()-t
        base_seconds[name]=secs;final_components[name]=pred
        np.savez_compressed(out/f'o{final}_{name}.npz',pred=pred,truth=yy)
        (out/f'o{final}_{name}.json').write_text(json.dumps(dict(seconds=secs,**meta),indent=2)+'\n')
        rows.append(dict(origin=final,model=name,seconds=secs,**metrics(pred,yy)))
        print(key,final,name,round(rows[-1]['wape']*100,3),round(secs,2),flush=True)
    b=final_components;original={'l2':b['l2'],'l1':b['l1'],'state':b['original_state']};pooled={k:b[k] for k in ['l2','l1','state']}
    funcs={'ase_original':lambda:ase._combine(original,selection['original'],ase._stages(H)),
           'ase_static':lambda:ase._combine(pooled,selection['static'],{'all':list(range(H))}),
           'equal_log':lambda:ase._combine_arr(original,np.ones(3)/3),
           'equal_arithmetic':lambda:np.mean(list(original.values()),axis=0),
           'selected_single':lambda:b[selection['best_single']]}
    for name,fn in funcs.items():
        t=time.perf_counter();pred=fn();seconds=time.perf_counter()-t
        np.savez_compressed(out/f'o{final}_{name}.npz',pred=pred,truth=yy)
        rows.append(dict(origin=final,model=name,seconds=seconds,**metrics(pred,yy)))
        print(key,final,name,round(rows[-1]['wape']*100,3),flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    if tag=='vn1':
        clients=[]
        for r in rows:
            if r['origin']!=final:continue
            z=np.load(out/f"o{final}_{r['model']}.npz")
            for client in ids.Client.unique():
                mask=ids.Client.to_numpy()==client;y=z['truth'][mask];pr=z['pred'][mask];den=y.sum()
                clients.append(dict(client=int(client),model=r['model'],wape=float(abs(pr-y).sum()/den) if den>0 else None,bias=float((pr-y).sum()/den) if den>0 else None))
        pd.DataFrame(clients).to_csv(out/'client_metrics.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'status':'completed','protocol_sha256':ph,'final_base_seconds':base_seconds,'state_search_seconds':search_seconds,'weight_selection_seconds':weight_seconds,'series':len(p['y']),'limitations':'ported feature adapter; added after observing first-pass tests'},indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['freshretail','vn1']);ap.add_argument('group',choices=['history','plan']);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.dataset=='vn1' and args.group!='history':ap.error('VN1 only uses history in this protocol')
    run(args.dataset,args.group,args.resume)
