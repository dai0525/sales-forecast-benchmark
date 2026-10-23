"""Single-model loss and calibration experiment. Protocol is frozen separately."""
import argparse,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from retail_expansion import ROOT,OUT as BASE,load,features,metrics
from audit_freshretail import digest
OUT=ROOT/'audit/results/loss_optimization'
MODELS=['log_l2','raw_l2','poisson','tweedie']


def ratio(pred,y):
    if pred.sum()<=0:return 1.
    return float(np.clip(y.sum()/pred.sum(),.5,2.))


def choose(dev0,dev1,y0,y1,metric):
    options={}
    for model in MODELS:
        c=ratio(dev0[model],y0)
        scores={str(a):metrics(dev1[model]*(1+a*(c-1)),y1)[metric] for a in [0.,.5,1.]}
        strength=float(min(scores,key=scores.get))
        options[model]={'strength':strength,'first_origin_factor':c,'second_origin_scores':scores,'selection_score':scores[str(strength)],'final_factor':ratio(np.concatenate([dev0[model],dev1[model]]),np.concatenate([y0,y1]))}
    selected=min(options,key=lambda m:options[m]['selection_score'])
    return {'options':options,'selected_model':selected,'metric':metric}


def fit(p,o,group,objective):
    H=p['H'];origins=sorted(set(list(range(28,o-H+1,7))+[o-H])) if H==7 else sorted([o-H-13*j for j in range(8) if o-H-13*j>=52])
    X=pd.concat([features(p,t,group) for t in origins],ignore_index=True)
    y=np.concatenate([p['y'][:,t:t+H].ravel() for t in origins])
    kwargs=dict(objective='regression' if objective=='raw_l2' else objective,n_estimators=250,learning_rate=.05,num_leaves=31,min_child_samples=80,reg_lambda=1,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    if objective=='tweedie':kwargs['tweedie_variance_power']=1.5
    model=lgb.LGBMRegressor(**kwargs);model.fit(X,y,categorical_feature=list(p['static']))
    future=features(p,o,group);assert list(future)==list(X)
    pred=np.maximum(model.predict(future),0).reshape(len(p['y']),H)
    return pred,{'params':model.get_params(),'features':list(X),'training_origins':origins,'training_rows':len(X)}


def run(tag,group,resume=False):
    out=OUT/(tag+'_'+group);ph=digest(OUT/'protocol.json')
    if out.exists() and not resume:raise RuntimeError('Refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    p=load(tag);ids=pd.read_csv(BASE/tag/'series_ids.csv');pd.testing.assert_frame_equal(ids,p['static'],check_dtype=False)
    ids.to_csv(out/'series_ids.csv',index=False)
    origins=p['origins'];H=p['H'];predictions={};rows=[];metric='vn1_score' if tag=='vn1' else 'wape'
    for o in origins:
        if o==origins[-1]:
            selection=choose(predictions[origins[0]],predictions[origins[1]],p['y'][:,origins[0]:origins[0]+H],p['y'][:,origins[1]:origins[1]+H],metric)
            selection['protocol_sha256']=ph
            path=out/'selection.json'
            if path.exists():assert json.loads(path.read_text())==selection
            else:path.write_text(json.dumps(selection,indent=2)+'\n')
        predictions[o]={};truth=p['y'][:,o:o+H]
        for model in MODELS:
            path=out/f'o{o}_{model}.npz';mp=path.with_suffix('.json')
            if path.exists() and mp.exists():
                if not resume:raise RuntimeError('Existing prediction')
                meta=json.loads(mp.read_text());assert meta['protocol_sha256']==ph
                z=np.load(path);np.testing.assert_array_equal(z['truth'],truth);pred=z['pred'];secs=meta['seconds']
            else:
                if model=='log_l2':
                    source=BASE/tag/f'o{o}_lgb_{group}.npz';z=np.load(source);np.testing.assert_array_equal(z['truth'],truth);pred=z['pred']
                    old=pd.read_csv(BASE/tag/'metrics.csv');secs=float(old[(old.origin==o)&(old.model=='lgb_'+group)].seconds.iloc[0]);meta={'source':str(source.relative_to(ROOT)),'source_sha256':digest(source),'cached':True}
                else:
                    t=time.perf_counter();pred,meta=fit(p,o,group,model);secs=time.perf_counter()-t
                np.savez_compressed(path,pred=pred,truth=truth)
                mp.write_text(json.dumps(dict(protocol_sha256=ph,seconds=secs,**meta),indent=2)+'\n')
            predictions[o][model]=pred
            rows.append(dict(origin=o,model=model,seconds=secs,**metrics(pred,truth)))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(tag,group,o,model,round(rows[-1]['wape']*100,3),round(rows[-1]['bias']*100,3),round(secs,2),flush=True)
    o=origins[-1];truth=p['y'][:,o:o+H]
    for model in MODELS:
        opt=selection['options'][model];factor=1+opt['strength']*(opt['final_factor']-1)
        pred=predictions[o][model]*factor
        np.savez_compressed(out/f'o{o}_{model}_calibrated.npz',pred=pred,truth=truth)
        rows.append(dict(origin=o,model=model+'_calibrated',seconds=0.,**metrics(pred,truth)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    if tag=='vn1':
        clients=[]
        for row in rows:
            if row['origin']!=o:continue
            z=np.load(out/f"o{o}_{row['model']}.npz")
            for client in ids.Client.unique():
                m=ids.Client.to_numpy()==client;yy=truth[m];pp=z['pred'][m]
                if yy.sum()>0:clients.append(dict(client=int(client),model=row['model'],**metrics(pp,yy)))
        pd.DataFrame(clients).to_csv(out/'client_metrics.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'status':'completed','protocol_sha256':ph,'selection_metric':metric,'final_origin':o,'series':len(ids),'selected_prediction':selection['selected_model']+'_calibrated','versions':{'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__}},indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['freshretail','vn1']);ap.add_argument('group',choices=['history','plan']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.dataset=='vn1' and a.group!='history':ap.error('VN1 only history')
    run(a.dataset,a.group,a.resume)
