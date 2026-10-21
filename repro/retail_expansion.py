"""Frozen first-pass retail experiment. python repro/retail_expansion.py freshretail|vn1
See audit/results/retail_expansion/protocol.json. No future realized covariates.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from fg_state import forecast as state_forecast
from fg_state_exog import forecast as exog_forecast
from audit_freshretail import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit/results/retail_expansion'
WEATHER = ['precpt','avg_temperature','avg_humidity','avg_wind_level']
PLAN = ['discount','activity_flag','holiday_flag']


def load(tag):
    if tag == 'vn1':
        path=ROOT/'data/raw/vn1/vn1.zip'
        assert digest(path)=='0b5fe01aca827fb223533afe50e234e568ab304164c22e5d93037a69289e23a9'
        with zipfile.ZipFile(path) as z:
            frames=[pd.read_csv(z.open(f'vn1-forecasting-competition/Phase {i} - Sales.csv')) for i in range(3)]
            prices=[pd.read_csv(z.open(f'vn1-forecasting-competition/Phase {i} - Price.csv')) for i in range(2)]
        static=frames[0].iloc[:,:3].copy()
        assert all(f.iloc[:,:3].equals(static) for f in frames+prices)
        y=np.concatenate([f.iloc[:,3:].to_numpy(float) for f in frames],axis=1)
        price=np.concatenate([f.iloc[:,3:].to_numpy(float) for f in prices],axis=1)
        # No Phase 2 price supplied or manufactured. Features only access :origin.
        dates=pd.to_datetime(sum([list(f.columns[3:]) for f in frames],[]))
        return dict(y=y,static=static,dates=dates,cov={'price':price},H=13,period=52,origins=[157,170,183])
    paths=[ROOT/f'data/raw/freshretail/{s}.parquet' for s in ['train','eval']]
    hashes=['6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d','1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e']
    assert all(digest(p)==h for p,h in zip(paths,hashes))
    ids=pd.read_parquet(paths[0],columns=['store_id','product_id']).drop_duplicates()
    ids['hash']=[hashlib.sha256(f'freshretail:{s}:{p}'.encode()).hexdigest() for s,p in ids.to_numpy()]
    ids=ids.sort_values('hash').head(3000).drop(columns='hash').reset_index(drop=True)
    ids['sid']=np.arange(len(ids))
    cols=['store_id','product_id','city_id','management_group_id','first_category_id','second_category_id','third_category_id','dt','sale_amount','stock_hour6_22_cnt']+PLAN+WEATHER
    df=pd.concat([pd.read_parquet(p,columns=cols).merge(ids,on=['store_id','product_id'],validate='many_to_one') for p in paths])
    df=df.sort_values(['sid','dt']);dates=pd.to_datetime(sorted(df.dt.unique()))
    assert len(df)==3000*97 and not df.duplicated(['sid','dt']).any()
    def panel(c):return df[c].to_numpy(float).reshape(3000,97)
    static=df.groupby('sid',sort=True)[['store_id','product_id','city_id','management_group_id','first_category_id','second_category_id','third_category_id']].first().reset_index(drop=True)
    return dict(y=panel('sale_amount'),static=static,dates=dates,cov={c:panel(c) for c in PLAN+WEATHER+['stock_hour6_22_cnt']},H=7,period=7,origins=[76,83,90])


def features(p,o,group):
    """Direct multi-step rows; every history feature is anchored before origin."""
    y=p['y'];n=len(y);H=p['H'];hist=y[:,:o];period=p['period']
    d={c:np.repeat(p['static'][c].to_numpy(),H) for c in p['static']}
    d['horizon']=np.tile(np.arange(1,H+1),n)
    date=p['dates'][o:o+H]
    for name,value in [('dow',date.dayofweek),('month',date.month),('week',date.isocalendar().week.to_numpy(dtype=int))]:d[name]=np.tile(value,n)
    for lag in sorted(set([1,2,7,period])):
        d[f'lag{lag}']=np.repeat(hist[:,-lag],H)
    for w in sorted(set([7,14,28,period])):
        a=hist[:,-min(w,o):]
        for name,v in [('mean',a.mean(1)),('std',a.std(1)),('zero',(a==0).mean(1))]:d[f'{name}{w}']=np.repeat(v,H)
    idx=o+np.arange(H)-period
    idx=np.where(idx>=o,idx-period,idx)
    assert (idx>=0).all() and (idx<o).all()
    d['seasonal_target_lag']=y[:,idx].ravel()
    c=p['cov']
    if group in ('plan','weather','stock'):
        for k in PLAN:d[k]=c[k][:,o:o+H].ravel()
    if group in ('weather','stock'):
        for k in WEATHER:
            d[k+'_history7']=np.repeat(c[k][:,o-7:o].mean(1),H)
    if group=='stock':
        d['stock_history7']=np.repeat(c['stock_hour6_22_cnt'][:,o-7:o].mean(1)/17,H)
    if group=='price':
        a=c['price'][:,:o]
        valid=np.isfinite(a);idx=np.where(valid,np.arange(o),-1).max(1)
        last=a[np.arange(n),np.maximum(idx,0)].copy();last[idx<0]=np.nan
        d['last_observed_price']=np.repeat(last,H)
        d['price_missing_recent13']=np.repeat((~valid[:,-13:]).mean(1),H)
    frame=pd.DataFrame(d)
    for k in p['static']:frame[k]=pd.Categorical(frame[k],categories=sorted(p['static'][k].unique()))
    return frame


def gbm(p,o,group):
    H=p['H']
    origins=sorted(set(list(range(28,o-H+1,7))+[o-H])) if H==7 else sorted([o-H-13*j for j in range(8) if o-H-13*j>=52])
    assert origins and max(origins)+H<=o
    X=pd.concat([features(p,t,group) for t in origins],ignore_index=True)
    y=np.concatenate([p['y'][:,t:t+H].ravel() for t in origins])
    model=lgb.LGBMRegressor(n_estimators=250,learning_rate=.05,num_leaves=31,min_child_samples=80,reg_lambda=1,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(X,np.log1p(y),categorical_feature=list(p['static']))
    future=features(p,o,group)
    assert list(future.columns)==list(X.columns)
    pred=np.maximum(0,np.expm1(model.predict(future))).reshape(len(p['y']),H)
    return pred,{'training_origins':origins,'training_rows':len(X),'features':list(X),'params':model.get_params()}


def state_exog(p,o,group):
    H=p['H'];cov=p['cov'];start=7
    train=[];future=[]
    if H==7:
        for k in PLAN:
            train.append(cov[k][:,start:o]);future.append(cov[k][:,o:o+H])
        if group=='weather':
            for k in WEATHER:
                # Known historical weather during fitting; persistence forecast at origin.
                train.append(cov[k][:,start:o]);future.append(np.repeat(cov[k][:,o-7:o].mean(1)[:,None],H,axis=1))
    else:
        # Forward fill within history, never backwards from future observations.
        price=pd.DataFrame(cov['price'][:,:o]).T.ffill().T.to_numpy()
        train.append(np.nan_to_num(price[:,start:o],nan=0.))
        future.append(np.repeat(np.nan_to_num(price[:,-1],nan=0.)[:,None],H,axis=1))
    xt=np.stack(train,axis=2);xf=np.stack(future,axis=2)
    mean=xt.mean((0,1));scale=np.maximum(xt.std((0,1)),1e-6)
    xt=(xt-mean)/scale;xf=(xf-mean)/scale
    pred,detail=exog_forecast(p['y'][:,start:o],xt,xf,qr=.02,beta_var=1,period=p['period'],harmonics=3 if H==7 else 2)
    return pred,{'train_start':start,'min_covariance_eigenvalue':detail['min_covariance_eigenvalue'],'note':'per-series regression coefficients; future weather/price persistence; conditional planned promotions'}


def metrics(pred,y):
    assert pred.shape==y.shape and np.isfinite(pred).all() and (pred>=0).all()
    den=y.sum(1);err=pred-y;valid=den>0
    return {'wape':float(abs(err).sum()/y.sum()),'bias':float(err.sum()/y.sum()),'mae':float(abs(err).mean()),'mean_series_wape':float((abs(err).sum(1)[valid]/den[valid]).mean()),'zero_denominator_series':int((~valid).sum()),'vn1_score':float((abs(err).sum()+abs(err.sum()))/y.sum())}


def run(tag):
    p=load(tag);out=OUT/tag;out.mkdir(parents=True,exist_ok=True)
    p['static'].to_csv(out/'series_ids.csv',index=False)
    groups=['history','plan','weather','stock'] if tag=='freshretail' else ['history','price']
    models=['seasonal_naive','last_value','mean_recent','state_fourier']+['lgb_'+g for g in groups]+(['state_plan','state_weather'] if tag=='freshretail' else ['state_price'])
    rows=[];clients=[]
    protocol_hash=digest(OUT/'protocol.json')
    for o in p['origins']:
        if o==p['origins'][-1]:
            # Freeze model choice before computing final predictions or scores.
            scores={m:sum(r['absolute_error'] for r in rows if r['model']==m)/sum(r['actual_total'] for r in rows if r['model']==m) for m in models}
            (out/'selection.json').write_text(json.dumps({'development_wape':scores,'selected':min(scores,key=scores.get),'final_origin':o,'protocol_sha256':protocol_hash},indent=2)+'\n')
        truth=p['y'][:,o:o+p['H']]
        for model in models:
            path=out/f'o{o}_{model}.npz';meta={};t=time.perf_counter()
            if path.exists():raise RuntimeError('Refusing to silently reuse predictions: '+str(path))
            if model=='seasonal_naive':
                idx=o+np.arange(p['H'])-p['period'];idx=np.where(idx>=o,idx-p['period'],idx);pred=p['y'][:,idx]
            elif model=='last_value':pred=np.repeat(p['y'][:,o-1:o],p['H'],axis=1)
            elif model=='mean_recent':pred=np.repeat(p['y'][:,max(0,o-(7 if tag=='freshretail' else 12)):o].mean(1)[:,None],p['H'],axis=1)
            elif model=='state_fourier':pred=state_forecast(p['y'][:,:o],h=p['H'],period=p['period'],harmonics=3 if tag=='freshretail' else 2,qr=.02)
            elif model.startswith('lgb_'):pred,meta=gbm(p,o,model[4:])
            else:pred,meta=state_exog(p,o,model[6:])
            elapsed=time.perf_counter()-t
            np.savez_compressed(path,pred=pred,truth=truth)
            row=dict(dataset=tag,origin=o,model=model,seconds=elapsed,absolute_error=float(abs(pred-truth).sum()),actual_total=float(truth.sum()),**metrics(pred,truth));rows.append(row)
            (out/f'o{o}_{model}.json').write_text(json.dumps(dict(protocol_sha256=protocol_hash,**meta),indent=2)+'\n')
            if tag=='vn1':
                for client in p['static'].Client.unique():
                    mask=p['static'].Client.to_numpy()==client
                    yy=truth[mask];pp=pred[mask];den=yy.sum()
                    clients.append(dict(origin=o,model=model,client=int(client),actual_total=float(den),wape=float(abs(pp-yy).sum()/den) if den>0 else None))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            if clients:pd.DataFrame(clients).to_csv(out/'client_metrics.csv',index=False)
            print(tag,o,model,round(row['wape']*100,3),round(elapsed,2),flush=True)
    (out/'manifest.json').write_text(json.dumps({'protocol_sha256':protocol_hash,'series':len(p['y']),'dates':[str(p['dates'][0]),str(p['dates'][-1])],'versions':{'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__},'status':'completed'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['freshretail','vn1']);run(parser.parse_args().dataset)
