"""Business-boundary reliability routing with exact chain sum-product.
This is a contextual expert ensemble, not a causal decomposition of sales.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import research_iteration as r

OUT=r.ROOT/'audit/results/reliability_bp'
NAMES=['l2','l1','open_l2','open_l1']

def chain_marginals(unary,edges,strength):
    """Exact marginals of prod exp(unary) prod exp(strength*edge*I[same])."""
    unary=np.asarray(unary,float);edges=np.asarray(edges,float)
    n,h,m=unary.shape
    if edges.shape!=(n,h-1) or not np.isfinite(unary).all() or strength<0:
        raise ValueError('Invalid chain')
    a=np.zeros_like(unary);b=np.zeros_like(unary);eye=np.eye(m)[None,:,:]
    a[:,0]=unary[:,0]-logsumexp(unary[:,0],axis=-1,keepdims=True)
    for t in range(1,h):
        pair=strength*edges[:,t-1,None,None]*eye
        v=unary[:,t]+logsumexp(a[:,t-1,:,None]+pair,axis=1)
        a[:,t]=v-logsumexp(v,axis=-1,keepdims=True)
    for t in range(h-2,-1,-1):
        pair=strength*edges[:,t,None,None]*eye
        v=logsumexp(pair+unary[:,t+1,None,:]+b[:,t+1,None,:],axis=2)
        b[:,t]=v-logsumexp(v,axis=-1,keepdims=True)
    return softmax(a+b,axis=-1)

def features(p,o,pred):
    h=pred.shape[1];t=np.arange(o,o+h);n=p.n
    def repeat(v):return np.broadcast_to(np.asarray(v),(n,h))
    mean=p.y[:,o-28:o].mean(1)
    cv=p.y[:,o-28:o].std(1)/np.maximum(mean,1)
    ratio=p.y[:,o-7:o].mean(1)/np.maximum(mean,1)
    logp=np.log1p(pred);center=logp.mean(2,keepdims=True)
    d=[p.promo[:,t],p.open_[:,t],p.school[:,t]]+[p.state[:,t]==k for k in [1,2,3]]
    names=['promo','open','school','state_a','state_b','state_c']
    for name,v in [('dow_sin',np.sin(2*np.pi*p.cal['dow'][t]/7)),('dow_cos',np.cos(2*np.pi*p.cal['dow'][t]/7)),('horizon',np.arange(1,h+1)/h),('history_cv',cv[:,None]),('history_ratio',ratio[:,None])]:
        d.append(repeat(v));names.append(name)
    for j,k in enumerate(NAMES):d.append(logp[:,:,j]-center[:,:,0]);names.append('relative_log_'+k)
    d.append(logp.std(2));names.append('disagreement')
    return np.stack(d,axis=2),names

def boundaries(p,o,h,kind='business'):
    op=p.open_[:,o:o+h];edge=(op[:,:-1]==1)&(op[:,1:]==1)
    if kind=='constant':return edge.astype(float)
    for arr in [p.promo,p.school,p.state]:
        x=arr[:,o:o+h];edge&=x[:,:-1]==x[:,1:]
    if kind=='shuffled':
        rng=np.random.default_rng(20261020)
        # Shuffle only eligible open/open links, preserving link count.
        eligible=(op[:,:-1]==1)&(op[:,1:]==1)
        for i in range(len(edge)):
            idx=np.flatnonzero(eligible[i])
            edge[i,idx]=rng.permutation(edge[i,idx])
    return edge.astype(float)

def load_pred(tag,o):
    return np.stack([np.load(r.OUT/tag/f'components_o{o}_{k}.npz')['pred'] for k in NAMES],axis=2)

def predict(p,o,pred,model,strength,kind='business'):
    X,names=features(p,o,pred)
    if names!=model['feature_names']:raise ValueError('Feature schema changed')
    z=(X-model['mean'])/model['scale']
    risk=z@model['coef'].T+model['intercept']
    unary=np.log(model['prior'])[None,None,:]-risk/.05
    marg=chain_marginals(unary,boundaries(p,o,pred.shape[1],kind),strength)
    weights=.5*marg+.5*model['prior'][None,None,:]
    out=np.expm1((weights*np.log1p(pred)).sum(2))
    return out,weights

def train():
    OUT.mkdir(parents=True,exist_ok=True)
    p=r.rf.b.Panel(r.rf.b.PANEL);pred=load_pred('rossmann',732)
    X,names=features(p,732,pred);y=p.y[:,732:774]
    loss=np.abs(pred-y[:,:,None])/np.maximum(np.abs(y).mean(1)[:,None,None],1)
    target=loss-loss.mean(2,keepdims=True)
    scaler=StandardScaler().fit(X.reshape(-1,X.shape[-1]))
    gate=Ridge(alpha=100).fit(scaler.transform(X.reshape(-1,X.shape[-1])),target.reshape(-1,4))
    sel=json.loads((r.OUT/'rossmann/selection.json').read_text())
    prior=np.array([sel['configs'][r.PRIMARY]['all'][k] for k in NAMES])
    model={'mean':scaler.mean_,'scale':scaler.scale_,'coef':gate.coef_,'intercept':gate.intercept_,'prior':prior,'feature_names':names}
    pv=load_pred('rossmann',774);grid=[]
    for strength in [0,.5,2,4]:
        yp,_=predict(p,774,pv,model,strength)
        grid.append({'lambda':strength,**r.metric(yp,p.y[:,774:816])})
    chosen=min(grid,key=lambda x:x['wape_mean'])['lambda']
    selection={'chosen_lambda':chosen,'grid':grid,'feature_names':names,'train_origin':732,'selection_origin':774,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()}
    r.dump(OUT/'selection.json',selection)
    np.savez_compressed(OUT/'gate.npz',**{k:v for k,v in model.items() if k!='feature_names'})
    print(json.dumps(selection,indent=2),flush=True)

def evaluate():
    selection=json.loads((OUT/'selection.json').read_text());z=np.load(OUT/'gate.npz')
    model={k:z[k] for k in z.files};model['feature_names']=selection['feature_names']
    strength=selection['chosen_lambda'];report={}
    for tag in ['rossmann','confirmation']:
        p=r.rf.b.Panel(r.rf.b.PANEL if tag=='rossmann' else r.OUT/'confirmation/panel.npz')
        rows={};errors={};timing=[]
        for o in [816,858,900]:
            pred=load_pred(tag,o);truth=p.y[:,o:o+42];methods={};weights={}
            for name,lam,kind in [('business_bp',strength,'business'),('no_messages',0,'business'),('constant_edges',strength,'constant'),('shuffled_edges',strength,'shuffled')]:
                t=time.perf_counter();methods[name],weights[name]=predict(p,o,pred,model,lam,kind);timing.append({'origin':o,'method':name,'elapsed_s':time.perf_counter()-t})
            base=np.load(r.OUT/tag/f'predictions_o{o}.npz')
            for name in [r.PRIMARY,'original_stage','validation_best_single']:methods[name]=base['pred_'+name]
            rows[str(o)]={k:r.metric(v,truth) for k,v in methods.items()}
            errors[o]={k:np.abs(v-truth).sum(1)/np.abs(truth).sum(1) for k,v in methods.items()}
            path=OUT/tag;path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/f'predictions_o{o}.npz',truth=truth,ids=np.array(p.ids),dates=p.days[o:o+42],**{'pred_'+k:v for k,v in methods.items()},**{'weights_'+k:v for k,v in weights.items()})
        from report_iteration import ci
        comparisons={}
        for k in methods:
            if k=='business_bp':continue
            d=np.mean([errors[o][k]-errors[o]['business_bp'] for o in errors],axis=0)
            comparisons[k]={'delta_pp':float(d.mean()*100),**ci(d)}
        report[tag]={'origins':rows,'mean_wape':{k:float(np.mean([v[k]['wape_mean'] for v in rows.values()])) for k in methods},'delta_positive_favors_business_bp':comparisons,'router_timing':timing,'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    r.dump(OUT/'summary.json',report)
    for tag,v in report.items():print(tag,v['mean_wape'],v['delta_positive_favors_business_bp'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','evaluate']);args=ap.parse_args()
    train() if args.mode=='train' else evaluate()
