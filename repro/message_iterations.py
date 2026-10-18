"""Same Gaussian model: batch conditioning versus iterative component backfitting.
This is a posterior-mean solver experiment, not SCMA or full-distribution BP.
"""
import json
import numpy as np
from fg_state import forecast
from audit_experiment import OUT, metrics
import fg_mp


def kernels(t,h,c):
    """Construct prior covariance and future/history cross covariance independently."""
    f=np.array([[1.,c['damp']],[0.,c['damp']]])
    q=np.diag([c['qr'],c['trend_q']])
    p=np.diag([1e6,1. if c['trend_q'] else 0.])
    length=t+h; cov=np.empty((length,length)); powers=[np.linalg.matrix_power(f,i) for i in range(length)]
    for i in range(length):
        if i:p=f@p@f.T+q
        for j in range(i,length):cov[j,i]=cov[i,j]=(powers[j-i]@p)[0,0]
    phase=2*np.pi*np.arange(length)[:,None]*np.arange(1,c['harmonics']+1)/52
    b=np.concatenate([np.cos(phase),np.sin(phase)],axis=1)
    seasonal=b@b.T
    return cov[:t,:t],seasonal[:t,:t],cov[t:,:t],seasonal[t:,:t]


def solve(y,c,steps=(1,2,5,10,25,50,100,200,500)):
    n,t=y.shape;scale=np.maximum(np.std(y,axis=1),.02*np.abs(y).mean(axis=1)+1e-6)
    z=(y/scale[:,None]).T
    l,s,fl,fs=kernels(t,13,c); eye=np.eye(t)
    alpha=np.linalg.solve(l+s+eye,z)
    exact=np.maximum(((fl+fs)@alpha).T*scale[:,None],0)
    # Conditional component updates: E[L | y,S], then E[S | y,L], or reverse.
    # Iterating these means solves the joint Gaussian posterior mean equations.
    il=np.linalg.solve(l+eye,eye);iss=np.linalg.solve(s+eye,eye)
    al=l@il;ass=s@iss;gl=fl@il;gs=fs@iss
    outputs={};diagnostics={}
    target_l=l@alpha;target_s=s@alpha
    for order in ['level_first','season_first']:
        ml=np.zeros_like(z);ms=np.zeros_like(z)
        pl=np.zeros((13,n));ps=pl.copy()
        for iteration in range(1,max(steps)+1):
            if order=='level_first':
                residual=z-ms;ml=al@residual;pl=gl@residual
                residual=z-ml;ms=ass@residual;ps=gs@residual
            else:
                residual=z-ml;ms=ass@residual;ps=gs@residual
                residual=z-ms;ml=al@residual;pl=gl@residual
            if iteration in steps:
                key=f'{order}_{iteration}'
                outputs[key]=np.maximum((pl+ps).T*scale[:,None],0)
                diagnostics[key]={'prediction_max_abs_to_exact':float(abs(outputs[key]-exact).max()),'component_mean_max_abs_normalized':float(max(abs(ml-target_l).max(),abs(ms-target_s).max()))}
    return exact,outputs,diagnostics


def main():
    y,truth,ids=fg_mp.load()
    previous=json.loads((OUT/'message_ablation.json').read_text())
    protocol={'fixed_configs':{o:r['selected']['full']['config'] for o,r in previous.items()},'steps':[1,2,5,10,25,50,100,200,500], 'orders':['level_first','season_first'],'selection':'reuse full model parameters chosen only in preceding validation windows; no iteration count selected on test','scope':'same Gaussian prior and likelihood; iterative conditional posterior MEANS, not full BP distributions or SCMA','initialization':'both component means zero, consistent with prior means','early_stopping':'select order and iteration count by mean WAPE over preceding three windows, holding the already validation-selected model config fixed'}
    (OUT/'message_iterations_protocol.json').write_text(json.dumps(protocol,indent=2))
    results={};saved={}
    for origin,config in protocol['fixed_configs'].items():
        o=int(origin);true=truth if o==200 else y[:,o:o+13]
        exact,ps,diag=solve(y[:,:o],config)
        kalman=forecast(y[:,:o],**config)
        validation={k:[] for k in ps}
        for v in [o-39,o-26,o-13]:
            _,vp,_=solve(y[:,:v],config)
            for k,p in vp.items():validation[k].append(metrics(p,y[:,v:v+13])['wape_mean'])
        chosen=min(validation,key=lambda k:np.mean(validation[k]))
        results[origin]={'exact':metrics(exact,true),'validation_selected_iteration':{'key':chosen,'validation_mean':float(np.mean(validation[chosen])),'test':metrics(ps[chosen],true)},'validation_scores':{k:float(np.mean(v)) for k,v in validation.items()},'batch_vs_kalman_max_abs':float(abs(exact-kalman).max()),'variants':{k:{**metrics(p,true),**diag[k]} for k,p in ps.items()}}
        saved[f'exact_{origin}']=exact
        saved.update({f'{k}_{origin}':p for k,p in ps.items()})
        (OUT/'message_iterations.json').write_text(json.dumps(results,indent=2))
        np.savez_compressed(OUT/'message_iterations_predictions.npz',ids=ids,**saved)
        print(origin,'batch vs Kalman',results[origin]['batch_vs_kalman_max_abs'],[(k,round(v['wape_mean']*100,4),v['prediction_max_abs_to_exact']) for k,v in results[origin]['variants'].items()],flush=True)
if __name__=='__main__':main()
