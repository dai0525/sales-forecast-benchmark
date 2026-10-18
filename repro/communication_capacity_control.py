"""Supplement: K=4 (same as final Gaussian model), 3**8 exact states.
Use one merged factor to avoid loop approximation; select amplitude on history.
Added after initial K<=2 results; explicitly exploratory capacity check.
"""
import json,time
import numpy as np
from discrete_mpa import prepare,exact_marginals,predict
from fg_state import forecast
from audit_experiment import OUT,metrics
import fg_mp


def main():
    y,truth,ids=fg_mp.load();cache={};rows={};predictions={}
    protocol={'added_after_initial_results':True,'harmonics':4,'alphabet':3,'amplitudes':[.5,1.,2.],'qr':.2,'trend_q':.0001,'damp':.98,'selection':'same three preceding windows per outer origin','purpose':'Check whether restricting initial MPA experiment to K<=2 caused failure; not a full K4 loopy BP sweep.'}
    (OUT/'communication_capacity_protocol.json').write_text(json.dumps(protocol,indent=2))
    def run(a,o):
        if (a,o) not in cache:
            t=time.perf_counter();p=prepare(y[:,:o],harmonics=4,alphabet=3,amplitude=a,qr=.2,group=200)
            m=exact_marginals(p['factors'],p['neighbors'],p['logprior']);discrete=predict(p,m)
            gaussian=forecast(y[:,:o],harmonics=4,qr=.2,trend_q=.0001,damp=.98,seasonal_var=p['prior_variance'])
            cache[a,o]=(discrete,gaussian,time.perf_counter()-t)
        return cache[a,o]
    for o in [161,174,187,200]:
        true=truth if o==200 else y[:,o:o+13];tune=[o-39,o-26,o-13];scores={};variants={}
        for method,index in [('discrete_exact',0),('continuous_matched_grid',1)]:
            scores[method]={str(a):float(np.mean([metrics(run(a,v)[index],y[:,v:v+13])['wape_mean'] for v in tune])) for a in protocol['amplitudes']}
            chosen=float(min(scores[method],key=scores[method].get))
            pred=run(chosen,o)[index];variants[method]={'amplitude':chosen,**metrics(pred,true)};predictions[f'{method}_{o}']=pred
        rows[str(o)]={'validation':scores,'results':variants}
        (OUT/'communication_capacity.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(OUT/'communication_capacity_predictions.npz',ids=ids,**predictions)
        print(o,variants,flush=True)
if __name__=='__main__':main()
