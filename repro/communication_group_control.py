"""Post-hoc diagnostic: same joint distribution, different factor grouping.
No prediction configuration is selected from these test diagnostics.
"""
import json,time
import numpy as np
from discrete_mpa import prepare,exact_marginals,mpa,predict
from audit_experiment import OUT,metrics
import fg_mp


def main():
    y,truth,_=fg_mp.load();c=json.loads((OUT/'communication_mpa.json').read_text())['200']['config'];rows={};ref=None
    for group in [5,13,26,52,200]:
        p=prepare(y,**c,group=group);exact=exact_marginals(p['factors'],p['neighbors'],p['logprior']);ep=predict(p,exact)
        if ref is None:ref=ep
        it=1 if group==200 else 60;damping=1. if group==200 else .3
        start=time.perf_counter();bs,diag=mpa(p['factors'],p['neighbors'],p['logprior'],steps=(it,),damping=damping)
        rows[str(group)]={'mpa':metrics(predict(p,bs[it]),truth),'exact':metrics(ep,truth),'exact_prediction_difference_across_groupings':float(abs(ep-ref).max()),'mpa_mean_tv':float(.5*abs(bs[it]-exact).sum(-1).mean()),'n_factors':len(p['factors']),'steps':it,'damping':damping,'mpa_s':time.perf_counter()-start,**diag[it]}
        (OUT/'communication_group_control.json').write_text(json.dumps(rows,indent=2));print(group,rows[str(group)],flush=True)
if __name__=='__main__':main()
