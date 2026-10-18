"""Controlled covariance-coupling ablation; no test-based selection."""
import json, time
import numpy as np
from fg_state import forecast
import fg_mp
from audit_experiment import OUT, metrics

def compare(a,b,true):
    delta=(np.abs(a-true).sum(1)-np.abs(b-true).sum(1))/np.abs(true).sum(1)
    rng=np.random.default_rng(42)
    boots=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return {'full_minus_block':float(delta.mean()),'bootstrap_95':np.quantile(boots,[.025,.975]).tolist()}

def main():
    y,truth,ids=fg_mp.load()
    grid=json.loads((OUT/'protocol.json').read_text())['grid']
    protocol={'modes':['full','block','information'],'block_definition':'After each observation, zero only covariance between level/trend and all seasonal coefficients; keep within-block covariance, shared observation innovation and identical priors/dynamics.', 'fixed_control':'Select full hyperparameters using preceding three validation windows; use same hyperparameters for block.', 'retuned_control':'Also select block independently using same grid and preceding windows.', 'outer_origins':[161,174,187,200], 'grid':grid, 'limitation':'Block approximation still uses observation messages; this is not removal of all message passing. Public test is previously observed.'}
    (OUT/'message_ablation_protocol.json').write_text(json.dumps(protocol,indent=2))
    results={};preds={}; cache={}
    def score(mode,c,o):
        key=(mode,tuple(c.values()),o)
        if key not in cache: cache[key]=metrics(forecast(y[:,:o],covariance_mode=mode,**c),y[:,o:o+13])['wape_mean']
        return cache[key]
    for origin in protocol['outer_origins']:
        tuning=[origin-39,origin-26,origin-13]
        selected={}
        for mode in ['full','block']:
            scores=[np.mean([score(mode,c,o) for o in tuning]) for c in grid]
            i=int(np.argmin(scores));selected[mode]={'config':grid[i],'validation_mean':float(scores[i])}
        c=selected['full']['config'];true=truth if origin==200 else y[:,origin:origin+13]
        cases={'full':('full',c),'block_fixed':('block',c),'block_retuned':('block',selected['block']['config']),'information':('information',c)}
        ps={k:forecast(y[:,:origin],covariance_mode=m,**conf) for k,(m,conf) in cases.items()}
        results[str(origin)]={'selected':selected,'metrics':{k:metrics(p,true) for k,p in ps.items()},'full_information_max_abs':float(abs(ps['full']-ps['information']).max()),'fixed_comparison':compare(ps['full'],ps['block_fixed'],true),'retuned_comparison':compare(ps['full'],ps['block_retuned'],true)}
        preds.update({f'{k}_{origin}':p for k,p in ps.items()})
        (OUT/'message_ablation.json').write_text(json.dumps(results,indent=2))
        np.savez_compressed(OUT/'message_ablation_predictions.npz',ids=ids,**preds)
        print(origin, {k:round(v['wape_mean']*100,6) for k,v in results[str(origin)]['metrics'].items()},'equivalence',results[str(origin)]['full_information_max_abs'],flush=True)
if __name__=='__main__':main()
