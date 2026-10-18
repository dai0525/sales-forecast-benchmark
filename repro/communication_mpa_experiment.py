"""Finite-codebook likelihood MPA, exact enumeration and matched Gaussian controls."""
import json,itertools,time,hashlib
import numpy as np
from discrete_mpa import prepare,exact_marginals,mpa,predict
from fg_state import forecast
import fg_mp
from audit_experiment import OUT,metrics


def paired(a,b,y):
    delta=(abs(a-y).sum(1)-abs(b-y).sum(1))/abs(y).sum(1)
    rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(y),(10000,len(y)))].mean(1)
    return {'delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'win_fraction':float((delta<0).mean())}


def main():
    y,truth,ids=fg_mp.load()
    configs=[dict(harmonics=k,alphabet=m,amplitude=a,qr=q,trend_q=.0001,damp=.98) for (k,m),a,q in itertools.product([(1,5),(2,3)],[.5,1.,2.],[.02,.2])]
    protocol={'configs':configs,'outer_origins':[161,174,187,200],'tuning':'three preceding nonoverlapping 13-step windows; exact enumeration chooses model config; MPA chooses damping/iterations on same validation windows only','damping':[.3,.7],'iterations':[1,5,20,60],'factor_group':13,'controls':['exact same discrete model','continuous Gaussian coefficients with matched prior variance and dynamics','one merged factor exact MPA','previous best State-Fourier and UCM'], 'decision':'Seek predictive benefit across historical folds and official test, plus practical runtime. No test-based winner selection. If absent, pause this transfer route; not a universal impossibility theorem.', 'limitations':'Known public test; finite grid; Gaussian dynamics + quantized seasonal coefficients, not business causal factors. Dense Fourier connectivity lacks SCMA sparsity. MPA structure chosen via exact validation model, not exhaustive joint tuning.'}
    (OUT/'communication_mpa_protocol.json').write_text(json.dumps(protocol,indent=2))
    cache={};bp_cache={};rows={};predictions={}
    def get(c,o):
        key=(tuple(c.values()),o)
        if key not in cache:
            start=time.perf_counter();p=prepare(y[:,:o],**c)
            marginal=exact_marginals(p['factors'],p['neighbors'],p['logprior']);pred=predict(p,marginal)
            cache[key]=(p,marginal,pred,time.perf_counter()-start)
        return cache[key]
    def getbp(c,o):
        key=(tuple(c.values()),o)
        if key not in bp_cache:
            p,exact,_,_=get(c,o);out={}
            for damping in protocol['damping']:
                start=time.perf_counter();bs,diags=mpa(p['factors'],p['neighbors'],p['logprior'],steps=protocol['iterations'],damping=damping)
                elapsed=time.perf_counter()-start
                for it,b in bs.items():
                    out[f'd{damping}_i{it}']=(predict(p,b),{'marginal_mean_tv_to_exact':float(.5*abs(b-exact).sum(-1).mean()),'marginal_max_tv_to_exact':float((.5*abs(b-exact).sum(-1)).max()),**diags[it],'entire_60_step_run_s':elapsed})
            bp_cache[key]=out
        return bp_cache[key]
    old=json.loads((OUT/'message_ablation.json').read_text())
    roll=np.load(OUT/'rolling_predictions.npz');official=np.load(OUT/'state_predictions.npz')
    for o in protocol['outer_origins']:
        tune=[o-39,o-26,o-13];truth_o=truth if o==200 else y[:,o:o+13]
        grid_scores=[float(np.mean([metrics(get(c,v)[2],y[:,v:v+13])['wape_mean'] for v in tune])) for c in configs]
        best=configs[int(np.argmin(grid_scores))]
        print('outer',o,'selected',best,flush=True)
        validation={}
        for v in tune:
            for key,(p,_) in getbp(best,v).items():validation.setdefault(key,[]).append(metrics(p,y[:,v:v+13])['wape_mean'])
        chosen=min(validation,key=lambda k:np.mean(validation[k]))
        prepared,marg,exact,exact_s=get(best,o);allbp=getbp(best,o);selected=allbp[chosen][0]
        continuous=forecast(y[:,:o],harmonics=best['harmonics'],qr=best['qr'],trend_q=best['trend_q'],damp=best['damp'],seasonal_var=prepared['prior_variance'])
        merged=[sum(prepared['factors'])]
        start=time.perf_counter();tree,_=mpa(merged,[prepared['neighbors'][0]],prepared['logprior'],steps=(1,),damping=1.)
        tree_s=time.perf_counter()-start;treepred=predict(prepared,tree[1])
        state=forecast(y[:,:o],**old[str(o)]['selected']['full']['config'])
        ucm=official['UCM52-saved'] if o==200 else roll[f'ucm_{o}']
        variants={'discrete_exact':exact,'mpa_selected':selected,'matched_continuous':continuous,'state_fourier':state,'ucm':ucm}
        rows[str(o)]={'config':best,'exact_grid_validation_scores':grid_scores,'mpa_validation_scores':{k:float(np.mean(v)) for k,v in validation.items()},'selected_mpa':chosen,'metrics':{k:metrics(p,truth_o) for k,p in variants.items()},'all_mpa_variants':{k:{**metrics(p,truth_o),**d} for k,(p,d) in allbp.items()},'tree_prediction_max_abs_to_exact':float(abs(treepred-exact).max()),'exact_prepare_and_predict_s':exact_s,'tree_mpa_only_s':tree_s,'paired':{k:paired(selected,p,truth_o) for k,p in variants.items() if k!='mpa_selected'}}
        predictions.update({f'{k}_{o}':p for k,p in variants.items()})
        (OUT/'communication_mpa.json').write_text(json.dumps(rows,indent=2))
        np.savez_compressed(OUT/'communication_mpa_predictions.npz',ids=ids,**predictions)
        print('RESULT',o,chosen,{k:round(v['wape_mean']*100,4) for k,v in rows[str(o)]['metrics'].items()},flush=True)
if __name__=='__main__':main()
