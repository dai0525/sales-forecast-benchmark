"""Known sparse signatures + finite symbols: communication positive control.
BPSK sparse superposition, not a reproduction of a published SCMA codebook.
"""
import json,time
import numpy as np
from discrete_mpa import assignments,exact_marginals,mpa
from audit_experiment import OUT


def main():
    rng=np.random.default_rng(20261018);n=1000;d=6;m=2
    neighbors=[[0,1,2],[0,3,4],[1,3,5],[2,4,5]]
    gains=rng.normal(size=(4,d))
    for f,nb in enumerate(neighbors):gains[f,[j for j in range(d) if j not in nb]]=0
    gains/=np.sqrt((gains**2).sum(0))[None,:]
    symbols=rng.integers(0,2,(n,d));values=2*symbols-1
    rows={}
    for noise in [.2,.7]:
        obs=values@gains.T+rng.normal(0,noise,(n,4));factors=[]
        for f,nb in enumerate(neighbors):
            local=2*assignments(m,len(nb))-1
            factors.append(-.5*((obs[:,f,None]-local@gains[f,nb])/noise)**2)
        prior=np.zeros((n,d,m));start=time.perf_counter();exact=exact_marginals(factors,neighbors,prior);exact_s=time.perf_counter()-start
        start=time.perf_counter();bs,diag=mpa(factors,neighbors,prior,steps=(1,5,20,60),damping=.7);mpa_s=time.perf_counter()-start
        results={'exact':{'bit_error_rate':float(np.mean(np.argmax(exact,-1)!=symbols))}}
        for i,b in bs.items():results[str(i)]={'bit_error_rate':float(np.mean(np.argmax(b,-1)!=symbols)),'marginal_mean_tv':float(.5*abs(b-exact).sum(-1).mean()),**diag[i]}
        rows[str(noise)]={'results':results,'exact_s':exact_s,'mpa_60_steps_s':mpa_s}
    result={'seed':20261018,'samples':n,'neighbors':neighbors,'gains':gains.tolist(),'note':'Known signatures and Gaussian noise, BPSK sparse superposition. Not learned from sales and not claimed to be standard SCMA codebook performance.','runs':rows}
    (OUT/'communication_positive_control.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
