"""Independent structural-coordinate check; run from repository root."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'repro'))
from fg_state import forecast

def structural(y,h,k,phi,qr,qt):
    # delta=phi*slope; rotate each static (cos,sin) coefficient pair.
    n,T=y.shape;d=2+2*k
    scale=np.maximum(np.std(y,axis=1),.02*np.abs(y).mean(axis=1)+1e-6)
    z=y/scale[:,None];F=np.eye(d);F[0,1]=1.;F[1,1]=phi
    for j in range(k):
        w=2*np.pi*(j+1)/52;c,s=np.cos(w),np.sin(w)
        F[2+2*j:4+2*j,2+2*j:4+2*j]=[[c,s],[-s,c]]
    Q=np.diag([qr,phi**2*qt]+[0.]*(d-2))
    P=np.diag([1e6,phi**2 if qt else 0.]+[1.]*(d-2))
    a=np.zeros(d);a[0]=1.;a[2::2]=1.;m=np.zeros((n,d))
    for t in range(T):
        if t:m=m@F.T;P=F@P@F.T+Q
        v=P@a;K=v/(1+a@v);m+=(z[:,t]-m@a)[:,None]*K
        J=np.eye(d)-np.outer(K,a);P=J@P@J.T+np.outer(K,K)
        P=(P+P.T)/2
    pred=[];var=[]
    for t in range(h):
        m=m@F.T;P=F@P@F.T+Q
        pred.append((m@a)*scale);var.append((1+a@P@a)*scale**2)
    return np.array(pred).T,np.array(var).T
rng=np.random.default_rng(47191);y=50+np.cumsum(rng.normal(size=(7,200)),axis=1)
results=[]
for k,phi,qt in [(0,.9,.001),(4,.98,.0001),(8,.9,0.),(4,1.,.001)]:
    _,a=forecast(y,13,harmonics=k,damp=phi,qr=.2,trend_q=qt,return_components=True)
    p,v=structural(y,13,k,phi,.2,qt)
    err=float(np.max(np.abs(a['raw_mean']-p)));ve=float(np.max(np.abs(a['variance']-v)))
    assert np.allclose(a['raw_mean'],p,rtol=1e-10,atol=1e-9)
    assert np.allclose(a['variance'],v,rtol=1e-10,atol=1e-9)
    results.append(dict(harmonics=k,damping=phi,trend_q=qt,max_abs_mean_error=err,max_abs_variance_error=ve))
Path(__file__).with_suffix('.json').write_text(json.dumps({'seed':47191,'scope':'Structural-coordinate equivalence, not accuracy experiment; same priors transformed exactly.','results':results},indent=2))
print(json.dumps(results,indent=2))
