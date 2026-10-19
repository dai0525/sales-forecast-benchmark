"""Joint Gaussian State-Fourier regression. Open is an observation gate.
Static store-specific coefficients; normalization uses training targets only.
"""
import numpy as np


def forecast(y,x_train,x_future,open_train=None,open_future=None,qr=.02,beta_var=1.,period=7,harmonics=3):
    y=np.asarray(y,float);xt=np.asarray(x_train,float);xf=np.asarray(x_future,float)
    if y.ndim!=2 or xt.ndim!=3 or xf.ndim!=3 or xt.shape[:2]!=y.shape or xf.shape[0]!=y.shape[0] or xt.shape[2]!=xf.shape[2]:
        raise ValueError('Target/exogenous shapes do not align')
    if not all(np.isfinite(a).all() for a in (y,xt,xf)) or qr<0 or beta_var<=0:
        raise ValueError('Nonfinite data or invalid prior')
    n,T=y.shape;h=xf.shape[1];k=xt.shape[2];d=1+2*harmonics+k
    ot=np.ones_like(y) if open_train is None else np.asarray(open_train,float)
    of=np.ones((n,h)) if open_future is None else np.asarray(open_future,float)
    if ot.shape!=y.shape or of.shape!=(n,h) or not np.isin(ot,[0,1]).all() or not np.isin(of,[0,1]).all():
        raise ValueError('Open must be aligned binary arrays')
    scale=np.maximum(y.std(1),.02*np.abs(y).mean(1)+1e-6);z=y/scale[:,None]
    state=np.zeros((n,d));P=np.broadcast_to(np.diag([1e6]+[1.]*(2*harmonics)+[beta_var]*k),(n,d,d)).copy()
    def design(t,x):
        angle=2*np.pi*np.arange(1,harmonics+1)*t/period
        return np.column_stack((np.ones(n),np.tile(np.r_[np.cos(angle),np.sin(angle)],(n,1)),x))
    for t in range(T):
        if t:P[:,0,0]+=qr
        a=design(t,xt[:,t])*ot[:,t,None]
        v=np.einsum('nij,nj->ni',P,a);s=np.einsum('ni,ni->n',a,v)+1.
        gain=v/s[:,None];state+=gain*(z[:,t]-np.einsum('ni,ni->n',state,a))[:,None]
        P-=np.einsum('ni,nj->nij',gain,v);P=(P+P.transpose(0,2,1))/2
    components=np.zeros((n,h,3+k))
    for j in range(h):
        a=design(T+j,xf[:,j]);components[:,j,0]=state[:,0]*scale
        components[:,j,1]=np.sum(state[:,1:1+2*harmonics]*a[:,1:1+2*harmonics],axis=1)*scale
        components[:,j,2:2+k]=state[:,1+2*harmonics:]*xf[:,j]*scale[:,None]
        latent=components[:,j,:-1].sum(1);components[:,j,-1]=(of[:,j]-1)*latent
    raw=components.sum(2);pred=np.maximum(raw,0)
    return pred,{'components':components,'raw_mean':raw,'clipping_adjustment':pred-raw,
                 'coefficients':state[:,1+2*harmonics:]*scale[:,None],
                 'min_covariance_eigenvalue':float(np.linalg.eigvalsh(P).min())}
