"""Discrete codebook sum-product (communication-style MPA), plus exact control.
No claim of SCMA sparsity for the sales Fourier graph. Gaussian level/trend is
analytically marginalized; only seasonal symbols use finite codebooks.
"""
import itertools
import numpy as np
from scipy.special import logsumexp


def normalize(logp):
    return logp-logsumexp(logp,axis=-1,keepdims=True)


def assignments(m,d):
    return np.array(list(itertools.product(range(m),repeat=d)),dtype=int)


def exact_marginals(factors,neighbors,logprior):
    n,d,m=logprior.shape; states=assignments(m,d)
    joint=sum(logprior[:,j,states[:,j]] for j in range(d))
    for phi,nb in zip(factors,neighbors):
        local=np.ravel_multi_index(states[:,nb].T,(m,)*len(nb))
        joint=joint+phi[:,local]
    joint=normalize(joint)
    marg=np.stack([np.stack([np.exp(logsumexp(joint[:,states[:,j]==v],axis=1)) for v in range(m)],axis=1) for j in range(d)],axis=1)
    return marg


def mpa(factors,neighbors,logprior,steps=(1,10,30,100),damping=.5):
    """Extrinsic sum-product messages, damped in probability space.

    factors[f]: batch x M**degree log potential; logprior: batch x variables x M.
    No max-product, hard decision or test-dependent stopping.
    """
    if not 0<damping<=1:raise ValueError('Invalid damping')
    n,d,m=logprior.shape
    adjacency=[[] for _ in range(d)]; localstates=[]; messages=[]
    for fi,(phi,nb) in enumerate(zip(factors,neighbors)):
        if phi.shape!=(n,m**len(nb)):raise ValueError('Invalid factor shape')
        localstates.append(assignments(m,len(nb)))
        messages.append(np.full((n,len(nb),m),-np.log(m)))
        for li,j in enumerate(nb):adjacency[j].append((fi,li))
    saved={};last=np.exp(normalize(logprior)); diag={}
    for it in range(1,max(steps)+1):
        total=logprior.copy()
        for j in range(d):
            for fi,li in adjacency[j]:total[:,j]+=messages[fi][:,li]
        updated=[];maxchange=0.
        for fi,(phi,nb,st) in enumerate(zip(factors,neighbors,localstates)):
            incoming=[normalize(total[:,j]-messages[fi][:,li]) for li,j in enumerate(nb)]
            joint=phi.copy()
            for li in range(len(nb)):joint+=incoming[li][:,st[:,li]]
            edge=[]
            for li in range(len(nb)):
                extrinsic=joint-incoming[li][:,st[:,li]]
                new=normalize(np.stack([logsumexp(extrinsic[:,st[:,li]==v],axis=1) for v in range(m)],axis=1))
                if damping<1:new=normalize(np.logaddexp(np.log(damping)+new,np.log1p(-damping)+messages[fi][:,li]))
                maxchange=max(maxchange,float(abs(np.exp(new)-np.exp(messages[fi][:,li])).max()))
                edge.append(new)
            updated.append(np.stack(edge,axis=1))
        messages=updated
        total=logprior.copy()
        for j in range(d):
            for fi,li in adjacency[j]:total[:,j]+=messages[fi][:,li]
        belief=np.exp(normalize(total))
        if it in steps:
            saved[it]=belief.copy();diag[it]={'max_message_probability_change':maxchange,'max_belief_probability_change':float(abs(belief-last).max())}
        last=belief
    return saved,diag


def prepare(y,harmonics=1,qr=.2,trend_q=.0001,damp=.98,alphabet=5,amplitude=1.,group=13):
    """Whiten y and Fourier basis under the same Gaussian level/trend model."""
    y=np.asarray(y,float);n,tmax=y.shape;d=2*harmonics
    scale=np.maximum(y.std(1),.02*abs(y).mean(1)+1e-6)
    tt=np.arange(tmax+13)[:,None];ang=2*np.pi*tt*np.arange(1,harmonics+1)/52
    basis=np.concatenate([np.cos(ang),np.sin(ang)],axis=1)
    inputs=np.concatenate([(y/scale[:,None]).T,basis[:tmax]],axis=1)
    f=np.array([[1.,damp],[0.,damp]]);q=np.diag([qr,trend_q]);p=np.diag([1e6,1. if trend_q else 0.]);state=np.zeros((2,n+d));white=[]
    for t in range(tmax):
        if t:state=f@state;p=f@p@f.T+q
        residual=inputs[t]-state[0];v=p[:,0];variance=p[0,0]+1.;gain=v/variance
        white.append(residual/np.sqrt(variance));state+=gain[:,None]*residual
        j=np.eye(2)-np.outer(gain,[1.,0.]);p=j@p@j.T+np.outer(gain,gain);p=(p+p.T)/2
    base=[];effects=[]
    for step in range(13):
        state=f@state;base.append(state[0,:n]);effects.append(basis[tmax+step]-state[0,n:])
    white=np.asarray(white);wy=white[:,:n];wb=white[:,n:]
    grid=np.arange(alphabet)-(alphabet-1)/2;codes=amplitude*grid
    prior=normalize(-.5*grid**2);states=assignments(alphabet,d);symbols=codes[states]
    factors=[]
    for start in range(0,tmax,group):
        a=wb[start:start+group];z=wy[start:start+group]
        # Drop -y'y/2: it is independent of all symbols.
        factors.append(z.T@a@symbols.T-.5*np.sum((a@symbols.T)**2,axis=0)[None,:])
    return {'factors':factors,'neighbors':[list(range(d)) for _ in factors], 'logprior':np.broadcast_to(prior,(n,d,alphabet)).copy(),'codes':codes,'base':np.asarray(base).T,'effects':np.asarray(effects),'scale':scale,'prior_variance':float(np.sum(np.exp(prior)*codes**2))}


def predict(prepared,marginals):
    coef=marginals@prepared['codes']
    return np.maximum((prepared['base']+coef@prepared['effects'].T)*prepared['scale'][:,None],0)
