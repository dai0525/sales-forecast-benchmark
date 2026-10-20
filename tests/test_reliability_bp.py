import itertools
import sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro/hybrid'))
from reliability_bp import chain_marginals,features,r


def test_chain_bp_matches_exhaustive_joint():
    unary=np.random.default_rng(4).normal(size=(1,4,3));edges=np.array([[1.,0.,1.]])
    bp=chain_marginals(unary,edges,2)[0];truth=np.zeros_like(bp)
    states=list(itertools.product(range(3),repeat=4))
    scores=[sum(unary[0,t,z[t]] for t in range(4))+sum(2*edges[0,t]*(z[t]==z[t+1]) for t in range(3)) for z in states]
    for z,p in zip(states,softmax(scores)):
        for t,k in enumerate(z):truth[t,k]+=p
    np.testing.assert_allclose(bp,truth,atol=1e-14)


def test_zero_messages_are_independent_softmax():
    u=np.random.default_rng(8).normal(size=(7,42,4))
    np.testing.assert_allclose(chain_marginals(u,np.ones((7,41)),0),softmax(u,axis=2),atol=1e-14)


def test_gate_features_do_not_read_future_sales():
    p=r.rf.b.Panel(r.rf.b.PANEL,n_series=4)
    pred=np.ones((4,42,4))*100
    x,n=features(p,140,pred);p.y[:,140:]=999999
    x2,n2=features(p,140,pred)
    np.testing.assert_array_equal(x,x2);assert n==n2
