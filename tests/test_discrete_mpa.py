import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
from discrete_mpa import mpa,exact_marginals,prepare,predict


def test_tree_sum_product_equals_enumeration():
    rng=np.random.default_rng(42)
    factors=[rng.normal(size=(3,4)),rng.normal(size=(3,4))]
    neighbors=[[0,1],[1,2]];prior=rng.normal(size=(3,3,2))
    exact=exact_marginals(factors,neighbors,prior)
    beliefs,_=mpa(factors,neighbors,prior,steps=(5,),damping=1.)
    np.testing.assert_allclose(beliefs[5],exact,atol=1e-12)


def test_one_resource_codebook_demapping():
    # Two BPSK users on one Gaussian resource: true sum-product marginalization.
    symbols=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    y=np.array([.3,1.5])
    factor=-.5*(y[:,None]-symbols@np.array([1.,.7]))**2/.4
    prior=np.zeros((2,2,2))
    exact=exact_marginals([factor],[[0,1]],prior)
    b,_=mpa([factor],[[0,1]],prior,steps=(1,),damping=1)
    np.testing.assert_allclose(b[1],exact,atol=1e-12)


def test_merging_sales_factors_preserves_joint_distribution():
    rng=np.random.default_rng(2);y=rng.normal(100,10,(2,70))
    a=prepare(y,group=13);b=prepare(y,group=200)
    ma=exact_marginals(a['factors'],a['neighbors'],a['logprior'])
    mb,_=mpa(b['factors'],b['neighbors'],b['logprior'],steps=(1,),damping=1)
    np.testing.assert_allclose(predict(a,ma),predict(b,mb[1]),atol=1e-9)
    np.testing.assert_allclose(ma.sum(-1),1,atol=1e-12)
