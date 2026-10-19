import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
from fg_state_exog import forecast


def test_static_state_filter_matches_joint_gaussian_regression():
    rng=np.random.default_rng(3);y=rng.normal(10,2,(2,20));x=rng.normal(size=(2,20,2));xf=rng.normal(size=(2,4,2))
    p,d=forecast(y,x,xf,qr=0,harmonics=0,beta_var=.3)
    scale=np.maximum(y.std(1),.02*np.abs(y).mean(1)+1e-6)
    for i in range(2):
        X=np.column_stack([np.ones(20),x[i]])
        mean=np.linalg.solve(np.diag([1e-6,1/.3,1/.3])+X.T@X,X.T@(y[i]/scale[i]))
        want=np.column_stack([np.ones(4),xf[i]])@mean*scale[i]
        np.testing.assert_allclose(d['raw_mean'][i],want,rtol=1e-8,atol=1e-7)
    np.testing.assert_allclose(p,d['components'].sum(2)+d['clipping_adjustment'])
    assert d['min_covariance_eigenvalue']>=-1e-8


def test_closed_day_gate_and_future_features_do_not_change_fit():
    y=np.arange(1,61,dtype=float)[None,:];x=np.ones((1,60,1));xf=np.ones((1,3,1))
    p,a=forecast(y,x,xf,open_future=np.array([[0,1,0]]))
    _,b=forecast(y,x,xf*100,open_future=np.ones((1,3)))
    np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
    np.testing.assert_array_equal(p[:,[0,2]],0)
    np.testing.assert_allclose(a['components'].sum(2),a['raw_mean'])


def test_no_factors_matches_original_state_fourier():
    from fg_state import forecast as original
    rng=np.random.default_rng(7);y=rng.uniform(10,100,(4,90))
    p,_=forecast(y,np.zeros((4,90,0)),np.zeros((4,13,0)),qr=.02)
    want=original(y,h=13,harmonics=3,qr=.02,trend_q=0,period=7)
    np.testing.assert_allclose(p,want,atol=2e-6)


def test_future_enrollment_and_competitor_changes_are_not_anticipated():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro/alt'))
    from rossmann_factors import FactorPanel
    # Exercise availability logic without requiring downloaded raw CSVs.
    p=object.__new__(FactorPanel);p.n=2;p.days=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-11'))
    p.extra=np.ones((2,10,7));p.promo2_start=np.array(['2020-01-08','2019-01-01'],dtype='datetime64[D]')
    p.competition_start=p.promo2_start.copy()
    train,future=p.at_origin(5,5)
    np.testing.assert_array_equal(future[0,:,5:7],0)
    np.testing.assert_array_equal(future[1,:,5:7],1)
    np.testing.assert_array_equal(train,p.extra[:,:5])
