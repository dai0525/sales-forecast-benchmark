import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'repro'))
from fg_state import forecast


def test_information_matches_joint_covariance_update():
    rng=np.random.default_rng(19)
    y=100+np.cumsum(rng.normal(size=(3,80)),axis=1)
    for trend in [0.,.0001]:
        p,c=forecast(y,harmonics=4,trend_q=trend,return_components=True)
        q,d=forecast(y,harmonics=4,trend_q=trend,covariance_mode='information',return_components=True)
        np.testing.assert_allclose(p,q,rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(c['variance'],d['variance'],rtol=1e-8,atol=1e-7)


def test_block_projection_only_affects_cross_component_dependence():
    rng=np.random.default_rng(20)
    y=100+np.cumsum(rng.normal(size=(3,80)),axis=1)
    a=forecast(y,harmonics=0,trend_q=.0001)
    b=forecast(y,harmonics=0,trend_q=.0001,covariance_mode='block')
    np.testing.assert_array_equal(a,b)
    a=forecast(y,harmonics=4,trend_q=.0001)
    b,c=forecast(y,harmonics=4,trend_q=.0001,covariance_mode='block',return_components=True)
    assert np.max(abs(a-b))>1e-3
    assert np.isfinite(b).all() and (c['variance']>0).all()


def test_iterative_component_means_converge_to_same_model():
    from message_iterations import solve
    rng=np.random.default_rng(39)
    y=100+4*np.sin(np.arange(80)[None,:]*2*np.pi/52)+rng.normal(size=(2,80))
    config=dict(harmonics=2,qr=.2,trend_q=.0001,damp=.98)
    exact,ps,diag=solve(y,config,steps=(1,500))
    np.testing.assert_allclose(exact,forecast(y,**config),atol=1e-5,rtol=1e-7)
    for order in ['level_first','season_first']:
        np.testing.assert_allclose(ps[f'{order}_500'],exact,atol=1e-4,rtol=1e-6)
        assert diag[f'{order}_500']['component_mean_max_abs_normalized'] < diag[f'{order}_1']['component_mean_max_abs_normalized']
