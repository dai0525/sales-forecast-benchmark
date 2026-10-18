import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
import fg_mp
from fg_state import forecast
from audit_experiment import metrics

def test_exact_messages_against_dense_gaussian_conditioning():
    # Independent covariance construction, not a second copy of the recursion.
    rng=np.random.default_rng(1); y=rng.normal(20,2,(1,18)); q=.2
    scale=max(y.std(),.02*abs(y).mean()+1e-6); z=y[0]/scale
    time=np.arange(19); B=np.column_stack([np.cos(2*np.pi*time/52),np.sin(2*np.pi*time/52)])
    cov=1e6+q*np.minimum.outer(time,time)+B@B.T
    cov+=np.eye(19)
    expected=cov[-1,:-1]@np.linalg.solve(cov[:-1,:-1],z)*scale
    expected_var=(cov[-1,-1]-cov[-1,:-1]@np.linalg.solve(cov[:-1,:-1],cov[:-1,-1]))*scale**2
    p,c=forecast(y,h=1,harmonics=1,qr=q,trend_q=0,return_components=True)
    np.testing.assert_allclose(p[0,0],max(0,expected),rtol=1e-7)
    np.testing.assert_allclose(c['variance'][0,0],expected_var,rtol=1e-7)

def test_state_series_independence_and_components():
    y=np.random.default_rng(2).normal(100,10,(3,130))
    p,c=forecast(y,h=60,trend_q=.001,return_components=True)
    np.testing.assert_allclose(p[:1],forecast(y[:1],h=60,trend_q=.001))
    np.testing.assert_allclose(c['raw_mean'],c['level']+c['season'])
    assert np.isfinite(p).all() and (c['variance']>0).all()

def test_old_fg_arbitrary_horizon_and_saved_reproduction():
    y,t,ids=fg_mp.load();m=fg_mp.fit(y,qr=.02,ridge=20,kappa=2)
    p=fg_mp.forecast(m,h=65,damp=.85)
    old=np.load(Path(fg_mp.ROOT)/'data/fg_preds.npz')
    np.testing.assert_allclose(p[:,:13],old['preds'],rtol=1e-12)

def test_metric_rejects_silent_missing_and_zero_targets():
    for t in [np.array([[np.nan]]),np.zeros((1,1))]:
        with pytest.raises(ValueError):metrics(np.ones((1,1)),t)

def test_daily_loader_rejects_weekly_cache(tmp_path,monkeypatch):
    import common
    monkeypatch.setattr(common,'DATA',tmp_path)
    np.savez(tmp_path/'m4daily.npz',ids=['W1'],train=np.ones((200,1)),test=np.ones((13,1)))
    with pytest.raises(ValueError,match='Daily'):common.load_m4_daily()

def test_daily_training_rows_include_last_target_and_custom_lag():
    import lgbm_global as l
    y=np.arange(80,dtype=float).reshape(2,40)
    i,c=l.training_rows(y,100,np.random.default_rng(0))
    assert c.min()==27 and c.max()==38
    assert (c==38).sum()==2

def test_weekly_deep_validation_targets_disjoint():
    from bench_deep import make_windows,H,INPUT
    y=np.arange(200,dtype=float)[None,:]
    _,labels=make_windows(y[:,:-H]); assert labels.max()==186
    assert y[:,-H:].min()==187

def test_weekly_prep_never_writes_daily(tmp_path,monkeypatch):
    import prep_weekly as p
    monkeypatch.setattr(p,'D',tmp_path)
    (tmp_path/'m4daily.npz').write_bytes(b'daily sentinel')
    import shutil
    root=Path(fg_mp.ROOT)
    for name in ['Weekly-train.csv','Weekly-test.csv']:shutil.copy(root/'data'/name,tmp_path/name)
    p.main(); assert (tmp_path/'m4daily.npz').read_bytes()==b'daily sentinel'
