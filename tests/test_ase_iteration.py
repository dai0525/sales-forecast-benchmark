import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repro/hybrid'))
import ase


def panel(season=52):
    y = np.random.default_rng(7).uniform(1,100,(4,220))
    return ase.Panel('test', y,13 if season==52 else 42,season,[161],148,['a','b','c','d'])


def test_weekly_feature_schema_and_actual_prediction_future_invariance():
    ds=panel()
    a=ase._row(ds,105,1,ds.y[:,:105]);b=ase._row(ds,148,1,ds.y[:,:148])
    assert list(a)==list(b)
    params=dict(ase.LGB_PARAMS,n_estimators=3,n_jobs=1)
    first=ase._gbm_predict(ds,148,dict(params),'log1p')
    ds.y[:,148:]=1e9
    second=ase._gbm_predict(ds,148,dict(params),'log1p')
    np.testing.assert_array_equal(first,second)


def test_same_weekday_uses_four_observed_days_and_stable_holiday_schema():
    ds=panel(7);ds.y=np.tile(np.arange(220,dtype=float),(4,1))
    ds.exog={'state':np.zeros_like(ds.y)};ds.exog['state'][:,145]=3
    for h in [1,7,8,14,42]:
        r=ase._row(ds,148,h,ds.y[:,:148])
        target=148+h-1
        idx=[i for i in range(148) if i%7==target%7][-4:]
        np.testing.assert_array_equal(r['same_dow_4w'],np.full(4,np.mean(idx)))
    assert list(ase._row(ds,105,1,ds.y[:,:105]))==list(ase._row(ds,148,1,ds.y[:,:148]))


def test_open_only_model_future_sales_invariance_and_closed_gate(monkeypatch):
    import research_iteration as r
    monkeypatch.setitem(r.rf.PARAMS,'n_estimators',3)
    p=r.rf.b.Panel(r.rf.b.PANEL,n_series=4)
    first=r.fit_ross(p,140,'open_l1')
    p.y[:,140:]=999999
    second=r.fit_ross(p,140,'open_l1')
    np.testing.assert_array_equal(first,second)
    assert np.all(first[p.open_[:,140:182]==0]==0)
    assert np.isfinite(first).all()


def test_ratio_probe_does_not_read_future_sales():
    from probe_rossmann import base_same_dow
    y=np.tile(np.arange(1,201,dtype=float),(2,1));opened=np.ones_like(y)
    for h in [1,8,21,42]:
        first=base_same_dow(y,opened,100,h)
        changed=y.copy();changed[:,100:]=1e12
        np.testing.assert_array_equal(first,base_same_dow(changed,opened,100,h))
