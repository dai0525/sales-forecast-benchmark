import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
import retail_ase as r


def checkpoint(tmp_path):
    p=dict(y=np.ones((2,60)),static=pd.DataFrame({'id':[1,2]}),dates=pd.date_range('2024-01-01',periods=60),cov={},H=7,period=7)
    pred=np.ones((2,7))*1.1
    np.savez_compressed(tmp_path/'o40_l2.npz',pred=pred,truth=p['y'][:,40:47])
    meta=dict(seconds=12.5,params=dict(r.ase.LGB_PARAMS,n_jobs=2),features=list(r.features(p,40,'history')),training_origins=[28])
    (tmp_path/'o40_l2.json').write_text(json.dumps(meta))
    return p,pred


def test_resume_preserves_training_time_without_refit(tmp_path,monkeypatch):
    p,pred=checkpoint(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Should reuse validated component')
    monkeypatch.setattr(r,'fit_tree',forbidden)
    actual,_,secs=r.tree_component(p,40,'history','l2',tmp_path,True)
    np.testing.assert_array_equal(actual,pred)
    assert secs==12.5


def test_resume_rejects_different_target(tmp_path):
    p,_=checkpoint(tmp_path);p['y'][0,40]=2
    with pytest.raises(AssertionError):r.tree_component(p,40,'history','l2',tmp_path,True)
