import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
from bench_bsts_mcmc import read_result,diagnostic_summary


def test_bsts_export_rejects_wrong_id_and_missing_predictions(tmp_path):
    r={'ok':True,'id':'W1','mean':[1]*13,'median':[1]*13,'lower':[0]*13,'upper':[2]*13}
    p=tmp_path/'x.json';p.write_text(json.dumps(r))
    assert read_result(p,'W1')['mean']==[1]*13
    with pytest.raises(ValueError,match='ID'):read_result(p,'W2')
    r['mean'][3]=None;p.write_text(json.dumps(r))
    with pytest.raises((ValueError,TypeError)):read_result(p,'W1')


def test_missing_mcmc_diagnostic_is_not_silently_passed():
    r={'diagnostics':{'variable':['sigma','level'],'rhat':[1.001,1.002],'ess_bulk':[600,700],'ess_tail':[None,500]}}
    d=diagnostic_summary(r)
    assert not d['passes_all'] and d['failed_variables']==['sigma']


def test_r_sampler_rejects_stale_cache_before_sampling(tmp_path):
    import os,shutil,subprocess
    import numpy as np
    rscript=shutil.which('Rscript')
    if not rscript:pytest.skip('Optional R bsts environment is not active')
    script=Path(__file__).resolve().parents[1]/'repro/bench_bsts_mcmc.R'
    (tmp_path/'repro').mkdir();shutil.copy2(script,tmp_path/'repro'/script.name)
    folder=tmp_path/'audit/results/bsts_mcmc';folder.mkdir(parents=True)
    np.savetxt(folder/'train.csv',np.arange(200)[None,:]+1,delimiter=',')
    (folder/'ids.json').write_text(json.dumps(['W1']))
    out=folder/'trend_harmonic_default_1000';out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'cache_signature':{'train_md5':'stale'}}))
    (out/'001.json').write_text(json.dumps({'ok':True,'id':'W1'}))
    r=subprocess.run([rscript,str(tmp_path/'repro'/script.name),str(tmp_path),'trend_harmonic','1000','1'],capture_output=True,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
    assert r.returncode!=0 and 'fingerprint mismatch' in r.stderr
    assert not (out/'001.rds').exists()
