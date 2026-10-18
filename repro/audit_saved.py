"""Recompute stored metrics against CSV truth; never infer IDs from row count."""
import json,hashlib
from pathlib import Path
import numpy as np
from prep_weekly import load
from audit_experiment import metrics
R=Path(__file__).resolve().parents[1]; out=R/'audit/results';out.mkdir(exist_ok=True,parents=True)
ids,tr=load(R/'data/Weekly-train.csv'); tids,te=load(R/'data/Weekly-test.csv')
z=np.load(R/'data/weekly.npz'); assert ids==tids==list(z['ids'])
assert np.array_equal(tr,z['train'].T,equal_nan=True)
assert np.array_equal(te,z['test'].T)
report={'weekly_csv_matches_npz':True,'weekly':{},'daily':{},'files':{}}
for name in ['fg','bsts']:
 p=R/f'data/{name}_preds.npz';a=np.load(p);assert list(a['ids'])==ids
 report['weekly'][name]=metrics(a['preds'],te)
# Daily original CSV test is sufficient for WAPE audit; no need to trust cached truth.
dids,dte=load(R/'data/Daily-test.csv'); positions={s:i for i,s in enumerate(dids)}
for file in ['bench_preds.npz','lgbm_preds.npz','lgbm_preds_v1.npz','lgbm_preds_v2.npz','preds/AutoARIMA.npz']:
 p=R/'data'/file;a=np.load(p); ai=list(a['ids']);truth=dte[[positions[s] for s in ai]]
 if 'test_truth' in a: assert np.allclose(truth,a['test_truth'],atol=.001)
 rr={}
 for k in a.files:
  if k in ['ids','test_truth'] or k.endswith('_val'):continue
  rr[k]=dict(metrics(np.maximum(a[k],0),truth),n=len(ai))
 report['daily'][file]=rr
 report['files'][file]=hashlib.sha256(p.read_bytes()).hexdigest()
# Compare every Daily baseline on the exact AutoARIMA subset, using ID joins.
a=np.load(R/'data/preds/AutoARIMA.npz');b=np.load(R/'data/bench_preds.npz')
ix=[list(b['ids']).index(s) for s in a['ids']];truth=dte[[positions[s] for s in a['ids']]]
report['daily_common_250']={k:metrics(np.maximum(b[k][ix],0),truth) for k in b.files if k not in ['ids','test_truth']}
report['daily_common_250']['AutoARIMA']=metrics(a['preds'],truth)
(out/'saved_metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
