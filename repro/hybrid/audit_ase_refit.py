"""Independent Rossmann validation + last-origin L1/L2 refit, preserving originals."""
import json
import platform
from pathlib import Path
import numpy as np
import lightgbm
import run_rossmann_final as r
from fg_state import forecast

OUT = r.ROOT/'audit/results/ase_review'
p = r.b.Panel(r.b.PANEL)
saved = np.load(r.OUT/'rossmann_final_preds.npz')
meta = json.loads((r.OUT/'rossmann_final.json').read_text())
r.PARAMS['n_jobs'] = 2
report = {'python':platform.python_version(), 'lightgbm':lightgbm.__version__, 'params':r.PARAMS, 'validation':{}, 'origin900':{}}
val = {}
for o in [774,900]:
    for name,obj in [('l2','l2'),('l1','regression_l1')]:
        pr,_,_,_ = r.fit(p,o,obj)
        metric = r.wape(pr,p.y[:,o:o+42])
        if o==774:
            val[name]=pr
            report['validation'][name] = metric
        else:
            report['origin900'][name] = {'wape':metric,'max_abs_difference_from_archive':float(np.max(np.abs(pr-saved[f'{o}_{name}'])))}
        print(o,name,metric,flush=True)
val['state'] = forecast(p.y[:,:774],h=42,period=7,harmonics=3,qr=.02,trend_q=0,damp=.98)
report['validation']['state'] = r.wape(val['state'],p.y[:,774:816])
report['weights'] = {}
for stage,idx in r.stages(42).items():
    w,sc,keep = r.fit_weights({k:v[:,idx] for k,v in val.items()},p.y[:,774:816][:,idx])
    report['weights'][stage] = {'weights':w,'scores':sc,'keep':keep,'matches_archive':w==meta['weights'][stage]}
w,sc,keep = r.fit_weights(val,p.y[:,774:816])
report['constant_validation_fitted_weights'] = w
report['constant_validation_fitted_test_wape'] = {str(o):r.wape(r.combine({k:saved[f'{o}_{k}'] for k in val},w),p.y[:,o:o+42]) for o in [816,858,900]}
OUT.mkdir(parents=True,exist_ok=True)
np.savez_compressed(OUT/'rossmann_validation_preds.npz',**val,truth=p.y[:,774:816],ids=np.array(p.ids))
(OUT/'rossmann_refit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
