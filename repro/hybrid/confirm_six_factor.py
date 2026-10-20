"""Additional existing six-factor LightGBM comparator; no new model selection.
Specified before any confirmation results. Mirrors bench_multifactor.py exactly.
"""
import json
import time
import numpy as np
import research_iteration as r
from rossmann_factors import FactorPanel,NAMES

p=FactorPanel(r.OUT/'confirmation/panel.npz')
b=r.rf.b
original_matrix=b.build_matrix
params=json.loads((b.OUTDIR/'metrics.json').read_text())['selection']['chosen']['LGBM-direct']['config']['params']
b.N_JOBS=2

def augmented(p,o,h,full,hist=None):
    cols=original_matrix(p,o,h,False,hist)
    target=o+h-1
    for j,name in enumerate(NAMES):
        value=p.extra[:,target,j].copy()
        if j==5:value*=(p.promo2_start<=p.days[o-1])
        if j==6:value=(p.competition_start<=p.days[o-1]).astype(float)
        cols[name]=value
    cols['Open']=p.open_[:,target].astype(float)
    return cols

b.build_matrix=augmented
try:
    for o in [816,858,900]:
        path=r.OUT/'confirmation'/f'six_factor_o{o}.npz'
        if path.exists():continue
        t=time.perf_counter();pred=np.maximum(b.lgbm_forecast(p,o,True,params),0)
        np.savez_compressed(path,preds=pred,truth=p.y[:,o:o+42],ids=np.array(p.ids),dates=p.days[o:o+42])
        m=r.metric(pred,p.y[:,o:o+42]);m['elapsed_s']=time.perf_counter()-t
        r.dump(path.with_suffix('.json'),m);print(o,m,flush=True)
finally:b.build_matrix=original_matrix
