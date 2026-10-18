"""Nested historical holdouts: selection precedes each evaluation origin."""
import json,time
import numpy as np
from joblib import Parallel,delayed
import fg_mp, bsts52
from fg_state import forecast
from audit_experiment import OUT,metrics

def main():
 y,_,ids=fg_mp.load(); configs=json.loads((OUT/'protocol.json').read_text())['grid']
 results={}; preds={}
 for origin in [161,174,187]:
  tune=[origin-39,origin-26,origin-13]
  scores=[]
  for c in configs:
   scores.append(np.mean([metrics(forecast(y[:,:o],**c),y[:,o:o+13])['wape_mean'] for o in tune]))
  best=configs[int(np.argmin(scores))]
  p=forecast(y[:,:origin],**best)
  t=time.perf_counter()
  raw=Parallel(n_jobs=4)(delayed(bsts52.fit_one)(v[:origin]) for v in y)
  ucm=np.array([r if r is not None else y[i,origin-52:origin-39] for i,r in enumerate(raw)])
  true=y[:,origin:origin+13]
  results[str(origin)]={'tuning_origins':tune,'selected':best,'State-Fourier':metrics(p,true),'UCM52':metrics(ucm,true),'ucm_s':time.perf_counter()-t,'ucm_failed':sum(r is None for r in raw)}
  preds[f'state_{origin}']=p;preds[f'ucm_{origin}']=ucm
  (OUT/'rolling.json').write_text(json.dumps(results,indent=2))
  np.savez_compressed(OUT/'rolling_predictions.npz',ids=ids,**preds)
  print(origin,results[str(origin)],flush=True)
if __name__=='__main__':main()
