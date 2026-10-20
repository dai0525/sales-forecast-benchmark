"""Read-only audit of d594f30 ASE artifacts; no fitted weights use test labels.
Run: python repro/hybrid/audit_ase.py
Weekly object archives are loaded only from this trusted repository.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import ase
import run_rossmann_final as rf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'audit/results/ase'
OUT = ROOT / 'audit/results/ase_review'

def per(p, y):
    den = np.abs(y).sum(1)
    assert np.all(den > 0) and np.isfinite(p).all()
    return np.abs(p-y).sum(1)/den

def ci(d):
    rng = np.random.default_rng(20261020)
    vals = d[rng.integers(len(d), size=(4000, len(d)))].mean(1)*100
    return np.quantile(vals, [.025, .975]).tolist()

def main():
    out = {'commit_reviewed': 'd594f30', 'metric': 'mean per-series WAPE; deltas in percentage points, positive favors ASE', 'datasets': {}}
    panel = rf.b.Panel(rf.b.PANEL)
    weekly = ase.load_weekly()
    for tag, summary_file, pred_file, ds in [('weekly', 'weekly_summary.json', 'weekly_preds.npz', weekly), ('rossmann', 'rossmann_final.json', 'rossmann_final_preds.npz', panel)]:
        summary = json.loads((SRC/summary_file).read_text())
        z = np.load(SRC/pred_file, allow_pickle=tag == 'weekly')
        rows, diffs = {}, []
        for o in summary['origins']:
            y = z[f'{o}_truth']
            truth = ds.truth(o) if tag == 'weekly' else ds.y[:,o:o+42]
            assert np.array_equal(y, truth), (tag, o, 'truth mismatch')
            models = z[f'{o}_models'].item() if tag == 'weekly' else {k:z[f'{o}_{k}'] for k in ['l2','l1','state']}
            saved = z[f'{o}_ase']
            rebuilt = ase._combine(models, summary['weights'], ase._stages(y.shape[1]))
            assert np.allclose(saved, rebuilt, rtol=1e-12, atol=1e-9)
            methods = dict(models, ase=saved)
            active = [k for k in models if any(w[k]>0 for w in summary['weights'].values())]
            methods['equal_active_log'] = np.expm1(np.mean([np.log1p(np.clip(models[k],0,None)) for k in active],axis=0))
            # Remove stage variation while preserving horizon-averaged learned weights.
            w = {k:sum(len(ase._stages(y.shape[1])[s])*v[k] for s,v in summary['weights'].items())/y.shape[1] for k in models}
            methods['constant_avg_stage_log'] = ase._combine_arr(models, np.array(list(w.values())))
            methods['equal_active_arithmetic'] = np.mean([models[k] for k in active], axis=0)
            if tag == 'rossmann':
                old = np.load(ROOT/f'audit/results/rossmann_multifactor/LGBM_six_o{o}.npz')
                assert np.array_equal(old['truth'], y)
                assert list(old['ids'].astype(str)) == list(panel.ids)
                methods['previous_six_factor_lgbm'] = old['preds']
            scores = {k:per(v,y) for k,v in methods.items()}
            diffs.append({k:v-scores['ase'] for k,v in scores.items() if k!='ase'})
            rows[str(o)] = {'wape':{k:float(v.mean()) for k,v in scores.items()}, 'truth_exact_match':True, 'ensemble_max_abs_reconstruction_error':float(np.max(np.abs(rebuilt-saved))), 'ci95_delta_pp':{k:ci(v-scores['ase']) for k,v in scores.items() if k!='ase'}}
        out['datasets'][tag] = {'origins':rows, 'mean_wape':{k:float(np.mean([r['wape'][k] for r in rows.values()])) for k in methods}, 'cluster_ci95_delta_pp':{k:ci(np.mean([d[k] for d in diffs],axis=0)) for k in diffs[0]}, 'weights':summary['weights']}
    # Actual feature/predict path with the same data and 1 tree isolates schema failure.
    try:
        ase._gbm_predict(weekly,148,dict(ase.LGB_PARAMS,n_estimators=1,n_jobs=2),'log1p')
        out['weekly_code_smoke'] = {'status':'passed'}
    except Exception as e:
        out['weekly_code_smoke'] = {'status':'failed', 'exception':type(e).__name__, 'message':str(e)}
    a = np.load(SRC/'rossmann_final_attrib.npz')
    meta = json.loads((SRC/'rossmann_final_attrib.json').read_text())
    reconstructed = meta['mean_base_log'] + sum(a[k] for k in a.files)
    mask = panel.open_[:,900:942] == 1
    z = np.load(SRC/'rossmann_final_preds.npz')
    out['attribution'] = {k:{'max_abs_log_error_open_days':float(np.max(np.abs(reconstructed[mask]-np.log1p(z[f'900_{k}'][mask])))), 'mean_abs_log_error_open_days':float(np.mean(np.abs(reconstructed[mask]-np.log1p(z[f'900_{k}'][mask]))))} for k in ['l1','ase']}
    out['sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.iterdir()) if p.is_file()}
    out['data_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/weekly.npz', rf.b.PANEL]}
    OUT.mkdir(exist_ok=True,parents=True)
    # Keep the original d594f30 source-failure record immutable after repairs.
    out['current_ase_source_sha256'] = hashlib.sha256(Path(ase.__file__).read_bytes()).hexdigest()
    (OUT/'archive_audit_current.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
