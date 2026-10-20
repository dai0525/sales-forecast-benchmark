"""Summarize frozen experiment, retaining negative findings and cohort limits."""
import json
from pathlib import Path
import numpy as np
import research_iteration as r

OUT=r.OUT

def per(p,y):return np.abs(p-y).sum(1)/np.abs(y).sum(1)

def ci(d):
    rng=np.random.default_rng(20261020)
    v=d[rng.integers(len(d),size=(4000,len(d)))].mean(1)*100
    return {'ci95_pp':np.quantile(v,[.025,.975]).tolist(),'ci97_5_pp':np.quantile(v,[.0125,.9875]).tolist()}

def summarize(tag):
    origins=[161,174,187,200] if tag=='weekly' else [816,858,900]
    scores={};rows={};aggregates={};ids=None
    for o in origins:
        z=np.load(OUT/tag/f'predictions_o{o}.npz');y=z['truth']
        if ids is None:ids=z['ids']
        assert np.array_equal(ids,z['ids'])
        preds={k[5:]:z[k] for k in z.files if k.startswith('pred_')}
        if tag=='weekly':
            old=np.load(r.ROOT/'audit/results/ase/weekly_preds.npz',allow_pickle=True)
            assert np.array_equal(y,old[f'{o}_truth'])
            preds['historical_archived_ase']=old[f'{o}_ase']
        elif tag=='rossmann':
            old=np.load(r.ROOT/f'audit/results/rossmann_multifactor/LGBM_six_o{o}.npz')
            assert np.array_equal(y,old['truth']) and np.array_equal(ids,old['ids'])
            preds['existing_six_factor_lgbm']=old['preds']
        else:
            old=np.load(OUT/tag/f'six_factor_o{o}.npz')
            assert np.array_equal(y,old['truth']) and np.array_equal(ids,old['ids'])
            preds['existing_six_factor_lgbm']=old['preds']
        scores[o]={k:per(v,y) for k,v in preds.items()}
        rows[o]={k:float(v.mean()) for k,v in scores[o].items()}
        aggregates[o]={k:r.metric(v,y)['wape_aggregate'] for k,v in preds.items()}
    primary=r.PRIMARY
    mean={k:float(np.mean([v[k] for v in rows.values()])) for k in preds}
    comparisons={}
    for k in preds:
        if k==primary:continue
        diff=np.mean([scores[o][k]-scores[o][primary] for o in origins],axis=0)
        comparisons[k]={'delta_pp':float(diff.mean()*100),'relative_wape_reduction_pct':float((mean[k]-mean[primary])/mean[k]*100),'series_mean_win_rate':float(np.mean(diff>0)),'window_wins':sum(rows[o][primary]<rows[o][k] for o in origins),**ci(diff)}
    times={}
    for path in (OUT/tag).glob('components_*.json'):
        v=json.loads(path.read_text());times.setdefault(v['model'],[]).append(v)
    return {'n_series':len(ids),'origins':rows,'mean_wape':mean,'aggregate_wape_by_origin':aggregates,'primary':primary,'comparisons_positive_favors_primary':comparisons,'timing_records':times}

def main():
    data={tag:summarize(tag) for tag in ['weekly','rossmann','confirmation']}
    r.dump(OUT/'summary.json',data)
    for tag,v in data.items():
        print(tag,{k:round(x*100,5) for k,x in v['mean_wape'].items()})
        for k in ['original_stage','validation_best_single','existing_six_factor_lgbm']:
            if k in v['comparisons_positive_favors_primary']:print(k,v['comparisons_positive_favors_primary'][k])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['validation_best_single','original_stage',r.PRIMARY,'pooled_stage_log','equal_all_arithmetic','equal_original_arithmetic','equal_open_arithmetic','existing_six_factor_lgbm']
    labels=['Validation-selected single','Original ASE','Proposed pooled blend','Pooled stage blend','Equal blend of all 4','50/50 original L1/L2','50/50 open-only L1/L2','Existing six-factor LGBM']
    vals=[data['confirmation']['mean_wape'][k]*100 for k in names]
    fig,ax=plt.subplots(figsize=(9,5.2));bars=ax.barh(labels,vals,color=['#94a3b8','#64748b','#0d9488']+['#94a3b8']*5)
    ax.bar_label(bars,fmt='%.4f%%',padding=4);ax.set_xlim(0,max(vals)*1.15);ax.invert_yaxis()
    ax.set_xlabel('Mean per-store WAPE (%) — lower is better')
    ax.set_title('534 stores outside the development cohort, 3 fixed windows\nWeights frozen on the original 400 stores; same calendar period')
    fig.tight_layout();fig.savefig(OUT/'confirmation_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
