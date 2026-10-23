"""Align static ASE validation objective with business score, reusing fixed components."""
import json,sys,time
import numpy as np
import pandas as pd
from retail_expansion import ROOT,metrics
from audit_freshretail import digest
sys.path.insert(0,str(ROOT/'repro/hybrid'))
import ase
OUT=ROOT/'audit/results/loss_optimization'


def blend(p,weights,space):
    if space=='log':return ase._combine_arr(p,np.array([weights[k] for k in p]))
    return sum(weights[k]*p[k] for k in p)


def fit_weights(p,y,metric,space):
    scores={k:metrics(v,y)[metric] for k,v in p.items()};best=min(scores.values())
    keep=[k for k in p if scores[k]<=1.5*best]
    best_score=np.inf;best_w=None
    for w in ase._simplex(len(keep),.05):
        pp={k:p[k] for k in keep};ww=dict(zip(keep,w));score=metrics(blend(pp,ww,space),y)[metric]
        if score<best_score:best_score=score;best_w=np.array(w)
    shrunk=(2/3)*best_w+(1/3)/len(keep)
    weights={k:0. for k in p};weights.update({k:float(w) for k,w in zip(keep,shrunk)})
    return dict(weights=weights,component_scores=scores,post_shrink_score=metrics(blend(p,weights,space),y)[metric])


def run():
    for track in ['freshretail_history','freshretail_plan','vn1_history']:
        source=ROOT/'audit/results/retail_ase'/track;out=OUT/('aligned_'+track)
        if out.exists():raise RuntimeError('Refusing overwrite')
        out.mkdir(parents=True)
        original=json.loads((source/'selection.json').read_text());final=original['final_origin'];origins=[76,83] if track.startswith('freshretail') else [157,170]
        metric='wape' if track.startswith('freshretail') else 'vn1_score';names=['l2','l1','state']
        p={k:np.concatenate([np.load(source/f'o{o}_{k}.npz')['pred'] for o in origins]) for k in names}
        y=np.concatenate([np.load(source/f'o{o}_l2.npz')['truth'] for o in origins])
        t=time.perf_counter();fits={space:fit_weights(p,y,metric,space) for space in ['log','arithmetic']}
        selected=min(fits,key=lambda s:fits[s]['post_shrink_score'])
        sel={'fits':fits,'selected_space':selected,'metric':metric,'final_origin':final,'seconds':time.perf_counter()-t,'protocol_sha256':digest(OUT/'ase_alignment_protocol.json')}
        (out/'selection.json').write_text(json.dumps(sel,indent=2)+'\n')
        fp={k:np.load(source/f'o{final}_{k}.npz')['pred'] for k in names};truth=np.load(source/f'o{final}_l2.npz')['truth'];rows=[]
        for space in fits:
            pred=blend(fp,fits[space]['weights'],space);np.savez_compressed(out/f'{space}.npz',pred=pred,truth=truth)
            rows.append(dict(model=space,**metrics(pred,truth)))
        old=np.load(source/f'o{final}_ase_static.npz');rows.append(dict(model='old_static',**metrics(old['pred'],old['truth'])))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        print(track,selected,rows,flush=True)


if __name__=='__main__':run()
