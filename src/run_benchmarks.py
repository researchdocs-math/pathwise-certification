"""Train search, certification and independent evaluation model banks."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,time,platform,sys,importlib.metadata as md
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import bank
ROOT=Path(__file__).resolve().parents[1]

def run(dataset,family,seed,cfg,device):
    name=f'{dataset}_{family}_{seed}';dest=ROOT/'results'/'banks';dest.mkdir(exist_ok=True)
    if (dest/(name+'.npz')).exists():print('SKIP',name,flush=True);return
    t0=time.perf_counter();a=np.load(ROOT/'data'/(dataset+'.npz'));X=a['X'];y=a['y'];rng=np.random.default_rng(seed)
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    if len(tr)>cfg['training_cap']:tr,_=train_test_split(tr,train_size=cfg['training_cap'],stratify=y[tr],random_state=seed+1)
    scaler=StandardScaler().fit(X[tr]);Z=scaler.transform(X).astype('float32');Xtr=Z[tr];ytr=y[tr]
    nsearch=cfg['search_models'];ncert=cfg['certification_models'];neval=cfg['evaluation_models'];K=cfg['path_points']
    seeds=np.random.SeedSequence([seed,{'logistic':1,'boosting':2,'mlp':3}[family]]).generate_state(nsearch+ncert+neval+1)
    # Base model and the search bank see the same fixed training table, with independent retraining seeds.
    before=time.perf_counter();base=bank(family,Xtr,ytr,Z,seeds[:1],cfg,device)[0]
    neg=te[base[te]==0];queries=rng.choice(neg,size=min(len(neg),cfg['queries']),replace=False)
    pos=tr[(y[tr]==1)&(base[tr]==1)]
    if len(pos)==0:raise RuntimeError('No positive anchor')
    anchors=rng.choice(pos,size=min(256,len(pos)),replace=False)
    searchpred=bank(family,Xtr,ytr,Z[anchors],seeds[1:1+nsearch],cfg,device)
    robust=searchpred.mean(0);eligible=np.flatnonzero(robust>=.95)
    if len(eligible)==0:eligible=np.flatnonzero(robust==robust.max())
    paths=[];costs=[];chosen=[]
    fractions=np.linspace(1/K,1,K,dtype='float32')
    for qi in queries:
        distances=np.mean(np.abs(Z[anchors[eligible]]-Z[qi]),axis=1)
        ai=int(anchors[eligible[np.argmin(distances)]])
        path=Z[qi]+fractions[:,None]*(Z[ai]-Z[qi]);path[-1]=Z[ai];paths.append(path);costs.append(np.mean(np.abs(path-Z[qi]),axis=1));chosen.append(ai)
    paths=np.stack(paths);P=np.concatenate([paths.reshape(-1,X.shape[1]),Z[te]])
    base_path=bank(family,Xtr,ytr,paths.reshape(-1,X.shape[1]),seeds[:1],cfg,device)[0].reshape(len(queries),K)
    search_seconds=time.perf_counter()-before;before=time.perf_counter()
    cert=bank(family,Xtr,ytr,P,seeds[1+nsearch:1+nsearch+ncert],cfg,device)
    cert_seconds=time.perf_counter()-before;before=time.perf_counter()
    ev=bank(family,Xtr,ytr,P,seeds[1+nsearch+ncert:],cfg,device)
    eval_seconds=time.perf_counter()-before;Q=len(queries)
    np.savez_compressed(dest/(name+'.npz'),nominal=base_path,cert=cert[:,:Q*K].reshape(ncert,Q,K),evaluation=ev[:,:Q*K].reshape(neval,Q,K),cost=np.array(costs),paths=paths,query_indices=queries,anchor_indices=chosen,train_indices=tr,test_indices=te,seeds=seeds,base_accuracy=np.mean(base[te]==y[te]),cert_accuracy=np.mean(cert[:,Q*K:]==y[te],axis=1),eval_accuracy=np.mean(ev[:,Q*K:]==y[te],axis=1))
    rec={'dataset':dataset,'family':family,'seed':seed,'train_n':len(tr),'test_n':len(te),'queries':Q,'search_seconds':search_seconds,'cert_seconds':cert_seconds,'eval_seconds':eval_seconds,'total_seconds':time.perf_counter()-t0,'device':device if family=='mlp' else 'cpu'}
    (dest/(name+'.json')).write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--families',nargs='+');p.add_argument('--datasets',nargs='+');p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--device',default='mps');args=p.parse_args()
    cfg=json.loads((ROOT/'configs/main.json').read_text())
    manifest={'python':sys.version,'platform':platform.platform(),'packages':{p:md.version(p) for p in ['numpy','scipy','scikit-learn','pandas','torch','threadpoolctl']},'config':cfg}
    (ROOT/'results/environment.json').write_text(json.dumps(manifest,indent=2))
    for dataset in args.datasets or cfg['datasets']:
      for seed in args.seeds or cfg['seeds']:
       for family in args.families or cfg['families']:run(dataset,family,seed,cfg,args.device)
if __name__=='__main__':main()
