"""Evaluate the same retraining seeds on monotone-cost, coordinatewise paths."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from models import bank
ROOT=Path(__file__).resolve().parents[1]
def coordinate_path(x,z,K):
 delta=z-x;order=np.argsort(-np.abs(delta),kind='stable');length=np.abs(delta[order]);end=np.cumsum(length);total=end[-1]
 out=[]
 for amount in np.linspace(total/K,total,K):
  point=x.copy();before=0.
  for j,ell in zip(order,length):
   if amount>=before+ell:point[j]=z[j]
   elif amount>before and ell>0:point[j]=x[j]+(amount-before)/ell*delta[j]
   before+=ell
  out.append(point)
 out[-1]=z.copy()
 return np.stack(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',default=['logistic','boosting','mlp']);p.add_argument('--device',default='mps');args=p.parse_args();cfg=json.loads((ROOT/'configs/main.json').read_text())
 for f in sorted((ROOT/'results/banks').glob('*.npz')):
  if '_coordinate' in f.stem:continue
  meta=json.loads(f.with_suffix('.json').read_text())
  if meta['family'] not in args.families:continue
  dest=f.with_name(f.stem+'_coordinate.npz')
  if dest.exists():continue
  start=time.perf_counter();a=dict(np.load(f));dat=np.load(ROOT/'data'/(meta['dataset']+'.npz'));tr=a['train_indices'];Z=StandardScaler().fit(dat['X'][tr]).transform(dat['X']).astype('float32');y=dat['y'];Q,K=a['cost'].shape
  paths=np.stack([coordinate_path(Z[q],Z[z],K) for q,z in zip(a['query_indices'],a['anchor_indices'])]);P=paths.reshape(-1,Z.shape[1]);seeds=a['seeds'];family=meta['family']
  nominal=bank(family,Z[tr],y[tr],P,seeds[:1],cfg,args.device)[0].reshape(Q,K)
  lo=1+cfg['search_models'];hi=lo+cfg['certification_models']
  t=time.perf_counter();cert=bank(family,Z[tr],y[tr],P,seeds[lo:hi],cfg,args.device).reshape(cfg['certification_models'],Q,K);tc=time.perf_counter()-t
  t=time.perf_counter();ev=bank(family,Z[tr],y[tr],P,seeds[hi:],cfg,args.device).reshape(cfg['evaluation_models'],Q,K);tv=time.perf_counter()-t
  a.update(paths=paths,cert=cert,evaluation=ev,nominal=nominal,cost=np.mean(np.abs(paths-Z[a['query_indices']][:,None,:]),axis=2));np.savez_compressed(dest,**a)
  meta.update(path_kind='coordinate',cert_seconds=tc,eval_seconds=tv,search_seconds=0.,total_seconds=time.perf_counter()-start)
  dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
