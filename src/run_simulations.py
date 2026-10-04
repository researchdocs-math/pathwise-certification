"""Exact-law Monte Carlo: threshold classifiers with model-specific failure notches."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from certification import e_threshold,cp_threshold,beta_threshold,suffix_outcomes
ROOT=Path(__file__).resolve().parents[1]
LOOKS=[32,64,96,128,192,256,384,512]
METHODS=['persistent_repeated_cp','persistent_anytime','persistent_bonferroni','persistent_spending','persistent_fixed','pointwise_fst','pointwise_beta','persistent_naive']
def trial(args):
 K,eta,R,seed=args;rng=np.random.default_rng(seed);ns=512;ks=np.arange(1,K+1)
 if eta<0:
  # All but the endpoint share an exact boundary-null probability.
  p=np.r_[np.repeat(.949,K-1),1.];g=p.copy()
 else:
  lo=max(1,K//8);hi=max(lo+1,3*K//4);notchlo=3*K//4
  ft=np.clip((ks-lo+1)/(hi-lo+1),0,1)
  npnt=np.where((ks>notchlo)&(ks<K),eta/(K-1-notchlo),0.)
  p=ft*(1-npnt);g=ft*(1-eta*np.clip((K-np.maximum(ks,notchlo+1))/(K-1-notchlo),0,1))
 records=[]
 for block in range(0,R,100):
  B=min(100,R-block)
  if eta<0:
   z=rng.random((B,ns))<.949;pred=np.repeat(z[:,:,None],K,axis=2);pred[:,:,-1]=True
  else:
   t=rng.integers(lo,hi+1,size=(B,ns));pred=ks[None,None,:]>=t[:,:,None]
   notch=rng.integers(notchlo+1,K,size=(B,ns));has=rng.random((B,ns))<eta
   bb,nn=np.nonzero(has);pred[bb,nn,notch[bb,nn]-1]=False
  raw=np.cumsum(pred,axis=1,dtype=np.int16);suf=np.cumsum(suffix_outcomes(pred),axis=1,dtype=np.int16)
  ever={m:np.zeros((B,K),bool) for m in ['persistent_repeated_cp','persistent_anytime','persistent_bonferroni','persistent_spending','pointwise_fst']}
  bad={m:np.zeros(B,bool) for m in METHODS};first={m:np.full(B,ns+1) for m in METHODS}
  bad_point={m:np.zeros(B,bool) for m in METHODS}
  for ell,n in enumerate(LOOKS,1):
   ever['persistent_repeated_cp']|=suf[:,n-1]>=cp_threshold(n)
   ever['persistent_anytime']|=suf[:,n-1]>=e_threshold(n)
   ever['persistent_bonferroni']|=suf[:,n-1]>=e_threshold(n,.05/K)
   ever['persistent_spending']|=suf[:,n-1]>=cp_threshold(n,.05/(ell*(ell+1)))
   ever['pointwise_fst']|=raw[:,n-1]>=e_threshold(n)
   masks={**ever,'pointwise_fst':suffix_outcomes(ever['pointwise_fst']),'persistent_fixed':suf[:,n-1]>=cp_threshold(n),'pointwise_beta':raw[:,n-1]>=beta_threshold(n),'persistent_naive':suf[:,n-1]>=np.ceil(.95*n)}
   for m,mask in masks.items():
    # Fixed procedures are inspected ONLY at their prespecified final n.
    if m in ['persistent_fixed','pointwise_beta'] and n!=ns:continue
    exists=mask.any(1);idx=np.argmax(mask,axis=1)
    false=(mask[:,g<.95]).any(1);falsep=(mask[:,p<.95]).any(1)
    bad[m]|=false;bad_point[m]|=falsep
    first[m]=np.where(exists&(first[m]>ns),n,first[m])
    if n==ns:
     for r in range(B):records.append({'K':K,'eta':eta,'replicate':block+r,'method':m,'false_persistent':int(bad[m][r]),'false_pointwise':int(bad_point[m][r]),'first_n':int(first[m][r]),'index':int(idx[r]+1) if exists[r] else -1,'g_selected':float(g[idx[r]]) if exists[r] else np.nan,'p_selected':float(p[idx[r]]) if exists[r] else np.nan})
 return records

def main():
 p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=5000);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 jobs=[(K,e,a.repetitions,20261003+i*101) for i,(K,e) in enumerate(( (k,e) for k in [16,64,256] for e in [-1.,0.,.1,.3] ))]
 records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  for job,rr in zip(jobs,ex.map(trial,jobs)):
   records.extend(rr);print(job[:2],len(rr),flush=True)
 pd.DataFrame(records).to_csv(ROOT/'results/simulation_trials.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 (ROOT/'results/simulation_config.json').write_text(json.dumps({'repetitions':a.repetitions,'jobs':jobs,'looks':LOOKS},indent=2))
if __name__=='__main__':main()
