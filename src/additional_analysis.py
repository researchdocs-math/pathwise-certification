"""Dataset-cluster bootstrap and path-grid sensitivity from archived predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from certification import curves,suffix_outcomes
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis'
def main():
 cfg=json.loads((ROOT/'configs/main.json').read_text());df=pd.read_csv(OUT/'benchmark_records.csv');df=df[df.n==512];rng=np.random.default_rng(81627);rows=[]
 for kind in ['linear','coordinate','both']:
  sub=df if kind=='both' else df[df.path_kind==kind]
  coverage=sub.pivot(index=['dataset','family','seed','query','path_kind'],columns='method',values='selected')
  costs=sub.pivot(index=['dataset','family','seed','query','path_kind'],columns='method',values='cost')
  datasets=sorted(sub.dataset.unique());samples=rng.choice(datasets,size=(10000,len(datasets)),replace=True)
  for comparator in ['persistent_bonferroni','persistent_fixed','persistent_spending','pointwise_fst']:
   delta=coverage.persistent_anytime.astype(float)-coverage[comparator].astype(float)
   rel=(costs.persistent_anytime/costs[comparator]-1).dropna()
   for metric,v in [('coverage_difference',delta),('relative_cost',rel)]:
    groups=v.groupby(level='dataset').agg(['sum','count']).reindex(datasets);positions=np.searchsorted(np.asarray(datasets),samples);sums=groups['sum'].to_numpy();counts=groups['count'].to_numpy();bs=sums[positions].sum(1)/counts[positions].sum(1);low,high=np.quantile(bs,[.025,.975])
    rows.append(dict(path_kind=kind,comparator=comparator,metric=metric,estimate=v.mean(),lower95=low,upper95=high,n_pairs=len(v)))
 pd.DataFrame(rows).to_csv(OUT/'cluster_intervals.csv',index=False,float_format='%.10g')
 rows=[]
 for file in sorted((ROOT/'results/banks').glob('*.npz')):
  a=np.load(file);meta=json.loads(file.with_suffix('.json').read_text());kind=meta.get('path_kind','linear');Q,K=a['cost'].shape
  for points in [16,32,64]:
   idx=np.arange(K//points-1,K,K//points);nom=suffix_outcomes(a['nominal'][:,idx]);ev=suffix_outcomes(a['evaluation'][:,:,idx]).mean(0)
   for tau in [.9,.95,.99]:
    for q in range(Q):
     eligible=np.flatnonzero(nom[q]);lo=int(eligible[0]) if len(eligible) else points
     for r in curves(a['cert'][:,q,:][:,idx],cfg['looks'],cfg['delta'],tau,min_index=lo):
      if r['n']!=512 or r['method'] not in ['persistent_anytime','persistent_bonferroni']:continue
      k=r['index'];rows.append(dict(path_kind=kind,dataset=meta['dataset'],family=meta['family'],seed=meta['seed'],query=q,points=points,tau=tau,method=r['method'],selected=k>=0,cost=float(a['cost'][q,idx[k]]) if k>=0 else np.nan,persistence=float(ev[q,k]) if k>=0 else np.nan))
 pd.DataFrame(rows).to_csv(OUT/'sensitivity_records.csv.gz',index=False,compression={'method':'gzip','mtime':0},float_format='%.10g')
 pd.DataFrame(rows).groupby(['path_kind','points','tau','method']).agg(certified=('selected','mean'),mean_cost=('cost','mean'),mean_persistent=('persistence','mean')).to_csv(OUT/'sensitivity_summary.csv',float_format='%.10g')
 print('Additional analyses complete.')
if __name__=='__main__':main()
