"""Recompute every numerical summary from the archived independent model banks."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
from scipy.stats import beta
from functools import lru_cache
from certification import curves,suffix_outcomes,cp_threshold,e_threshold
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(None)
def audit_limits(ne):
 successes=np.arange(ne+1);lower=beta.ppf(.005,successes,ne-successes+1);upper=beta.ppf(.995,successes+1,ne-successes);lower[0]=0.;upper[-1]=1.
 return lower,upper

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/analysis');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'configs/main.json').read_text());rows=[];settings=[];profiles=[]
 for file in sorted((ROOT/'results/banks').glob('*.npz')):
  a=np.load(file);meta=json.loads(file.with_suffix('.json').read_text());meta.setdefault('path_kind','linear');Q,K=a['cost'].shape
  assert 'nominal' in a,file
  ev=a['evaluation'].astype(bool);es=suffix_outcomes(ev);mean_p=ev.mean(0);mean_g=es.mean(0)
  meta['base_accuracy']=float(a['base_accuracy']);meta['mean_retrained_accuracy']=float(a['eval_accuracy'].mean());meta['negative_queries']=Q
  # Last-failure scan: number of queried points until the first negative, inclusive.
  cert=a['cert'].astype(bool);suf=suffix_outcomes(cert);last=K-suf.sum(2)
  meta['lazy_query_fraction']=float(np.minimum(K,K-last+1).mean()/K)
  meta['mean_max_backtracking']=float((mean_p-mean_g).max(1).mean())
  meta['max_backtracking']=float((mean_p-mean_g).max())
  settings.append(meta)
  for q in range(Q):
   nominal=suffix_outcomes(a['nominal'][q]);eligible=np.flatnonzero(nominal);minnom=int(eligible[0]) if len(eligible) else K
   for r in curves(cert[:,q,:],cfg['looks'],cfg['delta'],cfg['tau'],min_index=minnom):
    idx=r['index']
    if idx>=K:idx=-1
    r['index']=idx
    r.update(dataset=meta['dataset'],family=meta['family'],seed=meta['seed'],query=q,path_kind=meta['path_kind'],selected=idx>=0)
    if idx>=0:
     ne=len(ev);ng=int(es[:,q,idx].sum());npnt=int(ev[:,q,idx].sum())
     bounds=audit_limits(ne);lower=float(bounds[0][ng]);upper=float(bounds[1][ng])
     r.update(cost=float(a['cost'][q,idx]),fraction=(idx+1)/K,eval_pointwise=npnt/ne,eval_persistent=ng/ne,eval_gap=(npnt-ng)/ne,eval_lower99=lower,eval_upper99=upper,eval_estimated_shortfall=ng/ne<cfg['tau'],eval_supported_shortfall=upper<cfg['tau'])
    rows.append(r)
   for k in range(K):profiles.append({'dataset':meta['dataset'],'family':meta['family'],'seed':meta['seed'],'query':q,'path_kind':meta['path_kind'],'index':k,'cost':a['cost'][q,k],'pointwise':mean_p[q,k],'persistent':mean_g[q,k]})
  print(file.stem,flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'benchmark_records.csv',index=False,float_format='%.10g')
 sett=pd.DataFrame(settings);sett.to_csv(out/'settings.csv',index=False,float_format='%.10g')
 sett[sett.path_kind=='linear'].groupby(['dataset','family']).mean_retrained_accuracy.agg(mean_accuracy='mean',split_sd='std').to_csv(out/'accuracy_summary.csv',float_format='%.10g')
 pd.DataFrame(profiles).to_csv(out/'robustness_profiles.csv.gz',index=False,compression={'method':'gzip','mtime':0},float_format='%.10g')
 final=df[df.n==cfg['certification_models']]
 agg=dict(certified=('selected','mean'),mean_cost=('cost','mean'),mean_fraction=('fraction','mean'),mean_persistent=('eval_persistent','mean'),mean_pointwise=('eval_pointwise','mean'),mean_gap=('eval_gap','mean'),estimated_shortfall=('eval_estimated_shortfall','mean'),supported_shortfall=('eval_supported_shortfall','mean'))
 final.groupby(['path_kind','method']).agg(**agg).to_csv(out/'path_summary.csv',float_format='%.10g')
 final.groupby('method').agg(**agg).to_csv(out/'method_summary.csv',float_format='%.10g')
 final.groupby(['path_kind','dataset','family','method']).agg(**agg).to_csv(out/'dataset_summary.csv',float_format='%.10g')
 # Common-support comparison prevents different success rates from driving cost differences.
 pivot=final.pivot(index=['path_kind','dataset','family','seed','query'],columns='method',values='cost')
 ratios=[]
 for comparator in ['persistent_bonferroni','persistent_fixed','persistent_spending','pointwise_fst']:
  vals=pivot[['persistent_anytime',comparator]].dropna();rel=vals.persistent_anytime/vals[comparator]-1
  ratios.append({'comparator':comparator,'paired_queries':len(vals),'mean_relative_cost':rel.mean(),'median_relative_cost':rel.median()})
 pd.DataFrame(ratios).to_csv(out/'paired_costs.csv',index=False,float_format='%.10g')
 # Runtime and training counts include nominal/search/certification/evaluation models.
 summary={'settings':len(sett),'path_query_cases':int(sett.negative_queries.sum()),'unique_queries':int(sett[sett.path_kind=='linear'].negative_queries.sum()),'models_per_setting':1+cfg['search_models']+cfg['certification_models']+cfg['evaluation_models'],'model_instances_across_settings':len(sett[sett.path_kind=='linear'])*(1+cfg['search_models']+cfg['certification_models']+cfg['evaluation_models']),'search_seconds_sum':float(sett.search_seconds.sum()),'cert_seconds_sum':float(sett.cert_seconds.sum()),'evaluation_seconds_sum':float(sett.eval_seconds.sum()),'sum_setting_seconds':float(sett.total_seconds.sum()),'mean_lazy_query_fraction':float(sett.lazy_query_fraction.mean())}
 (out/'counts.json').write_text(json.dumps(summary,indent=2))
 simfile=ROOT/'results/simulation_trials.csv.gz'
 if simfile.exists():
  sim=pd.read_csv(simfile);g=sim.groupby(['K','eta','method']);simres=g.agg(repetitions=('replicate','count'),errors=('false_persistent','sum'),false_persistent=('false_persistent','mean'),false_pointwise=('false_pointwise','mean'),mean_index=('index','mean'),first_n=('first_n','mean'),mean_persistent=('g_selected','mean')).reset_index()
  x=simres.errors.to_numpy();n=simres.repetitions.to_numpy();simres['lower95']=np.where(x==0,0,beta.ppf(.025,x,n-x+1));simres['upper95']=np.where(x==n,1,beta.ppf(.975,x+1,n-x));simres.to_csv(out/'simulation_summary.csv',index=False,float_format='%.10g')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
