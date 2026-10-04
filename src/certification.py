"""Finite-path certificates based on ordered Bernoulli outcomes."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom,beta
@lru_cache(None)
def log_e_table(n,tau=.95):
    """Uniform mixture over q in (tau,1), evaluated without tail underflow."""
    ss=np.arange(n+1)
    logtail=np.logaddexp.accumulate(binom.logpmf(np.arange(n+2),n+1,tau))[:n+1]
    from scipy.special import betaln
    return betaln(ss+1,n-ss+1)+logtail-np.log1p(-tau)-ss*np.log(tau)-(n-ss)*np.log1p(-tau)
def log_evalue(s,n,tau=.95):
    return log_e_table(int(n),float(tau))[np.asarray(s,dtype=int)]
def suffix_outcomes(predictions):
    """Each row is one independent model; columns are an ordered, frozen path."""
    return np.logical_and.accumulate(np.asarray(predictions,dtype=bool)[...,::-1],axis=-1)[...,::-1]
@lru_cache(None)
def e_threshold(n,delta=.05,tau=.95):
    ss=np.arange(n+1);ii=np.flatnonzero(log_evalue(ss,n,tau)>=np.log(1/delta))
    return int(ii[0]) if len(ii) else n+1
@lru_cache(None)
def cp_threshold(n,delta=.05,tau=.95):
    ss=np.arange(n+1);ii=np.flatnonzero(binom.sf(ss-1,n,tau)<=delta)
    return int(ii[0]) if len(ii) else n+1
@lru_cache(None)
def beta_threshold(n,delta=.05,tau=.95):
    ss=np.arange(n+1);ii=np.flatnonzero(beta.cdf(tau,ss+.5,n-ss+.5)<=delta)
    return int(ii[0]) if len(ii) else n+1
def first_index(mask):
    idx=np.flatnonzero(mask);return int(idx[0]) if len(idx) else -1
def curves(predictions,looks,delta=.05,tau=.95,min_index=0):
    p=np.asarray(predictions,bool);N,K=p.shape
    a=suffix_outcomes(p);raw=np.cumsum(p,axis=0);suf=np.cumsum(a,axis=0)
    ever=np.zeros(K,bool);bonf=np.zeros(K,bool);point=np.zeros(K,bool);spend=np.zeros(K,bool);repeated=np.zeros(K,bool)
    out=[]
    for looknum,n in enumerate(looks,1):
        s=suf[n-1];r=raw[n-1]
        repeated |= s>=cp_threshold(n,delta,tau)
        ever |= s>=e_threshold(n,delta,tau)
        bonf |= s>=e_threshold(n,delta/K,tau)
        point |= r>=e_threshold(n,delta,tau)
        # Accept a suffix only after every point in it has crossed its own threshold.
        point_suffix=suffix_outcomes(point)
        spend |= s>=cp_threshold(n,delta/(looknum*(looknum+1)),tau)
        for method,mask in [('persistent_repeated_cp',repeated),('persistent_anytime',ever),('persistent_bonferroni',bonf),('persistent_spending',spend),('pointwise_fst',point_suffix),('persistent_fixed',s>=cp_threshold(n,delta,tau)),('pointwise_beta',r>=beta_threshold(n,delta,tau)),('persistent_naive',s>=np.ceil(tau*n))]:
            eligible=mask.copy();eligible[:min_index]=False
            out.append({'method':method,'n':int(n),'index':first_index(eligible)})
    return out

def last_failure(query,K):
    """Find the last failed point using a terminal-first black-box oracle."""
    for k in range(K-1,-1,-1):
        if not query(k):return k
    return -1
