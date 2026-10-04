"""Explicit bootstrap retraining laws for CPU estimators and batched neural nets."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor
import time

def fit_one(args):
    family,X,y,P,seed=args
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(y),size=len(y))
    if len(np.unique(y[idx]))<2:return np.full(len(P),y[idx][0],np.uint8)
    with threadpool_limits(limits=1):
        if family=='logistic':model=LogisticRegression(C=1.,max_iter=300,solver='lbfgs',random_state=seed)
        else:model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,max_depth=None,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=seed)
        model.fit(X[idx],y[idx]);return model.predict(P).astype(np.uint8)

def cpu_bank(family,X,y,P,seeds,workers=8):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vals=list(pool.map(fit_one,[(family,X,y,P,int(s)) for s in seeds],chunksize=4))
    return np.stack(vals)

def neural_bank(X,y,P,seeds,config,device='mps'):
    import torch
    torch.set_num_threads(1)
    if device=='mps' and not torch.backends.mps.is_available():device='cpu'
    out=[];N,D=X.shape;H=config['mlp_width'];batch=config['mlp_batch_size']
    xt=torch.tensor(X,dtype=torch.float32,device=device);yt=torch.tensor(y,dtype=torch.float32,device=device)
    pt=torch.tensor(P,dtype=torch.float32,device=device)
    for start in range(0,len(seeds),config['mlp_parallel_models']):
        ss=seeds[start:start+config['mlp_parallel_models']];B=len(ss)
        rngs=[np.random.default_rng(int(s)) for s in ss]
        # Independent bootstrap datasets; independent initial weights and minibatch indices.
        boots=np.stack([r.integers(0,N,size=N) for r in rngs])
        a=np.stack([r.normal(0,np.sqrt(2/D),(D,H)) for r in rngs]).astype('float32')
        b=np.zeros((B,1,H),dtype='float32')
        c=np.stack([r.normal(0,np.sqrt(1/H),(H,1)) for r in rngs]).astype('float32')
        d=np.zeros((B,1,1),dtype='float32')
        weights=[torch.tensor(w,device=device,requires_grad=True) for w in [a,b,c,d]]
        opt=torch.optim.Adam(weights,lr=.01)
        for step in range(config['mlp_steps']):
            ids=np.stack([boots[i,r.integers(0,N,size=batch)] for i,r in enumerate(rngs)])
            ids=torch.tensor(ids,dtype=torch.int64,device=device);xb=xt[ids];yb=yt[ids]
            w1,b1,w2,b2=weights;logits=(torch.bmm(torch.relu(torch.bmm(xb,w1)+b1),w2)+b2).squeeze(-1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction='none').mean(1).sum()
            loss=loss+1e-4*(w1.square().mean((1,2)).sum()+w2.square().mean((1,2)).sum())
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            preds=[]
            for lo in range(0,len(P),2048):
                xx=pt[lo:lo+2048].unsqueeze(0).expand(B,-1,-1)
                logits=(torch.bmm(torch.relu(torch.bmm(xx,weights[0])+weights[1]),weights[2])+weights[3]).squeeze(-1)
                preds.append((logits>=0).cpu().numpy().astype('uint8'))
            out.append(np.concatenate(preds,axis=1))
        del opt,weights
    return np.concatenate(out,axis=0)

def bank(family,X,y,P,seeds,config,device='mps'):
    if family=='mlp':return neural_bank(X,y,P,seeds,config,device)
    return cpu_bank(family,X,y,P,seeds,workers=config['cpu_workers'])
