"""Two-branch neural collaborative filtering with training-only sampling."""
import time
import numpy as np
import torch
from torch import nn
from .base import BaseModel, FitSummary, ModelError

class NeuMFNetwork(nn.Module):
    def __init__(self,users,items,gmf_dim,mlp_dim,hidden):
        super().__init__()
        self.gmf_user=nn.Embedding(users,gmf_dim); self.gmf_item=nn.Embedding(items,gmf_dim)
        self.mlp_user=nn.Embedding(users,mlp_dim); self.mlp_item=nn.Embedding(items,mlp_dim)
        layers=[]; size=2*mlp_dim
        for width in hidden: layers.extend([nn.Linear(size,width),nn.ReLU()]); size=width
        self.mlp=nn.Sequential(*layers); self.head=nn.Linear(gmf_dim+size,1)
        for embedding in (self.gmf_user,self.gmf_item,self.mlp_user,self.mlp_item): nn.init.normal_(embedding.weight,std=.01)

    def forward(self,users,items):
        gmf=self.gmf_user(users)*self.gmf_item(items)
        mlp=self.mlp(torch.cat([self.mlp_user(users),self.mlp_item(items)],dim=-1))
        return self.head(torch.cat([gmf,mlp],dim=-1)).squeeze(-1)


def sample_negatives(train,users,rng):
    result=np.empty(len(users),dtype=np.int64)
    # Group by user; dense rows use an explicit complement to avoid unbounded rejection.
    order=np.argsort(users,kind="stable")
    unique,starts=np.unique(users[order],return_index=True)
    ends=np.r_[starts[1:],len(users)]
    for user,lo,hi in zip(unique,starts,ends):
        positions=order[lo:hi]; seen=train.indices[train.indptr[user]:train.indptr[user+1]]
        if len(seen)>=train.shape[1]: raise ModelError(f'user {user} has no unobserved negative candidate')
        draws=rng.integers(train.shape[1],size=len(positions))
        for attempt in range(8):
            invalid=np.isin(draws,seen,assume_unique=False)
            if not invalid.any(): break
            draws[invalid]=rng.integers(train.shape[1],size=int(invalid.sum()))
        invalid=np.isin(draws,seen)
        if invalid.any():
            complement=np.setdiff1d(np.arange(train.shape[1]),seen,assume_unique=True)
            draws[invalid]=rng.choice(complement,size=int(invalid.sum()))
        result[positions]=draws
    return result

class NeuMF(BaseModel):
    def __init__(self,config=None,seed=42,device='cpu',**kwargs):
        super().__init__(config,seed); self.device=device; self.pair_batch_size=8192

    def fit(self,data):
        # L2 can drive inactive embedding values into the CPU subnormal range.
        # Flush only during regularized fitting; restore the reference mode.
        regularized = self.config.get('weight_decay', 0.0) > 0
        if regularized:
            torch.set_flush_denormal(True)
        try:
            return self._fit(data)
        finally:
            if regularized:
                torch.set_flush_denormal(False)

    def _fit(self,data):
        started=time.perf_counter(); self.bind(data); torch.manual_seed(self.seed)
        if self.device=='cuda' and not torch.cuda.is_available(): raise ModelError('requested CUDA unavailable')
        c=self.config
        self.network=NeuMFNetwork(data.num_users,data.num_items,c.get('gmf_dim',32),c.get('mlp_embedding_dim',32),c.get('hidden_sizes',[64,32,16])).to(self.device)
        optimizer=torch.optim.Adam(self.network.parameters(),lr=c.get('learning_rate',.001),weight_decay=c.get('weight_decay',0.0))
        criterion=nn.BCEWithLogitsLoss(); rng=np.random.default_rng(self.seed)
        pu,pi=data.train.nonzero(); ratio=c.get('negative_ratio',4); losses=[]
        for epoch in range(c.get('epochs',10)):
            nu=np.repeat(pu,ratio); ni=sample_negatives(data.train,nu,rng) if ratio else np.empty(0,dtype=np.int64)
            users=np.concatenate([pu,nu]); items=np.concatenate([pi,ni]); labels=np.concatenate([np.ones(len(pu)),np.zeros(len(nu))]).astype(np.float32)
            order=rng.permutation(len(users)); batch=c.get('batch_size',1024); total=0
            self.network.train()
            for start in range(0,len(order),batch):
                ix=order[start:start+batch]
                u=torch.as_tensor(users[ix],dtype=torch.long,device=self.device); i=torch.as_tensor(items[ix],dtype=torch.long,device=self.device)
                y=torch.as_tensor(labels[ix],device=self.device)
                optimizer.zero_grad(set_to_none=True); loss=criterion(self.network(u,i),y)
                if not torch.isfinite(loss): raise ModelError('non-finite NeuMF loss')
                loss.backward(); optimizer.step(); total+=float(loss.detach())*len(ix)
            losses.append(total/max(len(order),1))
        if any(not torch.isfinite(p).all() for p in self.network.parameters()): raise ModelError('non-finite NeuMF state')
        self.fitted=True
        return FitSummary(len(losses),time.perf_counter()-started,{'losses':losses,'device':self.device,'flush_denormal_during_fit':c.get('weight_decay',0.0)>0})

    def score(self,user_ids,item_ids):
        users,items=self.validate_score(user_ids,item_ids)
        out=np.empty((len(users),len(items)),dtype=np.float32); total=out.size
        self.network.eval()
        with torch.inference_mode():
            for start in range(0,total,self.pair_batch_size):
                ix=np.arange(start,min(start+self.pair_batch_size,total))
                u=torch.as_tensor(users[ix//len(items)],dtype=torch.long,device=self.device)
                i=torch.as_tensor(items[ix%len(items)],dtype=torch.long,device=self.device)
                out.ravel()[ix]=self.network(u,i).cpu().numpy()
        return out

    def save_checkpoint(self,directory):
        from pathlib import Path
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        torch.save(self.network.state_dict(),p/'state_dict.pt'); self.checkpoint_metadata(p)

