"""Collaborative topic regression with coupled EM-style topic/factor updates."""
import time
import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq
from sklearn.decomposition import LatentDirichletAllocation
from .base import BaseModel, FitSummary, ModelError


def factor_update(others,observed,a,b,regularization,prior=None,gram=None):
    chosen=others[observed]
    base=b*(others.T@others) if gram is None else gram
    lhs=base+(a-b)*(chosen.T@chosen)+regularization*np.eye(others.shape[1])
    rhs=a*chosen.sum(axis=0)
    if prior is not None: rhs=rhs+regularization*prior
    return solve(lhs,rhs,assume_a='pos',check_finite=False)


def update_theta(old,v,counts,regularization,floor,max_iterations,tolerance=1e-8):
    """Solve the concave simplex subproblem via its scalar Lagrange multiplier.

    The positive quadratic root enforces stationarity. A bracketed scalar solve
    is equivalent to constrained maximization, and only improving updates pass.
    """
    counts=np.maximum(counts,floor)
    def candidate(nu):
        z=nu-regularization*v
        disc=np.sqrt(z*z+4*regularization*counts)
        positive=z>=0
        out=np.empty_like(z)
        out[positive]=2*counts[positive]/(z[positive]+disc[positive])
        out[~positive]=(-z[~positive]+disc[~positive])/(2*regularization)
        return out
    low=float(np.min(regularization*v)-counts.sum()-regularization-1)
    high=float(np.max(regularization*v)+counts.sum()+regularization+1)
    nu,result=brentq(lambda x:candidate(x).sum()-1,low,high,xtol=tolerance,maxiter=max_iterations,full_output=True,disp=False)
    if not result.converged: raise ModelError('CTR theta multiplier solver did not converge; increase topic_iterations')
    theta=np.maximum(candidate(nu),floor); theta/=theta.sum()
    objective=lambda t: float(counts@np.log(np.maximum(t,floor))-.5*regularization*np.sum((v-t)**2))
    return theta if objective(theta)>=objective(old)-1e-10 else old.copy()

class CTR(BaseModel):
    def initialize_topics(self,data):
        c=self.config; k=c.get('topics',32); floor=c.get('numerical_floor',1e-12)
        if data.bow.nnz:
            lda=LatentDirichletAllocation(n_components=k,max_iter=c.get('lda_iterations',10),learning_method='batch',random_state=self.seed,n_jobs=1)
            theta=lda.fit_transform(data.bow)
            beta=np.maximum(lda.components_,floor); beta/=beta.sum(axis=1,keepdims=True)
        else:
            theta=np.full((data.num_items,k),1/k); beta=np.full((k,data.bow.shape[1]),1/data.bow.shape[1])
        empty=data.bow.getnnz(axis=1)==0; theta[empty]=1/k
        self.theta=np.maximum(theta,floor); self.theta/=self.theta.sum(axis=1,keepdims=True)
        self.beta=beta; self.initial_theta=self.theta.copy()

    def fit(self,data):
        started=time.perf_counter(); self.bind(data); self.initialize_topics(data)
        c=self.config; k=c.get('topics',32); a=c.get('confidence_observed',1.); b=c.get('confidence_unobserved',.01)
        lu=c.get('lambda_u',.1); lv=c.get('lambda_v',1.); floor=c.get('numerical_floor',1e-12)
        rng=np.random.default_rng(self.seed); self.user_factors=rng.normal(0,.01,(data.num_users,k)); self.item_factors=self.theta.copy()
        users=data.train; papers=data.train.T.tocsr(); history=[]; topic_changes=[]
        for outer in range(c.get('outer_iterations',10)):
            gram=b*(self.item_factors.T@self.item_factors)
            for user in range(data.num_users):
                observed=users.indices[users.indptr[user]:users.indptr[user+1]]
                self.user_factors[user]=factor_update(self.item_factors,observed,a,b,lu,gram=gram)
            gram=b*(self.user_factors.T@self.user_factors)
            for item in range(data.num_items):
                observed=papers.indices[papers.indptr[item]:papers.indptr[item+1]]
                self.item_factors[item]=factor_update(self.user_factors,observed,a,b,lv,self.theta[item],gram=gram)
            beta_counts=np.full(self.beta.shape,floor); before=self.theta.copy()
            for item in range(data.num_items):
                lo,hi=data.bow.indptr[item:item+2]
                if lo==hi: continue
                terms=data.bow.indices[lo:hi]; frequencies=data.bow.data[lo:hi]
                responsibilities=(self.theta[item,:,None]*self.beta[:,terms]).T
                responsibilities/=np.maximum(responsibilities.sum(axis=1,keepdims=True),floor)
                weighted=frequencies[:,None]*responsibilities
                counts=weighted.sum(axis=0)
                self.theta[item]=update_theta(self.theta[item],self.item_factors[item],counts,lv,floor,c.get('topic_iterations',20),c.get('tolerance',1e-4)*.01)
                beta_counts[:,terms]+=weighted.T
            self.beta=beta_counts/beta_counts.sum(axis=1,keepdims=True)
            objective=self.objective(data,a,b,lu,lv,floor)
            if not np.isfinite(objective): raise ModelError('non-finite CTR objective')
            history.append(objective); topic_changes.append(float(np.linalg.norm(self.theta-before)))
            if len(history)>1 and abs(history[-1]-history[-2])<=c.get('tolerance',1e-4)*max(1,abs(history[-2])): break
        self.fitted=True
        return FitSummary(len(history),time.perf_counter()-started,{'objectives':history,'topic_changes':topic_changes,'topic_solver':'bracketed simplex Lagrange multiplier','lda_seed':self.seed})

    def objective(self,data,a,b,lu,lv,floor):
        # Sum all missing-pair errors using Gram matrices, not a dense score matrix.
        u,v=self.user_factors,self.item_factors
        error=b*np.sum((u.T@u)*(v.T@v)); rows,cols=data.train.nonzero()
        scores=np.einsum('ij,ij->i',u[rows],v[cols])
        error+=np.sum(a*(1-scores)**2-b*scores**2)
        text=0.
        for item in range(data.num_items):
            lo,hi=data.bow.indptr[item:item+2]; terms=data.bow.indices[lo:hi]
            if lo!=hi: text+=data.bow.data[lo:hi]@np.log(np.maximum(self.theta[item]@self.beta[:,terms],floor))
        return float(text-.5*(error+lu*np.sum(u*u)+lv*np.sum((v-self.theta)**2)))

    def score(self,user_ids,item_ids):
        u,i=self.validate_score(user_ids,item_ids)
        return self.user_factors[u]@self.item_factors[i].T
