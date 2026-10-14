import copy
import time
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import ConvergenceWarning
from .core import finite_array

def transform_matrix(raw,scaler,path,batch=100000):
    x=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=raw.shape)
    for i in range(0,len(raw),batch):
        b=scaler.transform(raw[i:i+batch]).astype(np.float32); finite_array(b); x[i:i+len(b)]=b
    x.flush(); return x

def model_scores(model,kind,x,batch=16384):
    out=np.empty(len(x),dtype=np.float64)
    for i in range(0,len(x),batch):
        b=x[i:i+batch]
        if kind=='iforest': s=-model.score_samples(b)
        else:
            pred=model.predict(b) if kind=='ae' else model.inverse_transform(model.transform(b))
            residual=np.asarray(pred,dtype=np.float64)-b; s=np.mean(residual*residual,axis=1,dtype=np.float64)
        finite_array(s); out[i:i+len(b)]=s
    return out

def parameter_count(model):
    return sum(a.size for a in [*model.coefs_,*model.intercepts_]) if hasattr(model,'coefs_') else 0

def fit_ae(train,validation,hidden,seed,settings,score_batch,checkpoint=None):
    model=MLPRegressor(hidden_layer_sizes=tuple(hidden),activation='relu',solver='adam',alpha=settings['alpha'],
        batch_size=settings['batch_size'],learning_rate_init=settings['learning_rate_init'],max_iter=1,shuffle=False,
        random_state=seed,warm_start=True,tol=0.,n_iter_no_change=settings['max_epochs']+1)
    best=None; best_value=float('inf'); best_epoch=0; waiting=0; history=[]; start=time.perf_counter()
    for epoch in range(1,settings['max_epochs']+1):
        tick=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning); model.partial_fit(train,train)
        fit_seconds=time.perf_counter()-tick; tick=time.perf_counter()
        train_mse=float(model_scores(model,'ae',train,score_batch).mean())
        val_mse=float(model_scores(model,'ae',validation,score_batch).mean()); monitoring=time.perf_counter()-tick
        finite_array(np.array([model.loss_,train_mse,val_mse])); improved=val_mse<best_value-settings['min_delta']
        if improved: best=copy.deepcopy(model); best_value=val_mse; best_epoch=epoch; waiting=0
        else: waiting+=1
        history.append({'epoch':epoch,'optimizer_loss':float(model.loss_),'train_reconstruction_mse':train_mse,
            'validation_reconstruction_mse':val_mse,'accepted_best_mse':best_value,'improved':improved,
            'fit_seconds':fit_seconds,'monitoring_seconds':monitoring})
        if checkpoint: checkpoint(history,best,best_epoch)
        if waiting>=settings['patience']: break
    if best is None: raise RuntimeError('Nije dobijeno validno stanje')
    return best,{'best_epoch':best_epoch,'epochs_ran':len(history),'history':history,
        'fit_seconds':sum(r['fit_seconds'] for r in history),'monitoring_seconds':sum(r['monitoring_seconds'] for r in history),
        'training_with_monitoring_seconds':time.perf_counter()-start,'parameters':parameter_count(best)}

def fit_model(train,validation,kind,hidden,seed,cfg,checkpoint=None):
    if kind=='ae': return fit_ae(train,validation,hidden,seed,cfg['ae'],cfg['inference_batch_size'],checkpoint)
    start=time.perf_counter()
    if kind=='iforest':
        if len(train)<cfg['iforest']['max_samples']: raise ValueError('Premalo redova za IF max_samples')
        model=IsolationForest(**cfg['iforest'],random_state=seed,n_jobs=1).fit(train)
    elif kind=='pca': model=PCA(n_components=hidden[len(hidden)//2],svd_solver='full',whiten=False).fit(train)
    else: raise ValueError('Nepoznat model')
    elapsed=time.perf_counter()-start
    return model,{'best_epoch':None,'epochs_ran':None,'history':[],'fit_seconds':elapsed,'monitoring_seconds':0.,
        'training_with_monitoring_seconds':elapsed,'parameters':0}

def robust_scaler(train):
    scaler=RobustScaler(with_centering=True,with_scaling=True,quantile_range=(25,75),unit_variance=False).fit(train)
    q=np.percentile(train,[25,75],axis=0,method='linear')
    return scaler,np.flatnonzero(q[1]-q[0]==0).tolist()

def latent_activations(model,x):
    h=x
    for i in range(len(model.hidden_layer_sizes)//2+1): h=np.maximum(0,h@model.coefs_[i]+model.intercepts_[i])
    return h
