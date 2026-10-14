from pathlib import Path
import copy
import sys
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .core import environment,write_json,seal,source_hash,object_hash
from .data import validate_prepared
from .models import fit_ae,fit_model,model_scores
from .metrics import thresholds

def pilot(data,output,cfg):
    m=validate_prepared(data); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); scaler=joblib.load(Path(data)/'standard_scaler.joblib')
    samples=[]
    for role,n in [('train_normal',8192),('validation_normal',2048),('calibration_normal',2048)]:
        raw=np.load(Path(data)/role/'X.npy',mmap_mode='r'); ix=np.linspace(0,len(raw)-1,min(n,len(raw)),dtype=int)
        samples.append(scaler.transform(raw[ix]).astype(np.float32))
    train,val,cal=samples; small=copy.deepcopy(cfg); small['ae']['max_epochs']=3
    start=time.perf_counter()
    with threadpool_limits(limits=cfg['threads']):
        a,ar=fit_ae(train,val,cfg['architectures']['ae_l16'],42,small['ae'],cfg['inference_batch_size'])
        b,br=fit_ae(train,val,cfg['architectures']['ae_l16'],42,small['ae'],cfg['inference_batch_size'])
        sa=model_scores(a,'ae',cal); sb=model_scores(b,'ae',cal)
        if not np.allclose(sa,sb,rtol=1e-7,atol=1e-9): raise ValueError('Pilot nije numerički ponovljiv')
        baselines={}
        for kind in ['iforest','pca']:
            model,result=fit_model(train,val,kind,cfg['architectures']['ae_l16'],42,small)
            scores=model_scores(model,kind,cal)
            baselines[kind]={'training':result,'thresholds':thresholds(scores,cfg['percentiles'])}
        env=environment()
    elapsed=time.perf_counter()-start
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=int(peak if sys.platform=='darwin' else peak*1024)
    except ImportError: peak_bytes=None
    record={'purpose':'technical_pilot_not_research_result','dataset':m['dataset'],'code_hash':source_hash(),
        'config_hash':object_hash(cfg),'sample_rows':{'train':len(train),'validation':len(val),'calibration':len(cal)},
        'seed':42,'max_epochs':3,'repeat_max_absolute_difference':float(np.max(abs(sa-sb))),
        'repeat_rtol':1e-7,'repeat_atol':1e-9,'ae_training':ar,'baselines':baselines,'environment':env,
        'model_phase_seconds':elapsed,'process_peak_rss_bytes':peak_bytes,
        'limitations':'Pilot koristi mali uzorak i tri epohe. Vreme nije procena pune obuke. RSS uključuje prethodne operacije ovog procesa.'}
    write_json(output/'pilot.json',record); seal(output); return record
