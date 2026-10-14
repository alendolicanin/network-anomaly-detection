from pathlib import Path
import tempfile
import time
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from .core import read_json,write_json,digest,object_hash,source_hash,environment,seal,verify,data_fingerprint,runtime_versions
from .data import validate_prepared,iter_role
from .models import fit_model,transform_matrix,robust_scaler,model_scores,latent_activations
from .metrics import thresholds,evaluate,choose_architecture,choose_percentile,ranking_metrics

def archive_incomplete(directory,record_name,expected,identity_key=None):
    directory=Path(directory)
    if directory.exists() and not (directory/'integrity.json').exists():
        record=read_json(directory/record_name); actual=record[identity_key] if identity_key else record
        if actual!=expected: raise ValueError('Nedovršen izlaz pripada drugoj konfiguraciji')
        archive=directory.parent/'interrupted_runs'/f'{directory.name}_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True); directory.rename(archive)

def score_role(data,m,role,model,scaler,kind,batch,out):
    parts={k:[] for k in ['scores','source_id','timestamp_ns','label_code','day_code']}
    members=sorted({p['member'] for p in m['roles'][role]}); infer=scale_time=0.; activity={}; residuals={}
    total_start=time.perf_counter(); targets={'FTP-Patator','SSH-Patator','Botnet','Bot','SSH-BruteForce','FTP-BruteForce - Attempted'}
    for raw,meta in iter_role(data,m,role):
        values=[]
        for i in range(0,len(raw),batch):
            tick=time.perf_counter(); x=scaler.transform(raw[i:i+batch]).astype(np.float32); scale_time+=time.perf_counter()-tick
            tick=time.perf_counter(); s=model_scores(model,kind,x,batch); infer+=time.perf_counter()-tick; values.append(s)
            if kind=='ae' and role!='calibration_normal':
                h=latent_activations(model,x); codes=meta['label_code'][i:i+len(x)]
                for code in np.unique(codes):
                    name=m['labels'][int(code)]; mask=codes==code; a=h[mask]>0
                    old=activity.setdefault(name,{'n':0,'positive_counts':np.zeros(h.shape[1],dtype=np.int64)})
                    old['n']+=int(mask.sum()); old['positive_counts']+=a.sum(axis=0)
                    if name in targets:
                        pred=model.predict(x[mask]); residuals.setdefault(name,[]).append((np.asarray(pred,dtype=np.float64)-x[mask])**2)
        parts['scores'].append(np.concatenate(values))
        for k in ['source_id','timestamp_ns','label_code']: parts[k].append(meta[k])
        parts['day_code'].append(np.full(len(raw),members.index(str(meta['day'])),dtype=np.uint8))
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    np.savez_compressed(Path(out)/f'scores_{role}.npz',**arrays,members=np.asarray(members),labels=np.asarray(m['labels']))
    diagnostic={'latent':{},'residuals':{}}
    for name,v in activity.items():
        rates=v['positive_counts']/v['n']
        diagnostic['latent'][name]={'n':v['n'],'positive_fraction_per_unit':rates.tolist(),'mean_active_units':float(rates.sum())}
    for name,chunks in residuals.items():
        r=np.concatenate(chunks)
        diagnostic['residuals'][name]={'n':len(r),'mean_squared_residual':r.mean(axis=0).tolist(),'median_squared_residual':np.median(r,axis=0).tolist()}
    write_json(Path(out)/f'diagnostics_{role}.json',diagnostic)
    write_json(Path(out)/f'timing_{role}.json',{'rows':len(arrays['scores']),'inference_seconds':infer,'scaling_seconds':scale_time,
        'inference_rows_per_second':len(arrays['scores'])/infer,'total_with_io_and_diagnostics_seconds':time.perf_counter()-total_start})
    return arrays

def score_metrics(path,cutoffs):
    with np.load(path,allow_pickle=False) as z:
        scores=z['scores']; labels=z['labels'][z['label_code']]; days=z['members'][z['day_code']]
        attempted=np.char.find(labels.astype(str),'Attempted')>=0; y=((labels!='BENIGN')&~attempted).astype(np.int8)
    ranks={'primary':ranking_metrics(y,scores),'exclude_attempted':ranking_metrics(y[~attempted],scores[~attempted])}
    return {q:evaluate(y,scores,attempted,labels,days,v,ranks) for q,v in cutoffs.items()}

def normal_raw(data,role):
    return np.load(Path(data)/role/'X.npy',mmap_mode='r')

def run_fit(data,m,output,architecture,seed,cfg,kind='ae',scaling='standard',decision_hash=None):
    output=Path(output)
    identity={'dataset':m['dataset'],'architecture':architecture,'kind':kind,'seed':seed,
        'scaling':scaling,'config_hash':object_hash(cfg),'data_hash':data_fingerprint(data),'code_hash':source_hash(),
        'decision_hash':decision_hash,'runtime_versions':runtime_versions()}
    archive_incomplete(output,'run.json',identity,'identity')
    if output.exists():
        verify(output); record=read_json(output/'run.json')
        if record['identity']!=identity or record['status']!='complete': raise ValueError('Neodgovarajuće prethodno pokretanje')
        return record
    output.mkdir(parents=True); write_json(output/'run.json',{'identity':identity,'status':'incomplete'})
    train=normal_raw(data,'train_normal'); val=normal_raw(data,'validation_normal')
    if scaling=='standard': scaler=joblib.load(Path(data)/'standard_scaler.joblib'); zero_iqr=[]
    elif scaling=='robust' and m['dataset']=='cic2017': scaler,zero_iqr=robust_scaler(train)
    else: raise ValueError('Nedozvoljeno skaliranje')
    joblib.dump(scaler,output/'scaler.joblib'); hidden=cfg['architectures'].get(architecture,[]); env=environment()
    with tempfile.TemporaryDirectory(prefix='scaled_',dir=output) as td:
        tick=time.perf_counter(); x=transform_matrix(train,scaler,Path(td)/'train.npy'); v=transform_matrix(val,scaler,Path(td)/'validation.npy')
        scaling_seconds=time.perf_counter()-tick
        def checkpoint(history,best,epoch):
            write_json(output/'training_history.json',history); joblib.dump({'model':best,'best_epoch':epoch},output/'best_checkpoint.joblib')
        model,training=fit_model(x,v,kind,hidden,seed,cfg,checkpoint); del x,v
    joblib.dump(model,output/'model.joblib'); write_json(output/'training_history.json',training.pop('history'))
    cal=score_role(data,m,'calibration_normal',model,scaler,kind,cfg['inference_batch_size'],output)
    write_json(output/'thresholds.json',thresholds(cal['scores'],cfg['percentiles']))
    if kind=='ae': score_role(data,m,'validation_normal',model,scaler,kind,cfg['inference_batch_size'],output)
    record={'identity':identity,'status':'complete','training':training,'environment':env,'input_dimension':train.shape[1],
        'architecture':[train.shape[1],*hidden,train.shape[1]] if kind=='ae' else None,
        'scaling_train_validation_seconds':scaling_seconds,'zero_iqr_columns':zero_iqr}
    write_json(output/'run.json',record); seal(output); return record

def evaluation(data,m,fit_dir,out,role,cfg,decision_hash=None):
    out=Path(out); fit_dir=Path(fit_dir); verify(fit_dir); rec=read_json(fit_dir/'run.json'); ident=rec['identity']
    if (ident['data_hash']!=data_fingerprint(data) or ident['config_hash']!=object_hash(cfg)
        or ident['code_hash']!=source_hash() or ident['runtime_versions']!=runtime_versions()):
        raise ValueError('Model, podaci, kod ili okruženje nisu saglasni')
    if role!='development' and decision_hash is None: raise ValueError('Završna procena zahteva odluku')
    expected={'run_hash':digest(fit_dir/'integrity.json'),'role':role,'decision_hash':decision_hash}
    archive_incomplete(out,'evaluation.json',expected)
    if out.exists():
        verify(out)
        if read_json(out/'evaluation.json')!=expected: raise ValueError('Neodgovarajuća prethodna evaluacija')
        return read_json(out/'metrics.json')
    out.mkdir(parents=True); write_json(out/'evaluation.json',expected)
    model=joblib.load(fit_dir/'model.joblib'); scaler=joblib.load(fit_dir/'scaler.joblib')
    score_role(data,m,role,model,scaler,ident['kind'],cfg['inference_batch_size'],out)
    metrics=score_metrics(out/f'scores_{role}.npz',read_json(fit_dir/'thresholds.json'))
    write_json(out/'metrics.json',metrics); seal(out); return metrics

def develop(data,experiment,cfg):
    m=validate_prepared(data); e=Path(experiment)
    if m['dataset']!='cic2017': raise ValueError('Izbor nije dozvoljen na spoljnom skupu')
    if (e/'decision.json').exists(): raise FileExistsError('Odluka je zaključana')
    with threadpool_limits(limits=cfg['threads']):
        for architecture in [*cfg['architectures'],'iforest']:
            for seed in cfg['seeds']:
                name=f'{architecture}_seed{seed}'; fit=e/'fits'/name
                run_fit(data,m,fit,architecture,seed,cfg,'iforest' if architecture=='iforest' else 'ae')
                evaluation(data,m,fit,e/'development'/name,'development',cfg)
                print('development complete',name,flush=True)
    return lock_decision(data,e,cfg)

def lock_decision(data,experiment,cfg):
    e=Path(experiment)
    if (e/'decision.json').exists(): raise FileExistsError('Odluka se ne prepisuje')
    arch_rows=[]; all_rows={}; receipts={}
    for architecture in [*cfg['architectures'],'iforest']:
        all_rows[architecture]=[]
        for seed in cfg['seeds']:
            name=f'{architecture}_seed{seed}'; fit=e/'fits'/name; ev=e/'development'/name; verify(fit); verify(ev)
            r=read_json(fit/'run.json'); i=r['identity']; meta=read_json(ev/'evaluation.json'); metrics=read_json(ev/'metrics.json')
            if (i['architecture']!=architecture or i['seed']!=seed or i['scaling']!='standard' or i['dataset']!='cic2017'
                or i['code_hash']!=source_hash() or i['config_hash']!=object_hash(cfg) or i['data_hash']!=data_fingerprint(data)):
                raise ValueError('Nesaglasna razvojna pokretanja')
            if meta!={'run_hash':digest(fit/'integrity.json'),'role':'development','decision_hash':None}: raise ValueError('Izbor zahteva razvojnu ulogu')
            if set(metrics)!=set(map(str,cfg['percentiles'])): raise ValueError('Nepotpun niz pragova')
            for q in cfg['percentiles']:
                v=metrics[str(q)]['primary']; row={'architecture':architecture,'seed':seed,'role':'development','percentile':q,
                    'f1':v['f1'],'fpr':v['fpr'],'parameters':r['training']['parameters']}
                all_rows[architecture].append(row)
                if q==95 and architecture!='iforest': arch_rows.append(row)
            receipts[name]={'fit':digest(fit/'integrity.json'),'development':digest(ev/'integrity.json')}
    best,ranking=choose_architecture(arch_rows,cfg['seeds'])
    q,qr=choose_percentile(all_rows[best],cfg['seeds'],cfg['percentiles']); qi,qir=choose_percentile(all_rows['iforest'],cfg['seeds'],cfg['percentiles'])
    record={'architecture':best,'percentile_ae':q,'percentile_if':qi,'architecture_ranking':ranking,
        'ae_percentile_ranking':qr,'if_percentile_ranking':qir,'all_development_selection_rows':all_rows,'receipts':receipts,
        'config_hash':object_hash(cfg),'code_hash':source_hash(),'main_data_hash':data_fingerprint(data),'runtime_versions':runtime_versions()}
    write_json(e/'decision.json',record); write_json(e/'decision.sha256.json',{'sha256':digest(e/'decision.json')}); return record

def load_decision(experiment,cfg):
    e=Path(experiment); p=e/'decision.json'; d=read_json(p)
    if digest(p)!=read_json(e/'decision.sha256.json')['sha256']: raise ValueError('Promenjena odluka')
    if d['config_hash']!=object_hash(cfg) or d['code_hash']!=source_hash() or d['runtime_versions']!=runtime_versions():
        raise ValueError('Promenjen kod, konfiguracija ili okruženje')
    for name,r in d['receipts'].items():
        for area,key in [('fits','fit'),('development','development')]:
            p=e/area/name
            if digest(p/'integrity.json')!=r[key]: raise ValueError('Promenjen dokaz odluke')
            verify(p)
    return d,digest(e/'decision.json')

def final_main(data,experiment,cfg):
    m=validate_prepared(data); d,dh=load_decision(experiment,cfg); e=Path(experiment)
    if m['dataset']!='cic2017' or data_fingerprint(data)!=d['main_data_hash']: raise ValueError('Pogrešan glavni skup')
    names=sorted({'ae_l16','ae_l32',d['architecture']})
    with threadpool_limits(limits=cfg['threads']):
        for architecture in [*names,'iforest']:
            for seed in cfg['seeds']:
                name=f'{architecture}_seed{seed}'; evaluation(data,m,e/'fits'/name,e/'final_test'/name,'final_test',cfg,dh)
        for seed in cfg['seeds']:
            name=f'robust_{d["architecture"]}_seed{seed}'; fit=e/'fits'/name
            run_fit(data,m,fit,d['architecture'],seed,cfg,scaling='robust',decision_hash=dh)
            evaluation(data,m,fit,e/'final_test'/name,'final_test',cfg,dh)
        fit=e/'fits/pca'; run_fit(data,m,fit,d['architecture'],None,cfg,kind='pca',decision_hash=dh)
        evaluation(data,m,fit,e/'final_test/pca','final_test',cfg,dh)

def external(data,experiment,cfg):
    m=validate_prepared(data); d,dh=load_decision(experiment,cfg); e=Path(experiment)/'external'
    if m['dataset']!='cic2018': raise ValueError('Pogrešan spoljni skup')
    with threadpool_limits(limits=cfg['threads']):
        for architecture in [*sorted({'ae_l16','ae_l32',d['architecture']}),'iforest']:
            for seed in cfg['seeds']:
                name=f'{architecture}_seed{seed}'; fit=e/'fits'/name
                run_fit(data,m,fit,architecture,seed,cfg,'iforest' if architecture=='iforest' else 'ae',decision_hash=dh)
                for role in ['external_test','external_diagnostic']: evaluation(data,m,fit,e/role/name,role,cfg,dh)
                print('external complete',name,flush=True)
