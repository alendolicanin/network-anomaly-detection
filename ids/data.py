from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import shutil
import zipfile
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from .core import ROOT,read_json,write_json,digest,seal,verify,finite_array,environment,object_hash

OBJECTS={'Flow ID','Src IP','Dst IP','Timestamp','Label'}
EXCLUDED=OBJECTS|{'id','Attempted Category','Src Port','Dst Port','Protocol'}

def clean_chunk(frame,columns,low,high):
    if list(frame.columns)!=columns: raise ValueError('Neočekivana šema')
    numeric=[x for x in columns if x not in OBJECTS]
    num=frame[numeric].apply(pd.to_numeric,errors='coerce')
    ts=pd.to_datetime(frame.Timestamp,format='mixed',errors='coerce',utc=True)
    labels=frame.Label.astype('string').str.strip()
    ok=np.isfinite(num.to_numpy(dtype=float)).all(axis=1)&ts.notna()&~frame.isna().any(axis=1)&labels.notna()&labels.ne('')
    if ((ts[ok]<pd.Timestamp(low,tz='UTC'))|(ts[ok]>pd.Timestamp(high,tz='UTC'))).any():
        raise ValueError('Vreme izvan auditovanog opsega')
    c=frame.loc[ok].copy(); c[numeric]=num.loc[ok]; c['Timestamp']=ts.loc[ok]; c['Label']=labels.loc[ok]
    if (c.Label.str.contains('Attempted',case=False)!=(c['Attempted Category']!=-1)).any():
        raise ValueError('Nesaglasne Attempted oznake')
    for name in ['id','Protocol','Attempted Category']:
        if (c[name]%1!=0).any(): raise ValueError('Neceo identifikator ili kategorija')
    return c,int((~ok).sum())

def chronological_order(timestamps,ids):
    timestamps=np.asarray(timestamps); ids=np.asarray(ids)
    order=np.lexsort((ids,timestamps)); t=timestamps[order]; n=len(t)
    if n<5: raise ValueError('Premalo normalnih tokova')
    i=int(np.searchsorted(t,t[int(n*.6)-1],side='right'))
    j=int(np.searchsorted(t,t[int(n*.8)-1],side='right'))
    if not 0<i<j<n: raise ValueError('Jednaka vremena onemogućavaju podelu')
    return order,[i,j]

def training_spec(raw,proto,columns,order,train_end,chunk_size):
    lo=np.full(raw.shape[1],np.inf); hi=np.full(raw.shape[1],-np.inf); categories=set()
    for start in range(0,train_end,chunk_size):
        ix=order[start:min(start+chunk_size,train_end)]; b=raw[ix]
        lo=np.minimum(lo,b.min(axis=0)); hi=np.maximum(hi,b.max(axis=0)); categories.update(map(int,proto[ix]))
    keep=np.flatnonzero(lo!=hi).tolist(); names=[columns[i] for i in keep]
    return {'numeric_features':names,'constant_features':[columns[i] for i in range(len(columns)) if i not in keep],
        'protocol_categories':sorted(categories),'feature_names':names+[f'Protocol_{p}' for p in sorted(categories)],
        'numeric_indices':keep,'input_dimension':len(names)+len(categories)}

def matrix(raw,proto,spec):
    return np.column_stack([np.asarray(raw[:,spec['numeric_indices']],dtype=np.float64),
        *[(proto==p).astype(float) for p in spec['protocol_categories']]])

@contextmanager
def open_source(root,dataset,member):
    if dataset=='cic2017':
        with zipfile.ZipFile(Path(root)/'CICIDS2017_improved.zip') as z:
            with z.open(member) as f: yield f
    else:
        with (Path(root)/'improved2018'/member).open('rb') as f: yield f

def stream(root,dataset,member,reference,chunk_size):
    spec=reference['files'][member]; seen=np.zeros(spec['rows']+1,dtype=bool)
    total=valid=0; labels=Counter()
    with open_source(root,dataset,member) as f:
        for frame in pd.read_csv(f,chunksize=chunk_size,low_memory=False):
            total+=len(frame)
            c,_=clean_chunk(frame,reference['columns'],spec['min_timestamp'],spec['max_timestamp'])
            ids=c.id.to_numpy(dtype=np.int64)
            if (ids<1).any() or (ids>spec['rows']).any() or len(np.unique(ids))!=len(ids) or seen[ids].any():
                raise ValueError('Neispravni ili duplirani identifikatori')
            seen[ids]=True; valid+=len(c); labels.update(c.Label.value_counts().to_dict())
            yield c
    if total!=spec['rows'] or valid!=spec['valid_rows'] or dict(labels)!=spec['labels']:
        raise ValueError('Brojevi ili oznake se razlikuju od audita')

def save_meta(path,ids,ts,label_codes,day):
    np.savez_compressed(path,source_id=ids,timestamp_ns=ts,label_code=label_codes,day=np.asarray(day))

def identity_from(m,entries):
    return {'dataset':m['dataset'],'schema':m['schema'],
        'roles':m['roles'],'labels':m['labels'],'source_reference_hash':object_hash(m.get('source',{})),
        'payload_hashes':{k:v for k,v in entries.items() if k not in ['manifest.json','data_identity.json']}}

def seal_data(directory):
    directory=Path(directory); seal(directory)
    entries=read_json(directory/'integrity.json'); m=read_json(directory/'manifest.json')
    write_json(directory/'data_identity.json',identity_from(m,entries))
    entries['data_identity.json']=digest(directory/'data_identity.json')
    write_json(directory/'integrity.json',entries)

def prepare(root,dataset,output,chunk_size=100000):
    started=time.perf_counter(); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    reference=read_json(ROOT/'configs/sources.json')[dataset]
    if dataset=='cic2017':
        if digest(Path(root)/'CICIDS2017_improved.zip')!=reference['archive_sha256']:
            raise ValueError('Neočekivan SHA-256 glavne arhive')
    else:
        for name,s in reference['files'].items():
            p=Path(root)/'improved2018'/name
            if p.stat().st_size!=s['bytes'] or digest(p)!=s['sha256']: raise ValueError('Neočekivan SHA-256 CSV-a')
    output.mkdir(parents=True); stage=output/'_stage'; stage.mkdir()
    member=reference['normal_member']; n=reference['files'][member]['labels']['BENIGN']
    candidates=[x for x in reference['columns'] if x not in EXCLUDED]
    raw=np.lib.format.open_memmap(stage/'raw.npy',mode='w+',dtype='float64',shape=(n,len(candidates)))
    ids=np.empty(n,dtype=np.int64); ts=np.empty(n,dtype=np.int64); proto=np.empty(n,dtype=np.int64); at=0
    for c in stream(root,dataset,member,reference,chunk_size):
        c=c[c.Label=='BENIGN']; end=at+len(c)
        if end>n: raise ValueError('Previše normalnih tokova')
        raw[at:end]=c[candidates].to_numpy(dtype=float); ids[at:end]=c.id
        ts[at:end]=c.Timestamp.astype('int64'); proto[at:end]=c.Protocol; at=end
    if at!=n: raise ValueError('Nepotpun normalni blok')
    order,bounds=chronological_order(ts,ids)
    if [bounds[0],bounds[1]-bounds[0],n-bounds[1]]!=reference['schema']['split_counts']:
        raise ValueError('Promenjene granice')
    spec=training_spec(raw,proto,candidates,order,bounds[0],chunk_size)
    if spec['feature_names']!=reference['schema']['feature_names']: raise ValueError('Promenjena trening šema')
    labels=sorted({lab for s in reference['files'].values() for lab in s['labels']}); codes={lab:i for i,lab in enumerate(labels)}
    m={'dataset':dataset,'source':reference,'schema':spec,'labels':labels,'roles':{},'production':True,'chunk_size':chunk_size}
    scaler=StandardScaler(); edges=[0,*bounds,n]
    for k,role in enumerate(['train_normal','validation_normal','calibration_normal']):
        target=output/role; target.mkdir(); ix=order[edges[k]:edges[k+1]]
        x=np.lib.format.open_memmap(target/'X.npy',mode='w+',dtype='float64',shape=(len(ix),spec['input_dimension']))
        for start in range(0,len(ix),chunk_size):
            z=ix[start:start+chunk_size]; b=matrix(raw[z],proto[z],spec); x[start:start+len(z)]=b
            if role=='train_normal': scaler.partial_fit(b)
        x.flush(); del x
        save_meta(target/'meta.npz',ids[ix],ts[ix],np.full(len(ix),codes['BENIGN'],dtype=np.uint16),member)
        m['roles'][role]=[{'path':role,'rows':len(ix),'member':member,'min_timestamp_ns':int(ts[ix].min()),
            'max_timestamp_ns':int(ts[ix].max()),'unknown_protocols':int((~np.isin(proto[ix],spec['protocol_categories'])).sum())}]
    joblib.dump(scaler,output/'standard_scaler.joblib')
    del raw,ids,ts,proto,order; shutil.rmtree(stage)
    for role,members in reference['roles'].items():
        m['roles'][role]=[]
        for member in members:
            for i,c in enumerate(stream(root,dataset,member,reference,chunk_size)):
                if role=='external_diagnostic': c=c[c.Label!='BENIGN']
                if not len(c): continue
                rel=f'{role}/{member[:-4]}_{i:04d}'; target=output/rel; target.mkdir(parents=True)
                np.save(target/'X.npy',matrix(c[candidates].to_numpy(dtype=float),c.Protocol.to_numpy(),spec))
                save_meta(target/'meta.npz',c.id.to_numpy(dtype=np.int64),c.Timestamp.astype('int64').to_numpy(),c.Label.map(codes).to_numpy(dtype=np.uint16),member)
                m['roles'][role].append({'path':rel,'rows':len(c),'member':member,
                    'min_timestamp_ns':int(c.Timestamp.astype('int64').min()),'max_timestamp_ns':int(c.Timestamp.astype('int64').max()),
                    'unknown_protocols':int((~c.Protocol.isin(spec['protocol_categories'])).sum())})
            print('prepared',dataset,role,member,flush=True)
    m['preparation_seconds']=time.perf_counter()-started; m['environment']=environment()
    write_json(output/'manifest.json',m); seal_data(output); validate_prepared(output,False)
    return m

def validate_prepared(directory,check_hashes=True):
    directory=Path(directory)
    if check_hashes: verify(directory)
    m=read_json(directory/'manifest.json')
    if not m['production']: raise ValueError('Nisu produkcioni podaci')
    if read_json(directory/'data_identity.json')!=identity_from(m,read_json(directory/'integrity.json')):
        raise ValueError('Nesaglasan identitet podataka')
    seen={}; last=None
    for role in ['train_normal','validation_normal','calibration_normal']:
        parts=m['roles'][role]
        if len(parts)!=1: raise ValueError('Neispravan normalni blok')
        p=parts[0]
        if last is not None and p['min_timestamp_ns']<=last: raise ValueError('Vremensko preklapanje')
        last=p['max_timestamp_ns']
    for role,parts in m['roles'].items():
        for p in parts:
            x=np.load(directory/p['path']/'X.npy',mmap_mode='r')
            with np.load(directory/p['path']/'meta.npz',allow_pickle=False) as z:
                ids=z['source_id']; ts=z['timestamp_ns']; lc=z['label_code']; day=str(z['day'])
                if x.shape!=(p['rows'],m['schema']['input_dimension']) or any(len(a)!=len(x) for a in [ids,ts,lc]): raise ValueError('Neispravne dimenzije')
                if day!=p['member'] or len(np.unique(ids))!=len(ids): raise ValueError('Nesaglasan identitet člana')
                prior=seen.setdefault(day,set()); current=set(map(int,ids))
                if current&prior: raise ValueError('Preklapanje identiteta između blokova')
                prior.update(current)
                if role.endswith('_normal'):
                    if not np.all(lc==m['labels'].index('BENIGN')): raise ValueError('Napad u normalnom bloku')
                    if np.any(np.diff(ts)<0): raise ValueError('Nehoronološki normalni blok')
                if role in ('final_test','external_test') and int(ts.min())<=last: raise ValueError('Test nije kasniji od kalibracije')
            for start in range(0,len(x),100000): finite_array(x[start:start+100000])
    return m

def iter_role(directory,manifest,role):
    if role not in manifest['roles']: raise ValueError('Nedozvoljena uloga')
    for part in manifest['roles'][role]:
        p=Path(directory)/part['path']
        with np.load(p/'meta.npz',allow_pickle=False) as meta:
            yield np.load(p/'X.npy',mmap_mode='r'),{k:meta[k] for k in meta.files}
