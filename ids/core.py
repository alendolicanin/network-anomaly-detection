from pathlib import Path
import hashlib
import json
import os
import platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def config(path=None):
    c = read_json(path or ROOT/'configs/experiment.json')
    expected_arch = {'ae_l4':[64,32,4,32,64], 'ae_l8':[64,32,8,32,64],
        'ae_l16':[64,32,16,32,64], 'ae_l32':[64,32,32,32,64],
        'ae_compact_l16':[32,16,32], 'ae_wide_l16':[128,64,32,16,32,64,128]}
    expected_ae = {'max_epochs':200,'patience':8,'min_delta':1e-5,'batch_size':512,
        'learning_rate_init':.001,'alpha':1e-5,'shuffle':False}
    expected_if = {'n_estimators':300,'max_samples':4096,'max_features':1.,'bootstrap':False,'contamination':'auto'}
    if (c['seeds'] != [7,21,42,84,123]
        or c['percentiles'] != [90,92,94,95,96,97,98,99,99.5,99.9]
        or c['architectures'] != expected_arch or c['ae'] != expected_ae or c['iforest'] != expected_if):
        raise ValueError('Parametri se razlikuju od definisanog eksperimentalnog postupka')
    if type(c['threads']) is not int or c['threads'] < 1 or c['inference_batch_size'] < 1:
        raise ValueError('Neispravni resursni parametri')
    return c

def source_hash():
    return object_hash({str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'ids').glob('*.py'))})

def data_fingerprint(directory):
    return digest(Path(directory)/'data_identity.json')

def runtime_versions():
    return {n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl']}

def environment():
    from threadpoolctl import threadpool_info
    cpu = platform.processor()
    p = Path('/proc/cpuinfo')
    if p.exists():
        cpu = next((s.split(':',1)[1].strip() for s in p.read_text().splitlines() if s.startswith('model name')), cpu)
    return {'utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
        'os':platform.platform(), 'cpu':cpu, 'logical_cpus':os.cpu_count(),
        'ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES') if hasattr(os,'sysconf') else None,
        'libraries':runtime_versions(), 'threadpools':threadpool_info()}

def seal(directory, name='integrity.json'):
    directory = Path(directory)
    entries = {str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p.name != name and not p.name.endswith('.tmp')
        and not any(s.startswith('.') for s in p.relative_to(directory).parts)}
    write_json(directory/name, entries)
    return digest(directory/name)

def verify(directory, name='integrity.json'):
    directory = Path(directory)
    for rel, expected in read_json(directory/name).items():
        p = directory/rel
        if not p.is_file() or digest(p) != expected:
            raise ValueError('Promenjen ili nedostajući izlaz: '+rel)

def finite_array(a):
    if not np.isfinite(a).all():
        raise ValueError('Nekonačne numeričke vrednosti')
