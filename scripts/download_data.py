"""Download the source files specified by the configured checksums."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,urllib.request,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['cic2017','cic2018'],required=True); p.add_argument('--root',default='.'); a=p.parse_args()
    root=Path(a.root).resolve(); root.mkdir(parents=True,exist_ok=True); source=json.loads((ROOT/'configs/sources.json').read_text())[a.dataset]
    data=root/'data_sources'; data.mkdir(exist_ok=True)
    if a.dataset=='cic2017':
        target=data/'CICIDS2017_improved.zip'
        if not target.exists():
            temporary=target.with_suffix('.part')
            with urllib.request.urlopen(source['source_url'],timeout=120) as r, temporary.open('wb') as f: shutil.copyfileobj(r,f)
            temporary.replace(target)
        if sha(target)!=source['archive_sha256']: raise ValueError('SHA-256 arhive se ne podudara; datoteka nije prepisana')
    else:
        if not all((data/'improved2018'/name).exists() for name in source['files']):
            subprocess.run([sys.executable,str(ROOT/'scripts/download_improved2018_members.py')],cwd=root,check=True)
        for name,r in source['files'].items():
            target=data/'improved2018'/name
            if target.stat().st_size!=r['bytes'] or sha(target)!=r['sha256']: raise ValueError('SHA-256 člana se ne podudara: '+name)
    (data/(a.dataset+'_verified.json')).write_text(json.dumps({'dataset':a.dataset,'verified_utc':datetime.now(timezone.utc).isoformat(),'source':source},indent=2))
    print('SHA-256 provera uspešna:',a.dataset)
if __name__=='__main__': main()
