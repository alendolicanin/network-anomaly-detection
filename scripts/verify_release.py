from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())
for rel,entry in manifest['files'].items():
    p=root/rel
    if not p.is_file() or p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError('Promenjena ili nedostajuća datoteka: '+rel)
for area in ['ids','scripts','tests']:
    for p in (root/area).glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
print('Proveren paket:',len(manifest['files']),'datoteka; Python sintaksa ispravna.')
