import urllib.request,urllib.parse,concurrent.futures,hashlib,json,struct,zlib,zipfile,time
from pathlib import Path
D=Path('data_sources');E=Path('tmp/data_audit');D.mkdir(exist_ok=True);E.mkdir(parents=True,exist_ok=True)
U='https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/'
SIZE=10426851729
ETAG='"26d7d2191-5f86fb8b66b7b"'
def read_range(a,b):
 for attempt in range(3):
  try:
   req=urllib.request.Request(U+'CSECICIDS2018_improved.zip',headers={'Range':f'bytes={a}-{b}','If-Range':ETAG})
   with urllib.request.urlopen(req,timeout=90) as r:
    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{SIZE}':raise RuntimeError('Range/version mismatch')
    v=r.read()
   if len(v)!=b-a+1:raise RuntimeError('Length mismatch')
   return v
  except Exception:
   if attempt==2:raise
print('Fetching index',flush=True)
tail=read_range(SIZE-131072,SIZE-1)
with (E/'index.sparse').open('wb') as f:f.seek(SIZE-len(tail));f.write(tail)
with zipfile.ZipFile(E/'index.sparse') as z:
 info=[dict(name=x.filename,size=x.file_size,compressed=x.compress_size,offset=x.header_offset,crc=x.CRC) for x in z.infolist()]
(E/'improved2018_members.json').write_text(json.dumps(info,indent=2))
wanted=[x for x in info if x['name'] in ['Wednesday-14-02-2018.csv','Thursday-15-02-2018.csv']]
jobs=[]
for x in wanted:
 h=read_range(x['offset'],x['offset']+255);n,e=struct.unpack_from('<HH',h,26)
 assert h[:4]==b'PK\x03\x04'
 x['start']=x['offset']+30+n+e
 x['parts']=[]
 for i,off in enumerate(range(0,x['compressed'],16*1024*1024)):
  a=x['start']+off;b=min(a+16*1024*1024-1,x['start']+x['compressed']-1)
  p=E/(x['name']+f'.part{i:03d}')
  x['parts'].append(p);jobs.append((a,b,p))
def part(j):
 a,b,p=j;v=read_range(a,b);p.write_bytes(v);return len(v)
with concurrent.futures.ThreadPoolExecutor(8) as pool:
 n=0
 for f in concurrent.futures.as_completed([pool.submit(part,j) for j in jobs]):
  n+=f.result()
  if n%(64*1024*1024)<16*1024*1024:print('Compressed downloaded MB',n//1048576,flush=True)
for x in wanted:
 h=hashlib.sha256();ch=hashlib.sha256();crc=0;n=0;dec=zlib.decompressobj(-15)
 out=D/'improved2018'/x['name'];out.parent.mkdir(exist_ok=True)
 with out.open('wb') as f:
  for p in x['parts']:
   with p.open('rb') as g:
    while b:=g.read(1024*1024):
     ch.update(b);v=dec.decompress(b);f.write(v);h.update(v);crc=zlib.crc32(v,crc);n+=len(v)
   p.unlink()
  v=dec.flush();f.write(v);h.update(v);crc=zlib.crc32(v,crc);n+=len(v)
 assert n==x['size'] and crc==x['crc'] and dec.eof
 result={k:v for k,v in x.items() if k!='parts'}|{'sha256':h.hexdigest(),'compressed_sha256':ch.hexdigest(),'crc_verified':True,'source_url':U+'CSECICIDS2018_improved.zip','archive_size':SIZE,'archive_etag':ETAG,'archive_sha256':None,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (E/(x['name']+'.download.json')).write_text(json.dumps(result,indent=2));print('DONE',result,flush=True)

(E/'index.sparse').unlink(missing_ok=True)
