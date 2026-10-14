import argparse
from .core import config

def main():
    p=argparse.ArgumentParser(description='Network anomaly detection experiments'); p.add_argument('--config')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--source-root',required=True); q.add_argument('--dataset',choices=['cic2017','cic2018'],required=True); q.add_argument('--output',required=True); q.add_argument('--chunk-size',type=int,default=100000)
    q=sub.add_parser('verify-data'); q.add_argument('--data',required=True)
    q=sub.add_parser('pilot'); q.add_argument('--data',required=True); q.add_argument('--output',required=True)
    for name in ['develop','final','external']:
        q=sub.add_parser(name); q.add_argument('--data',required=True); q.add_argument('--experiment',required=True)
    for name in ['summarize','figures']:
        q=sub.add_parser(name); q.add_argument('--experiment',required=True); q.add_argument('--external',action='store_true'); q.add_argument('--diagnostic',action='store_true')
    a=p.parse_args(); cfg=config(a.config)
    if a.command=='prepare':
        from .data import prepare
        prepare(a.source_root,a.dataset,a.output,a.chunk_size)
    elif a.command=='verify-data':
        from .data import validate_prepared
        m=validate_prepared(a.data); print('Provera uspešna:',m['dataset'])
    elif a.command=='pilot':
        from .pilot import pilot
        pilot(a.data,a.output,cfg)
    elif a.command in ['develop','final','external']:
        from .pipeline import develop,final_main,external
        {'develop':develop,'final':final_main,'external':external}[a.command](a.data,a.experiment,cfg)
    else:
        if a.diagnostic and not a.external: p.error('--diagnostic zahteva --external')
        from .report import summarize,figures
        print({'summarize':summarize,'figures':figures}[a.command](a.experiment,cfg,a.external,a.diagnostic))

if __name__=='__main__': main()
