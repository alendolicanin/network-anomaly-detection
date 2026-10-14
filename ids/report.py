from pathlib import Path
import csv
import numpy as np
from .core import read_json,write_json,verify,digest,seal
from .pipeline import load_decision
from .metrics import paired_statistics

METRICS=['tn','fp','fn','tp','precision','recall','f1','fpr','specificity','balanced_accuracy','accuracy','roc_auc','average_precision','macro_recall']

def csv_rows(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def collect(experiment,cfg,external=False,diagnostic=False):
    d,dh=load_decision(experiment,cfg); base=Path(experiment)/'external' if external else Path(experiment)
    role=('external_diagnostic' if diagnostic else 'external_test') if external else 'final_test'
    models=[*sorted({'ae_l16','ae_l32',d['architecture']}),'iforest']
    if not external: models += ['robust_'+d['architecture'],'pca']
    rows=[]; classes=[]; days=[]
    for model in models:
        for seed in ([None] if model=='pca' else cfg['seeds']):
            name=model if seed is None else f'{model}_seed{seed}'; fit=base/'fits'/name; ev=base/role/name
            verify(fit); verify(ev)
            expected={'role':role,'decision_hash':dh,'run_hash':digest(fit/'integrity.json')}
            if read_json(ev/'evaluation.json')!=expected: raise ValueError('Pogrešan evaluacioni izlaz')
            metrics=read_json(ev/'metrics.json')
            if set(metrics)!=set(map(str,cfg['percentiles'])): raise ValueError('Nepotpuni pragovi')
            for q in cfg['percentiles']:
                v=metrics[str(q)]; key={'model':model,'seed':seed,'percentile':q}
                for policy in ['primary','exclude_attempted']:
                    values={k:v['macro_recall'] if k=='macro_recall' else v[policy][k] for k in METRICS}
                    rows.append({**key,'policy':policy,**values})
                for label,r in v['per_label'].items(): classes.append({**key,'label':label,**r})
                for day,r in v['per_day'].items(): days.append({**key,'day':day,**r})
    return d,dh,base,role,rows,classes,days

def summarize(experiment,cfg,external=False,diagnostic=False):
    d,dh,base,role,rows,classes,days=collect(experiment,cfg,external,diagnostic)
    out=base/('summary_'+role)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); csv_rows(out/'per_run.csv',rows); csv_rows(out/'per_class.csv',classes); csv_rows(out/'per_day.csv',days)
    aggregate=[]
    for model in sorted({r['model'] for r in rows}):
        for q in cfg['percentiles']:
            for policy in ['primary','exclude_attempted']:
                group=[r for r in rows if r['model']==model and r['percentile']==q and r['policy']==policy]
                expected=[None] if model=='pca' else cfg['seeds']
                if [r['seed'] for r in group]!=expected: raise ValueError('Nedostajuća ili duplirana ponavljanja')
                row={'model':model,'percentile':q,'policy':policy,'n_runs':len(group)}
                for metric in METRICS:
                    vals=[r[metric] for r in group]; defined=all(x is not None for x in vals)
                    row[metric+'_mean']=float(np.mean(vals)) if defined else None
                    row[metric+'_sd']=float(np.std(vals,ddof=1)) if defined and len(vals)>1 else None
                aggregate.append(row)
    csv_rows(out/'mean_sd.csv',aggregate); paired={}
    for q in [95,99]:
        for left,right in [('ae_l32','ae_l16'),(d['architecture'],'iforest')]:
            for policy in ['primary','exclude_attempted']:
                def values(model):
                    return [next(r['f1'] for r in rows if r['model']==model and r['seed']==s and r['percentile']==q and r['policy']==policy) for s in cfg['seeds']]
                a,b=values(left),values(right)
                paired[f'{left}_minus_{right}_p{q}_{policy}']=paired_statistics(np.asarray(a)-b) if None not in a+b else {'undefined':True}
    write_json(out/'paired_statistics.json',paired)
    write_json(out/'selected_operating_points.json',{'architecture':d['architecture'],'percentile_ae':d['percentile_ae'],
        'percentile_if':d['percentile_if'],'reference_percentiles':[95,99],'decision_hash':dh,'scope':role,
        'note':'Ostali percentili su analiza osetljivosti; nema ponovnog izbora na testu.'})
    seal(out); return out

def figures(experiment,cfg,external=False,diagnostic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from sklearn.metrics import roc_curve,precision_recall_curve
    d,dh,base,role,rows,classes,days=collect(experiment,cfg,external,diagnostic)
    out=base/('figures_'+role)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    def save(fig,name):
        for ax in fig.axes:
            for axis in [ax.xaxis,ax.yaxis]: axis.set_major_formatter(FuncFormatter(lambda x,pos: f'{x:g}'.replace('.',',')))
            ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/name); plt.close(fig)
    roc,ra=plt.subplots(); pr,pa=plt.subplots()
    for model in [d['architecture'],'iforest']:
        with np.load(base/role/f'{model}_seed42'/f'scores_{role}.npz',allow_pickle=False) as z:
            scores=z['scores']; labels=z['labels'][z['label_code']]
        y=(labels!='BENIGN')&(np.char.find(labels.astype(str),'Attempted')<0)
        if y.any() and (~y).any():
            fpr,tpr,_=roc_curve(y,scores); ra.plot(fpr,tpr,label=model)
        if y.any():
            precision,recall,_=precision_recall_curve(y,scores); pa.plot(recall,precision,label=model)
        fig,ax=plt.subplots(); transformed=np.log10(np.maximum(scores,np.finfo(float).tiny))
        for mask,label in [(~y,'BENIGN i Attempted'),(y,'Efektivni napadi')]:
            if mask.any(): ax.hist(transformed[mask],bins=80,density=True,alpha=.45,label=label)
        cuts=read_json(base/'fits'/f'{model}_seed42'/'thresholds.json')
        for q in [95,99]: ax.axvline(np.log10(max(cuts[str(q)],np.finfo(float).tiny)),label=f'p{q}',linestyle='--')
        ax.set(xlabel='log10(skor)',ylabel='Gustina',title=f'{model}, seed 42'); ax.legend(); save(fig,f'scores_{model}.pdf')
    ra.set(xlabel='FPR',ylabel='Odziv',title='ROC — seed 42'); ra.legend(); save(roc,'roc_seed42.pdf')
    pa.set(xlabel='Odziv',ylabel='Preciznost',title='PR — seed 42'); pa.legend(); save(pr,'pr_seed42.pdf')
    for model in sorted({'ae_l32',d['architecture']}):
        hist=read_json(base/'fits'/f'{model}_seed42'/'training_history.json'); fig,ax=plt.subplots()
        for metric,label in [('train_reconstruction_mse','Trening'),('validation_reconstruction_mse','Validacija')]:
            ax.plot([r['epoch'] for r in hist],[r[metric] for r in hist],label=label)
        ax.set(xlabel='Epoha',ylabel='Rekonstrukcioni MSE',title=f'{model}, seed 42'); ax.legend(); save(fig,f'training_{model}.pdf')
    fig,ax=plt.subplots()
    for model in [d['architecture'],'iforest']:
        means=[np.mean([r['f1'] for r in rows if r['model']==model and r['percentile']==q and r['policy']=='primary']) for q in cfg['percentiles']]
        ax.plot(cfg['percentiles'],means,marker='o',label=model)
    ax.set(xlabel='Percentil',ylabel='Srednji F1',title='Osetljivost na prag — pet seed vrednosti'); ax.legend(); save(fig,'percentile_f1.pdf')
    write_json(out/'figure_context.json',{'decision_hash':dh,'role':role,'illustrative_seed':42,'policy':'primary'}); seal(out); return out
