"""Validate development outputs and export metrics and figures."""
from pathlib import Path
import sys,json,shutil,csv
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Validate development runs and export summaries')
parser.add_argument('--code',type=Path,required=True,help='Repository root containing the ids package')
parser.add_argument('--experiment',type=Path,required=True,help='Direktorijum eksperimenta')
parser.add_argument('--output',type=Path,required=True,help='Direktorijum zbirnih izlaza')
args=parser.parse_args()
sys.path.insert(0,str(args.code.resolve()))
from ids.core import config,read_json,write_json,verify,digest,source_hash,environment
from ids.pipeline import load_decision
from ids.metrics import binary_metrics
EXP=args.experiment.resolve(); OUT=args.output.resolve()

def csv_write(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
 cfg=config(); d,dh=load_decision(EXP,cfg)
 if (EXP/'final_test').exists() or (EXP/'external').exists(): raise ValueError('Neočekivani završni rezultati u razvojnom koraku')
 OUT.mkdir(parents=True,exist_ok=True); rows=[]; runs=[]; class_rows=[]; day_rows=[]; checks=0
 all_metrics=['tn','fp','fn','tp','accuracy','precision','recall','f1','fpr','specificity','balanced_accuracy','roc_auc','average_precision']
 for arch in [*cfg['architectures'],'iforest']:
  for seed in cfg['seeds']:
   name=f'{arch}_seed{seed}'; fit=EXP/'fits'/name; ev=EXP/'development'/name; verify(fit);verify(ev)
   r=read_json(fit/'run.json'); vals=read_json(ev/'metrics.json'); cut=read_json(fit/'thresholds.json'); timing=read_json(ev/'timing_development.json')
   if r['identity']['seed']!=seed or r['identity']['architecture']!=arch or r['status']!='complete': raise ValueError(name)
   with np.load(ev/'scores_development.npz',allow_pickle=False) as z:
    scores=z['scores']; labels=z['labels'][z['label_code']]; attempted=np.char.find(labels.astype(str),'Attempted')>=0; y=(labels!='BENIGN')&~attempted
    if len(scores)!=818718 or not np.isfinite(scores).all(): raise ValueError('Neispravni razvojni skorovi')
    if len(np.unique(z['members']))!=2: raise ValueError('Pogrešan broj razvojnih članova')
   with np.load(fit/'scores_calibration_normal.npz',allow_pickle=False) as z:
    cal=z['scores']; recalculated=np.percentile(cal,cfg['percentiles'],method='linear')
   np.testing.assert_allclose([cut[str(q)] for q in cfg['percentiles']],recalculated,rtol=0,atol=0); checks+=len(cut)
   for q in cfg['percentiles']:
    v=vals[str(q)]
    for policy,mask in [('primary',np.ones(len(y),dtype=bool)),('exclude_attempted',~attempted)]:
     predicted=scores[mask]>cut[str(q)]; truth=y[mask]
     tn=int((~truth&~predicted).sum()); fp=int((~truth&predicted).sum()); fn=int((truth&~predicted).sum()); tp=int((truth&predicted).sum())
     assert [v[policy][k] for k in ['tn','fp','fn','tp']]==[tn,fp,fn,tp]
     expected={'f1':2*tp/(2*tp+fp+fn),'fpr':fp/(fp+tn),'recall':tp/(tp+fn)}
     for metric,value in expected.items(): np.testing.assert_allclose(v[policy][metric],value,rtol=0,atol=1e-14)
     checks+=7
     rows.append({'architecture':arch,'seed':seed,'percentile':q,'threshold':cut[str(q)],'policy':policy,
                  **{k:v[policy][k] for k in all_metrics},'macro_recall':v['macro_recall']})
    macro=np.mean([v['per_label'][lab]['alarm_rate'] for lab in np.unique(labels[y])]); np.testing.assert_allclose(v['macro_recall'],macro,rtol=0,atol=1e-14);checks+=1
    for label,stats in v['per_label'].items(): class_rows.append({'architecture':arch,'seed':seed,'percentile':q,'label':label,**stats})
    for day,stats in v['per_day'].items():day_rows.append({'architecture':arch,'seed':seed,'percentile':q,'day':day,**stats})
   training=r['training']; history=read_json(fit/'training_history.json')
   if arch!='iforest':
    with np.load(fit/'scores_validation_normal.npz',allow_pickle=False) as z: val_mse=float(z['scores'].mean())
    best=training['best_epoch']; np.testing.assert_allclose(val_mse,history[best-1]['validation_reconstruction_mse'],rtol=1e-10,atol=1e-12)
    assert training['epochs_ran']==len(history) and 1<=best<=len(history)<=200;checks+=2
   runs.append({'architecture':arch,'seed':seed,'parameters':training['parameters'],'best_epoch':training['best_epoch'],'epochs_ran':training['epochs_ran'],
    'fit_seconds':training['fit_seconds'],'monitoring_seconds':training['monitoring_seconds'],'training_with_monitoring_seconds':training['training_with_monitoring_seconds'],
    'inference_seconds_development':timing['inference_seconds'],'scaling_seconds_development':timing['scaling_seconds'],'development_rows_per_second':timing['inference_rows_per_second'],
    'model_bytes':(fit/'model.joblib').stat().st_size,'cpu':r['environment']['cpu'],'logical_cpus':r['environment']['logical_cpus'],'ram_bytes':r['environment']['ram_bytes'],'python':r['environment']['python'],'execution_utc':r['environment']['utc'],'identity_hash':digest(fit/'integrity.json')})
 aggregate=[]
 for arch in [*cfg['architectures'],'iforest']:
  for q in cfg['percentiles']:
   for policy in ['primary','exclude_attempted']:
    group=[r for r in rows if r['architecture']==arch and r['percentile']==q and r['policy']==policy]
    assert [r['seed'] for r in group]==cfg['seeds']; row={'architecture':arch,'percentile':q,'policy':policy,'n_runs':5}
    for metric in [*all_metrics,'macro_recall']:
     a=[r[metric] for r in group]; row[metric+'_mean']=float(np.mean(a)) if None not in a else None;row[metric+'_sd']=float(np.std(a,ddof=1)) if None not in a else None
    aggregate.append(row)
 ranking=sorted([r for r in aggregate if r['percentile']==95 and r['policy']=='primary' and r['architecture']!='iforest'],
  key=lambda r:(-round(r['f1_mean'],12),round(r['fpr_mean'],12),next(x['parameters'] for x in runs if x['architecture']==r['architecture']),r['architecture']))
 assert ranking[0]['architecture']==d['architecture']; checks+=1
 for arch,key in [(d['architecture'],'percentile_ae'),('iforest','percentile_if')]:
  rr=sorted([r for r in aggregate if r['architecture']==arch and r['policy']=='primary'],key=lambda r:(-round(r['f1_mean'],12),round(r['fpr_mean'],12),-r['percentile']))
  assert rr[0]['percentile']==d[key]; checks+=1
 csv_write(OUT/'development_per_run.csv',rows);csv_write(OUT/'development_mean_sd.csv',aggregate);csv_write(OUT/'training_runs.csv',runs)
 csv_write(OUT/'development_per_class.csv',class_rows);csv_write(OUT/'development_per_day.csv',day_rows)
 csv_write(OUT/'architecture_ranking_p95.csv',ranking)
 for filename in ['decision.json','decision.sha256.json']:shutil.copy2(EXP/filename,OUT/filename)
 write_json(OUT/'verification.json',{'status':'passed','checks':checks,'unique_training_runs':len(runs),'code_hash':source_hash(),'decision_hash':dh,
  'scope':'development only','final_test_metrics_created':False,'calibration_percentiles_recomputed':True,'confusion_matrices_recomputed_all_percentiles_both_policies':True,
  'saved_best_validation_mse_verified':True,'independent_selection_verified':True,'environment':environment()})
 plots(d,aggregate)
 print('VERIFIED',len(runs),'runs;',checks,'checks;',d['architecture'],d['percentile_ae'],d['percentile_if'])

def plots(d,aggregate):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import FuncFormatter
 def save(fig,name):
  for ax in fig.axes:
   ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')));ax.grid(axis='y',alpha=.2)
  fig.tight_layout();fig.savefig(OUT/name);plt.close(fig)
 cfg=config();fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 names=list(cfg['architectures'])
 for ax,q in zip(axes,[95,99]):
  r=[next(r for r in aggregate if r['architecture']==a and r['percentile']==q and r['policy']=='primary') for a in names]
  ax.errorbar(np.arange(len(r)),[x['f1_mean'] for x in r],yerr=[x['f1_sd'] for x in r],fmt='o',capsize=4)
  ax.set_xticks(np.arange(len(r)),names,rotation=40,ha='right');ax.set(title=f'Razvojni p{q}',ylabel='F1 — sredina ± SD',ylim=(0,1))
 save(fig,'development_architecture_f1.pdf')
 fig,axes=plt.subplots(1,3,figsize=(12,3.7))
 for ax,metric,label in zip(axes,['f1','recall','fpr'],['F1','Odziv','FPR']):
  for arch in [d['architecture'],'iforest']:
   r=[r for r in aggregate if r['architecture']==arch and r['policy']=='primary']
   ax.plot([x['percentile'] for x in r],[x[metric+'_mean'] for x in r],marker='o',label=arch)
  ax.set(xlabel='Percentil',ylabel=label,title='Razvoj — pet seed vrednosti');ax.legend(fontsize=8)
 save(fig,'development_thresholds.pdf')
 for arch in sorted({'ae_l32',d['architecture']}):
  h=read_json(EXP/'fits'/f'{arch}_seed42'/'training_history.json');fig,ax=plt.subplots()
  for metric,label in [('train_reconstruction_mse','Trening'),('validation_reconstruction_mse','Validacija')]:ax.plot([x['epoch'] for x in h],[x[metric] for x in h],label=label)
  ax.set(xlabel='Epoha',ylabel='Rekonstrukcioni MSE',title=f'{arch} — seed 42');ax.legend();save(fig,f'training_{arch}_seed42.pdf')

if __name__=='__main__': main()
