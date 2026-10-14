import itertools
import numpy as np
from scipy.stats import rankdata, t as student_t, binomtest
from sklearn.metrics import roc_auc_score, average_precision_score
from .core import finite_array

def thresholds(scores, percentiles):
    scores = np.asarray(scores,dtype=np.float64)
    finite_array(scores)
    if not len(scores): raise ValueError('Prazna kalibracija')
    return {str(q):float(np.percentile(scores,q,method='linear')) for q in percentiles}

def ranking_metrics(y,scores):
    n0=int((y==0).sum()); n1=int((y==1).sum())
    return {'roc_auc':float(roc_auc_score(y,scores)) if n0 and n1 else None,
        'average_precision':float(average_precision_score(y,scores)) if n1 else None}

def binary_metrics(y,scores,threshold,ranking=None):
    y=np.asarray(y); scores=np.asarray(scores)
    if y.shape!=scores.shape or y.ndim!=1 or not np.isin(y,[0,1]).all():
        raise ValueError('Neispravne oznake ili dimenzije')
    finite_array(scores)
    if not np.isfinite(threshold): raise ValueError('Neispravan prag')
    p=scores>threshold
    tn=int(((y==0)&~p).sum()); fp=int(((y==0)&p).sum())
    fn=int(((y==1)&~p).sum()); tp=int(((y==1)&p).sum())
    n0=tn+fp; n1=fn+tp; recall=tp/n1 if n1 else None; fpr=fp/n0 if n0 else None
    return {'rows':len(y),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'accuracy':(tn+tp)/len(y) if len(y) else None,
        'precision':tp/(tp+fp) if tp+fp else 0.,'precision_zero_convention':tp+fp==0,
        'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'fpr':fpr,'specificity':1-fpr if n0 else None,
        'balanced_accuracy':(recall+1-fpr)/2 if n0 and n1 else None,
        **(ranking_metrics(y,scores) if ranking is None else ranking)}

def evaluate(y,scores,attempted,labels,days,threshold,rankings=None):
    y=np.asarray(y); scores=np.asarray(scores); attempted=np.asarray(attempted,dtype=bool)
    labels=np.asarray(labels); days=np.asarray(days); out={}
    for policy,mask in [('primary',np.ones(len(y),bool)),('exclude_attempted',~attempted)]:
        out[policy]=binary_metrics(y[mask],scores[mask],threshold,None if rankings is None else rankings[policy])
    classes={}
    for label in np.unique(labels):
        m=labels==label
        classes[str(label)]={'n':int(m.sum()),'alarm_rate':float(np.mean(scores[m]>threshold)),
            'effective_attack':bool(np.all(y[m]==1))}
    rates=[v['alarm_rate'] for v in classes.values() if v['effective_attack']]
    out['macro_recall']=float(np.mean(rates)) if rates else None
    out['per_label']=classes; out['per_day']={}
    for day in np.unique(days):
        m=days==day; neg=m&(y==0); benign=neg&~attempted; att=m&attempted
        out['per_day'][str(day)]={'primary_negative_n':int(neg.sum()),
            'primary_fpr':float(np.mean(scores[neg]>threshold)) if neg.any() else None,
            'benign_n':int(benign.sum()),'benign_fpr':float(np.mean(scores[benign]>threshold)) if benign.any() else None,
            'attempted_n':int(att.sum()),'attempted_alarm_rate':float(np.mean(scores[att]>threshold)) if att.any() else None}
    return out

def choose_architecture(rows,seeds):
    ranking=[]
    for name in sorted({r['architecture'] for r in rows}):
        g=[r for r in rows if r['architecture']==name]
        if sorted(r['seed'] for r in g)!=sorted(seeds): raise ValueError('Nedostajući ili duplirani seed')
        if any(r['role']!='development' or r['percentile']!=95 for r in g): raise ValueError('Izbor zahteva razvojni p95')
        f1=float(np.mean([r['f1'] for r in g])); fpr=float(np.mean([r['fpr'] for r in g]))
        finite_array(np.array([f1,fpr]))
        if len({r['parameters'] for r in g})!=1: raise ValueError('Nejednaki modeli iste konfiguracije')
        ranking.append({'architecture':name,'mean_f1':f1,'mean_fpr':fpr,'parameters':g[0]['parameters']})
    ranking.sort(key=lambda r:(-round(r['mean_f1'],12),round(r['mean_fpr'],12),r['parameters'],r['architecture']))
    return ranking[0]['architecture'],ranking

def choose_percentile(rows,seeds,percentiles):
    if len(rows)!=len(seeds)*len(percentiles): raise ValueError('Nepotpun niz pragova')
    ranking=[]
    for q in percentiles:
        g=[r for r in rows if r['percentile']==q]
        if sorted(r['seed'] for r in g)!=sorted(seeds) or any(r['role']!='development' for r in g):
            raise ValueError('Izbor zahteva potpune razvojne rezultate')
        f1=float(np.mean([r['f1'] for r in g])); fpr=float(np.mean([r['fpr'] for r in g]))
        finite_array(np.array([f1,fpr]))
        ranking.append({'percentile':q,'mean_f1':f1,'mean_fpr':fpr})
    ranking.sort(key=lambda r:(-round(r['mean_f1'],12),round(r['mean_fpr'],12),-r['percentile']))
    return ranking[0]['percentile'],ranking

def paired_statistics(differences):
    d=np.asarray(differences,dtype=float); finite_array(d)
    if len(d)!=5: raise ValueError('Potrebno je svih pet uparenih razlika')
    mean=float(d.mean()); sd=float(d.std(ddof=1)); margin=float(student_t.ppf(.975,4)*sd/np.sqrt(5))
    nz=d[d!=0]; m=len(nz)
    if m:
        ranks=rankdata(abs(nz),method='average'); obs=float(ranks[nz>0].sum()); mid=float(ranks.sum()/2)
        sums=[float(np.dot(bits,ranks)) for bits in itertools.product([0,1],repeat=m)]
        p=sum(abs(v-mid)>=abs(obs-mid)-1e-12 for v in sums)/len(sums)
        sign=float(binomtest(int((nz>0).sum()),m,.5).pvalue)
    else: p=sign=1.
    return {'differences':d.tolist(),'mean':mean,'sd_ddof1':sd,'ci95_t':[mean-margin,mean+margin],
        'wilcoxon_enumerated_two_sided_p':p,'sign_two_sided_p':sign,'nonzero_pairs':m}
