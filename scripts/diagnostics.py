"""Export latent-activity and reconstruction-residual tables."""
from pathlib import Path
import argparse,csv,json
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--external',action='store_true');a=ap.parse_args()
def read(p):return json.loads(p.read_text())
def save(name,rows):
 if not rows:return
 with (a.output/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 d=read(a.experiment/'decision.json');base=a.experiment/('external' if a.external else '')
 features=read(a.data/'manifest.json')['schema']['feature_names'];latent=[];residual=[];score_rows=[]
 models=sorted({d['architecture'],'ae_l32'})
 if not a.external:models+=['robust_'+d['architecture']]
 for model in models:
  for seed in [7,21,42,84,123]:
   name=f'{model}_seed{seed}'
   roles=['validation_normal','external_test','external_diagnostic'] if a.external else ['validation_normal','final_test']+([] if model.startswith('robust_') else ['development'])
   for role in roles:
    folder=base/'fits'/name if role=='validation_normal' else base/role/name
    diag=read(folder/f'diagnostics_{role}.json')
    for label,v in diag['latent'].items():
     rates=np.array(v['positive_fraction_per_unit']);latent.append({'model':model,'seed':seed,'role':role,'label':label,'n':v['n'],
       'latent_dimension':len(rates),'mean_active_units':v['mean_active_units'],'mean_active_fraction':float(rates.mean()),
       'units_never_active':int((rates==0).sum()),'units_active_below_one_percent':int((rates<.01).sum())})
    for label,v in diag['residuals'].items():
     values=np.array(v['mean_squared_residual']);order=np.argsort(-values);total=values.sum()
     for rank,j in enumerate(order,1):
      residual.append({'model':model,'seed':seed,'role':role,'label':label,'n':v['n'],'feature':features[j],'rank_by_mean':rank,
         'mean_squared_residual':float(values[j]),'median_squared_residual':v['median_squared_residual'][j],
         'share_of_total_mean_squared_residual':float(values[j]/total) if total else None})
    with np.load(folder/f'scores_{role}.npz',allow_pickle=False) as z:
     scores=z['scores'];codes=z['label_code'];labels=z['labels']
     for code in np.unique(codes):
      v=scores[codes==code];score_rows.append({'model':model,'seed':seed,'role':role,'label':str(labels[code]),'n':len(v),
       'mean_score':float(v.mean()),**{f'score_p{q}':float(np.percentile(v,q,method='linear')) for q in [25,50,75,95,99]}})
 a.output.mkdir(parents=True,exist_ok=True);save('latent_activity.csv',latent);save('residuals_by_feature.csv',residual);save('score_distributions_by_label.csv',score_rows)
 print('DIAGNOSTICS',len(latent),len(residual),len(score_rows),flush=True)
if __name__=='__main__':main()
