import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from ids.core import config,read_json,write_json,digest
from ids.data import save_meta,seal_data
from ids.pipeline import develop,final_main,external,load_decision
from ids.report import summarize,figures

def fixture(root,dataset):
    root.mkdir(); rng=np.random.default_rng(12); labels=['BENIGN','FTP-Patator','Botnet','X - Attempted']
    m={'dataset':dataset,'production':False,'schema':{'input_dimension':82},'labels':labels,'roles':{},'source':{'synthetic_fixture':True}}
    specs=[('train_normal',96),('validation_normal',32),('calibration_normal',32)]
    specs += [('development',32),('final_test',32)] if dataset=='cic2017' else [('external_test',32),('external_diagnostic',32)]
    for k,(role,n) in enumerate(specs):
        p=root/role; p.mkdir(); x=rng.normal(size=(n,82)); codes=np.zeros(n,dtype=np.uint16)
        if not role.endswith('_normal'): codes=np.repeat(np.arange(4,dtype=np.uint16),8); x[8:24]+=1.5
        np.save(p/'X.npy',x); save_meta(p/'meta.npz',np.arange(n),np.arange(n)+k*1000,codes,role+'.csv')
        m['roles'][role]=[{'path':role,'rows':n,'member':role+'.csv','min_timestamp_ns':k*1000,'max_timestamp_ns':k*1000+n-1}]
        if role=='train_normal': joblib.dump(StandardScaler().fit(x),root/'standard_scaler.joblib')
    write_json(root/'manifest.json',m); seal_data(root); return m

class EndToEndTests(unittest.TestCase):
    def test_full_pipeline_on_synthetic_fixture(self):
        c=copy.deepcopy(config()); c['ae'].update(max_epochs=2,batch_size=32); c['iforest'].update(n_estimators=3,max_samples=32)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); main=root/'main'; ext=root/'external'; e=root/'experiment'
            m=fixture(main,'cic2017'); me=fixture(ext,'cic2018')
            def check(p): return {str(main):m,str(ext):me}[str(p)]
            with patch('ids.pipeline.validate_prepared',side_effect=check):
                d=develop(main,e,c); before=digest(e/'decision.json'); final_main(main,e,c); external(ext,e,c)
                summary=summarize(e,c); summarize(e,c,True); summarize(e,c,True,True); fig=figures(e,c)
                self.assertTrue((fig/'roc_seed42.pdf').exists()); self.assertEqual(before,digest(e/'decision.json')); load_decision(e,c)
                n=len(list(e.rglob('run.json'))); self.assertEqual(n,56 if d['architecture'] in ['ae_l16','ae_l32'] else 61)
                final_main(main,e,c); external(ext,e,c); self.assertEqual(n,len(list(e.rglob('run.json'))))
                for v in read_json(summary/'paired_statistics.json').values(): self.assertEqual(len(v['differences']),5)
                with self.assertRaises(FileExistsError): develop(main,e,c)
                with self.assertRaises(ValueError): develop(ext,e,c)
                (e/'external/external_test/iforest_seed7/metrics.json').unlink()
                with self.assertRaises(ValueError): external(ext,e,c)

if __name__=='__main__': unittest.main()
