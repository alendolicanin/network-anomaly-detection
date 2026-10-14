import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ids.core import config,ROOT,read_json,write_json,seal,verify,digest,object_hash,source_hash,data_fingerprint,runtime_versions
from ids.metrics import binary_metrics,thresholds,evaluate,choose_architecture,choose_percentile,paired_statistics
from ids.data import chronological_order,training_spec,matrix,clean_chunk
from ids.models import fit_ae,model_scores,latent_activations,robust_scaler
from ids.pipeline import archive_incomplete,lock_decision,load_decision

class MetricTests(unittest.TestCase):
    def test_hand_matrix_and_equality(self):
        r=binary_metrics([0,0,1,1],[.1,.7,.5,.9],.5)
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']],[1,1,1,1]); self.assertEqual(r['f1'],.5)
    def test_sklearn_agreement(self):
        rng=np.random.default_rng(4); y=rng.integers(0,2,400); s=rng.random(400); p=s>.5; r=binary_metrics(y,s,.5)
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']],confusion_matrix(y,p).ravel().tolist())
        for name,value in [('precision',precision_score(y,p)),('recall',recall_score(y,p)),('f1',f1_score(y,p)),('roc_auc',roc_auc_score(y,s))]: self.assertAlmostEqual(r[name],value)
    def test_single_class(self):
        r=binary_metrics([0,0],[0.,0.],1.); self.assertIsNone(r['f1']); self.assertIsNone(r['recall']); self.assertIsNone(r['roc_auc']); self.assertEqual(r['precision'],0); self.assertTrue(r['precision_zero_convention'])
        r=binary_metrics([1,1],[0.,0.],1.); self.assertIsNone(r['fpr']); self.assertEqual(r['f1'],0)
    def test_monotonic(self):
        rng=np.random.default_rng(7); y=rng.integers(0,2,100); s=rng.random(100); rs=[binary_metrics(y,s,t) for t in [.2,.5,.9]]
        for name in ['recall','fpr']: self.assertTrue(np.all(np.diff([r[name] for r in rs])<=0))
    def test_linear_percentile(self): self.assertEqual(thresholds([0,10],[95])['95'],9.5)
    def test_attempted_macro(self):
        r=evaluate([0,0,1,1,1],[.1,.9,.7,.2,.8],[0,1,0,0,0],['BENIGN','FTP - Attempted','A','A','B'],['day']*5,.5)
        self.assertEqual(r['primary']['fp'],1); self.assertEqual(r['exclude_attempted']['fp'],0); self.assertEqual(r['macro_recall'],.75)
    def test_nonfinite(self):
        with self.assertRaises(ValueError): binary_metrics([0,1],[np.nan,1],.5)
    def test_exact_five(self):
        r=paired_statistics([1,2,3,4,5]); self.assertEqual(r['sign_two_sided_p'],.0625); self.assertEqual(r['wilcoxon_enumerated_two_sided_p'],.0625)
    def test_zeros_and_ties(self):
        r=paired_statistics([0]*5); self.assertEqual(r['sign_two_sided_p'],1); self.assertEqual(r['nonzero_pairs'],0)
        r=paired_statistics([1,-1,0,2,-2]); self.assertEqual(r['wilcoxon_enumerated_two_sided_p'],1); self.assertEqual(r['nonzero_pairs'],4)

class DataTests(unittest.TestCase):
    def test_ties(self):
        t=np.array([9,2,3,4,5,6,6,8,1,10]); order,b=chronological_order(t,np.arange(10)); self.assertEqual(b,[7,8]); self.assertLess(t[order[b[0]-1]],t[order[b[0]]])
    def test_impossible_ties(self):
        with self.assertRaises(ValueError): chronological_order(np.ones(10),np.arange(10))
    def test_training_only_schema(self):
        x=np.array([[0,1],[1,1],[2,1],[3,100]]); p=np.array([6,6,17,99]); spec=training_spec(x,p,['a','b'],np.arange(4),3,2)
        self.assertEqual(spec['numeric_features'],['a']); self.assertEqual(spec['protocol_categories'],[6,17]); np.testing.assert_array_equal(matrix(x,p,spec)[-1],[3,0,0])
    def test_scaler_not_refitted(self):
        s=StandardScaler().fit([[0.,1],[2,3]]); mean=s.mean_.copy(); s.transform([[500,900]]); np.testing.assert_array_equal(s.mean_,mean)
    def frame(self,t='2018-02-15 00:10:00'):
        cols=read_json(ROOT/'configs/sources.json')['cic2018']['columns']; row=dict.fromkeys(cols,0)
        row.update({'Flow ID':'x','Src IP':'a','Dst IP':'b','Timestamp':t,'Label':'BENIGN','Attempted Category':-1,'id':1,'Protocol':6})
        return pd.DataFrame([row]),cols
    def test_invalid_date(self):
        f,c=self.frame('1970-01-01')
        with self.assertRaises(ValueError): clean_chunk(f,c,'2018-02-14','2018-02-15 01:00:00')
    def test_utc_midnight(self):
        f,c=self.frame(); result,n=clean_chunk(f,c,'2018-02-14','2018-02-15 01:00:00'); self.assertEqual(len(result),1); self.assertEqual(str(result.Timestamp.dtype),'datetime64[ns, UTC]')
    def test_inf_removed(self):
        f,c=self.frame(); f['Flow Bytes/s']=np.inf; result,n=clean_chunk(f,c,'2018-02-14','2018-02-15 01:00:00'); self.assertEqual(n,1); self.assertEqual(len(result),0)
    def test_attempted_mismatch(self):
        f,c=self.frame(); f['Label']='FTP - Attempted'
        with self.assertRaises(ValueError): clean_chunk(f,c,'2018-02-14','2018-02-15 01:00:00')
    def test_hash_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npy'; np.save(p,[1,2]); seal(td); verify(td); np.save(p,[1,3])
            with self.assertRaises(ValueError): verify(td)

class SelectionTests(unittest.TestCase):
    def rows(self):
        return [{'architecture':a,'seed':s,'role':'development','percentile':95,'f1':.5,'fpr':f,'parameters':p} for a,f,p in [('a',.2,10),('b',.1,20),('c',.1,15)] for s in config()['seeds']]
    def test_architecture_ties(self): self.assertEqual(choose_architecture(self.rows(),config()['seeds'])[0],'c')
    def test_missing_seed(self):
        with self.assertRaises(ValueError): choose_architecture(self.rows()[:-1],config()['seeds'])
    def test_final_role_rejected(self):
        rows=self.rows(); rows[0]['role']='final_test'
        with self.assertRaises(ValueError): choose_architecture(rows,config()['seeds'])
    def test_shared_percentile(self):
        c=config(); rows=[{'seed':s,'percentile':q,'role':'development','f1':.5,'fpr':.1} for q in c['percentiles'] for s in c['seeds']]
        self.assertEqual(choose_percentile(rows,c['seeds'],c['percentiles'])[0],99.9)
    def test_config_change(self):
        with tempfile.TemporaryDirectory() as td:
            c=config(); c['ae']['shuffle']=True; p=Path(td)/'c.json'; write_json(p,c)
            with self.assertRaises(ValueError): config(p)
    def test_incomplete_archive(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'fit'; write_json(d/'run.json',{'identity':{'seed':42}}); archive_incomplete(d,'run.json',{'seed':42},'identity')
            self.assertFalse(d.exists()); self.assertEqual(len(list((Path(td)/'interrupted_runs').iterdir())),1)
    def test_decision_never_reads_final(self):
        c=config()
        with tempfile.TemporaryDirectory() as td:
            e=Path(td)/'exp'; data=Path(td)/'data'; write_json(data/'data_identity.json',{'test':True})
            for arch in [*c['architectures'],'iforest']:
                for s in c['seeds']:
                    name=f'{arch}_seed{s}'; f=e/'fits'/name; v=e/'development'/name
                    identity={'architecture':arch,'seed':s,'scaling':'standard','dataset':'cic2017','code_hash':source_hash(),'config_hash':object_hash(c),'data_hash':data_fingerprint(data),'runtime_versions':runtime_versions()}
                    write_json(f/'run.json',{'identity':identity,'status':'complete','training':{'parameters':100}}); seal(f)
                    write_json(v/'evaluation.json',{'run_hash':digest(f/'integrity.json'),'role':'development','decision_hash':None})
                    write_json(v/'metrics.json',{str(q):{'primary':{'f1':.9 if arch=='ae_l8' else .5,'fpr':.1}} for q in c['percentiles']}); seal(v)
            def guard(p):
                self.assertNotIn('final_test',str(p)); return read_json(p)
            with patch('ids.pipeline.read_json',side_effect=guard): d=lock_decision(data,e,c)
            self.assertEqual(d['architecture'],'ae_l8'); self.assertEqual(d['percentile_ae'],99.9); load_decision(e,c)
            d['architecture']='ae_l32'; write_json(e/'decision.json',d)
            with self.assertRaises(ValueError): load_decision(e,c)

class TrainingTests(unittest.TestCase):
    def arrays(self):
        rng=np.random.default_rng(9); return rng.normal(size=(96,5)).astype('float32'),rng.normal(size=(24,5)).astype('float32')
    def test_repeat_and_best(self):
        x,v=self.arrays(); settings=config()['ae']; settings.update(max_epochs=4,patience=2,batch_size=16)
        with threadpool_limits(limits=1):
            a,ar=fit_ae(x,v,[4,2,4],42,settings,64); b,br=fit_ae(x,v,[4,2,4],42,settings,64)
        np.testing.assert_allclose(model_scores(a,'ae',v),model_scores(b,'ae',v),rtol=1e-7,atol=1e-9)
        self.assertAlmostEqual(model_scores(a,'ae',v).mean(),ar['history'][ar['best_epoch']-1]['validation_reconstruction_mse'],places=10); self.assertEqual(latent_activations(a,v).shape,(24,2))
    def test_early_stop(self):
        x,v=self.arrays(); settings=config()['ae']; settings.update(max_epochs=10,patience=2,min_delta=1e9,batch_size=16)
        with threadpool_limits(limits=1): a,r=fit_ae(x,v,[4,2,4],42,settings,64)
        self.assertEqual(r['best_epoch'],1); self.assertEqual(r['epochs_ran'],3)
    def test_robust_zero_iqr(self):
        x=np.array([[1,1],[1,2],[1,3]],dtype=float); s,zero=robust_scaler(x); self.assertEqual(zero,[0]); self.assertTrue(np.isfinite(s.transform(x)).all())

if __name__=='__main__': unittest.main()
