import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
TOOL=Path(__file__).with_name('model_quality.py')
class QualityTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def run_tool(self,*args):
        p=subprocess.run([sys.executable,str(TOOL),*map(str,args)],capture_output=True,text=True)
        return p,json.loads(p.stdout) if p.stdout.strip() else None
    def capture(self,name,kind,logits,ids=(0,1),vocab=None):
        d=self.root/(name+'-raw');d.mkdir();np.asarray(ids,dtype='<u4').tofile(d/'ids.u32');a=np.asarray(logits,dtype='<f4');a.tofile(d/'logits.f32')
        args=['freeze','--kind',kind,'--ids',d/'ids.u32','--logits',d/'logits.f32','--vocab-size',a.shape[1],'--out',self.root/name]
        if vocab is not None:np.asarray(vocab,dtype='<u4').tofile(d/'vocab.u32');args+=['--vocab-ids',d/'vocab.u32']
        p,_=self.run_tool(*args);self.assertEqual(p.returncode,0,p.stderr);return self.root/name
    def pair(self,c=None,r=None,ids=(0,1),vocab=None):
        r=np.log([[.8,.2],[.3,.7]]) if r is None else r
        c=np.log([[.6,.4],[.5,.5]]) if c is None else c
        return self.capture('candidate','candidate',c,ids,vocab),self.capture('reference','reference',r)
    def compare(self,c,r,*extra):return self.run_tool('compare','--candidate',c,'--reference',r,*extra)
    def test_analytic_probabilities(self):
        c,r=self.pair();p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr);m=j['metrics']
        expected=(.8*math.log(.8/.6)+.2*math.log(.2/.4)+.3*math.log(.3/.5)+.7*math.log(.7/.5))/2
        self.assertAlmostEqual(m['mean_reference_to_candidate_kl'],expected,places=6)
        self.assertAlmostEqual(m['reference_mean_nll'],-math.log(.2),places=6)
        self.assertAlmostEqual(m['candidate_mean_nll'],-math.log(.4),places=6)
        self.assertEqual(m['top1_agreement'],.5);self.assertEqual(j['positions'][1]['candidate_top1'],0)
        self.assertIsNone(j['quality_pass']);self.assertEqual(j['status'],'diagnostic_missing_provenance')
    def test_large_shared_offset_preserves_probability_normalization(self):
        c,r=self.pair(c=[[1e30,1e30],[1e30,1e30]],r=[[0.,0.],[0.,0.]])
        p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr);m=j['metrics']
        self.assertAlmostEqual(m['candidate_mean_nll'],math.log(2.),places=7)
        self.assertAlmostEqual(m['mean_reference_to_candidate_kl'],0.,places=7)
        self.assertAlmostEqual(m['maximum_absolute_probability_error'],0.,places=7)
    def test_topk_cutoff_ties_choose_low_token_ids(self):
        c=self.capture('candidate','candidate',[[3.,2.,2.,0.],[0.,0.,0.,0.]],vocab=(0,1,2,3))
        r=self.capture('reference','reference',[[3.,2.,0.,2.],[0.,0.,0.,0.]],vocab=(0,1,2,3))
        p,j=self.compare(c,r,'--diagnostic','--top-k','2');self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(j['metrics']['mean_topk_overlap'],1.)
    def test_offset_invariance_is_reported_separately(self):
        r=np.array([[1.,2.],[3.,1.]],dtype=np.float32);c,r=self.pair(r+7,r);p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr);m=j['metrics']
        self.assertAlmostEqual(m['logit_rmse'],7.,places=7);self.assertAlmostEqual(m['centered_logit_rmse'],0.,places=7);self.assertAlmostEqual(m['mean_reference_to_candidate_kl'],0.,places=7);self.assertEqual(m['top1_agreement'],1.)
    def test_vocab_column_alignment(self):
        r=np.array([[1.,2.],[3.,1.]],dtype=np.float32);c,r=self.pair(r[:,::-1],r,vocab=(1,0));p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(j['metrics']['logit_rmse'],0.)
    def test_missing_provenance_is_not_a_pass(self):
        c,r=self.pair();p,j=self.compare(c,r);self.assertEqual(p.returncode,2,p.stderr);self.assertIsNone(j['quality_pass']);self.assertTrue(j['missing_provenance']['reference'])
    def test_declared_limit_failure_is_nonzero(self):
        c,r=self.pair();p,j=self.compare(c,r,'--diagnostic','--max-mean-kl','0');self.assertEqual(p.returncode,3,p.stderr);self.assertFalse(j['declared_criteria'][0]['passes'])
    def test_large_representable_perplexity_is_preserved(self):
        logits=[[0.,-700.],[0.,0.]]
        c,r=self.pair(c=logits,r=logits)
        p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr)
        m=j['metrics'];self.assertEqual(m['candidate_mean_nll'],700.)
        self.assertAlmostEqual(m['candidate_perplexity']/math.exp(700.),1.,places=12)
        self.assertAlmostEqual(m['reference_perplexity']/math.exp(700.),1.,places=12)
        self.assertFalse(m['perplexity_overflow'])
    def test_perplexity_overflow_is_explicit(self):
        logits=[[0.,-710.],[0.,0.]]
        c,r=self.pair(c=logits,r=logits)
        p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,0,p.stderr)
        m=j['metrics'];self.assertEqual(m['candidate_mean_nll'],710.)
        self.assertIsNone(m['candidate_perplexity']);self.assertIsNone(m['reference_perplexity'])
        self.assertTrue(m['perplexity_overflow'])
    def test_changed_frozen_bytes_rejected(self):
        c,r=self.pair();f=c/'logits.f32';b=bytearray(f.read_bytes());b[0]^=1;f.write_bytes(b);p,_=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,2);self.assertIn('frozen artifact changed',p.stderr)
    def test_different_teacherforced_ids_rejected(self):
        c,r=self.pair(ids=(1,1));p,_=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,2);self.assertIn('identical teacher-forced',p.stderr)
    def test_equal_invalid_first_input_is_rejected(self):
        c=self.capture('candidate','candidate',[[0.,1.],[1.,0.]],ids=(99,1))
        r=self.capture('reference','reference',[[0.,1.],[1.,0.]],ids=(99,1))
        p,_=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,2)
        self.assertIn('input token IDs absent',p.stderr)
    def test_complete_declarations_do_not_attest_execution(self):
        c,r=self.pair(c=[[0.,1.],[1.,0.]],r=[[0.,1.],[1.,0.]])
        for directory in (c,r):
            path=directory/'manifest.json';m=json.loads(path.read_text())
            m['metadata']={'teacher_forced':True,'base_model':{'id':'declared','revision':'a'*40},'tokenizer':{'sha256':'b'*64},'checkpoint':{'files':[{'source_uri':'declared:model','sha256':'c'*64}]},'implementation':{'name':'declared','revision':'d'*40,'independent_reference':True},'command':['declared-command']}
            path.write_text(json.dumps(m))
        for args in ((),('--diagnostic',)):
            p,j=self.compare(c,r,*args,'--max-rmse','0');self.assertEqual(p.returncode,0,p.stderr)
            self.assertTrue(j['declared_limits_pass']);self.assertIsNone(j['quality_pass'])
            self.assertFalse(j['independent_checkpoint_evidence'])
            self.assertEqual(j['provenance_status'],'producer_declared_unattested')
    def test_nonfinite_candidate_fails(self):
        c,r=self.pair(c=[[float('nan'),0.],[0.,1.]]);p,j=self.compare(c,r,'--diagnostic');self.assertEqual(p.returncode,3,p.stderr);self.assertFalse(j['quality_pass']);self.assertEqual(j['candidate_nonfinite'],1)
if __name__=='__main__':unittest.main()
