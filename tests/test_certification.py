import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from certification import *
class CertificateTests(unittest.TestCase):
 def test_last_failure(self):
  a=np.array([[0,1,0,1,1],[1,1,1,1,1],[1,0,1,1,0]],bool)
  expected=np.array([[0,0,0,1,1],[1,1,1,1,1],[0,0,0,0,0]],bool)
  np.testing.assert_array_equal(suffix_outcomes(a),expected)
 def test_martingale_null(self):
  for n in [1,10,100,512]:
   s=np.arange(n+1);w=binom.pmf(s,n,.95)
   self.assertAlmostEqual(float(np.dot(w,np.exp(log_evalue(s,n)))),1.,places=10)
 def test_supermartingale(self):
  for p in [.5,.8,.94,.95]:
   q=np.linspace(.951,.999,20)
   self.assertTrue(np.all(p*q/.95+(1-p)*(1-q)/.05<=1+1e-12))
 def test_monotonicity(self):
  for n in [10,100,512]:self.assertTrue(np.all(np.diff(log_evalue(np.arange(n+1),n))>0))
 def test_exact_cp_level(self):
  for n in [32,64,128,512]:
   b=cp_threshold(n);self.assertLessEqual(binom.sf(b-1,n,.95),.05+1e-14)
 def test_no_certificate_without_evidence(self):self.assertEqual(first_index(np.zeros(64,bool)),-1)
 def test_one_point_equivalence(self):
  p=np.random.default_rng(53).binomial(1,.99,(512,1))
  d=curves(p,[128,256,512]);a=[r['index'] for r in d if r['method']=='persistent_anytime'];b=[r['index'] for r in d if r['method']=='persistent_bonferroni'];self.assertEqual(a,b)
 def test_oracle_scan(self):
  rng=np.random.default_rng(73)
  for _ in range(100):
   x=rng.random(64)>.1;fail=np.flatnonzero(~x)
   self.assertEqual(last_failure(lambda k:x[k],64),int(fail[-1]) if len(fail) else -1)
 def test_nominal_eligibility(self):
  p=np.ones((512,3),bool);p[:,1]=False
  rows=curves(p,[512],min_index=1)
  self.assertEqual(next(r['index'] for r in rows if r['method']=='pointwise_beta'),2)
 def test_integral(self):
  from scipy.integrate import quad
  for n in [2,10,40]:
   for s in [0,n//2,n]:
    value=quad(lambda q:(q/.95)**s*((1-q)/.05)**(n-s)/.05,.95,1)[0]
    self.assertAlmostEqual(np.exp(log_evalue(s,n)),value,places=9)
if __name__=='__main__':unittest.main()
