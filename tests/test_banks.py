import unittest,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class BankIntegrityTests(unittest.TestCase):
 def test_archived_banks(self):
  files=sorted((ROOT/'results/banks').glob('*.npz'));self.assertEqual(len(files),144)
  total=0
  for f in files:
   with np.load(f) as a:
    q,k=a['cost'].shape;total+=q
    self.assertEqual(k,64);self.assertEqual(a['cert'].shape,(512,q,k));self.assertEqual(a['evaluation'].shape,(1024,q,k));self.assertEqual(a['nominal'].shape,(q,k))
    for name in ['cert','evaluation','nominal']:self.assertTrue(np.isin(a[name],[0,1]).all())
    self.assertTrue(np.isfinite(a['paths']).all());self.assertTrue((np.diff(a['cost'],axis=1)>=-1e-6).all());self.assertTrue(a['nominal'][:,-1].all())
    self.assertFalse(set(a['train_indices'])&set(a['test_indices']));self.assertTrue(set(a['query_indices'])<=set(a['test_indices']));self.assertTrue(set(a['anchor_indices'])<=set(a['train_indices']))
    self.assertEqual(len(np.unique(a['seeds'])),1569)
    if '_coordinate' in f.stem:
     with np.load(f.with_name(f.name.replace('_coordinate',''))) as b:
      np.testing.assert_array_equal(a['query_indices'],b['query_indices']);np.testing.assert_array_equal(a['seeds'],b['seeds']);np.testing.assert_array_equal(a['cert'][:,:,-1],b['cert'][:,:,-1]);np.testing.assert_array_equal(a['evaluation'][:,:,-1],b['evaluation'][:,:,-1])
  self.assertEqual(total,3450)
 def test_numeric_data(self):
  for f in (ROOT/'data').glob('*.npz'):
   a=np.load(f);self.assertTrue(np.isfinite(a['X']).all());self.assertEqual(set(a['y']),{0,1})
if __name__=='__main__':unittest.main()
