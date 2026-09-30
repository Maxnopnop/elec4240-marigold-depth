"""Meaningful evaluation checks: affine alignment, exact metrics, invalid input."""
import unittest
import numpy as np
from experiment import evaluate_depth

class ProtocolTest(unittest.TestCase):
    def test_recovers_affine_depth(self):
        y,x=np.mgrid[:480,:640];gt=1+x/200+y/300
        pred=(gt-.7)/2.3
        m,aligned,mask=evaluate_depth(pred,gt)
        self.assertLess(m['abs_rel'],1e-10);self.assertEqual(m['delta1'],1.)
        self.assertEqual(m['valid_pixels'],426*560)
        self.assertAlmostEqual(m['scale'],2.3)
    def test_rejects_nan(self):
        gt=np.ones((480,640));pred=gt.copy();pred[100,100]=np.nan
        with self.assertRaises(ValueError):evaluate_depth(pred,gt)
    def test_flat_prediction_is_finite(self):
        y,x=np.mgrid[:480,:640];gt=1+x/200
        m,_,_=evaluate_depth(np.ones_like(gt),gt)
        self.assertTrue(np.isfinite(m['abs_rel']));self.assertGreater(m['abs_rel'],.01)
if __name__=='__main__':unittest.main()
