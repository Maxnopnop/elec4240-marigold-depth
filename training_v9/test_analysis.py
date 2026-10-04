"""Meaningful statistical edge cases and independent metric sanity checks."""
import unittest
import numpy as np
from .analyze import holm,bootstrap,depth_metrics,normal_metrics


class TestAnalysis(unittest.TestCase):
    def test_paired_zero_and_constant_effect(self):
        rng=np.random.default_rng(42);s=rng.integers(0,3,(20000,3));x=rng.integers(0,32,(20000,32))
        zero=bootstrap(np.zeros((3,32)),s,x)
        self.assertEqual(zero['raw_p_two_sided_centered'],1.)
        effect=bootstrap(np.full((3,32),-.02),s,x)
        self.assertAlmostEqual(effect['difference'],-.02)
        self.assertLess(effect['raw_p_two_sided_centered'],.001)
        self.assertLess(effect['unadjusted_symmetric_95_interval'][1],0)

    def test_holm_order_invariance_and_bounds(self):
        p=np.array([.03,.01,.04]);expected=np.array([.06,.03,.06])
        np.testing.assert_allclose(holm(p),expected)
        index=np.array([2,0,1]);np.testing.assert_allclose(holm(p[index]),expected[index])
        np.testing.assert_allclose(holm([1,1,1]),[1,1,1])

    def test_independent_metrics_known_geometry(self):
        mask=np.ones((12,12),bool);gt=np.full((12,12),2.)
        d=depth_metrics(gt*.5,gt,mask)
        self.assertEqual(d['abs_rel'],.5);self.assertEqual(d['rmse_m'],1.);self.assertEqual(d['delta1'],0)
        n=np.zeros((3,12,12));n[2]=-1
        self.assertEqual(normal_metrics(n,n,mask)['mean_deg'],0)
        opposite=normal_metrics(-n,n,mask)
        self.assertEqual(opposite['mean_deg'],180);self.assertEqual(opposite['within_30'],0)


if __name__=='__main__':unittest.main()
