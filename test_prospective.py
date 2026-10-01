import unittest
import numpy as np
from prepare_prospective import decode_depth
from prospective_metrics import aligned_metrics,metrics,valid_mask
from prospective_statistics import broad_location_bootstrap

class ExternalProtocolChecks(unittest.TestCase):
    def test_official_bit_rotation(self):
        mm=np.array([0,1,1000,2500,9999,65535],dtype=np.uint32)
        encoded=(((mm<<3)|(mm>>13))&65535).astype(np.uint16)
        np.testing.assert_allclose(decode_depth(encoded),mm/1000,atol=1e-5)

    def test_alignment_and_fixed_calibration_are_distinct(self):
        gt=np.linspace(.2,8.,300).reshape(15,20)
        pred=(gt-1)/3
        self.assertLess(aligned_metrics(pred,gt)['abs_rel'],1e-12)
        self.assertLess(metrics(pred,gt,3,1)['abs_rel'],1e-12)
        self.assertGreater(metrics(pred,gt)['abs_rel'],.1)

    def test_sensor_mask_excludes_missing_depth(self):
        gt=np.array([0.,.1,.11,9.99,10.,np.nan])
        np.testing.assert_array_equal(valid_mask(gt),[False,False,True,True,False,False])

    def test_proxy_cluster_preserves_constant_difference(self):
        r=broad_location_bootstrap(np.full((3,5,6),-.02),['a','a','a','b','c','c'],replicates=1000)
        self.assertAlmostEqual(r['ci_low'],-.02)
        self.assertLess(r['p_bootstrap'],.002)

if __name__=='__main__':unittest.main()
