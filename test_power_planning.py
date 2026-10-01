import unittest
import numpy as np
from plan_prospective_power import errors, holm_rows, TRANSFORM
from robustness_stats import holm

class PlanningChecks(unittest.TestCase):
    def test_holm_matches_frozen_scalar_implementation(self):
        rows=np.random.default_rng(42).uniform(0,.1,(40,6))
        np.testing.assert_allclose(holm_rows(rows),np.array([holm(p) for p in rows]))

    def test_additive_crossed_variance(self):
        d=np.array([-1.,0.,1.])
        s=np.arange(5.)-2
        i=np.arange(7.)-3
        x=(d[:,None,None]+s[None,:,None]+i[None,None,:])[...,None]*np.array([1.,2.,3.,4.])
        observed=errors(x,5,5,100,np.random.default_rng(444),count=20000)
        expected=d.var()/5+s.var()/5+i.var()/100
        self.assertLess(abs(observed[:,0].var()/expected-1),.04)
        np.testing.assert_allclose(observed[:,4],observed[:,0]-observed[:,2],atol=1e-12)
        np.testing.assert_allclose(observed[:,5],observed[:,1]-observed[:,3],atol=1e-12)

    def test_constant_pilot_has_no_resampling_noise(self):
        for kind in ['nested','crossed']:
            observed=errors(np.ones((3,5,7,4)),3,5,31,np.random.default_rng(21),count=50,kind=kind)
            np.testing.assert_array_equal(observed,0)

if __name__=='__main__':unittest.main()
