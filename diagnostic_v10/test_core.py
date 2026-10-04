import unittest
import numpy as np
import torch
from .common import *
from reliability_v8.core import shuffled_weights,native_weights


class MechanismTests(unittest.TestCase):
    def test_gradient_geometry(self):
        v=dict(depth=np.array([1.,0]),normal=np.array([-1.,1.]),uniform_geometry=np.array([0.,1.]),weighted_geometry=np.array([0.,2.]))
        s=vector_stats(v)
        self.assertAlmostEqual(s['cosines']['depth__normal'],-1/np.sqrt(2))
        self.assertAlmostEqual(s['geometry_to_supervised_ratio']['uniform'],.2)
        self.assertAlmostEqual(s['geometry_to_supervised_ratio']['weighted'],.4)

    def test_labels_masks_and_render_are_controlled(self):
        torch.set_num_threads(2);ray=rays(480,640,read(V7OUT/'camera.json')['intrinsics'])
        cases=[synthetic_case('plane',201,c,ray) for c in CONDITIONS]
        for other in cases[1:]:
            np.testing.assert_array_equal(cases[0]['image'],other['image'])
            np.testing.assert_array_equal(cases[0]['mask'],other['mask'])
            np.testing.assert_array_equal(cases[0]['depth'],other['depth'])
            self.assertFalse(np.array_equal(cases[0]['observed'],other['observed']))
        np.testing.assert_array_equal(cases[0]['observed'],cases[0]['depth'])
        clean_error=angles(cases[0]['target_normal'],cases[0]['normal'])[cases[0]['mask']].mean()
        self.assertLess(clean_error,.1)
        for case in cases:
            w,m=native_weights(case['weight'],case['mask']);s=shuffled_weights(w.numpy(),m.numpy(),9)
            np.testing.assert_array_equal(np.sort(s[m.numpy()]),np.sort(w.numpy()[m.numpy()]))


if __name__=='__main__':unittest.main()
