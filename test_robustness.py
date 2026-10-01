"""Small independent checks for calibration and multiple-comparison bookkeeping."""
import unittest
import numpy as np
from robustness_stats import holm, bootstrap
from crossed_bootstrap_sensitivity import crossed_bootstrap
from run_robustness import fit_calibration, fixed_metrics, perturb, matrix, conditions


class RobustnessTests(unittest.TestCase):
    def test_holm_known_example(self):
        np.testing.assert_allclose(holm([.03, .001, .02]), [.04, .003, .04])

    def test_constant_paired_bootstrap(self):
        result = bootstrap(np.full((3, 5, 7), -.02), replicates=1000)
        for name in ['scene', 'hierarchical']:
            self.assertAlmostEqual(result[name]['ci_low'], -.02)
            self.assertLess(result[name]['p_bootstrap'], .002)

    def test_crossed_seed_variability_is_preserved(self):
        seed_effect = np.array([-2., -1., 0., 1., 2.])
        delta = np.broadcast_to(seed_effect[None, :, None], (3, 5, 7))
        nested = bootstrap(delta, replicates=2000)['hierarchical']
        crossed = crossed_bootstrap(delta, replicates=2000)
        self.assertGreater(crossed['ci_high']-crossed['ci_low'], nested['ci_high']-nested['ci_low'])

    def test_calibration_recovers_fixed_mapping(self):
        pred = np.tile(np.linspace(.1, .9, 640), (480, 1))
        gt = 3*pred+1
        c = fit_calibration([(pred, gt), (pred*.8, pred*.8*3+1)])
        self.assertAlmostEqual(c['scale'], 3)
        self.assertAlmostEqual(c['shift'], 1)
        self.assertLess(fixed_metrics(pred, gt, c['scale'], c['shift'])['abs_rel'], 1e-12)

    def test_metric_scoring_does_not_fit_test(self):
        pred = np.ones((480, 640))
        gt = np.full_like(pred, 2)
        self.assertAlmostEqual(fixed_metrics(pred, gt)['abs_rel'], .5)
        self.assertAlmostEqual(fixed_metrics(pred, gt, 2, 0)['abs_rel'], 0)

    def test_fixed_counts_and_perturbation(self):
        configs = matrix()
        self.assertEqual(sum(c['mode'] in ['high512', 'mixed'] for c in configs), 30)
        self.assertEqual(sum(len(conditions(c, 'test')) for c in configs), 64)
        self.assertEqual(sum(len(conditions(c, 'corruption')) for c in configs), 34)
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(perturb(rgb, 'dark'), 50)


if __name__ == '__main__':
    unittest.main()
