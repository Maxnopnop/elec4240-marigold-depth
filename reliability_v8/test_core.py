import unittest
import numpy as np
import torch
import torch.nn.functional as F
from multitask_v7.common import rays, depth_to_normals
from .core import native_weights, shuffled_weights, geometry_loss, stability

CAMERA = {'fx_rgb': 518.8579011745019, 'fy_rgb': 519.4696111212749,
          'cx_rgb': 325.58244941119034, 'cy_rgb': 253.73616633400465}


class TestReliability(unittest.TestCase):
    def test_analytic_slanted_plane(self):
        ray = rays(96, 128, CAMERA)
        depth = 3 / (1 + .3*ray[:, 0:1] - .2*ray[:, 1:2])
        n = depth_to_normals(depth, ray)[:, :, 3:-3, 3:-3]
        target = F.normalize(torch.tensor([-.3, .2, -1.]), dim=0)[None, :, None, None]
        self.assertLess(float((n-target).abs().max()), .0001)

    def test_plane_stability(self):
        ray = rays(96, 128, CAMERA)
        _, u, w = stability(np.full((96, 128), 3, np.float32), ray)
        self.assertLess(float(u[10:-10, 10:-10].max()), .05)
        self.assertGreater(float(w[10:-10, 10:-10].min()), .99)

    def test_native_support_and_invalid_values(self):
        valid = np.ones((32, 32), bool); valid[:8] = False
        w = np.full((32, 32), .3, np.float32); w[~valid] = 1e6
        out, support = native_weights(w, valid, (8, 8))
        torch.testing.assert_close(out[support], torch.full_like(out[support], .3))
        self.assertTrue(bool((out[~support] == 0).all()))
        self.assertEqual(int(support.sum()), 48)

    def test_shuffle_conserves_distribution(self):
        w = np.arange(144).reshape(12, 12).astype(np.float32) / 144
        m = np.ones((12, 12), bool); m[0] = False
        shuffled = shuffled_weights(w, m, 41)
        np.testing.assert_array_equal(np.sort(shuffled[m]), np.sort(w[m]))
        self.assertFalse(np.array_equal(shuffled[m], w[m]))
        np.testing.assert_array_equal(shuffled, shuffled_weights(w, m, 41))

    def test_uniform_identity_scaling_and_gradients(self):
        torch.manual_seed(7)
        ray = rays(24, 32, CAMERA)
        d = (2 + torch.rand(1, 1, 24, 32)).requires_grad_()
        n = torch.randn(1, 3, 24, 32, requires_grad=True)
        m = torch.ones_like(d, dtype=torch.bool); m[..., :2, :] = False
        w = torch.ones_like(d, requires_grad=True)
        expected = (1-(depth_to_normals(d, ray)*F.normalize(n, dim=1)).sum(1).clamp(-1, 1))[m[:,0]].mean()
        actual = geometry_loss(d, n, ray, w, m)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, geometry_loss(d, n, ray, w*.17, m))
        actual.backward()
        self.assertIsNone(w.grad)
        self.assertTrue(bool(torch.isfinite(d.grad).all() and torch.isfinite(n.grad).all()))
        self.assertGreater(float(d.grad.abs().sum()), 0)
        self.assertGreater(float(n.grad.abs().sum()), 0)

    def test_zero_weights_fail_explicitly(self):
        d = torch.ones(1,1,24,32)
        with self.assertRaises(ValueError):
            geometry_loss(d, d.repeat(1,3,1,1), rays(24,32,CAMERA), d*0, d.bool())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
