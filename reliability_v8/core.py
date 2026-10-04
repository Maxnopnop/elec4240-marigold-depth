"""Training-label stability proxy and normalized spatial geometry loss."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from multitask_v7.common import depth_to_normals


def angles(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=0), -1, 1)))


def stability(depth, ray, sigmas=(1., 2., 4.), temperature_deg=10.):
    """Full-resolution sigma units are pixels, weights dimensionless and detached."""
    normals = []
    with torch.no_grad():
        for sigma in sigmas:
            smooth = gaussian_filter(np.asarray(depth, dtype=np.float32), sigma=sigma, truncate=2.)
            normals.append(depth_to_normals(torch.from_numpy(smooth)[None, None], ray)[0].numpy())
    disagreement = np.mean([angles(normals[a], normals[b])
                            for a, b in [(0, 1), (0, 2), (1, 2)]], axis=0)
    weight = np.exp(-disagreement / temperature_deg).astype(np.float32)
    return normals[0], disagreement, weight


def original_support(depth, valid):
    """Synthetic analogue of V7 erosion and depth-jump filtering."""
    jumps = np.zeros_like(valid)
    for axis in [0, 1]:
        diff = np.diff(depth, axis=axis)
        a, b = (depth[1:], depth[:-1]) if axis == 0 else (depth[:, 1:], depth[:, :-1])
        jump = (abs(diff) > .1) & (abs(diff) > .05 * np.minimum(a, b))
        if axis == 0:
            jumps[1:] |= jump; jumps[:-1] |= jump
        else:
            jumps[:, 1:] |= jump; jumps[:, :-1] |= jump
    mask = binary_erosion(valid, iterations=3) & ~binary_dilation(jumps, iterations=2)
    return mask


def native_weights(weights, valid, size=(192, 256)):
    """Area average only valid contributors; exactly retain V7 native support."""
    mask = torch.as_tensor(valid, dtype=torch.float32)[None, None]
    w = torch.as_tensor(weights, dtype=torch.float32)[None, None]
    coverage = F.interpolate(mask, size=size, mode='area')
    total = F.interpolate(w * mask, size=size, mode='area')
    support = coverage > .9
    out = torch.where(support, total / coverage.clamp_min(1e-8), 0.)
    return out.detach(), support


def shuffled_weights(weights, valid, seed):
    out = np.zeros_like(weights)
    values = np.asarray(weights)[valid].copy()
    out[valid] = np.random.default_rng(seed).permutation(values)
    return out


def geometry_loss(depth, normal, ray, weights, mask):
    nd = depth_to_normals(depth.float(), ray)
    normal = F.normalize(normal.float(), dim=1, eps=1e-6)
    error = 1 - (nd * normal).sum(1).clamp(-1, 1)
    effective = (weights.detach().float() * mask.detach().float())[:, 0]
    total_weight = effective.sum()
    if not torch.isfinite(effective).all() or (effective < 0).any():
        raise ValueError('Weights must be finite and nonnegative')
    if float(total_weight) <= 1e-8:
        raise ValueError('Empty or negligible geometry weight; do not silently change objective')
    return (effective * error).sum() / total_weight


def weight_stats(w):
    w = np.asarray(w, dtype=np.float64)
    return {'mean_weight': float(w.mean()), 'min_weight': float(w.min()),
            'effective_fraction': float(w.sum()**2 / (len(w) * np.square(w).sum())),
            'fraction_below_0_1': float((w < .1).mean())}
