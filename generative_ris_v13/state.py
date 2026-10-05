"""Deterministic exposure schedule and atomic, protocol-bound training state."""
import hashlib
import random
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def schedule(image_ids, updates=8192):
    ids = sorted(image_ids)
    assert len(ids) == len(set(ids)) and ids
    result = []
    for epoch in range((updates + len(ids) - 1) // len(ids)):
        order = ids.copy()
        random.Random(424113 + epoch).shuffle(order)
        result.extend(order)
    return result[:updates]


def save(path, params, optimizer, step, history, protocol_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(adapter={k: p.detach().cpu().clone() for k, p in params.items()},
                    optimizer=optimizer.state_dict(), step=step, history=history,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    protocol_sha256=protocol_sha256), tmp)
    tmp.replace(path)


def restore(path, params, optimizer, protocol_sha256):
    d = torch.load(path, map_location='cpu', weights_only=True)
    assert d['protocol_sha256'] == protocol_sha256
    assert set(d['adapter']) == set(params)
    assert len(d['history']) == d['step']
    with torch.no_grad():
        for k, p in params.items():
            assert p.shape == d['adapter'][k].shape
            p.copy_(d['adapter'][k].to(p.device))
    optimizer.load_state_dict(d['optimizer'])
    torch.set_rng_state(d['rng'])
    torch.cuda.set_rng_state(d['cuda_rng'])
    return d['step'], d['history']
