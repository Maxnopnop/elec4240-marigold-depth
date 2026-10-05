"""Training-only actual next-update equivalence test, including AdamW state."""
import argparse
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch

from generative_ris_v12.train import update, masks
from multitask_v7.engine import setup
from .state import save, restore, schedule, sha


def main(root):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    repo = root / 'outputs/marigold-depth'
    dest = root / 'work/scaleup_research_v13'
    pilot = json.loads((dest / 'pilot_protocol.json').read_text())
    for path, digest in pilot['source_sha256'].items():
        assert sha(path) == digest, path
    selection = json.loads((dest / 'selection.json').read_text())
    exposure = {}
    for size, key in [(2048, 'small_train_images'), (8192, 'large_train_images')]:
        order = schedule(selection[key])
        assert len(order) == 8192
        assert len(set(order[:2048])) == 2048
        assert len(set(order)) == size
        assert order == schedule(list(reversed(selection[key])))
        exposure[str(size)] = {'unique_at_2048': 2048, 'unique_at_8192': size}
    rows = json.loads((repo / 'results/generative_ris_v12/data_manifest.json').read_text())['records']
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['split'] == 'train':
            groups[r['image_id']][r['ann_id']].append(r)
    iid = sorted(groups)[0]
    pair = [v[0] for v in groups[iid].values()]
    cache_path = root / 'work/marigold-local/generative_ris_v12/cache.pt'
    assert sha(cache_path) == pilot['cache_sha256']
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    targets = masks(pair)
    initial = root / 'work/marigold-local/generative_ris_v11/seed17/adapter_1000.pt'
    ph = sha(dest / 'pilot_protocol.json')
    results = []
    for arm in ['pixel', 'pair_always', 'pair_ready']:
        pipe, params = setup(17, initial)
        pipe.unet.train()
        opt = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=.01)
        first = update(pipe, params, opt, cache, targets, pair, arm, 17, 1)
        path = dest / f'resume_test_{arm}.pt'
        save(path, params, opt, 1, [first], ph)
        expected_loss = update(pipe, params, opt, cache, targets, pair, arm, 17, 2)
        expected = {k: p.detach().cpu().clone() for k, p in params.items()}
        del pipe, params, opt
        gc.collect(); torch.cuda.empty_cache()
        pipe, params = setup(17, initial)
        pipe.unet.train()
        opt = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=.01)
        step, history = restore(path, params, opt, ph)
        assert step == 1 and history == [first]
        actual_loss = update(pipe, params, opt, cache, targets, pair, arm, 17, 2)
        maximum = max((p.detach().cpu() - expected[k]).abs().max().item() for k, p in params.items())
        assert actual_loss == expected_loss and maximum == 0, (arm, maximum, actual_loss, expected_loss)
        results.append({'arm': arm, 'next_update_exact': True, 'max_parameter_difference': maximum})
        del pipe, params, opt
        gc.collect(); torch.cuda.empty_cache()
    receipt = {'status': 'passed', 'training_only': True, 'formal_training_started': False,
               'results': results, 'exposure': exposure,
               'source_sha256': {str(p): sha(p) for p in Path(__file__).parent.glob('*.py')}}
    (dest / 'resume_test.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root)
    except Exception as e:
        (args.root / 'work/scaleup_research_v13/resume_test_error.json').write_text(
            json.dumps({'type': type(e).__name__, 'message': str(e)}, indent=2))
        raise
