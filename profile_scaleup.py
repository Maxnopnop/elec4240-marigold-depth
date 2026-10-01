"""Training-data-only preflight; smoke checkpoints never enter reported runs."""
import argparse
import gc
from pathlib import Path
import torch
from experiment import load_pipe, load_sample, cache_latents, infer_marigold, evaluate_depth
from train_scaleup import MODES, train
from run_expanded import read, write, sha, restore_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    a = p.parse_args()
    rows = [r for r in read('results/expanded_v1/split_manifest.json')['samples'] if r['split'] == 'train'][:2]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    reports = []
    for mode in MODES:
        pipe = load_pipe(read(a.assets/'model_path.json')['path'])
        caches = {res: cache_latents(pipe, rows, a.assets, res) for res in [256, 512]}
        stats = train(pipe, caches, mode, 17, a.work/mode, steps=4)
        image, gt = load_sample(a.assets, rows[0])
        pred, seconds, memory = infer_marigold(pipe, image, 1, 512, 17+rows[0]['id'])
        evaluate_depth(pred, gt)
        del pipe, caches
        gc.collect(); torch.cuda.empty_cache()
        pipe = load_pipe(read(a.assets/'model_path.json')['path'])
        restore_adapter(pipe, 'lora', a.work/mode/'adapter.pt')
        pred2, _, _ = infer_marigold(pipe, image, 1, 512, 17+rows[0]['id'])
        import numpy as np
        assert np.array_equal(pred, pred2), 'Preflight restoration mismatch'
        stats['restoration_max_difference'] = float(abs(pred-pred2).max())
        stats['inference_512_peak_mib'] = memory
        assert stats['peak_allocated_mib'] < 6800, 'Insufficient safety margin'
        reports.append(stats)
        del pipe
        gc.collect(); torch.cuda.empty_cache()
    write('results/scaleup_v2/preflight.json', {'status': 'passed', 'training_frame_ids': [r['id'] for r in rows],
          'source_sha256': sha(Path('train_scaleup.py')), 'runs': reports})
    print('PREFLIGHT_PASSED', [(r['mode'], round(r['peak_allocated_mib'])) for r in reports], flush=True)


if __name__ == '__main__':
    main()
