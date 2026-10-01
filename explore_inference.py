"""Frozen validation-only factorial study of LoRA merging and inference resolution."""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from experiment import load_pipe, load_sample, infer_marigold, evaluate_depth, seed_all
from run_expanded import read, write, sha, restore_adapter, utc


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--expanded-work', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--results', default='results/validation_exploration_v1', type=Path)
    return p.parse_args()


def configurations():
    return [{'label': f'{mode}_r{resolution}_s{steps}', 'mode': mode,
             'resolution': resolution, 'steps': steps}
            for mode in ['base', 'unmerged', 'merged']
            for resolution in [256, 512] for steps in [1, 4]]


def inputs(a):
    manifest_path = Path('results/expanded_v1/split_manifest.json')
    rows = [r for r in read(manifest_path)['samples'] if r['split'] == 'val']
    assert len(rows) == 16 and len({r['scene'] for r in rows}) == 16
    for row in rows:
        assert sha(a.assets / 'subset' / row['file']) == row['sha256']
    checkpoint = a.expanded_work / 'checkpoints/lora64_seed17/adapter.pt'
    assert sha(checkpoint) == read('results/expanded_v1/runs/lora64_seed17/training.json')['checkpoint_sha256']
    return rows, checkpoint, manifest_path


def main():
    a = arguments()
    rows, checkpoint, manifest_path = inputs(a)
    configs = configurations()
    rng = np.random.default_rng(4244)
    orders = [[configs[i]['label'] for i in rng.permutation(len(configs))] for _ in range(3)]
    root = Path(__file__).parent
    protocol = {'version': 'validation_exploration_v1', 'validation_ids': [r['id'] for r in rows],
                'checkpoint': 'lora64_seed17', 'checkpoint_sha256': sha(checkpoint),
                'manifest_sha256': sha(manifest_path), 'inference_seed_offset': 17,
                'warmup_images': 2, 'timing_rounds': 3, 'orders': orders, 'configurations': configs,
                'protocol_sha256': sha(root / 'EXPLORATION_PROTOCOL.md'),
                'source_sha256': {name: sha(root / name) for name in
                                  ['explore_inference.py', 'experiment.py', 'run_expanded.py']}}
    config_path = a.results / 'protocol.json'
    if config_path.exists():
        assert read(config_path) == protocol, 'Frozen exploration changed'
    else:
        write(config_path, protocol)
    write(a.results / 'environment.json', read('results/expanded_v1/environment.json'))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    by_label = {c['label']: c for c in configs}
    for round_index, order in enumerate(orders, 1):
        for label in order:
            cfg = by_label[label]
            run = f'round{round_index}_{label}'
            dest = a.results / 'runs' / run
            pred_dir = a.work / 'predictions' / run
            complete = dest / 'complete.json'
            if complete.exists():
                mark = read(complete)
                assert mark['protocol_sha256'] == sha(config_path)
                assert mark['metrics_sha256'] == sha(dest / 'per_image.json')
                assert all(sha(pred_dir / filename) == checksum
                           for filename, checksum in mark['prediction_sha256'].items())
                print('SKIP_COMPLETE', run, flush=True)
                continue
            write(a.results / 'status.json', {'state': 'running', 'run': run, 'updated_utc': utc()})
            seed_all(17)
            pipe = load_pipe(read(a.assets / 'model_path.json')['path'])
            if cfg['mode'] != 'base':
                restore_adapter(pipe, 'lora', checkpoint)
            before = sum(isinstance(m, BaseTunerLayer) for m in pipe.unet.modules())
            if cfg['mode'] == 'merged':
                assert before > 0
                pipe.unet.fuse_lora(safe_fusing=True)
                assert all(m.merged for m in pipe.unet.modules() if isinstance(m, BaseTunerLayer))
                pipe.unet.unload_lora()
            after = sum(isinstance(m, BaseTunerLayer) for m in pipe.unet.modules())
            assert (after > 0) == (cfg['mode'] == 'unmerged')
            def infer(image, idx):
                return infer_marigold(pipe, image, cfg['steps'], cfg['resolution'], 17 + idx)
            for row in rows[:2]:
                image, _ = load_sample(a.assets, row)
                infer(image, row['id'])
            pred_dir.mkdir(parents=True, exist_ok=True)
            records, hashes = [], {}
            for row in rows:
                image, gt = load_sample(a.assets, row)
                pred, seconds, memory = infer(image, row['id'])
                metrics, _, _ = evaluate_depth(pred, gt)
                filename = f"{row['id']:04d}.npy"
                np.save(pred_dir / filename, pred)
                hashes[filename] = sha(pred_dir / filename)
                records.append({**cfg, 'round': round_index, 'id': row['id'], 'scene': row['scene'],
                                'split': 'val', **metrics, 'inference_seconds': seconds,
                                'peak_allocated_mib': memory})
            write(dest / 'per_image.json', records)
            write(complete, {'protocol_sha256': sha(config_path),
                            'metrics_sha256': sha(dest / 'per_image.json'),
                            'prediction_sha256': hashes, 'images': len(records),
                            'adapter_wrappers_before': before, 'adapter_wrappers_after': after,
                            'finished_utc': utc()})
            print('COMPLETE', run, 'AbsRel', np.mean([r['abs_rel'] for r in records]),
                  'seconds', np.mean([r['inference_seconds'] for r in records]), flush=True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    write(a.results / 'status.json', {'state': 'complete', 'timed_predictions': 576, 'finished_utc': utc()})
    print('EXPLORATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
