"""Sequential larger local runs; fresh test is gated by a technical validation audit."""
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiment import load_pipe, load_sample, cache_latents, infer_marigold, evaluate_depth, seed_all
from run_expanded import read, write, sha, restore_adapter, utc
from train_scaleup import MODES, train

SEEDS = [17, 29, 43]


def matrix():
    return [{'run_id': 'base', 'mode': 'base', 'seed': None}] + [
        {'run_id': f'{mode}_seed{seed}', 'mode': mode, 'seed': seed}
        for mode in MODES for seed in SEEDS] + [{'run_id': 'expert', 'mode': 'expert', 'seed': None}]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--results', default='results/scaleup_v2', type=Path)
    p.add_argument('--stage', choices=['validation', 'test'], required=True)
    return p.parse_args()


def fingerprint(out):
    return {'version': 'scaleup_v2', 'matrix': matrix(), 'training_steps': 320,
            'train_images': 128, 'inference_resolutions': [256, 512], 'denoising_steps': 1,
            'manifest_sha256': sha(out/'split_manifest.json'),
            'protocol_sha256': sha(Path('SCALEUP_PROTOCOL.md')),
            'source_sha256': {n: sha(Path(n)) for n in
                              ['run_scaleup.py', 'train_scaleup.py', 'experiment.py', 'run_expanded.py']}}


def labels(cfg):
    return [('expert', None)] if cfg['mode'] == 'expert' else [(f"{cfg['run_id']}_r{r}", r) for r in [256, 512]]


def complete(out, work, stage, label, ids):
    rd = out/'evaluations'/stage/label
    markpath = rd/'complete.json'
    if not markpath.exists():
        return False
    mark = read(markpath)
    assert mark['protocol_sha256'] == sha(out/'protocol.json')
    assert mark['metrics_sha256'] == sha(rd/'per_image.json')
    assert [r['id'] for r in read(rd/'per_image.json')] == ids
    assert set(mark['prediction_sha256']) == {f'{idx:04d}.npy' for idx in ids}
    for filename, checksum in mark['prediction_sha256'].items():
        assert sha(work/'predictions'/stage/label/filename) == checksum
    return True


def execute(a, cfg, manifest):
    mode, rid = cfg['mode'], cfg['run_id']
    rows = [r for r in manifest['samples'] if r['split'] == ('val' if a.stage == 'validation' else 'test')]
    conditions = labels(cfg)
    if all(complete(a.results, a.work, a.stage, lab, [r['id'] for r in rows]) for lab, _ in conditions):
        print('SKIP_COMPLETE', a.stage, rid, flush=True)
        return
    rd = a.results/'runs'/rid
    config = {**cfg, 'protocol_sha256': sha(a.results/'protocol.json')}
    if (rd/'config.json').exists():
        assert read(rd/'config.json') == config
    else:
        write(rd/'config.json', config)
    write(a.results/'status.json', {'stage': a.stage, 'state': 'running', 'run_id': rid, 'updated_utc': utc()})
    seed_all(cfg['seed'] or 17)
    print('START', a.stage, rid, flush=True)
    if mode == 'expert':
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(str(a.assets/'expert'), local_files_only=True, use_fast=False)
        net = AutoModelForDepthEstimation.from_pretrained(str(a.assets/'expert'), local_files_only=True).to('cuda').eval()
        @torch.inference_mode()
        def expert_infer(image, idx):
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
            inputs = processor(images=image, return_tensors='pt', size={'height': 256, 'width': 256}).to('cuda')
            pred = net(**inputs).predicted_depth
            pred = F.interpolate(pred[:, None], size=image.shape[:2], mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
            torch.cuda.synchronize()
            return pred, time.perf_counter()-start, torch.cuda.max_memory_allocated()/2**20
    else:
        pipe = load_pipe(read(a.assets/'model_path.json')['path'])
        if mode != 'base':
            checkpoint = a.work/'checkpoints'/rid/'adapter.pt'
            if (rd/'training.json').exists():
                stats = read(rd/'training.json')
                assert stats['config_sha256'] == sha(rd/'config.json')
                assert stats['checkpoint_sha256'] == sha(checkpoint)
                restore_adapter(pipe, 'lora', checkpoint)
            else:
                assert a.stage == 'validation', 'Test cannot create new checkpoints'
                tr = [r for r in manifest['samples'] if r['split'] == 'train']
                resolutions = [256] if mode == 'low256' else [512] if mode == 'high512' else [256, 512]
                start = time.perf_counter()
                caches = {r: cache_latents(pipe, tr, a.assets, r) for r in resolutions}
                seconds = time.perf_counter()-start
                stats = train(pipe, caches, mode, cfg['seed'], checkpoint.parent)
                stats.update({'latent_cache_seconds': seconds, 'checkpoint_sha256': sha(checkpoint),
                              'config_sha256': sha(rd/'config.json'), 'training_ids': [r['id'] for r in tr]})
                write(rd/'training.json', stats)
                del caches
            print('TRAIN_READY', rid, round(stats['training_seconds'], 1), 'seconds', flush=True)
    for label, res in conditions:
        if complete(a.results, a.work, a.stage, label, [r['id'] for r in rows]):
            continue
        infer = expert_infer if mode == 'expert' else lambda image, idx: infer_marigold(pipe, image, 1, res, 17+idx)
        for row in rows[:2]:
            image, _ = load_sample(a.assets, row)
            infer(image, row['id'])
        records, hashes = [], {}
        pred_dir = a.work/'predictions'/a.stage/label
        pred_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            image, gt = load_sample(a.assets, row)
            pred, seconds, memory = infer(image, row['id'])
            metrics, _, _ = evaluate_depth(pred, gt)
            records.append({**cfg, 'label': label, 'resolution': res, 'id': row['id'], 'scene': row['scene'],
                            'split': row['split'], **metrics, 'inference_seconds': seconds, 'peak_allocated_mib': memory})
            filename = f"{row['id']:04d}.npy"
            np.save(pred_dir/filename, pred)
            hashes[filename] = sha(pred_dir/filename)
        dest = a.results/'evaluations'/a.stage/label
        write(dest/'per_image.json', records)
        write(dest/'complete.json', {'protocol_sha256': sha(a.results/'protocol.json'),
              'metrics_sha256': sha(dest/'per_image.json'), 'prediction_sha256': hashes, 'finished_utc': utc()})
        print('EVAL_COMPLETE', a.stage, label, len(records), flush=True)


def main():
    a = arguments()
    fixed = fingerprint(a.results)
    if (a.results/'protocol.json').exists():
        assert read(a.results/'protocol.json') == fixed
    else:
        assert a.stage == 'validation'
        write(a.results/'protocol.json', fixed)
    assert read(a.results/'preflight.json')['status'] == 'passed'
    assert read(a.results/'preflight.json')['source_sha256'] == sha(Path('train_scaleup.py'))
    manifest = read(a.results/'split_manifest.json')
    assert manifest['counts'] == {'train': 128, 'val': 32, 'test': 64}
    if a.stage == 'test':
        gate = read(a.results/'validation_gate.json')
        assert gate['status'] == 'passed' and gate['protocol_sha256'] == sha(a.results/'protocol.json')
        for name, checksum in gate['audited_artifact_sha256'].items():
            assert sha(a.results/name) == checksum
        for rid, checksum in gate['checkpoint_sha256'].items():
            assert sha(a.work/'checkpoints'/rid/'adapter.pt') == checksum
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    write(a.results/'environment.json', read('results/expanded_v1/environment.json'))
    for cfg in matrix():
        execute(a, cfg, manifest)
        gc.collect(); torch.cuda.empty_cache()
    write(a.results/'status.json', {'stage': a.stage, 'state': 'complete', 'finished_utc': utc()})
    print('SCALEUP_STAGE_COMPLETE', a.stage, flush=True)


if __name__ == '__main__':
    main()
