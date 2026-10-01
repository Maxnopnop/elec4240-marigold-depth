"""Resumable v3 replication; no test execution before a frozen calibration/audit gate."""
import argparse
import gc
import importlib.metadata
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from experiment import load_pipe, load_sample, cache_latents, infer_marigold, evaluate_depth, mask_for, seed_all
from run_expanded import read, write, sha, restore_adapter, utc
from train_scaleup import train

SEEDS = [17, 29, 43, 59, 71]
MODES = ['high512', 'mixed']
DRAWS = ['draw1', 'draw2', 'draw3']


def matrix():
    return [{'run_id': 'base', 'mode': 'base', 'draw': None, 'seed': None}] + [
        {'run_id': f'{d}_{m}_seed{s}', 'mode': m, 'draw': d, 'seed': s}
        for d in DRAWS for s in SEEDS for m in MODES] + [
        {'run_id': f'expert{r}', 'mode': 'expert', 'draw': None, 'seed': None, 'input_long_side': r}
        for r in [256, 504]]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--results', default='results/robustness_v3', type=Path)
    p.add_argument('--stage', choices=['validation', 'test', 'corruption'], required=True)
    p.add_argument('--restore', action='store_true')
    return p.parse_args()


def fingerprint(out):
    return {'version': 'robustness_v3', 'matrix': matrix(), 'training_steps': 320,
            'resolutions': [256, 512], 'inference_seed_offset': 17,
            'manifest_sha256': sha(out/'split_manifest.json'),
            'protocol_sha256': sha('ROBUSTNESS_PROTOCOL.md'),
            'source_sha256': {n: sha(n) for n in ['run_robustness.py', 'prepare_robustness.py',
                                               'train_scaleup.py', 'experiment.py', 'run_expanded.py', 'robustness_stats.py']}}


def conditions(cfg, stage):
    if stage == 'corruption':
        if cfg['mode'] not in ['base', 'mixed'] and cfg['run_id'] != 'expert504':
            return []
        base = cfg['run_id'] if cfg['mode'] == 'expert' else cfg['run_id']+'_r512'
        return [(base+'_'+c, 512, c) for c in ['dark', 'blur']]
    if cfg['mode'] == 'expert':
        return [(cfg['run_id'], cfg['input_long_side'], 'clean')]
    return [(cfg['run_id']+f'_r{r}', r, 'clean') for r in [256, 512]]


def rows_for(manifest, stage):
    if stage == 'validation':
        return manifest['validation']
    if stage == 'test':
        return manifest['fresh31'] + manifest['observed64']
    return manifest['fresh31']


def perturb(image, kind):
    if kind == 'dark':
        return np.rint(image.astype(np.float32)*.5).clip(0, 255).astype(np.uint8)
    if kind == 'blur':
        return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=2)))
    assert kind == 'clean'
    return image


def fixed_metrics(pred, gt, scale=1., shift=0.):
    pred = np.asarray(pred, dtype=np.float64)
    assert pred.shape == gt.shape and np.isfinite(pred).all()
    mask = mask_for(gt)
    y = gt[mask].astype(np.float64)
    z = np.clip(pred[mask]*scale+shift, .001, 10)
    return {'abs_rel': float(np.mean(abs(z-y)/y)), 'rmse_m': float(np.sqrt(np.mean((z-y)**2))),
            'delta1': float(np.mean(np.maximum(z/y, y/z) < 1.25))}


def fit_calibration(pairs):
    moments = []
    for pred, gt in pairs:
        mask = mask_for(gt)
        x = pred[mask].astype(np.float64)
        y = gt[mask].astype(np.float64)
        moments.append([x.mean(), y.mean(), np.mean(x*x), np.mean(x*y)])
    mx, my, mxx, mxy = np.mean(moments, axis=0)
    scale = float((mxy-mx*my)/(mxx-mx*mx)) if mxx-mx*mx > 1e-12 else 0.
    return {'scale': scale, 'shift': float(my-scale*mx), 'constant_depth_m': float(my),
            'calibration_images': len(moments), 'objective': 'equal-image-weighted squared depth error'}


def complete(a, label, rows):
    dest = a.results/'evaluations'/a.stage/label
    if not (dest/'complete.json').exists():
        return False
    mark = read(dest/'complete.json')
    assert mark['protocol_sha256'] == sha(a.results/'protocol.json')
    assert mark['metrics_sha256'] == sha(dest/'per_image.json')
    assert [r['id'] for r in read(dest/'per_image.json')] == [r['id'] for r in rows]
    assert set(mark['prediction_sha256']) == {f"{r['id']:04d}.npy" for r in rows}
    if a.stage != 'validation':
        assert mark['calibration_sha256'] == sha(a.results/'calibration.json')
    for filename, checksum in mark['prediction_sha256'].items():
        assert sha(a.work/'predictions'/a.stage/label/filename) == checksum
    return True


def run(a, cfg, manifest):
    cases = conditions(cfg, a.stage)
    rows = rows_for(manifest, a.stage)
    if not cases or all(complete(a, label, rows) for label, _, _ in cases):
        print('SKIP_COMPLETE', a.stage, cfg['run_id'], flush=True)
        return
    rid = cfg['run_id']
    rd = a.results/'runs'/rid
    config = {**cfg, 'protocol_sha256': sha(a.results/'protocol.json')}
    if (rd/'config.json').exists():
        assert read(rd/'config.json') == config
    else:
        write(rd/'config.json', config)
    write(a.results/'status.json', {'stage': a.stage, 'run_id': rid, 'state': 'running', 'updated_utc': utc()})
    seed_all(cfg['seed'] or 17)
    print('START', a.stage, rid, flush=True)
    if cfg['mode'] == 'expert':
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(str(a.assets/'expert'), local_files_only=True, use_fast=False)
        net = AutoModelForDepthEstimation.from_pretrained(str(a.assets/'expert'), local_files_only=True).to('cuda').eval()
        @torch.inference_mode()
        def infer(image, idx, res):
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
            if cfg['input_long_side'] == 504:
                resized = Image.fromarray(image).resize((504, 378), Image.Resampling.BICUBIC)
                inputs = proc(images=resized, return_tensors='pt', do_resize=False).to('cuda')
                assert tuple(inputs.pixel_values.shape[-2:]) == (378, 504)
            else:
                inputs = proc(images=image, return_tensors='pt', size={'height': 256, 'width': 256}).to('cuda')
                assert tuple(inputs.pixel_values.shape[-2:]) == (252, 336)
            pred = net(**inputs).predicted_depth
            pred = F.interpolate(pred[:, None], size=image.shape[:2], mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
            torch.cuda.synchronize()
            return pred, time.perf_counter()-started, torch.cuda.max_memory_allocated()/2**20
    else:
        pipe = load_pipe(read(a.assets/'model_path.json')['path'])
        if cfg['mode'] != 'base':
            checkpoint = a.work/'checkpoints'/rid/'adapter.pt'
            if (rd/'training.json').exists():
                stats = read(rd/'training.json')
                assert stats['config_sha256'] == sha(rd/'config.json')
                assert stats['checkpoint_sha256'] == sha(checkpoint)
                restore_adapter(pipe, 'lora', checkpoint)
            else:
                assert a.stage == 'validation'
                tr = manifest['draws'][cfg['draw']]
                start = time.perf_counter()
                resolutions = [512] if cfg['mode'] == 'high512' else [256, 512]
                caches = {r: cache_latents(pipe, tr, a.assets, r) for r in resolutions}
                seconds = time.perf_counter()-start
                stats = train(pipe, caches, cfg['mode'], cfg['seed'], checkpoint.parent)
                stats.update({'latent_cache_seconds': seconds, 'config_sha256': sha(rd/'config.json'),
                              'checkpoint_sha256': sha(checkpoint), 'training_ids': [r['id'] for r in tr]})
                write(rd/'training.json', stats)
                del caches
            print('TRAIN_READY', rid, round(stats['training_seconds'], 2), flush=True)
        def infer(image, idx, res):
            return infer_marigold(pipe, image, 1, res, 17+idx)
    calibration = read(a.results/'calibration.json') if a.stage != 'validation' else None
    for label, resolution, corruption in cases:
        if complete(a, label, rows):
            continue
        for row in rows[:2]:
            rgb, _ = load_sample(a.assets, row)
            infer(perturb(rgb, corruption), row['id'], resolution)
        hashes, records = {}, []
        pred_dir = a.work/'predictions'/a.stage/label
        pred_dir.mkdir(parents=True, exist_ok=True)
        clean_label = label if corruption == 'clean' else label.rsplit('_', 1)[0]
        for row in rows:
            rgb, gt = load_sample(a.assets, row)
            pred, seconds, memory = infer(perturb(rgb, corruption), row['id'], resolution)
            metrics, _, _ = evaluate_depth(pred, gt)
            record = {**cfg, 'label': label, 'resolution': resolution, 'corruption': corruption,
                      'id': row['id'], 'scene': row['scene'], 'cohort': row['cohort'], **metrics,
                      'inference_seconds': seconds, 'peak_allocated_mib': memory}
            if calibration is not None:
                c = calibration['conditions'][clean_label]
                record.update({'calibrated_'+k: v for k, v in fixed_metrics(pred, gt, c['scale'], c['shift']).items()})
                if cfg['mode'] == 'expert':
                    record.update({'native_'+k: v for k, v in fixed_metrics(pred, gt).items()})
            filename = f"{row['id']:04d}.npy"
            np.save(pred_dir/filename, pred)
            hashes[filename] = sha(pred_dir/filename)
            records.append(record)
        dest = a.results/'evaluations'/a.stage/label
        write(dest/'per_image.json', records)
        write(dest/'complete.json', {'protocol_sha256': sha(a.results/'protocol.json'),
              'metrics_sha256': sha(dest/'per_image.json'), 'prediction_sha256': hashes,
              'calibration_sha256': sha(a.results/'calibration.json') if calibration else None,
              'finished_utc': utc()})
        print('EVAL_COMPLETE', a.stage, label, len(records), flush=True)


def main():
    a = arguments()
    fixed = fingerprint(a.results)
    if (a.results/'protocol.json').exists():
        assert read(a.results/'protocol.json') == fixed
    else:
        assert a.stage == 'validation'
        write(a.results/'protocol.json', fixed)
    manifest = read(a.results/'split_manifest.json')
    if a.stage != 'validation':
        gate = read(a.results/'validation_gate.json')
        assert gate['status'] == 'passed' and gate['protocol_sha256'] == sha(a.results/'protocol.json')
        assert gate['calibration_sha256'] == sha(a.results/'calibration.json')
        for name, checksum in gate['audited_artifact_sha256'].items():
            assert sha(a.results/name) == checksum
        for rid, checksum in gate['checkpoint_sha256'].items():
            assert sha(a.work/'checkpoints'/rid/'adapter.pt') == checksum
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    env = {'python': platform.python_version(), 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'total_vram_mib': torch.cuda.get_device_properties(0).total_memory/2**20,
           'packages': {n: importlib.metadata.version(n) for n in
                        ['torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'accelerate', 'numpy', 'Pillow', 'h5py', 'scipy']}}
    if (a.results/'environment.json').exists():
        assert read(a.results/'environment.json') == env
    else:
        write(a.results/'environment.json', env)
    for cfg in matrix():
        run(a, cfg, manifest)
        gc.collect(); torch.cuda.empty_cache()
    write(a.results/'status.json', {'stage': a.stage, 'state': 'complete', 'finished_utc': utc()})
    print('ROBUSTNESS_STAGE_COMPLETE', a.stage, flush=True)


if __name__ == '__main__':
    main()
