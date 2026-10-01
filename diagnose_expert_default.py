"""Post-hoc evaluation of the pinned specialist's unmodified processor defaults."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from run_expanded import read, write, sha, utc
from run_robustness import fit_calibration, fixed_metrics
from experiment import load_sample, evaluate_depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--results', default='results/expert_default_diagnostic_v1', type=Path)
    a = p.parse_args()
    out = a.results
    manifest_path = Path('results/robustness_v3/split_manifest.json')
    assert read('results/robustness_v3/verification.json')['metrics_recomputed'] == 9182
    manifest = read(manifest_path)
    protocol = {'design': 'post-hoc official-default preprocessing, no new confirmatory claims',
                'model_id': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
                'revision': '8078d68a9c75a972131914f6afd0c1723be0da7f',
                'manifest_sha256': sha(manifest_path), 'script_sha256': sha(__file__),
                'protocol_sha256': sha('EXPERT_DEFAULT_DIAGNOSTIC.md'),
                'processor_config_sha256': sha(a.assets/'expert/preprocessor_config.json')}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json') == protocol
    else:
        write(out/'protocol.json', protocol)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    proc = AutoImageProcessor.from_pretrained(str(a.assets/'expert'), local_files_only=True, use_fast=False)
    net = AutoModelForDepthEstimation.from_pretrained(str(a.assets/'expert'), local_files_only=True).to('cuda').eval()
    shapes = set()
    @torch.inference_mode()
    def infer(image):
        torch.cuda.synchronize(); start = time.perf_counter()
        inputs = proc(images=image, return_tensors='pt').to('cuda')
        shapes.add(tuple(inputs.pixel_values.shape[-2:]))
        pred = net(**inputs).predicted_depth
        pred = F.interpolate(pred[:, None], size=image.shape[:2], mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
        torch.cuda.synchronize()
        return pred, time.perf_counter()-start
    calibration = None
    all_records, hashes = [], {}
    for stage, rows in [('validation', manifest['validation']), ('test', manifest['fresh31']+manifest['observed64'])]:
        dest = out/stage
        rawdir = a.work/'predictions'/stage
        rawdir.mkdir(parents=True, exist_ok=True)
        records, pairs = [], []
        for row in rows[:2]:
            infer(load_sample(a.assets, row)[0])
        for row in rows:
            rgb, gt = load_sample(a.assets, row)
            assert sha(a.assets/'subset'/row['file']) == row['sha256']
            pred, seconds = infer(rgb)
            filename = f"{row['id']:04d}.npy"
            np.save(rawdir/filename, pred)
            hashes[f'{stage}/{filename}'] = sha(rawdir/filename)
            aligned, _, _ = evaluate_depth(pred, gt)
            record = {'id': row['id'], 'scene': row['scene'], 'cohort': row['cohort'], 'stage': stage,
                      'inference_seconds': seconds, **aligned,
                      **{'native_'+k: v for k, v in fixed_metrics(pred, gt).items()}}
            if stage == 'validation':
                pairs.append((pred, gt))
            else:
                record.update({'calibrated_'+k: v for k, v in fixed_metrics(pred, gt, calibration['scale'], calibration['shift']).items()})
            records.append(record)
        if stage == 'validation':
            calibration = fit_calibration(pairs)
            write(out/'calibration.json', {'fitted_utc': utc(), 'validation_ids': [r['id'] for r in rows], **calibration})
        write(dest/'per_image.json', records)
        all_records.extend(records)
        print('DEFAULT_EXPERT_COMPLETE', stage, len(rows), 'input_shapes', shapes, flush=True)
    total = 0
    indexed = {r['id']: r for r in manifest['validation']+manifest['fresh31']+manifest['observed64']}
    for rec in all_records:
        path = a.work/'predictions'/rec['stage']/f"{rec['id']:04d}.npy"
        assert sha(path) == hashes[f"{rec['stage']}/{path.name}"]
        pred = np.load(path)
        _, gt = load_sample(a.assets, indexed[rec['id']])
        metrics, _, _ = evaluate_depth(pred, gt)
        metrics.update({'native_'+k: v for k, v in fixed_metrics(pred, gt).items()})
        if rec['stage'] == 'test':
            metrics.update({'calibrated_'+k: v for k, v in fixed_metrics(pred, gt, calibration['scale'], calibration['shift']).items()})
        assert all(np.isclose(rec[k], v, atol=1e-10, rtol=0) for k, v in metrics.items())
        total += 1
    assert total == 127 and shapes == {(518, 686)}
    summary = []
    for cohort in ['fresh31', 'observed64']:
        rows = [r for r in all_records if r['cohort'] == cohort]
        summary.append({'cohort': cohort, 'images': len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in
                       ['abs_rel', 'native_abs_rel', 'native_rmse_m', 'native_delta1', 'calibrated_abs_rel', 'calibrated_rmse_m', 'calibrated_delta1', 'inference_seconds']}})
    write(out/'summary.json', summary)
    write(out/'verification.json', {'status': 'passed', 'predictions_recomputed': total, 'actual_input_height_width': [518, 686],
          'protocol_sha256': sha(out/'protocol.json'), 'calibration_sha256': sha(out/'calibration.json'),
          'per_image_sha256': {s: sha(out/s/'per_image.json') for s in ['validation', 'test']}, 'prediction_sha256': hashes})
    lines = ['# Official-default specialist preprocessing diagnostic', '',
             'This post-hoc diagnostic was added after the clean robustness_v3 results were observed. It evaluates the pinned '
             'Depth Anything V2 indoor model with its unchanged processor defaults. Actual input is 518x686, larger than Marigold 384x512. '
             'It is a descriptive reference with a different pixel budget, not an added confirmatory comparison.', '',
             'The model is metric-fine-tuned on Hypersim. Fixed calibration uses only the same 32validation scenes and is saved before test prediction. '
             'All 127 predictions and recomputed metrics passed the audit.', '',
             '| Cohort | GT-aligned AbsRel | Native AbsRel | Native RMSE (m) | Calibrated AbsRel | Calibrated RMSE (m) |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['cohort']} | {r['abs_rel']:.5f} | {r['native_abs_rel']:.5f} | {r['native_rmse_m']:.4f} | {r['calibrated_abs_rel']:.5f} | {r['calibrated_rmse_m']:.4f} |")
    lines += ['', 'These numbers do not replace the predefined custom-input comparisons. Compare all input settings, distinguish native '
              'from calibrated outputs, and do not attribute preprocessing differences solely to model architecture. No cross-dataset or universal superiority claim follows.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('DEFAULT_EXPERT_VERIFIED', summary, flush=True)


if __name__ == '__main__':
    main()
