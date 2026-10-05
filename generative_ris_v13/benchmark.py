"""Deterministic timing gate; only old training pixels, no holdout prediction."""
from .common import *
import gc
import time
import numpy as np
from collections import defaultdict
from generative_ris_v12.train import update, masks
from multitask_v7.engine import setup, encode
from generative_ris_v11.train_baseline import latent_prediction, text_prompt, SIZE
from PIL import Image

def main():
    deterministic()
    pilot = read(WORK / 'pilot_protocol.json')
    for path, digest in pilot['source_sha256'].items():
        assert sha(path) == digest
    rr = [r for r in rows() if r['split'] == 'train' and r['image_id'] in pilot['training_images']]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rr: grouped[r['image_id']][r['ann_id']].append(r)
    pairs = [[v[0] for v in grouped[i].values()] for i in sorted(grouped)]
    path = ROOT / 'work/marigold-local/generative_ris_v12/cache.pt'
    assert sha(path) == pilot['cache_sha256']
    cache = torch.load(path, map_location='cpu', weights_only=True)
    targets = masks(rr); results = []
    for arm in ARMS:
        pipe, params = setup(17, OLD / 'seed17/adapter_1000.pt')
        pipe.unet.train()
        opt = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=.01)
        times = []; torch.cuda.reset_peak_memory_stats()
        for step in range(1, 65):
            torch.cuda.synchronize(); begin = time.perf_counter()
            result = update(pipe, params, opt, cache, targets, pairs[step-1], arm, 17, step)
            torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
        results.append({'arm': arm, 'warm_mean_seconds': float(np.mean(times[16:])),
                        'warm_p90_seconds': float(np.quantile(times[16:], .9)),
                        'peak_mib': torch.cuda.max_memory_allocated()/2**20})
        print(results[-1], flush=True)
        if arm == ARMS[-1]:
            pipe.unet.eval(); timings = []
            with torch.inference_mode():
                for pair in pairs[:16]:
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    for r in pair:
                        rgb = cache['rgb'][r['image_id']].cuda()
                        noise = torch.zeros_like(rgb)
                        latent_prediction(pipe, rgb, cache['text'][r['sent_id']].cuda(), noise)
                    torch.cuda.synchronize(); timings.append(time.perf_counter()-begin)
            pair_seconds = float(np.mean(timings[2:]))
        del pipe, params, opt; gc.collect(); torch.cuda.empty_cache()
    training = max(r['warm_p90_seconds'] for r in results)*18*8192/3600
    # Reserve 50% inference margin + one blank prediction/image, and 4h cache/IO/audit.
    evaluation = pair_seconds*1.5*36*320*1.5/3600
    estimate = training + evaluation + 4
    write(WORK/'deterministic_benchmark.json', {'status':'complete', 'training_only':True,
          'results':results, 'evaluation_pair_seconds':pair_seconds,
          'conservative_training_hours':training, 'estimated_evaluation_hours':evaluation,
          'cache_io_audit_reserve_hours':4, 'estimated_total_hours':estimate,
          'budget_gate_pass':estimate<=48, 'source_sha256':sha(__file__)})

if __name__ == '__main__':
    try: main()
    except Exception as e:
        write(WORK/'benchmark_error.json', {'type':type(e).__name__, 'message':str(e)}); raise
