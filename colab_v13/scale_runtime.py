"""Cloud scale shared primitives; imports alone never initialize CUDA."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['USE_TF'] = '0'
import gc
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from prepare_scale_cache import ROOT, DRIVE, sha, read, write, copy_verified, EXPECTED_MANIFEST

WORK = DRIVE / 'scale_experiment'
OUT = WORK
ARMS = ['pixel', 'pair_always', 'pair_ready']
SEEDS = [17, 29]
SIZES = [2048, 4096]
STEPS = 4096
SIZE = (192, 256)


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def label(arm, size, seed):
    return f'{arm}_n{size}_s{seed}'


def schedule(ids, updates=4096):
    ids = sorted(ids)
    assert ids and len(set(ids)) == len(ids)
    result = []
    for epoch in range((updates + len(ids) - 1) // len(ids)):
        part = ids.copy()
        random.Random(424113 + epoch).shuffle(part)
        result.extend(part)
    return result[:updates]


def rows():
    assert sha(ROOT/'cloud_data_manifest.json') == EXPECTED_MANIFEST
    return read(ROOT/'cloud_data_manifest.json')['records']


def setup(seed, checkpoint):
    from frozen_functions import setup as frozen_setup
    return frozen_setup(seed, checkpoint)


def text_prompt(text):
    return 'A binary segmentation mask of '+text+', white foreground and black background.'


def encode(pipe, tensor):
    return pipe.vae.encode(tensor.to(torch.bfloat16)).latent_dist.mode()*pipe.vae.config.scaling_factor


def latent_prediction(pipe, rgb, embedding, noise):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        v = pipe.unet(torch.cat([rgb, noise], 1), 999, embedding).sample.float()
        z = pipe.scheduler.step(v, 999, noise.float(), eta=0.).prev_sample
        return pipe.vae.decode(z.to(torch.bfloat16)/pipe.vae.config.scaling_factor).sample.float().mean(1, keepdim=True)


def iou(pred, truth):
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum()/union) if union else 1.


def groups(rr):
    result = defaultdict(lambda: defaultdict(list))
    for r in rr:
        if r['split'] == 'train':
            result[r['image_id']][r['ann_id']].append(r)
    assert len(result) == 4096 and all(len(v) == 2 for v in result.values())
    return {i: [sorted(v[a], key=lambda r:r['sent_id']) for a in sorted(v)] for i,v in result.items()}


def load_cache():
    """Load all BF16 text embeddings on CPU; verify every durable/local shard."""
    receipt = read(DRIVE/'scale_cache/cache_audit.json')
    assert receipt['status'] == 'complete' and receipt['manifest_sha256'] == EXPECTED_MANIFEST
    cache = {'rgb': {}, 'mask': {}, 'text': {}}
    for name, record in receipt['shards'].items():
        local = ROOT/'scale_cache'/name
        assert sha(DRIVE/'scale_cache'/name) == record['sha256']
        copy_verified(DRIVE/'scale_cache'/name, local)
        shard = torch.load(local, map_location='cpu', weights_only=True)
        assert shard['manifest_sha256'] == EXPECTED_MANIFEST
        for k in cache:
            assert not set(cache[k]) & set(shard[k]), 'Duplicate cache key'
            cache[k].update(shard[k])
    assert len(cache['rgb']) == 4096 and len(cache['mask']) == 8192
    return cache


def update(pipe, params, opt, cache, pair, arm, seed, step):
    from frozen_functions import update as frozen_update
    targets = {}
    for r in pair:
        a = np.array(Image.open(r['mask']).resize((256,192), Image.Resampling.NEAREST), copy=True)
        targets[r['ann_id']] = torch.from_numpy(a)[None,None].float()/255
    torch.cuda.synchronize()
    begin = time.monotonic()
    result = frozen_update(pipe, params, opt, cache, targets, pair, arm, seed, step)
    torch.cuda.synchronize()
    return dict(result, step=step, seconds=time.monotonic()-begin, image_id=pair[0]['image_id'], sent_ids=[r['sent_id'] for r in pair])


def save_state(dest, params, opt, step, history, protocol_sha):
    """Immutable generations; commit the pointer only after durable hash verification."""
    dest.mkdir(parents=True, exist_ok=True)
    local = ROOT/'state-staging.pt'
    torch.save({'adapter': {k:p.detach().cpu().clone() for k,p in params.items()},
                'optimizer':opt.state_dict(), 'step':step, 'history':history,
                'rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(),
                'protocol_sha256':protocol_sha}, local)
    digest = sha(local)
    name = f'resume_{step:05d}_{digest[:16]}.pt'
    copy_verified(local, dest/name)
    # Keep all generations: a partial write never destroys the previous committed state.
    write(dest/'latest.json', {'file':name, 'sha256':digest, 'step':step, 'protocol_sha256':protocol_sha})
    return dest/name


def restore(path, params, opt, protocol_sha):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['protocol_sha256'] == protocol_sha
    assert len(state['history']) == state['step'] and set(state['adapter']) == set(params)
    with torch.no_grad():
        for k,p in params.items(): p.copy_(state['adapter'][k].to(p.device))
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    return state['step'], state['history']


def latest_state(dest, protocol_sha):
    try:
        pointer=read(dest/'latest.json')
    except (FileNotFoundError,json.JSONDecodeError):
        # Recover from immutable generations if Drive lost the mutable pointer.
        candidates=[]
        for path in dest.glob('resume_*.pt'):
            parts=path.stem.split('_')
            if len(parts)==3 and parts[1].isdigit():candidates.append((int(parts[1]),parts[2],path))
        for step,prefix,path in sorted(candidates,reverse=True):
            digest=sha(path)
            assert digest.startswith(prefix),'Immutable state hash mismatch'
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==protocol_sha and state['step']==step and len(state['history'])==step
            return path,step
        return None,0
    assert pointer['protocol_sha256']==protocol_sha
    path=dest/pointer['file']
    assert path.parent==dest and sha(path)==pointer['sha256']
    return path,pointer['step']


class Budget:
    """Precharge bounded chunks so runtime loss cannot erase unrecorded compute.

    A completed chunk refunds unused reservation. An interrupted chunk stays charged.
    The watchdog kills the owned process before its precharged deadline.
    """
    def __init__(self, path, initial_seconds, cap=48*3600):
        self.path, self.cap = path, cap
        self.journal=path.with_name(path.stem+'_journal')
        self.journal.mkdir(parents=True,exist_ok=True)
        entries=sorted(self.journal.glob('entry_*.json'))
        if entries:
            self.state=read(entries[-1])
            self.sequence=int(entries[-1].stem.split('_')[1])
        else:
            self.state=read(path) if path.exists() else {'charged_seconds':initial_seconds,'reservations':0}
            self.sequence=0
            self.persist()
        self.active = None

    def persist(self):
        self.sequence+=1
        entry=self.journal/f'entry_{self.sequence:08d}.json'
        assert not entry.exists(),'Budget journal sequence collision'
        write(entry,self.state)
        assert read(entry)==self.state
        write(self.path,self.state) # display summary only; journal is authoritative

    def reserve(self, seconds, stage):
        assert self.active is None
        assert self.state['charged_seconds'] + seconds <= self.cap, 'Cumulative budget exhausted'
        assert not (DRIVE/'PAUSE').exists(), 'Drive PAUSE marker'
        self.state['charged_seconds'] += seconds
        self.state['reservations'] += 1
        self.state['stage'] = stage
        self.persist()
        self.active = (time.monotonic(), seconds)
        # Hard stop requires no main-thread progress; GPU stalls cannot bypass cap.
        import threading
        self.timer = threading.Timer(seconds, lambda: os._exit(124))
        self.timer.daemon = True
        self.timer.start()

    def settle(self):
        assert self.active is not None
        begin, seconds = self.active
        elapsed = time.monotonic()-begin
        assert elapsed <= seconds
        self.timer.cancel()
        self.state['charged_seconds'] -= seconds-elapsed
        self.persist()
        self.active = None


def verify_training_barrier(protocol_sha):
    for size in SIZES:
        for seed in SEEDS:
            for arm in ARMS:
                dest = WORK/'runs'/label(arm,size,seed)
                receipt = read(dest/'trained.json')
                assert receipt['updates'] == STEPS and receipt['protocol_sha256'] == protocol_sha
                for step in [2048,4096]:
                    assert sha(dest/f'adapter_{step}.pt') == receipt['checkpoints'][str(step)]
