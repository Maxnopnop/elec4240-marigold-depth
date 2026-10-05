"""Cloud-only training cache engineering; no optimizer or holdout inference.

Each immutable shard is verified after copying to the authorized Drive folder.
An interrupted shard is reconstructed; completed shards are reused by hash.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['USE_TF'] = '0'
import hashlib
import json
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path('/content/v13_cloud_scale')
DRIVE = Path('/content/drive/MyDrive/ELEC4240/v13-cloud-scale-2026-10-05')
EXPECTED_MANIFEST = 'd1f063ffba5a4ebca6d8b0a9145c038838bfc612691c88001429b61bd2599a30'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def copy_verified(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    if target.exists():
        assert sha(target) == digest, ('Existing durable file differs', target)
    else:
        tmp = target.with_name(target.name + '.part')
        shutil.copyfile(source, tmp)
        assert sha(tmp) == digest
        tmp.replace(target)
    assert sha(target) == digest
    return digest


def main():
    assert sys.platform == 'linux' and ROOT.is_dir(), 'Cloud only; local GPU remains paused'
    assert DRIVE.is_dir(), 'Authorized project Drive directory must be mounted'
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from importlib.metadata import version
    started = time.monotonic()
    dest = ROOT / 'scale_cache'
    durable = DRIVE / 'scale_cache'
    dest.mkdir(exist_ok=True)
    durable.mkdir(exist_ok=True)
    lock = dest / 'RUNNING.json'
    if lock.exists():
        previous = read(lock)
        try:
            p = psutil.Process(previous['pid'])
            assert p.create_time() != previous['create_time'] or p.status() == 'zombie', 'Cache process already active'
        except psutil.NoSuchProcess:
            pass
    write(lock, {'pid': os.getpid(), 'create_time': psutil.Process().create_time()})
    audit = read(ROOT / 'migration_audit.json')
    assert audit['status'] == 'passed'
    assert sha(ROOT / 'cloud_data_manifest.json') == audit['manifest_sha256'] == EXPECTED_MANIFEST
    assert read(DRIVE / 'persistence_check/PERSISTENCE_VERIFIED.json')['status'] == 'passed'
    source_hash = sha(__file__)
    pins = read(ROOT / 'BUNDLE_MANIFEST.json')
    for name in ['frozen_functions.py', 'initial_seed17.pt', 'initial_seed29.pt', 'scale_plan.json']:
        assert sha(ROOT / name) == pins[name], name
    for name in ['cloud_data_manifest.json', 'migration_audit.json', 'scale_plan.json', 'frozen_functions.py', 'initial_seed17.pt', 'initial_seed29.pt']:
        copy_verified(ROOT / name, DRIVE / 'scale_inputs' / name)
    copy_verified(__file__, DRIVE / 'scale_sources' / (source_hash + '.py'))
    plan = read(ROOT / 'scale_plan.json')
    groups = defaultdict(list)
    for row in read(ROOT / 'cloud_data_manifest.json')['records']:
        if row['split'] == 'train':
            groups[row['image_id']].append(row)
    ids = sorted(groups)
    assert len(ids) == 4096 and set(ids) == set(plan['large_train_images'])
    assert set(plan['small_train_images']) < set(ids)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = snapshot_download('prs-eth/marigold-depth-v1-1', revision='9571e7123e258cf052b4e54241f17971c290e9a8', local_files_only=True)
    write(ROOT / 'model_path.json', {'path': model})
    sys.path.insert(0, str(ROOT))
    from frozen_functions import setup
    pipe, _ = setup(17, ROOT / 'initial_seed17.pt')
    pipe.vae.eval()
    pipe.text_encoder.eval()
    environment = {'gpu': torch.cuda.get_device_name(), 'packages': {k: version(k) for k in ['torch','diffusers','transformers','peft','accelerate']}}
    receipt_path = durable / 'cache_audit.json'
    receipt = read(receipt_path) if receipt_path.exists() else {'source_sha256': source_hash, 'manifest_sha256': EXPECTED_MANIFEST, 'environment': environment, 'shards': {}, 'sessions': []}
    assert receipt['source_sha256'] == source_hash and receipt['manifest_sha256'] == EXPECTED_MANIFEST
    assert receipt['environment'] == environment, 'Environment differs; do not mix cache silently'
    prior_seconds = sum(s['seconds'] for s in receipt['sessions'])
    session = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': 0}
    receipt['sessions'].append(session)

    def tick(count):
        session['seconds'] = time.monotonic() - started
        assert prior_seconds + session['seconds'] < 7200, 'Cumulative cache engineering cap: 2h'
        assert not (DRIVE / 'PAUSE').exists(), 'Project Drive pause requested'
        write(ROOT / 'scale_cache_status.json', {'images': count, 'total': len(ids), 'session_seconds': session['seconds'], 'cumulative_seconds': prior_seconds + session['seconds'], 'optimizer_updates': 0, 'heldout_inference': False})
        write(receipt_path, receipt)

    def encode(tensor):
        return (pipe.vae.encode(tensor.cuda().to(torch.bfloat16)).latent_dist.mode() * pipe.vae.config.scaling_factor).float().cpu()

    with torch.no_grad():
        for start in range(0, len(ids), 64):
            tick(start)
            name = f'shard_{start:05d}.pt'
            path = dest / name
            if name in receipt['shards']:
                record = receipt['shards'][name]
                assert sha(durable / name) == record['sha256']
                copy_verified(durable / name, path)
                continue
            shard = {'manifest_sha256': EXPECTED_MANIFEST, 'source_sha256': source_hash, 'rgb': {}, 'mask': {}, 'text': {}}
            for iid in ids[start:start + 64]:
                tick(start + len(shard['rgb']))
                for r in groups[iid]:
                    aid, sid = r['ann_id'], r['sent_id']
                    if iid not in shard['rgb']:
                        assert sha(r['image']) == r['image_sha256']
                        a = np.array(Image.open(r['image']).convert('RGB').resize((256,192), Image.Resampling.BILINEAR), copy=True)
                        shard['rgb'][iid] = encode(torch.from_numpy(a).permute(2,0,1)[None].float()/127.5-1)
                    if aid not in shard['mask']:
                        a = np.array(Image.open(r['mask']), copy=True)
                        assert hashlib.sha256(a.tobytes()).hexdigest() == r['mask_pixel_sha256']
                        a = np.array(Image.fromarray(a).resize((256,192), Image.Resampling.NEAREST), copy=True)
                        shard['mask'][aid] = encode((torch.from_numpy(a)[None,None].float()/127.5-1).repeat(1,3,1,1))
                    if sid not in shard['text']:
                        prompt = 'A binary segmentation mask of '+r['text']+', white foreground and black background.'
                        assert len(pipe.tokenizer(prompt, truncation=False).input_ids) <= 77
                        tokens = pipe.tokenizer(prompt, padding='max_length', max_length=77, return_tensors='pt').input_ids.cuda()
                        shard['text'][sid] = pipe.text_encoder(tokens)[0].cpu()
            assert len(shard['rgb']) == 64 and len(shard['mask']) == 128
            assert all(torch.isfinite(t).all() for key in ['rgb','mask','text'] for t in shard[key].values())
            tmp = path.with_suffix('.tmp')
            torch.save(shard, tmp)
            tmp.replace(path)
            digest = copy_verified(path, durable / name)
            receipt['shards'][name] = {'sha256': digest, 'bytes': path.stat().st_size, 'counts': {k: len(shard[k]) for k in ['rgb','mask','text']}}
            tick(start + 64)
            print('CACHE', start + 64, '/', len(ids), round(session['seconds'], 1), flush=True)
    tick(4096)
    receipt.update(status='complete', training_images=4096, optimizer_updates=0, heldout_inference=False)
    write(receipt_path, receipt)
    write(dest / 'cache_audit.json', receipt)
    print('TRAINING CACHE COMPLETE; formal runner and budget gates still required', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        report = {'error': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}
        write(ROOT / 'scale_cache_error.json', report)
        if DRIVE.is_dir():
            write(DRIVE / 'scale_cache_error.json', report)
        raise
