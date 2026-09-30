"""Download official NYUv2 labeled data, official splits and pinned public models."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests

NYU_URL = 'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SPLIT_URL = 'https://raw.githubusercontent.com/cleinc/bts/master/utils/splits.mat'
MODEL_ID = 'prs-eth/marigold-depth-v1-1'
MODEL_REVISION = '9571e7123e258cf052b4e54241f17971c290e9a8'

def download(url, path, expected_size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        print('Already downloaded:', path.name, flush=True)
        return
    tmp = path.with_suffix(path.suffix + '.partial')
    offset = tmp.stat().st_size if tmp.exists() else 0
    with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=(30,120)) as r:
        r.raise_for_status()
        resume = offset > 0 and r.status_code == 206
        if not resume:
            offset = 0
        last = time.monotonic()
        with tmp.open('ab' if resume else 'wb') as out:
            for chunk in r.iter_content(4 * 1024 * 1024):
                out.write(chunk)
                offset += len(chunk)
                if time.monotonic() - last > 15:
                    print(f'{path.name}: {offset / 1e6:.0f} MB', flush=True)
                    last = time.monotonic()
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError('Download has unexpected size')
    tmp.replace(path)
    print('Downloaded:', path.name, path.stat().st_size, flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--asset', choices=['data','model'], required=True)
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if a.asset=='data':
        download(NYU_URL,root/'nyu_depth_v2_labeled.mat',2972037809)
        download(SPLIT_URL,root/'splits.mat')
    else:
        # local_dir avoids Windows symlink permissions in the global hub cache.
        from huggingface_hub import snapshot_download
        path=snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(root/'marigold-v1-1'),
          allow_patterns=['*.json','tokenizer/*','*/*.fp16.safetensors'], max_workers=3)
        (root/'model_path.json').write_text(json.dumps({'model_id':MODEL_ID,'revision':MODEL_REVISION,'path':path},indent=2))
        print('MODEL_READY',path,flush=True)
if __name__=='__main__':main()
