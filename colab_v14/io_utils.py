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

ROOT = Path('/content/v14_pretraining_lora')
OLD_DRIVE = Path('/content/drive/MyDrive/ELEC4240/v13-cloud-scale-2026-10-05')
DRIVE = OLD_DRIVE / 'pretraining_lora_v14'
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


