"""V13 paths and deterministic execution, independent of frozen old outputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import torch
from .state import sha

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('ELEC4240_ROOT', REPO.parents[1]))
WORK = ROOT / 'work/scaleup_research_v13'
OUT = REPO / 'results/generative_ris_v13'
OLD = ROOT / 'work/marigold-local/generative_ris_v11'
ARMS = ['pixel', 'pair_always', 'pair_ready']
SEEDS = [17, 29, 43]
SIZES = [2048, 8192]
STEPS = 8192

def deterministic():
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def rows():
    audit = read(WORK / 'input_audit.json')
    assert audit['status'] == 'verified'
    assert sha(WORK / 'data_manifest.json') == audit['manifest_sha256']
    return read(WORK / 'data_manifest.json')['records']

def label(arm, size, seed):
    return f'{arm}_n{size}_s{seed}'
