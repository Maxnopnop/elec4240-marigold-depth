"""Build a self-contained, engineering-only Colab notebook from the local probe."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
probe = (HERE / 'preflight.py').read_text(encoding='utf-8')
helper = '''from diffusers import MarigoldDepthPipeline
from peft import LoraConfig
import random
import numpy as np
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
MODEL_ID = 'prs-eth/marigold-depth-v1-1'
REVISION = '9571e7123e258cf052b4e54241f17971c290e9a8'

def setup(seed):
    assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > T4 GPU first.'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    pipe = MarigoldDepthPipeline.from_pretrained(MODEL_ID, revision=REVISION, variant='fp16', torch_dtype=DTYPE).to('cuda')
    pipe.vae.requires_grad_(False); pipe.unet.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian', target_modules=['to_q','to_k','to_v','to_out.0']))
    params = {n:p for n,p in pipe.unet.named_parameters() if p.requires_grad}
    for p in params.values(): p.data = p.data.float()
    pipe.unet.enable_gradient_checkpointing()
    pipe.scheduler.set_timesteps(1, device='cuda')
    assert pipe.scheduler.timesteps.tolist() == [999]
    assert float(pipe.scheduler.alphas_cumprod[999]) == 0.
    assert pipe.scheduler.config.prediction_type == 'v_prediction'
    return pipe, params

def encode(pipe, tensor):
    return pipe.vae.encode(tensor.to(DTYPE)).latent_dist.mode() * pipe.vae.config.scaling_factor
'''
probe = probe.replace('from multitask_v7.engine import setup, encode', helper)
probe = probe.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path('/content/generative_ris_v11')")
probe = probe.replace("OUT = ROOT / 'results/generative_ris_v11'", "OUT = ROOT / 'results'")
# Preserve helper's dtype selection while replacing the original BF16-only calls.
probe = probe.replace('to(torch.bfloat16)', 'to(DTYPE)').replace('dtype=torch.bfloat16', 'dtype=DTYPE, enabled=(DTYPE != torch.float32)')
probe = probe.replace("for p in [Path(__file__), ROOT/'multitask_v7/engine.py', ROOT/'multitask_v7/common.py', ROOT/'experiment.py']", "for p in [Path(__file__)]")
probe = probe.replace("'gpu': torch.cuda.get_device_name(),", "'gpu': torch.cuda.get_device_name(), 'dtype': str(DTYPE), 'model_id': MODEL_ID, 'model_revision': REVISION,")
ast.parse(probe)
(HERE/'colab_preflight.py').write_text(probe,encoding='utf-8')
def md(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text):
    ast.parse(text)
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(True)}
cells=[md('''# ELEC4240: Generative RIS — Colab engineering preflight

The image-generation-derived Marigold backbone directly processes RGB and referring text toward a generated mask target. This notebook checks runtime compatibility only. **It does not train a referring segmentation model or validate a condition-selection policy.**

Select **Runtime → Change runtime type → T4 GPU** (free resources when available), then run the cells. No Drive mount, credentials, private dataset upload, paid service, or background keep-alive is required. A pinned public model is downloaded from Hugging Face. Dtype follows PyTorch support detection, which may include BF16 emulation on T4. The actual dtype is recorded; this does not establish native BF16 acceleration or cross-device numerical equivalence.

The final download preserves the result, environment versions and executed source. Next research gate: real text-mask pairs and a trained baseline that switches target correctly when its expression changes.
'''),code('''import subprocess, sys, os
os.environ['USE_TF'] = '0'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.35.2', 'transformers==4.57.1', 'peft==0.17.1', 'accelerate==1.11.0', 'safetensors'], check=True)
'''),code('''import torch, json, platform, importlib.metadata
from pathlib import Path
assert torch.cuda.is_available(), 'Connect a GPU runtime before running.'
out = Path('/content/generative_ris_v11'); out.mkdir(exist_ok=True)
environment = {'python':platform.python_version(), 'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda, 'packages':{n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers','peft','accelerate']}}
(out/'environment.json').write_text(json.dumps(environment,indent=2))
print(json.dumps(environment,indent=2))
'''),code("from pathlib import Path\nimport subprocess,sys,hashlib\nsource = "+repr(probe)+"\np = Path('/content/generative_ris_v11/colab_preflight.py')\np.write_text(source)\nprint('Probe SHA256:',hashlib.sha256(p.read_bytes()).hexdigest())\nsubprocess.run([sys.executable,str(p)],check=True)\n"),md('''## Interpretation
Finite gradients and prompt-dependent outputs establish a computational path only. A mask VAE round trip is not segmentation prediction accuracy. No optimizer update is performed. Do not interpret this notebook as a trained RIS baseline or as evidence that budgeted verification works.
'''),code('''import shutil
from google.colab import files
archive = shutil.make_archive('/content/generative_ris_v11_preflight', 'zip', '/content/generative_ris_v11')
files.download(archive)
''')]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},'colab':{'name':'ELEC4240_Generative_RIS_Preflight.ipynb','provenance':[]}},'cells':cells}
for i,c in enumerate(cells):c['id']=f'v11-{i}'
path=HERE/'ELEC4240_Generative_RIS_Preflight.ipynb'
path.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'notebook':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'code_cells_ast_valid':True}))
