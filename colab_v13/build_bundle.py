"""CPU-only export of existing training-only tensors; no local experiment run."""
import ast, hashlib, json, shutil, zipfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parents[1]
WORK=ROOT/'work/scaleup_research_v13'
DEST=ROOT/'output/colab_v13_engineering_2026-10-05'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    DEST.mkdir(exist_ok=True,parents=True)
    pilot=json.loads((WORK/'pilot_protocol.json').read_text())
    old=ROOT/'work/marigold-local/generative_ris_v12/cache.pt'
    assert sha(old)==pilot['cache_sha256']
    rr=json.loads((REPO/'results/generative_ris_v12/data_manifest.json').read_text())['records']
    rr=[r for r in rr if r['split']=='train' and r['image_id'] in pilot['training_images']]
    cache=torch.load(old,map_location='cpu',weights_only=True)
    subset={key:{i:cache[key][i] for i in {r[field] for r in rr}} for key,field in [('rgb','image_id'),('mask','ann_id'),('text','sent_id')]}
    targets={}
    for r in rr:
        if r['ann_id'] not in targets:
            a=np.array(Image.open(r['mask']).resize((256,192),Image.Resampling.NEAREST),copy=True)
            targets[r['ann_id']]=torch.from_numpy(a)[None,None].float()/255
    portable=[{k:r[k] for k in ['image_id','ann_id','sent_id','text','image_sha256','mask_sha256']} for r in rr]
    torch.save({'cache':subset,'targets':targets,'rows':portable},DEST/'training_only_inputs.pt')
    shutil.copy2(ROOT/'work/marigold-local/generative_ris_v11/seed17/adapter_1000.pt',DEST/'initial.pt')
    imports='import json, random\nfrom pathlib import Path\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\nfrom diffusers import MarigoldDepthPipeline\nfrom diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor\nfrom peft import LoraConfig\nASSETS=Path(__file__).parent\nread=lambda p:json.loads(Path(p).read_text())\nPROMPTS={"depth":"A metric depth map of the indoor scene, encoded as logarithmic depth.","normal":"A camera-facing surface normal map of the indoor scene, encoded in RGB."}\n'
    sources={};pieces=[imports]
    for rel,names in [('experiment.py',['FloatResizeProcessor','seed_all','load_pipe']),('multitask_v7/engine.py',['setup']),('generative_ris_v12/train.py',['pixel_loss','pair_term','readiness','update'])]:
        p=REPO/rel;s=p.read_text(encoding='utf-8');sources[rel]=sha(p)
        for node in ast.parse(s).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:pieces.append(ast.get_source_segment(s,node))
    (DEST/'frozen_functions.py').write_text('\n\n'.join(pieces),encoding='utf-8')
    shutil.copy2(REPO/'colab_v13/cloud_probe.py',DEST/'cloud_probe.py')
    manifest={'scope':'64 previously used training images only; no holdout;64updates perarm seed17; actual resume at32 replay33',
        'arms':['pixel','pair_always','pair_ready'],'dtype':'same bfloat16 as local, no silent FP16 fallback',
        'source_hashes':sources,'local_reference':'deterministic_benchmark.json; same sorted64 image order and first sentence per annotation',
        'files':{p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.suffix in ['.pt','.py']}}
    (DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(DEST/'cloud_training_probe.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest['files'])+['MANIFEST.json']:z.write(DEST/name,name)
    print(json.dumps({'bundle':str(DEST/'cloud_training_probe.zip'),'sha256':sha(DEST/'cloud_training_probe.zip'),'bytes':(DEST/'cloud_training_probe.zip').stat().st_size}))
if __name__=='__main__':main()
