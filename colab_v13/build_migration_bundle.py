"""CPU-only export of fixed metadata and existing seed adapters for cloud migration."""
import hashlib,json,shutil,zipfile,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parents[1]
OUT=ROOT/'output/colab_v13_migration_2026-10-05'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    work=ROOT/'work/scaleup_research_v13';plan=json.loads((work/'cloud_scale_resource_plan.json').read_text(encoding='utf-8'))
    manifest=work/'data_manifest.json';assert sha(manifest)=='54e4e86b02155988089f3bf4f85abeebd3fb54598650eb542cd1dec1ef7f1dce'
    ids=set(plan['large_train_images'])|set(plan['fresh_holdout_images'])
    rows=[r for r in json.loads(manifest.read_text(encoding='utf-8'))['records'] if r['image_id'] in ids]
    paths={r['mask'] for r in rows}
    def pixels(path):return path,hashlib.sha256(np.array(Image.open(path),dtype=np.uint8).tobytes()).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:hashes=dict(pool.map(pixels,paths))
    clean=[dict({k:v for k,v in r.items() if k not in ['image','mask']},mask_pixel_sha256=hashes[r['mask']]) for r in rows]
    selection=json.loads((work/'selection.json').read_text(encoding='utf-8'))
    files={'train':'train-00000-of-00001-4fe3e6340cfb69ed.parquet','validation':'validation-00000-of-00001-15168dfe7b5961e5.parquet'}
    data={'original_manifest_sha256':sha(manifest),'records':clean,'parquets':{s:{'url':f'https://huggingface.co/datasets/jxu124/refcocog/resolve/55319436ad54b9480cefda6b9d64397de92456dd/data/{name}','sha256':selection['raw_parquet_sha256'][s]} for s,name in files.items()}}
    (OUT/'migration_inputs.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    (OUT/'scale_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for name in ['persistence_check.py','prepare_migration.py']:shutil.copy2(REPO/'colab_v13'/name,OUT/name)
    shutil.copy2(ROOT/'output/colab_v13_engineering_2026-10-05/frozen_functions.py',OUT/'frozen_functions.py')
    for seed in [17,29]:shutil.copy2(ROOT/f'work/marigold-local/generative_ris_v11/seed{seed}/adapter_1000.pt',OUT/f'initial_seed{seed}.pt')
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.py','.pt','.json'] and p.name!='BUNDLE_MANIFEST.json'}
    (OUT/'BUNDLE_MANIFEST.json').write_text(json.dumps(files,indent=2))
    with zipfile.ZipFile(OUT/'v13_cloud_migration.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(files)+['BUNDLE_MANIFEST.json']:z.write(OUT/name,name)
    print(json.dumps({'sha256':sha(OUT/'v13_cloud_migration.zip'),'bytes':(OUT/'v13_cloud_migration.zip').stat().st_size,'records':len(rows),'images':len(ids)}))
if __name__=='__main__':main()
