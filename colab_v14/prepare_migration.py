"""Colab-only original-data acquisition and label-content audit; never trains."""
import concurrent.futures,hashlib,json,sys,time
from pathlib import Path
import requests,numpy as np,pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as coco_mask
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,d):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def fetch(url,path,digest):
    if path.exists():assert sha(path)==digest;return path
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=90);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==digest
            tmp=path.with_suffix('.part');tmp.write_bytes(r.content);tmp.replace(path);return path
        except (requests.RequestException,AssertionError):
            if attempt==2:raise
            time.sleep(2)
def main():
    assert sys.platform=='linux' and Path('/content').exists(),'Cloud-only preparation; local experiments paused'
    expected=json.loads((ROOT/'migration_inputs.json').read_text());records=expected['records']
    raw={};lookup={}
    for split,item in expected['parquets'].items():
        path=fetch(item['url'],ROOT/'data'/f'{split}.parquet',item['sha256'])
        for r in pq.read_table(path).to_pylist():lookup[r['ann_id']]=r
    groups={}
    for r in records:groups.setdefault(r['image_id'],[]).append(r)
    def acquire(pair):
        first=pair[0];image=fetch(first['image_url'],ROOT/'data/images'/f"{first['image_id']}.jpg",first['image_sha256'])
        result=[]
        for r in pair:
            original=lookup[r['ann_id']];ann=json.loads(original['raw_anns']);info=json.loads(original['raw_image_info'])
            assert ann['image_id']==r['image_id']==info['id']
            assert any(s['sent_id']==r['sent_id'] and s['sent']==r['text'] for s in original['sentences'])
            h,w=info['height'],info['width'];seg=ann['segmentation']
            if isinstance(seg,list):rle=coco_mask.merge(coco_mask.frPyObjects(seg,h,w))
            elif isinstance(seg['counts'],list):rle=coco_mask.frPyObjects(seg,h,w)
            else:rle=seg
            mask=coco_mask.decode(rle)
            if mask.ndim==3:mask=mask.any(2)
            mask=(mask>0).astype(np.uint8)*255
            assert hashlib.sha256(mask.tobytes()).hexdigest()==r['mask_pixel_sha256']
            with Image.open(image) as im:assert im.size==(w,h)
            path=ROOT/'data/masks'/f"{r['ann_id']}.png";path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert np.array_equal(np.array(Image.open(path)),mask)
            else:Image.fromarray(mask).save(path)
            result.append(dict(r,image=str(image),mask=str(path),cloud_mask_png_sha256=sha(path)))
        return result
    output=[];start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,part in enumerate(pool.map(acquire,groups.values())):
            output.extend(part)
            if (i+1)%64==0:write('migration_status.json',{'stage':'original_data_acquisition','images':i+1,'total':len(groups),'seconds':time.time()-start})
    train={r['image_id'] for r in output if r['split']=='train'};test={r['image_id'] for r in output if r['split']=='fresh_holdout'}
    assert len(train)==2048 and len(test)==320 and not train&test
    assert not {r['image_sha256'] for r in output if r['split']=='train'} & {r['image_sha256'] for r in output if r['split']=='fresh_holdout'}
    write('cloud_data_manifest.json',{'records':output,'original_input_sha256':sha(ROOT/'migration_inputs.json')})
    write('migration_audit.json',{'status':'passed','training_images':2048,'fresh_holdout_images':320,'original_image_bytes_exact':True,
        'mask_pixels_exact':True,'original_text_exact':True,'manifest_sha256':sha(ROOT/'cloud_data_manifest.json'),'seconds':time.time()-start,'training_started':False,'heldout_inference':False})
    print('MIGRATION INPUT AUDIT COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:write('migration_error.json',{'type':type(e).__name__,'message':str(e)});raise

