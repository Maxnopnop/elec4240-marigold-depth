"""Pinned RefCOCOg mirror annotations and original COCO pixels; image-disjoint subsets."""
import concurrent.futures
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]/'work/marigold-local/generative_ris_v11'
OUT=ROOT/'results/generative_ris_v11/training'
REV='55319436ad54b9480cefda6b9d64397de92456dd'
FILES={'train':'train-00000-of-00001-4fe3e6340cfb69ed.parquet','validation':'validation-00000-of-00001-15168dfe7b5961e5.parquet'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def fetch(url,path):
    if not path.exists():
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=60);r.raise_for_status()
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(r.content);tmp.replace(path)
                break
            except requests.RequestException:
                if attempt==2:raise
    return path

def decode_mask(row):
    ann=json.loads(row['raw_anns']);info=json.loads(row['raw_image_info'])
    assert ann['image_id']==row['image_id']==info['id'] and ann['id']==row['ann_id']
    h,w=info['height'],info['width'];seg=ann['segmentation']
    if isinstance(seg,list):rle=coco_mask.merge(coco_mask.frPyObjects(seg,h,w))
    elif isinstance(seg['counts'],list):rle=coco_mask.frPyObjects(seg,h,w)
    else:rle=seg
    m=coco_mask.decode(rle)
    if m.ndim==3:m=m.any(2)
    assert m.shape==(h,w) and 0<int(m.sum())<h*w
    return m.astype(np.uint8)

def groups(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['sentences'] and not json.loads(r['raw_anns']).get('iscrowd',0):
            grouped[r['image_id']][r['category_id']].append(r)
    return {i:{c:rs for c,rs in cs.items() if len({r['ann_id'] for r in rs})>=2}
            for i,cs in grouped.items() if any(len({r['ann_id'] for r in rs})>=2 for rs in cs.values())}

def main():
    if (OUT/'data_manifest.json').exists():
        print('Existing manifest retained.');return
    raw={}
    for split,name in FILES.items():
        p=fetch(f'https://huggingface.co/datasets/jxu124/refcocog/resolve/{REV}/data/{name}',WORK/'data'/f'{split}.parquet')
        raw[split]=pq.read_table(p).to_pylist()
    train_ids={r['image_id'] for r in raw['train']};val_ids={r['image_id'] for r in raw['validation']}
    assert not train_ids&val_ids
    gg={k:groups(v) for k,v in raw.items()};rng=random.Random(424011)
    tr=sorted(gg['train']);va=sorted(gg['validation']);rng.shuffle(tr);rng.shuffle(va)
    assert len(tr)>=512 and len(va)>=128
    selected={'train':tr[:512],'dev':va[:64],'reserved':va[64:128]}
    # Freeze IDs before pixel acquisition: unavailable images stop, never silently replace.
    write(OUT/'selection.json',{'seed':424011,'images':selected,'mirror_revision':REV,
        'selection':'images with >=2 annotated same-category instances; bounded difficulty-enriched subset, not full benchmark'})
    jobs=[]
    for split,ids in selected.items():
        g=gg['train' if split=='train' else 'validation']
        for image_id in ids:
            cats=sorted(g[image_id]);cat=cats[rng.randrange(len(cats))]
            rr=sorted(g[image_id][cat],key=lambda r:r['ann_id']);rng.shuffle(rr)
            jobs.append((split,image_id,rr[:2]))
    def acquire(job):
        split,image_id,rr=job;info=json.loads(rr[0]['raw_image_info'])
        url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+info['file_name']
        image_path=fetch(url,WORK/'data/images'/info['file_name'])
        image=Image.open(image_path).convert('RGB');assert image.size==(info['width'],info['height'])
        records=[]
        for row in rr:
            mask=decode_mask(row);mp=WORK/'data/masks'/f"{row['ann_id']}.png";mp.parent.mkdir(exist_ok=True,parents=True)
            Image.fromarray(mask*255).save(mp)
            # Native shape, source identity and area are checked; annotations are not manually relabelled.
            sentences=sorted(row['sentences'],key=lambda s:s['sent_id'])[:2 if split=='train' else 1]
            for sent in sentences:
                records.append({'split':split,'image_id':image_id,'ann_id':row['ann_id'],'category_id':row['category_id'],
                    'sent_id':sent['sent_id'],'text':sent['sent'],'image':str(image_path.relative_to(WORK)),
                    'mask':str(mp.relative_to(WORK)),'image_sha256':sha(image_path),'mask_sha256':sha(mp),
                    'height':info['height'],'width':info['width'],'mask_pixels':int(mask.sum()),'image_url':url})
        return records
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,part in enumerate(pool.map(acquire,jobs)):
            records.extend(part)
            if (i+1)%32==0:
                write(OUT/'preparation_status.json',{'status':'acquiring','completed_images':i+1,'total_images':len(jobs)})
                print('acquired',i+1,flush=True)
    for split in selected:
        rr=[r for r in records if r['split']==split]
        assert len({r['image_id'] for r in rr})==len(selected[split])
    manifest={'dataset':'RefCOCOg annotation mirror jxu124/refcocog + original COCO train2014 images',
        'mirror_revision':REV,'mirror_caveat':'Raw IDs/polygons and image-disjoint train/validation verified; original release archive byte equivalence not verified.',
        'original_split_counts':{k:len(v) for k,v in raw.items()},'original_splits_image_disjoint':True,
        'parquet_sha256':{k:sha(WORK/'data'/f'{k}.parquet') for k in raw},
        'image_counts':{k:len(v) for k,v in selected.items()},
        'expression_counts':{k:sum(r['split']==k for r in records) for k in selected},'records':records}
    write(OUT/'data_manifest.json',manifest)
    # Training-only contact sheet; reserved masks never used for modelling decisions.
    canvas=Image.new('RGB',(768,3*240),'white');draw=ImageDraw.Draw(canvas)
    for i,job in enumerate(jobs[:3]):
        rr=[r for r in records if r['image_id']==job[1]]
        img=Image.open(WORK/rr[0]['image']).convert('RGB').resize((256,192))
        canvas.paste(img,(0,i*240))
        distinct={r['ann_id']:r for r in rr}
        for j,r in enumerate(distinct.values()):
            m=Image.open(WORK/r['mask']).resize((256,192),Image.Resampling.NEAREST).convert('RGB')
            canvas.paste(m,((j+1)*256,i*240))
            draw.text(((j+1)*256+3,i*240+194),r['text'][:36],fill='black')
    canvas.save(OUT/'training_data_check.png')
    write(OUT/'preparation_status.json',{'status':'complete','completed_images':len(jobs),'expressions':len(records),'manifest_sha256':sha(OUT/'data_manifest.json')})
    print(json.dumps({k:manifest[k] for k in ['image_counts','expression_counts']}),flush=True)

if __name__=='__main__':main()
