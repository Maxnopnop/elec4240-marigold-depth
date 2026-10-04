import concurrent.futures
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from generative_ris_v11.prepare_training import ROOT, WORK as OLDWORK, OUT as OLDOUT, fetch, groups, decode_mask, sha, write

WORK=OLDWORK.parent/'generative_ris_v12'
OUT=ROOT/'results/generative_ris_v12'
EXTERNAL={
 'refcoco':('9fba8200c5326e996f789191f095bd464ef1d09e','validation-00000-of-00001-bfeafdc84ca37aa2.parquet'),
 'refcocoplus':('12e20b0b6039fbf656e89e2f26597e84c1037847','validation-00000-of-00001-8c57d66282bc60c9.parquet')}

def original_rows():return json.loads((OLDOUT/'data_manifest.json').read_text())['records']

def convert_old(r):
    r=dict(r)
    for key in ['image','mask']:r[key]=str((OLDWORK/r[key]).resolve())
    r['dataset']='refcocog';return r

def main():
    if (OUT/'data_manifest.json').exists():audit();return
    old=original_rows();records=[convert_old(r) for r in old]
    raw={split:pq.read_table(OLDWORK/f'data/{split}.parquet').to_pylist() for split in ['train','validation']}
    old_manifest=json.loads((OLDOUT/'data_manifest.json').read_text())
    for split in raw:assert sha(OLDWORK/f'data/{split}.parquet')==old_manifest['parquet_sha256'][split]
    gg={split:groups(rr) for split,rr in raw.items()}
    rng=random.Random(424025);used={r['image_id'] for r in old};jobs=[]
    candidates=sorted(set(gg['train'])-used);rng.shuffle(candidates)
    assert len(candidates)>=1536
    train_new=candidates[:1536]
    candidates=sorted(set(gg['validation'])-used);rng.shuffle(candidates)
    assert len(candidates)>=192
    reserved_new=candidates[:192]
    for split,ids,g in [('train',train_new,gg['train']),('reserved',reserved_new,gg['validation'])]:
        for image_id in ids:
            cats=sorted(g[image_id]);cat=rng.choice(cats)
            choices=sorted(g[image_id][cat],key=lambda r:r['ann_id']);rng.shuffle(choices)
            jobs.append(('refcocog',split,image_id,choices[:2]))
    used.update(train_new);used.update(reserved_new)
    excluded_by_dataset={};raw_hashes={};selected_external={}
    for dataset,(revision,name) in EXTERNAL.items():
        path=fetch(f'https://huggingface.co/datasets/jxu124/{dataset}/resolve/{revision}/data/{name}',WORK/'raw'/f'{dataset}_validation.parquet')
        raw_hashes[dataset]=sha(path)
        rows=pq.read_table(path).to_pylist();byimage=defaultdict(list)
        for r in rows:
            if r['sentences'] and not json.loads(r['raw_anns']).get('iscrowd',0):byimage[r['image_id']].append(r)
        excluded_by_dataset[dataset]=sorted(set(byimage)&used)
        ids=sorted(set(byimage)-used);rng.shuffle(ids);assert len(ids)>=128
        selected_external[dataset]=ids[:128]
        for image_id in ids[:128]:
            choices=sorted(byimage[image_id],key=lambda r:r['ann_id']);rng.shuffle(choices)
            anchor=choices[0]
            other=next((r for r in choices[1:] if r['category_id']==anchor['category_id'] and r['ann_id']!=anchor['ann_id']),None)
            jobs.append((dataset,dataset,image_id,[anchor]+([other] if other else [])))
        used.update(ids[:128])
    selection={'seed':424025,'train_new':train_new,'reserved_new':reserved_new,'external':selected_external,
        'external_revisions':EXTERNAL,'raw_sha256':raw_hashes,'excluded_overlap_image_ids':excluded_by_dataset,
        'selection':'retain all640oldimages; add1536train/192reserved same-category pair images;128random eligible external validation images each, one random anchor + same-category partner if available; no performance-based exclusions',
        'jobs':[{'dataset':ds,'split':split,'image_id':iid,'ann_ids':[r['ann_id'] for r in rr]} for ds,split,iid,rr in jobs]}
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(selection))
    else:write(path,selection)
    def acquire(job):
        dataset,split,iid,rr=job;info=json.loads(rr[0]['raw_image_info']);name=info['file_name']
        url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+name
        path=fetch(url,WORK/'images'/name)
        with Image.open(path) as im:assert im.size==(info['width'],info['height'])
        result=[]
        for r in rr:
            m=decode_mask(r);mp=WORK/'masks'/dataset/f"{r['ann_id']}.png";mp.parent.mkdir(parents=True,exist_ok=True)
            if mp.exists():assert np.array_equal(np.array(Image.open(mp)),m*255)
            else:Image.fromarray(m*255).save(mp)
            for sent in sorted(r['sentences'],key=lambda s:s['sent_id'])[:2 if split=='train' else 1]:
                result.append({'dataset':dataset,'split':split,'image_id':iid,'ann_id':r['ann_id'],'category_id':r['category_id'],
                    'sent_id':sent['sent_id'],'text':sent['sent'],'image':str(path.resolve()),'mask':str(mp.resolve()),
                    'image_sha256':sha(path),'mask_sha256':sha(mp),'height':info['height'],'width':info['width'],
                    'mask_pixels':int(m.sum()),'image_url':url})
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,part in enumerate(pool.map(acquire,jobs)):
            records.extend(part)
            if (i+1)%32==0:
                write(OUT/'preparation_status.json',{'stage':'downloading_original_images','done':i+1,'total':len(jobs)})
                print('acquired',i+1,'/',len(jobs),flush=True)
    write(OUT/'data_manifest.json',{'records':records,'selection_sha256':sha(OUT/'selection.json'),
        'source_manifest_sha256':sha(OLDOUT/'data_manifest.json'),'raw_sha256':raw_hashes,
        'note':'Public annotation mirrors, decoded original polygons; original archive byte equivalence unverified. RefCOCO datasets share COCO image domain; external sets are image-disjoint here, not a new visual domain.'})
    audit()

def audit():
    manifest=json.loads((OUT/'data_manifest.json').read_text());rr=manifest['records']
    assert sha(OUT/'selection.json')==manifest['selection_sha256']
    assert sha(OLDOUT/'data_manifest.json')==manifest['source_manifest_sha256']
    for ds,h in manifest['raw_sha256'].items():assert sha(WORK/'raw'/f'{ds}_validation.parquet')==h
    filehash={};split_ids=defaultdict(set);split_hash=defaultdict(set)
    for r in rr:
        split_ids[r['split']].add(r['image_id']);split_hash[r['split']].add(r['image_sha256'])
        for key in ['image','mask']:
            p=r[key];h=r[key+'_sha256']
            if p not in filehash:assert sha(p)==h;filehash[p]=h
        with Image.open(r['mask']) as im:
            a=np.array(im)
            assert a.shape==(r['height'],r['width']) and set(np.unique(a))<={0,255}
            assert int((a>0).sum())==r['mask_pixels']
    counts={k:len(v) for k,v in split_ids.items()}
    assert counts=={'train':2048,'dev':64,'reserved':256,'refcoco':128,'refcocoplus':128},counts
    for a in split_ids:
        for b in split_ids:
            if a!=b:assert not split_ids[a]&split_ids[b] and not split_hash[a]&split_hash[b]
    result={'stage':'verified','image_counts':counts,'expression_counts':{k:sum(r['split']==k for r in rr) for k in counts},
        'unique_files':len(filehash),'manifest_sha256':sha(OUT/'data_manifest.json'),'original_data_unchanged':True}
    write(OUT/'input_audit.json',result);write(OUT/'preparation_status.json',result);print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':main()
