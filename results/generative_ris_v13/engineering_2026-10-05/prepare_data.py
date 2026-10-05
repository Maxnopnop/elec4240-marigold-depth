"""Acquire exactly the preregistered images; no replacement or outcome filtering."""
import sys,json,concurrent.futures
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'outputs/marigold-depth';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
from generative_ris_v11.prepare_training import WORK,fetch,decode_mask,sha,write

def main():
    selection=json.loads((OUT/'selection.json').read_text());design=json.loads((OUT/'scale_design.json').read_text())
    assert sha(OUT/'selection.json')==design['selection_sha256']
    old_path=REPO/'results/generative_ris_v12/data_manifest.json';assert sha(old_path)==selection['prior_manifest_sha256']
    old=json.loads(old_path.read_text())['records'];records=[dict(r,scale_membership='both' if r['split']=='train' else 'development') for r in old if r['split'] in ['train','dev']]
    oldhash={r['image_sha256'] for r in old}
    gqa=json.loads((REPO/'results/generative_ris_v12/conditions/data_manifest.json').read_text())
    grows=gqa['records']
    oldhash.update(r['image_sha256'] for r in grows if 'image_sha256' in r)
    lookup={}
    for split in ['train','validation']:
        path=WORK/f'data/{split}.parquet';assert sha(path)==selection['raw_parquet_sha256'][split]
        for row in pq.read_table(path).to_pylist():lookup[row['ann_id']]=row
    def acquire(job):
        rr=[lookup[aid] for aid in job['ann_ids']];info=json.loads(rr[0]['raw_image_info']);iid=job['image_id']
        assert all(r['image_id']==iid for r in rr)
        url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+info['file_name'];image=fetch(url,OUT/'data/images'/info['file_name'])
        ih=sha(image);assert ih not in oldhash,('duplicate previous image pixels',iid)
        with Image.open(image) as im:assert im.size==(info['width'],info['height'])
        result=[]
        for r in rr:
            target=decode_mask(r)*255;mp=OUT/'data/masks'/f"{r['ann_id']}.png";mp.parent.mkdir(parents=True,exist_ok=True)
            if mp.exists():assert np.array_equal(np.array(Image.open(mp)),target)
            else:Image.fromarray(target).save(mp)
            for sent in sorted(r['sentences'],key=lambda s:s['sent_id'])[:2 if job['split']=='train' else 1]:
                result.append({'dataset':'refcocog','split':job['split'],'scale_membership':'large_only' if job['split']=='train' else 'fresh_holdout','image_id':iid,'ann_id':r['ann_id'],'category_id':r['category_id'],'sent_id':sent['sent_id'],'text':sent['sent'],'image':str(image),'mask':str(mp),'image_sha256':ih,'mask_sha256':sha(mp),'height':info['height'],'width':info['width'],'mask_pixels':int((target>0).sum()),'image_url':url})
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,part in enumerate(pool.map(acquire,selection['jobs'])):
            records.extend(part)
            if (i+1)%64==0:write(OUT/'data_status.json',{'stage':'acquiring_original_images','done':i+1,'total':len(selection['jobs'])});print(i+1,'/',len(selection['jobs']),flush=True)
    ids=defaultdict(set);hashes=defaultdict(set);audited={}
    for r in records:
        ids[r['split']].add(r['image_id']);hashes[r['split']].add(r['image_sha256'])
        for key in ['image','mask']:
            p=r[key]
            if p not in audited:assert sha(p)==r[key+'_sha256'];audited[p]=r[key+'_sha256']
    assert {k:len(v) for k,v in ids.items()}=={'train':8192,'dev':64,'fresh_holdout':320}
    for a in ids:
        for b in ids:
            if a!=b:assert not(ids[a]&ids[b]) and not(hashes[a]&hashes[b])
    result={'records':records,'selection_sha256':sha(OUT/'selection.json'),'original_data_unchanged':True,'source_note':'Pinned public RefCOCOg annotation mirror and original COCO pixels; original upstream archive byte equivalence not established. Difficulty-enriched same-category-pair subset, not full official benchmark.'}
    path=OUT/'data_manifest.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write(path,result)
    write(OUT/'input_audit.json',{'status':'verified','counts':{k:len(v) for k,v in ids.items()},'files':len(audited),'manifest_sha256':sha(path),'new_pixels_checked_against_previous_images':True,'selection_before_pixels':True,'no_new_evaluation':True})
    write(OUT/'data_status.json',{'stage':'complete','images':8576,'new_images':6464});print('DATA COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:write(OUT/'data_error.json',{'type':type(e).__name__,'message':str(e)});raise
