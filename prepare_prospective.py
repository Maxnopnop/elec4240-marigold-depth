"""Acquire a fixed space-disjoint external cohort without model predictions."""
import argparse,io,json,hashlib,struct,zlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import numpy as np
from PIL import Image
from remote_archive import HTTPRangeFile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def rank(seed,text):return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()
def pixel_hash(rgb):return hashlib.sha256(np.array(rgb.shape,dtype=np.int64).tobytes()+rgb.tobytes()).hexdigest()
def decode_depth(encoded):
    v=np.asarray(encoded,dtype=np.uint32)
    return (((v>>3)|(v<<13))&65535).astype(np.float32)/1000

def location_proxy(space):
    if space.startswith('mit_'):
        if space.startswith('mit_dorm_'):return '_'.join(space.split('_')[:3])
        if space.startswith('mit_w85'):return 'mit_w85'
        return '_'.join(space.split('_')[:2])
    if space.startswith('harvard_'):return 'harvard_unspecified_building'
    if space.startswith('brown_'):return '_'.join(space.split('_')[:2])
    return space

def member(remote,entry,cache):
    dest=cache/(hashlib.sha256(entry['name'].encode()).hexdigest()+'.bin')
    if dest.exists():
        data=dest.read_bytes()
        assert len(data)==entry['bytes'] and zlib.crc32(data)==entry['crc32']
        return data
    for attempt in range(3):
        try:
            header,total=remote.fetch(entry['offset'],entry['offset']+30)
            parts=struct.unpack('<4s5H3I2H',header)
            assert parts[0]==b'PK\x03\x04' and not parts[2]&1 and total==remote.size
            method=parts[3];name_len,extra_len=parts[-2:]
            start=entry['offset']+30
            payload,_=remote.fetch(start,start+name_len+extra_len+entry['compressed_bytes'])
            assert payload[:name_len].decode('utf-8')==entry['name']
            compressed=payload[name_len+extra_len:]
            data=zlib.decompress(compressed,-15) if method==8 else compressed
            assert method in [0,8] and len(data)==entry['bytes'] and zlib.crc32(data)==entry['crc32']
            dest.write_bytes(data)
            return data
        except Exception:
            if attempt==2:raise
            time.sleep(1+attempt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path);p.add_argument('--data',required=True,type=Path)
    p.add_argument('--results',default=Path('results/prospective_v4'),type=Path);a=p.parse_args();out=a.results
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json')
        for r in m['samples']:assert sha(a.data/r['file'])==r['sha256']
        print('EXTERNAL_DATA_ALREADY_FROZEN',len(m['samples']),flush=True);return
    inventory=read(out/'sunrgbd_sun3d_inventory.json');entries=inventory['entries'];byname={e['name']:e for e in entries}
    groups=defaultdict(list);depths=defaultdict(list)
    for e in entries:
        if '/image/' in e['name']:groups[e['name'].split('/')[3]].append(e)
        if '/depth/' in e['name']:depths[e['name'].split('/depth/')[0]].append(e)
    ordered=sorted(groups,key=lambda s:rank(4271,s))
    assert len(ordered)==207
    selection={'target_scenes':200,'space_seed':4271,'frame_seed':4272,'candidate_spaces':ordered,
               'inventory_sha256':sha(out/'sunrgbd_sun3d_inventory.json'),
               'candidates':{s:[e['name'] for e in sorted(groups[s],key=lambda e:rank(4272,e['name']))[:3]] for s in ordered},
               'selection_rule':'first 200 eligible groups, one raw RGB/depth pair each; all QC before model inference',
               'source_sha256':sha(__file__),'protocol_sha256':sha('PROSPECTIVE_PROTOCOL.md')}
    if (out/'selection_plan.json').exists():assert read(out/'selection_plan.json')==selection
    else:write(out/'selection_plan.json',selection)
    decision={'target_scenes':200,'training_draws':3,'training_seeds':5,'new_training_runs':0,
              'planning_source':'fresh31','assumed_target_gain':.009,'assumed_error_sd_multiplier':1.,
              'selection_reason':'first candidate N reaching approximately80% planning power with existing 3x5 models; preserve all checkpoints',
              'limitations':'Not80% power under every scenario. SDx1.5/effect.009 requires750 candidates; external domain shift is unknown.',
              'power_results_sha256':sha(out/'power/design_and_results.json'),'external_model_predictions_exist':False}
    write(out/'sample_size_decision.json',decision)
    old_hashes=set();old_ids=set()
    old_manifest=read('results/robustness_v3/split_manifest.json')
    for key in ['validation','fresh31','observed64']:
        old_ids.update(r['id'] for r in old_manifest[key])
    for rows in old_manifest['draws'].values():old_ids.update(r['id'] for r in rows)
    # Include every prior locally extracted NYUv2 image, not just the v3 selection.
    for path in (a.assets/'subset').glob('*.npz'):
        with np.load(path) as f:old_hashes.add(pixel_hash(f['image']))
    a.data.mkdir(parents=True,exist_ok=True);cache=a.data/'raw_members';cache.mkdir(exist_ok=True)
    remote=HTTPRangeFile(inventory['url']);assert remote.size==inventory['archive_bytes']
    def acquire(space,banned=old_hashes):
        rejected=[]
        for name in selection['candidates'][space]:
            try:
                rgb_entry=byname[name];paired=depths[name.split('/image/')[0]];assert len(paired)==1
                depth_entry=paired[0]
                rgb_bytes=member(remote,rgb_entry,cache);depth_bytes=member(remote,depth_entry,cache)
                rgb=np.array(Image.open(io.BytesIO(rgb_bytes)).convert('RGB'))
                encoded=np.array(Image.open(io.BytesIO(depth_bytes)))
                assert encoded.dtype==np.uint16 and encoded.shape==rgb.shape[:2], 'Invalid depth encoding/dimensions'
                dep=decode_depth(encoded);fraction=float(np.mean((dep>.1)&(dep<10)))
                assert fraction>=.1,'Too few measured valid pixels'
                pixel=pixel_hash(rgb);assert pixel not in banned,'Exact RGB duplicate'
                return {'space':space,'location_proxy':location_proxy(space),'image_member':name,'depth_member':depth_entry['name'],
                        'rgb_sha256':hashlib.sha256(rgb_bytes).hexdigest(),'depth_png_sha256':hashlib.sha256(depth_bytes).hexdigest(),
                        'rgb_pixel_sha256':pixel,'valid_fraction':fraction,'height':rgb.shape[0],'width':rgb.shape[1],
                        'image_crc32':rgb_entry['crc32'],'depth_crc32':depth_entry['crc32'],
                        'image':rgb,'depth':dep,'rejections':rejected}
            except Exception as error:
                rejected.append({'space':space,'image_member':name,'reason':str(error)})
        return {'space':space,'rejections':rejected,'unavailable':True}
    accepted=[];excluded=[];pixels=set(old_hashes);availability=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for position,item in enumerate(pool.map(acquire,ordered),1):
            space=item['space']
            if not item.get('unavailable') and item['rgb_pixel_sha256'] in pixels:
                item=acquire(space,pixels)
            excluded.extend(item['rejections'])
            availability.append({'space':space,'available':not item.get('unavailable',False),'selected':False})
            if not item.get('unavailable') and len(accepted)<200:
                rgb=item.pop('image');dep=item.pop('depth');item.pop('rejections')
                idx=len(accepted)+1;filename=f'{idx:04d}.npz';np.savez_compressed(a.data/filename,image=rgb,depth=dep)
                item.update({'id':idx,'file':filename,'sha256':sha(a.data/filename)})
                accepted.append(item);pixels.add(item['rgb_pixel_sha256']);availability[-1]['selected']=True
            if position%10==0:print('EXTERNAL_ACQUISITION',position,len(ordered),'selected',len(accepted),flush=True)
    write(out/'acquisition_audit.json',{'availability':availability,'exclusions':excluded,'earlier_rgb_hashes_checked':len(old_hashes),
                                      'source_index_sha256':sha(out/'sunrgbd_sun3d_inventory.json')})
    assert len(accepted)==200,'Insufficient independent groups; no external inference is authorized by this protocol yet'
    manifest={'dataset':'SUNRGBD SUN3D-source raw measured-depth subset','space_definition':'first component after sun3ddata; one frame per group',
              'sampling_seed':4271,'frame_seed':4272,'samples':accepted,'selection_plan_sha256':sha(out/'selection_plan.json'),
              'broader_location_counts':dict(Counter(r['location_proxy'] for r in accepted)),
              'independence_limit':'Scene groups can share buildings; broader location labels are conservative proxies, not verified building IDs.'}
    write(out/'manifest.json',manifest)
    print('EXTERNAL_DATA_READY',len(accepted),'proxy_locations',len(manifest['broader_location_counts']),'exclusions',len(excluded),flush=True)

if __name__=='__main__':main()
