"""Select original GQA scene graphs before fetching unchanged image ZIP members."""
import io
import concurrent.futures
import struct
import threading
import zlib
import json
import random
import time
import zipfile
import requests
from PIL import Image
from .data import OUT,WORK,fetch,sha,write

DEST=OUT/'conditions';DATA=WORK/'conditions'
COLORS=['black','white','red','green','blue','yellow','brown','gray','orange','pink','purple']
IMAGES='https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'

class HTTPRange(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.session=requests.Session();self.pos=0
        r=self.session.get(url,headers={'Range':'bytes=-65536'},timeout=60);r.raise_for_status()
        assert r.status_code==206;self.size=int(r.headers['Content-Range'].split('/')[-1]);self.tail=r.content
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if self.pos>=self.size-len(self.tail):
            start=self.pos-(self.size-len(self.tail));self.pos+=n;return self.tail[start:start+n]
        end=self.pos+n-1
        for attempt in range(3):
            try:
                r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{end}'},timeout=90);r.raise_for_status();break
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2*(attempt+1))
        assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {self.pos}-{end}/')
        assert len(r.content)==n;self.pos+=n;return r.content

def eligible(graph):
    objects=graph['objects'];result=[]
    for oid,o in sorted(objects.items()):
        colors=[c for c in COLORS if c in o['attributes']]
        if len(colors)!=1 or min(o['w'],o['h'])<16:continue
        others=[k for k,v in objects.items() if k!=oid and v['name']==o['name'] and min(v['w'],v['h'])>=16 and len([c for c in COLORS if c in v['attributes']])==1]
        if not others:continue
        rels=[r for r in o['relations'] if r['name'] in ['to the left of','to the right of'] and r['object'] in objects and r['object']!=oid]
        if rels:result.append((oid,sorted(others)[0],rels[0]['object'],rels[0]['name'],colors[0]))
    return result

def main():
    if (DEST/'data_manifest.json').exists():return
    raw=fetch('https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip',DATA/'raw/sceneGraphs.zip')
    meta=fetch('https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip',DATA/'raw/image_data.json.zip')
    with zipfile.ZipFile(meta) as z:
        metadata=json.loads(z.read(next(n for n in z.namelist() if n.endswith('.json'))))
    coco_by_vg={str(r['image_id']):r.get('coco_id') for r in metadata}
    urls_by_vg={str(r['image_id']):r['url'] for r in metadata}
    old=json.loads((OUT/'data_manifest.json').read_text())['records'];used={r['image_id'] for r in old};oldhash={r['image_sha256'] for r in old}
    rng=random.Random(424028);selection=[];counts={}
    with zipfile.ZipFile(raw) as z:
        for split,filename,n in [('train','train_sceneGraphs.json',576),('test','val_sceneGraphs.json',128)]:
            graphs=json.loads(z.read(filename));ids=sorted(graphs);rng.shuffle(ids);count=0
            for iid in ids:
                if coco_by_vg.get(iid) in used:continue
                possibilities=eligible(graphs[iid])
                if not possibilities:continue
                target,other,ref,relation,color=rng.choice(possibilities);g=graphs[iid]
                splitname='calibration' if split=='train' and count>=512 else split
                selection.append({'image_id':iid,'split':splitname,'target':target,'distractor':other,'reference':ref,'relation':relation,'color':color,
                    'width':g['width'],'height':g['height'],'objects':g['objects'],'coco_id':coco_by_vg.get(iid),
                    'query':f"the {color} {g['objects'][target]['name']} {relation} the {g['objects'][ref]['name']}"})
                count+=1
                if count==n:break
            assert count==n,(split,count);counts[split]=count
    manifest={'scope':'derived condition/box-grounding diagnostic, NOT official GQA question or mask benchmark',
        'selection_rule':'seed424028,512train/64calibration from officialtrain,128test fromofficialval; two same-name >=16pixel boxes with exactly1listedcolor, original left/right relation; exclude V12COCO imageIDs; selection beforeimages/inference',
        'scenegraphs_sha256':sha(raw),'metadata_sha256':sha(meta),'image_archive_url':IMAGES,'records':selection}
    p=DEST/'selection.json'
    if p.exists():assert json.loads(p.read_text())==manifest
    else:write(p,manifest)
    remote=HTTPRange(IMAGES)
    with zipfile.ZipFile(remote) as z:
        infos={i.filename.split('/')[-1]:i for i in z.infolist() if i.filename.endswith('.jpg')}
    write(DATA/'raw/selected_zip_index.json',{'archive_bytes':remote.size,'members':{r['image_id']:{'name':infos[r['image_id']+'.jpg'].filename,'crc32':infos[r['image_id']+'.jpg'].CRC,'bytes':infos[r['image_id']+'.jpg'].file_size} for r in selection}})
    thread=threading.local()
    def acquire(r):
            info=infos[r['image_id']+'.jpg'];name=info.filename;p=DATA/'images'/f"{r['image_id']}.jpg";p.parent.mkdir(parents=True,exist_ok=True)
            existing=p.read_bytes() if p.exists() else None
            if existing is None or zlib.crc32(existing)!=info.CRC:
                if not hasattr(thread,'session'):thread.session=requests.Session()
                # Original per-image URL from official metadata; require exact ZIP size/CRC.
                url=urls_by_vg[r['image_id']]
                for attempt in range(3):
                    try:
                        response=thread.session.get(url,timeout=60);response.raise_for_status();content=response.content
                        assert len(content)==info.file_size and zlib.crc32(content)==info.CRC
                        break
                    except requests.RequestException:
                        if attempt==2:raise
                        time.sleep(2*(attempt+1))
                tmp=p.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(p)
            with Image.open(p) as im:assert im.size==(r['width'],r['height'])
            digest=sha(p);assert digest not in oldhash,'Image bytes overlap V12, stop rather than replace'
            r.update(image=str(p.resolve()),image_sha256=digest,zip_member=name,zip_crc32=info.CRC,original_metadata_url=urls_by_vg[r['image_id']])
            return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(acquire,selection)):
            if (i+1)%32==0:
                write(DEST/'preparation_status.json',{'stage':'fetching_original_zip_members','done':i+1,'total':len(selection)})
                print('GQA images',i+1,'/',len(selection),flush=True)
    manifest['selection_sha256']=sha(DEST/'selection.json');manifest['image_archive_bytes']=remote.size
    write(DEST/'data_manifest.json',manifest)
    write(DEST/'preparation_status.json',{'stage':'complete','images':len(selection),'manifest_sha256':sha(DEST/'data_manifest.json')})

if __name__=='__main__':main()
