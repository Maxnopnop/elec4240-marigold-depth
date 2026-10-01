"""Read selected frames directly from NYU's official HDF5 using HTTP ranges.

Uses the conventional 795/654 split distributed by BTS (SHA256 checked).
Selects one frame per scene; validation scenes come only from training split.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import requests
from scipy.io import loadmat
from download_assets import NYU_URL, SPLIT_URL, download

SPLIT_SHA256='6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee'

class RangeFile(io.RawIOBase):
    """Read-only seekable HTTP file with a persistent block cache."""
    def __init__(self, url, cache):
        self.url=url; self.pos=0; self.size=2972037809; self.bs=262144
        self.cache=Path(cache); self.cache.mkdir(parents=True,exist_ok=True)
        self.blocks={}; self.downloaded=0; self.session=requests.Session()
    def seek(self,pos,whence=0):
        self.pos=pos if whence==0 else self.pos+pos if whence==1 else self.size+pos
        return self.pos
    def tell(self):return self.pos
    def readable(self):return True
    def seekable(self):return True
    def getblock(self,block):
        path=self.cache/f'{block:08d}.block'
        expected=min(self.bs,self.size-block*self.bs)
        if path.exists() and path.stat().st_size==expected:
            return path.read_bytes()
        start=block*self.bs;end=start+expected-1
        for attempt in range(3):
            try:
                r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,60))
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=expected:
                    raise IOError('Server did not honor byte range')
                tmp=path.with_name(path.name+f'.{os.getpid()}.partial')
                tmp.write_bytes(r.content);os.replace(tmp,path);self.downloaded+=len(r.content)
                return r.content
            except (requests.RequestException,IOError):
                if attempt==2:raise
                time.sleep(attempt+1)
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        out=[]
        while n>0 and self.pos<self.size:
            block=self.pos//self.bs
            if block not in self.blocks:self.blocks[block]=self.getblock(block)
            p=self.pos%self.bs; m=min(n,len(self.blocks[block])-p)
            out.append(self.blocks[block][p:p+m]);self.pos+=m;n-=m
        return b''.join(out)
    def prefetch(self,blocks):
        missing=sorted(set(blocks)-self.blocks.keys())
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i,(b,data) in enumerate(zip(missing,ex.map(self.getblock,missing))):
                self.blocks[b]=data
                if (i+1)%100==0:print(f'Prefetched {i+1}/{len(missing)} blocks',flush=True)

def extract_frame(args):
    root,row=args;root=Path(root);dest=root/'subset';dest.mkdir(exist_ok=True)
    path=dest/row['file']
    if not path.exists():
        remote=RangeFile(NYU_URL,root/'range_cache')
        with h5py.File(remote,'r') as f:
            i=row['id']-1
            image=f['images'][i].transpose(2,1,0)
            depth=f['depths'][i].T
            assert image.shape==(480,640,3) and depth.shape==(480,640)
        tmp=path.with_suffix('.npz.partial')
        with tmp.open('wb') as stream:np.savez_compressed(stream,image=image,depth=depth)
        os.replace(tmp,path)
    with np.load(path) as sample:
        assert sample['image'].shape==(480,640,3) and sample['depth'].shape==(480,640)
    row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--train',type=int,default=32);p.add_argument('--val',type=int,default=8);p.add_argument('--test',type=int,default=24)
    p.add_argument('--seed',type=int,default=4240);a=p.parse_args();root=Path(a.root)
    root.mkdir(parents=True,exist_ok=True);download(SPLIT_URL,root/'splits.mat')
    assert hashlib.sha256((root/'splits.mat').read_bytes()).hexdigest()==SPLIT_SHA256
    split=loadmat(root/'splits.mat');train=split['trainNdxs'].ravel()-1;test=split['testNdxs'].ravel()-1
    assert len(train)==795 and len(test)==654 and not set(train)&set(test)
    remote=RangeFile(NYU_URL,root/'range_cache')
    with h5py.File(remote,'r') as f:
        refs=f['scenes'][0];scenes={}
        def scene(i):
            i=int(i)
            if i not in scenes:scenes[i]=''.join(chr(int(c)) for c in f[refs[i]][()].ravel())
            return scenes[i]
        rng=np.random.default_rng(a.seed)
        def choose(indices,count):
            order=rng.permutation(indices);selected=[];seen=set()
            for i in order:
                name=scene(i)
                if name not in seen:
                    selected.append(int(i));seen.add(name)
                    print('Selected scene',len(selected),name,flush=True)
                if len(selected)==count:return selected
            raise ValueError('Not enough distinct scenes')
        tr=choose(train,a.train+a.val);te=choose(test,a.test)
        assert not {scene(i) for i in tr}&{scene(i) for i in te}, 'Selected scene overlap'
        groups={'train':tr[:a.train],'val':tr[a.train:],'test':te}
        rows=[{'id':i+1,'scene':scenes[i],'split':group,'file':f'{i+1:04d}.npz'}
              for group,ids in groups.items() for i in ids]
    # Direct indexed reads avoid a costly full HDF5 chunk-index enumeration.
    with ProcessPoolExecutor(max_workers=4) as ex:
        extracted=[]
        for row in ex.map(extract_frame,[(str(root),r) for r in rows]):
            extracted.append(row);print('Saved',row['split'],row['id'],row['scene'],flush=True)
    rows=extracted
    manifest={'dataset':'NYU Depth V2 labeled','source_url':NYU_URL,'split_url':SPLIT_URL,
              'split_sha256':SPLIT_SHA256,'seed':a.seed,'one_frame_per_scene':True,
              'official_train_count':795,'official_test_count':654,'samples':rows}
    path=Path(a.manifest);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('SUBSET_READY',len(rows),'downloaded_MB',round(remote.downloaded/1e6,1),flush=True)
if __name__=='__main__':main()
