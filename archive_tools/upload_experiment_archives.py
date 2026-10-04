"""Package and publish project-generated experiment artifacts, with resumable receipts.

Credentials are read from Git Credential Manager into memory, never written to disk.
Original datasets, pretrained assets and Python environments are not packaged.
"""
import concurrent.futures, hashlib, json, os, re, subprocess, threading, time, zipfile
from pathlib import Path
import requests

ROOT=Path(r'E:\Codex\2026-09-27\yo')
REPO=ROOT/'outputs/marigold-depth'
OUT=ROOT/'output/github_all_experiments_2026-10-04'
OUT.mkdir(parents=True,exist_ok=True)
TAG='all-experiments-2026-10-04'
API='https://api.github.com/repos/Maxnopnop/elec4240-marigold-depth'
LIMIT=900*1024*1024
LOCK=threading.Lock()
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)

def log(msg):
    with LOCK:print(time.strftime('%Y-%m-%d %H:%M:%S'),msg,flush=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def headers():
    r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,cwd=REPO,check=True)
    c=dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
    return {'Authorization':'Bearer '+c['password'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}

def req(method,url,**kw):
    r=requests.request(method,url,headers=headers(),timeout=120,**kw)
    r.raise_for_status();return r.json()

class Parts:
    def __init__(self,name):self.name=name;self.paths=[];self.f=None;self.pos=0;self.used=0
    def write(self,b):
        n=len(b);view=memoryview(b)
        while view:
            if self.f is None or self.used==LIMIT:
                if self.f:self.f.close()
                p=OUT/(self.name+'.zip.part'+str(len(self.paths)+1).zfill(3));self.paths.append(p)
                self.f=p.open('wb');self.used=0
            take=min(len(view),LIMIT-self.used);self.f.write(view[:take]);view=view[take:];self.used+=take;self.pos+=take
        return n
    def tell(self):return self.pos
    def flush(self):
        if self.f:self.f.flush()
    def close(self):
        if self.f:self.f.close()

def included(p,base):
    rel=p.relative_to(base)
    if any(x in {'__pycache__','.git','.venv','assets','upstream'} for x in rel.parts):return False
    if base.name in {'generative_ris_v11','generative_ris_v12'}:
        if any(x in {'data','images','masks','raw'} for x in rel.parts):return False
    if p.name in {'train_cache.pt','cache.pt','cache_clean.pt','cache_heterogeneous_noise.pt','cache_smooth_bias.pt','condition_features.pt','run.lock'}:return False
    if p.suffix in {'.pyc','.partial'}:return False
    if p.name=='.env' or any(x in p.name.lower() for x in ['credential','cookies']):raise RuntimeError('Sensitive file name: '+str(rel))
    return True

def pack(base):
    name=base.name; receipt=OUT/(name+'.package.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if all((OUT/a['name']).exists() and digest(OUT/a['name'])==a['sha256'] for a in old['assets']):return old
        raise RuntimeError('Existing package checksum mismatch: '+name)
    files=[p for p in sorted(base.rglob('*')) if p.is_file() and included(p,base)]
    manifest=[];parts=Parts(name)
    log('Packing '+name+' ('+str(len(files))+' files)')
    with zipfile.ZipFile(parts,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in files:
            rel=p.relative_to(base).as_posix();h=hashlib.sha256();size=0
            with p.open('rb') as src,z.open(name+'/'+rel,'w',force_zip64=True) as dst:
                for b in iter(lambda:src.read(4*1024*1024),b''):
                    if p.suffix.lower() in {'.txt','.log','.py','.json','.md','.ps1','.html','.ipynb','.csv'} and SECRET.search(b):raise RuntimeError('Potential secret in '+name+'/'+rel)
                    h.update(b);size+=len(b);dst.write(b)
            manifest.append({'path':name+'/'+rel,'bytes':size,'sha256':h.hexdigest()})
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    parts.close()
    if len(parts.paths)==1:
        dest=OUT/(name+'.zip');os.replace(parts.paths[0],dest);parts.paths=[dest]
    assets=[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in parts.paths]
    result={'experiment':name,'files':manifest,'assets':assets}
    save(receipt,result);log('Packaged '+name+' '+str(sum(x['bytes'] for x in assets))+' bytes')
    return result

def upload_asset(release,path,expected):
    current=req('GET',API+'/releases/'+str(release['id'])+'/assets?per_page=100')
    matches=[a for a in current if a['name']==path.name]
    if matches:
        a=matches[0]
        if a['state']=='uploaded' and a['size']==path.stat().st_size and a.get('digest')=='sha256:'+expected:
            return a
        raise RuntimeError('Remote asset mismatch, refusing overwrite: '+path.name)
    for attempt in range(3):
        try:
            with path.open('rb') as f:
                h=headers();h['Content-Type']='application/octet-stream'
                response=requests.post(release['upload_url'].split('{')[0],params={'name':path.name},headers=h,data=f,timeout=(60,3600))
            response.raise_for_status();a=response.json()
            if a['size']!=path.stat().st_size or a.get('digest')!='sha256:'+expected:raise RuntimeError('Uploaded asset SHA256 mismatch: '+path.name)
            log('Verified GitHub asset '+path.name);return a
        except (requests.ConnectionError,requests.Timeout):
            matches=[a for a in req('GET',API+'/releases/'+str(release['id'])+'/assets?per_page=100') if a['name']==path.name]
            if matches:
                a=matches[0]
                if a.get('digest')=='sha256:'+expected:return a
                raise RuntimeError('Interrupted remote asset needs review: '+path.name)
            if attempt==2:raise
            time.sleep(5)

def main():
    r=requests.get(API+'/releases/tags/'+TAG,headers=headers(),timeout=60)
    if r.status_code==404:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        release=req('POST',API+'/releases',json={'tag_name':TAG,'target_commitish':head,'name':'Complete ELEC4240 experiment archive through V12 (2026-10-04)','draft':True,'body':'All historical project-generated predictions, trained checkpoints and logs. Original datasets, downloaded base models, environments and recomputable top-level caches excluded. See EXPERIMENT_ARCHIVE.md in the repository. Split ZIP parts must be concatenated in numerical order before extraction. SHA256 checksums and per-file inventories accompany the archive. Includes negative and inconclusive findings; no claim of universal method superiority.'})
    else:r.raise_for_status();release=r.json()
    save(OUT/'release.json',{'id':release['id'],'html_url':release['html_url'],'draft':release['draft']})
    bases=[p for p in sorted((ROOT/'work/marigold-local').iterdir()) if p.is_dir() and p.name not in {'.venv','assets','upstream','__pycache__'}]
    def job(base):
        package=pack(base);uploaded=[]
        for a in package['assets']:
            result=upload_asset(release,OUT/a['name'],a['sha256'])
            uploaded.append({'name':a['name'],'sha256':a['sha256'],'bytes':a['bytes'],'url':result['browser_download_url'],'github_digest':result['digest']})
        save(OUT/(base.name+'.uploaded.json'),uploaded);return package,uploaded
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(job,b):b for b in bases}
        for f in concurrent.futures.as_completed(futures):
            completed.append(f.result());log('Finished '+futures[f].name)
    manifest={'tag':TAG,'exclusions':['original datasets/annotations','downloaded pretrained models','Python environments','top-level recomputable training caches','unrelated projects'],'experiments':[p for p,a in completed],'uploads':[a for p,items in completed for a in items]}
    save(OUT/'ALL_EXPERIMENTS_MANIFEST.json',manifest)
    upload_asset(release,OUT/'ALL_EXPERIMENTS_MANIFEST.json',digest(OUT/'ALL_EXPERIMENTS_MANIFEST.json'))
    release=req('PATCH',API+'/releases/'+str(release['id']),json={'draft':False})
    save(OUT/'COMPLETION.json',{'status':'complete','release_url':release['html_url'],'experiments':len(completed),'assets':len(manifest['uploads'])+1,'files':sum(len(p['files']) for p,a in completed),'bytes':sum(a['bytes'] for a in manifest['uploads']),'all_github_digests_verified':True})
    log('COMPLETE '+release['html_url'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'ERROR.json',{'type':type(e).__name__,'message':str(e),'time':time.time()});raise
