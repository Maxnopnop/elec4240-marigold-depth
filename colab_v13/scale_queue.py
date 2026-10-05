"""Wait for the existing cache process; then test and run only if all gates pass."""
import os
import sys
import time
import subprocess
import traceback
import psutil
from pathlib import Path
from prepare_scale_cache import ROOT, DRIVE, read, write, sha


def main():
    assert sys.platform=='linux' and ROOT.is_dir() and DRIVE.is_dir(),'Cloud only'
    lock=ROOT/'scale_queue_process.json'
    if lock.exists():
        old=read(lock)
        try:
            p=psutil.Process(old['pid'])
            assert p.create_time()!=old['create_time'] or p.status()=='zombie','Queue already active'
        except psutil.NoSuchProcess:pass
    write(lock,{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    pins=read(ROOT/'SCALE_BUNDLE_MANIFEST.json')
    for name,digest in pins.items():assert sha(ROOT/name)==digest,name
    identity=read(ROOT/'scale_cache_process.json')
    started=time.monotonic()
    while True:
        assert not (DRIVE/'PAUSE').exists(),'Drive pause marker'
        audit=DRIVE/'scale_cache/cache_audit.json'
        if audit.exists() and read(audit).get('status')=='complete':break
        assert time.monotonic()-started<7200,'Cache wait exceeded2h'
        p=psutil.Process(identity['pid'])
        assert p.create_time()==identity['create_time'] and p.status()!='zombie','Cache exited before completion'
        write(ROOT/'scale_queue_status.json',{'stage':'waiting_for_existing_cache','cache_pid':p.pid})
        time.sleep(30)
    for name,digest in pins.items():assert sha(ROOT/name)==digest,name
    write(ROOT/'scale_queue_status.json',{'stage':'production_preflight'})
    subprocess.run([sys.executable,'-u',str(ROOT/'scale_preflight.py')],check=True,timeout=7500)
    for name,digest in pins.items():assert sha(ROOT/name)==digest,name
    write(ROOT/'scale_queue_status.json',{'stage':'freeze_budget_then_formal'})
    subprocess.run([sys.executable,'-u',str(ROOT/'scale_run.py')],check=True,timeout=48*3600)
    write(ROOT/'scale_queue_status.json',{'stage':'complete'})


if __name__=='__main__':
    try:main()
    except Exception as e:
        error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        write(ROOT/'scale_queue_error.json',error)
        if DRIVE.is_dir():write(DRIVE/'scale_queue_error.json',error)
        raise
