"""Finite batch, complete audits and archive before user-authorized non-forced shutdown."""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from .data import ROOT,OUT,WORK,OLDWORK,OLDOUT,sha,write,audit
from .train import ARMS,SEEDS,STEPS,freeze,verify,run_training,final_evaluation

def archive():
    target=ROOT.parents[1]/'output/generative_ris_v12_delivery';target.mkdir(parents=True,exist_ok=True)
    paths={}
    for label,folder in [('source',ROOT/'generative_ris_v12'),('results',OUT),('work',WORK)]:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.lock','.tmp','.log'] and p.name not in ['status.json','run.lock']:
                paths[f'{label}/{p.relative_to(folder).as_posix()}']=p
    # Include referenced original inputs without moving or modifying them.
    rows=json.loads((OUT/'data_manifest.json').read_text())['records']
    for r in rows:
        for key in ['image','mask']:
            p=Path(r[key])
            if OLDWORK in p.parents:paths[f'v11_inputs/{p.relative_to(OLDWORK).as_posix()}']=p
    for seed in SEEDS:paths[f'initial_adapters/seed{seed}_1000.pt']=OLDWORK/f'seed{seed}/adapter_1000.pt'
    for name in ['data_manifest.json','training_protocol.json']:paths['v11_inputs/'+name]=OLDOUT/name
    for name in ['experiment.py','multitask_v7/engine.py','multitask_v7/common.py','generative_ris_v11/train_baseline.py','generative_ris_v11/prepare_training.py']:
        paths['dependencies/'+name]=ROOT/name
    entries=[{'archive':name,'source':str(p),'sha256':sha(p),'size':p.stat().st_size} for name,p in sorted(paths.items())]
    dest=target/'experiment_files.zip';tmp=target/'experiment_files.partial'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for r in entries:z.write(r['source'],r['archive'])
        z.writestr('MANIFEST.json',json.dumps(entries,indent=2))
    with tmp.open('rb+') as f:os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as z:
        for r in entries:
            h=hashlib.sha256()
            with z.open(r['archive']) as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==r['sha256'] and sha(r['source'])==r['sha256'],r['archive']
    tmp.replace(dest)
    receipt={'state':'all_batch_tasks_saved_and_verified','archive':str(dest),'sha256':sha(dest),'files':len(entries),'bytes':dest.stat().st_size,
        'original_pretrained_model_retained_at':'work/marigold-local/assets; not duplicated','report':str(OUT/'RESULTS.html')}
    write(target/'COMPLETION.json',receipt);return receipt

def main(shutdown=False):
    lock=WORK/'run.lock';WORK.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as f:f.write(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        protocol=freeze();audit();deadline=time.time()+36*3600
        assert json.loads((OUT/'smoke.json').read_text())['status']=='passed'
        for seed in SEEDS:
            for arm in ARMS:
                if not (OUT/'runs'/f'{arm}_s{seed}'/'trained.json').exists():run_training(arm,seed,protocol,deadline)
        write(OUT/'status.json',{'stage':'final_heldout_evaluation'})
        for seed in SEEDS:
            for arm in ARMS:
                if not (OUT/'runs'/f'{arm}_s{seed}'/'complete.json').exists():final_evaluation(arm,seed,protocol)
        # Additional explicitly requested directions must be registered and finish before shutdown.
        plan=json.loads((OUT/'extra_plan.json').read_text())
        for relative,digest in plan['source_sha256'].items():assert sha(ROOT/relative)==digest
        subprocess.run([sys.executable,'-m',plan['module']],cwd=ROOT,check=True)
        extra=json.loads((OUT/'conditions/complete.json').read_text());assert extra['status']=='complete'
        from .report import main as report
        verification=report();verify(protocol);audit()
        receipt=archive()
        write(OUT/'completion.json',{'status':'complete','verification':verification,'archive':receipt,'extra_experiments':extra})
        plans=ROOT.parents[1]/'work/.planning/marigold-local'
        with (plans/'progress.md').open('a',encoding='utf-8') as f:
            f.write('\nV12 finite batch COMPLETE:15x2048updates,heldoutmatrix,condition probes,independentmaskmetrics,checkpointrestores,inputhashes,verifiedlocalarchive. No positive-result requirement. Shutdown requested='+str(shutdown)+'\n');f.flush();os.fsync(f.fileno())
        if shutdown and not (WORK/'CANCEL_SHUTDOWN').exists():
            write(OUT/'status.json',{'stage':'saved_shutdown_requested','archive':receipt['archive'],'command':'shutdown.exe /s /t 0 (no force)'})
            result=subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/shutdown.exe'),'/s','/t','0'],capture_output=True,text=True)
            write(OUT/'shutdown_result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:raise RuntimeError('Windows declined shutdown; no alternate bypass attempted')
        else:write(OUT/'status.json',{'stage':'complete_saved_without_shutdown'})
    except BaseException:
        write(OUT/'status.json',{'stage':'failed_no_shutdown','error':traceback.format_exc()});raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shutdown-after-success',action='store_true');args=parser.parse_args();main(args.shutdown_after_success)
