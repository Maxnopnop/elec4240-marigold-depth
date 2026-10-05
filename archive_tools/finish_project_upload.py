"""Finalize an explicitly authorized complete-project upload; optional non-forced shutdown.

No training or dataset mutations. No shutdown fallback or retry.
"""
import argparse, ctypes, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import upload_experiment_archives as u

def git(*args):
    return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=u.REPO,text=True,stderr=subprocess.STDOUT).strip()

def native_shutdown():
    from ctypes import wintypes
    class LUID(ctypes.Structure):
        _fields_=[('LowPart',wintypes.DWORD),('HighPart',wintypes.LONG)]
    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_=[('Luid',LUID),('Attributes',wintypes.DWORD)]
    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_=[('PrivilegeCount',wintypes.DWORD),('Privileges',LUID_AND_ATTRIBUTES*1)]
    adv=ctypes.WinDLL('advapi32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    adv.OpenProcessToken.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    adv.LookupPrivilegeValueW.argtypes=[wintypes.LPCWSTR,wintypes.LPCWSTR,ctypes.POINTER(LUID)]
    adv.AdjustTokenPrivileges.argtypes=[wintypes.HANDLE,wintypes.BOOL,ctypes.POINTER(TOKEN_PRIVILEGES),wintypes.DWORD,ctypes.c_void_p,ctypes.c_void_p]
    adv.InitiateSystemShutdownExW.argtypes=[wintypes.LPWSTR,wintypes.LPWSTR,wintypes.DWORD,wintypes.BOOL,wintypes.BOOL,wintypes.DWORD]
    token=wintypes.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(),0x20|0x8,ctypes.byref(token)):raise ctypes.WinError(ctypes.get_last_error())
    try:
        privileges=TOKEN_PRIVILEGES();privileges.PrivilegeCount=1;privileges.Privileges[0].Attributes=2
        if not adv.LookupPrivilegeValueW(None,'SeShutdownPrivilege',ctypes.byref(privileges.Privileges[0].Luid)):raise ctypes.WinError(ctypes.get_last_error())
        ctypes.set_last_error(0)
        if not adv.AdjustTokenPrivileges(token,False,ctypes.byref(privileges),0,None,None):raise ctypes.WinError(ctypes.get_last_error())
        error=ctypes.get_last_error()
        if error:raise ctypes.WinError(error)
        if not adv.InitiateSystemShutdownExW(None,'ELEC4240: GitHub upload and archive verification completed. Non-forced shutdown in 60 seconds.',60,False,False,0x80000000):raise ctypes.WinError(ctypes.get_last_error())
        return {'state':'native_shutdown_request_accepted','timeout_seconds':60,'force_apps_closed':False,'restart':False,'physical_shutdown_confirmed':False}
    finally:kernel.CloseHandle(token)

def finalize(shutdown):
    completion_path=u.OUT/'COMPLETION.json'
    deadline=time.monotonic()+24*3600
    while not completion_path.exists():
        error=u.ROOT/'work/upload_archives_v3.stderr.log'
        if error.exists() and error.stat().st_size:raise RuntimeError('Upload worker failed; no shutdown. See upload_archives_v3.stderr.log')
        if time.monotonic()>deadline:raise RuntimeError('Upload completion timeout; no shutdown')
        time.sleep(10)
    completion=json.loads(completion_path.read_text());manifest=json.loads((u.OUT/'ALL_EXPERIMENTS_MANIFEST.json').read_text())
    if completion['status']!='complete' or not completion['all_github_digests_verified']:raise RuntimeError('Upload completion not verified')
    release=u.req('GET',u.API+'/releases/tags/'+u.TAG)
    if release['draft']:raise RuntimeError('Release still draft')
    assets=u.all_assets(release['id']);remote={a['name']:a for a in assets}
    for item in manifest['uploads']:
        a=remote[item['name']]
        if a['state']!='uploaded' or a['size']!=item['bytes'] or a.get('digest')!='sha256:'+item['sha256']:raise RuntimeError('Remote asset verification failed: '+item['name'])
    manifest_asset=remote['ALL_EXPERIMENTS_MANIFEST.json']
    if manifest_asset.get('digest')!='sha256:'+u.digest(u.OUT/'ALL_EXPERIMENTS_MANIFEST.json'):raise RuntimeError('Remote manifest mismatch')
    delivery=u.REPO/'delivery/2026-10-04';delivery.mkdir(parents=True,exist_ok=True)
    for name in ['COMPLETION.json','ALL_EXPERIMENTS_MANIFEST.json']:
        shutil.copy2(u.OUT/name,delivery/name)
    for name in ['upload_experiment_archives.py','finish_project_upload.py','test_upload_guard.py']:
        shutil.copy2(u.ROOT/'work'/name,u.REPO/'archive_tools'/name)
    summary=f"\n\nUpload verified on 2026-10-04: {completion['experiments']} historical experiment work directories, {completion['files']:,} generated files, {completion['assets']} release assets including the inventory, {completion['bytes']:,} archive bytes. All GitHub SHA-256 digests matched. [Upload receipt](delivery/2026-10-04/COMPLETION.json).\n"
    index=u.REPO/'EXPERIMENT_ARCHIVE.md'
    if 'Upload verified on 2026-10-04:' not in index.read_text(encoding='utf-8'):
        with index.open('a',encoding='utf-8') as f:f.write(summary)
    git('add','--','delivery/2026-10-04','archive_tools/upload_experiment_archives.py','archive_tools/finish_project_upload.py','archive_tools/test_upload_guard.py','EXPERIMENT_ARCHIVE.md')
    if git('diff','--cached','--name-only'):git('commit','-m','Verify complete historical experiment release and upload receipts','--quiet')
    git('push','origin','HEAD')
    local=git('rev-parse','HEAD');remote_head=git('ls-remote','origin','refs/heads/main').split()[0]
    if local!=remote_head:raise RuntimeError('Final remote commit mismatch')
    receipt={'status':'all_uploads_and_final_commit_verified','commit':local,'release_url':release['html_url'],'assets':len(assets),'shutdown_requested':shutdown,'original_data_modified':False}
    u.save(u.OUT/'FINAL_VERIFICATION.json',receipt)
    plan=u.ROOT/'work/.planning/marigold-local'
    for name in ['progress.md','task_plan.md']:
        with (plan/name).open('a',encoding='utf-8') as f:f.write('\nPhase26 upload COMPLETE: '+json.dumps(receipt)+'\n')
    u.log('ALL UPLOADS VERIFIED '+local)
    if shutdown:
        if (u.OUT/'CANCEL_SHUTDOWN').exists() or (u.ROOT/'work/marigold-local/generative_ris_v12/CANCEL_SHUTDOWN').exists():
            u.save(u.OUT/'SHUTDOWN.json',{'state':'cancelled_by_marker'});return
        shutdown_receipt=u.OUT/'SHUTDOWN.json'
        if shutdown_receipt.exists():raise RuntimeError('Shutdown already recorded; refusing duplicate attempt')
        u.save(shutdown_receipt,{'state':'native_shutdown_attempt_started','force_apps_closed':False})
        try:
            result=native_shutdown();u.save(shutdown_receipt,result);u.log(result['state'])
        except Exception as e:
            u.save(shutdown_receipt,{'state':'native_shutdown_failed','type':type(e).__name__,'error':str(e),'winerror':getattr(e,'winerror',None),'alternate_attempted':False})
            u.log('Native shutdown failed; no fallback: '+str(e))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shutdown-after-success',action='store_true');args=parser.parse_args()
    try:finalize(args.shutdown_after_success)
    except Exception as e:
        u.save(u.OUT/'FINALIZATION_ERROR.json',{'type':type(e).__name__,'error':str(e),'shutdown_attempted':False});raise

