"""One-shot background execution of the already authorized experiment matrix.

This is a finite training job, not a scheduler or an agent automation. It never
changes a failed protocol, purchases compute, publishes files, or shuts down.
"""
import argparse,ctypes,datetime,json,os,subprocess,sys,traceback
from pathlib import Path
from .protocol import ROOT,OUT,WORK,SEEDS,read,write,sha


def status(state,**details):
    result={'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pid':os.getpid(),**details}
    write(OUT/'pipeline_status.json',result)
    print(json.dumps(result),flush=True)


def wait_for_baseline(pid):
    if os.name!='nt':raise RuntimeError('Existing-process wait is Windows-specific')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
    if not handle:
        complete=all((OUT/'runs'/f'{m}_seed{s}'/'complete.json').exists() for m in ['depth','normal'] for s in SEEDS)
        if complete:return
        raise RuntimeError(f'Cannot attach to baseline PID{pid}; six completion records are not present')
    status('waiting_for_baseline',baseline_pid=pid,protocol_b_sha256=sha(OUT/'protocol_b.json'))
    try:
        while True:
            signal=kernel.WaitForSingleObject(handle,30000)
            if signal==0:break
            if signal!=258:raise RuntimeError(f'WaitForSingleObject failed: {signal}')
        code=ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        if code.value!=0:raise RuntimeError(f'Baseline process failed with exit code {code.value}; no automatic retry')
    finally:kernel.CloseHandle(handle)


def run(module,*args):
    status('executing',module=module,arguments=list(args))
    subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-baseline-pid',type=int);args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    lock=WORK/'pipeline.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.wait_baseline_pid:wait_for_baseline(args.wait_baseline_pid)
        else:
            run('training_v9.protocol','freeze_b')
            run('training_v9.run','--stage','b')
        run('training_v9.analyze','--stage','b')
        run('training_v9.report','--stage','b')
        if not read(OUT/'baseline_gate.json')['passed']:
            status('baseline_gate_failed',report=str(OUT/'RESULTS_b.html'),message='Results retained. Joint training not started; diagnose before changing the frozen protocol.')
            return
        run('training_v9.test_analysis')
        run('training_v9.protocol','freeze_c')
        run('training_v9.run','--stage','c')
        run('training_v9.analyze','--stage','c')
        run('training_v9.report','--stage','c')
        result=read(OUT/'analysis_c.json')
        assert result['verification']['independently_recomputed_predictions']==1536
        assert result['verification']['exact_restored_tasks']==36
        assert result['verification']['history_steps']==26880
        assert len(result['comparisons'])==6
        status('complete',report=str(OUT/'RESULTS.html'),analysis_sha256=sha(OUT/'analysis_c.json'),
               completed_trainings=21,verification=result['verification'],
               note='Computed report and figures available. No new independent-data claim; no GitHub publication or shutdown.')
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
