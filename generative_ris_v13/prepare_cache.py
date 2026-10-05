"""Measure actual cache preparation before deciding the unchanged matrix budget."""
from .common import *
import ctypes
import time
import psutil
from .train import build_cache
from .run import verify_inputs
from multitask_v7.engine import setup


def main():
    deterministic();start=time.monotonic()
    # 2h engineering cap; this phase performs no optimizer update or heldout inference.
    def tick(stage,**detail):
        elapsed=time.monotonic()-start
        write(WORK/'cache_status.json',{'stage':stage,'elapsed_seconds':elapsed,**detail})
        assert elapsed<7200,'Cache engineering2h cap'
        assert not (WORK/'PAUSE').exists(),'User pause marker'
    rr=verify_inputs();pipe,params=setup(17,OLD/'seed17/adapter_1000.pt')
    cache=build_cache(pipe,rr,tick)
    elapsed=time.monotonic()-start
    benchmark=read(WORK/'deterministic_benchmark.json')
    # Keep the original conservative optimizer/inference estimates unchanged.
    # Measured cache time replaces the rough4h reserve;1h covers remaining IO/report/setup.
    estimated=benchmark['conservative_training_hours']+benchmark['estimated_evaluation_hours']+elapsed/3600+1
    write(WORK/'budget_review.json',{'status':'complete','original_estimate_hours':benchmark['estimated_total_hours'],
          'training_hours':benchmark['conservative_training_hours'],'evaluation_hours':benchmark['estimated_evaluation_hours'],
          'measured_cache_and_input_audit_hours':elapsed/3600,'remaining_io_setup_report_allowance_hours':1,
          'estimated_total_hours':estimated,'budget_gate_pass':estimated<=48,
          'basis':'Replace unmeasured4h cache/IO placeholder with actual full cache+input audit and1h remaining allowance. No change to18runs/8192steps or conservative P90 training and inference margins.',
          'source_sha256':sha(__file__),'benchmark_sha256':sha(WORK/'deterministic_benchmark.json'),
          'cache_audit_sha256':sha(WORK/'cache_audit.json')})
    tick('complete',images=8192);print('CACHE ENGINEERING COMPLETE',estimated,flush=True)


if __name__=='__main__':
    write(WORK/'cache_process.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'command':psutil.Process().cmdline()})
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main()
    except Exception as e:
        write(WORK/'cache_error.json',{'type':type(e).__name__,'message':str(e)});raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
