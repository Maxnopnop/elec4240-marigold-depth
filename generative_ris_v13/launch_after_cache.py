"""One bounded handoff from verified engineering preparation to formal runner."""
from .common import *
import time
import psutil
from .run import main as run


def main():
    identity={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(WORK/'queue_process.json',identity)
    frozen=read(WORK/'launch_source_hashes.json')
    expected=read(WORK/'cache_process.json')
    started=time.monotonic()
    while psutil.pid_exists(expected['pid']):
        p=psutil.Process(expected['pid'])
        if abs(p.create_time()-expected['create_time'])>.1:break
        assert p.cmdline()==expected['command'],'Cache PID command changed'
        assert time.monotonic()-started<7500,'Cache wait timeout'
        assert not (WORK/'PAUSE').exists(),'User pause marker'
        time.sleep(10)
    assert not (WORK/'cache_error.json').exists(),'Cache preparation failed; no formal launch'
    assert read(WORK/'cache_status.json')['stage']=='complete'
    assert read(WORK/'budget_review.json')['budget_gate_pass'],'Budget gate failed; no formal launch'
    for path,digest in frozen.items():assert sha(REPO/path)==digest,path
    assert read(WORK/'protocol_tests.json')['status']=='passed'
    run()


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(WORK/'queue_error.json',{'type':type(e).__name__,'message':str(e)});raise
