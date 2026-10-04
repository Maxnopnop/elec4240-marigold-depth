import ctypes
import os
import subprocess
import sys
import traceback
from .common import *


def main():
    WORK.mkdir(parents=True,exist_ok=True);lock=WORK/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        freeze()
        for module in ['regional','gradients','noise','report']:
            write(OUT/'status.json',dict(state='running',module=module,pid=os.getpid()))
            subprocess.run([sys.executable,'-u','-m','diagnostic_v10.'+module],cwd=ROOT,check=True)
        write(OUT/'status.json',dict(state='complete',pid=os.getpid(),receipt=read(OUT/'complete.json')))
    except BaseException as exc:
        write(OUT/'status.json',dict(state='failed',message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
