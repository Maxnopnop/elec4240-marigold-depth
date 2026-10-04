"""Run finite training with a process-scoped Windows sleep-prevention request."""
import ctypes
import os
from .train_baseline import main

if __name__=='__main__':
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main()
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
