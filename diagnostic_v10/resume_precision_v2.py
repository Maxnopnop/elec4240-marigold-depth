"""Resume unchanged experiments with a documented gradient-precision diagnostic."""
import ctypes
import os
import subprocess
import sys
import traceback
from .common import *
from .gradients_precision_v2 import verify_amendment


def main():
    verify_amendment();lock=WORK/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        a=read(OUT/'numerical_amendment.json')
        assert sha(OUT/'regional.json')==a['completed_regional_sha256']
        for module in ['gradients_precision_v2','noise','report']:
            verify_amendment()
            write(OUT/'status.json',dict(state='running',module=module,pid=os.getpid(),amendment='numerical_amendment.json'))
            subprocess.run([sys.executable,'-u','-m','diagnostic_v10.'+module],cwd=ROOT,check=True)
        gradients=read(OUT/'gradients.json')['records'];flags=[r for r in gradients if r['precision_flag']]
        assert all(r['fp32_reference']['sum_relative_error']<.001 for r in flags)
        note=(f'<p class="box"><strong>Numerical diagnostic amendment:</strong> {len(flags)} of56 BF16 probes exceeded the original2% gradient-sum discrepancy screen. '
              'All flagged BF16 measurements were retained, and each was additionally checked in full FP32 with relative discrepancy below0.1%. '
              'This is a numerical sensitivity analysis, not a change to any training method, seed, data, loss or budget. '
              'Interpret near-zero gradient cosines cautiously. See <a href="numerical_amendment.json">the preserved amendment</a> and <a href="gradients.json">per-probe FP32 references</a>.</p>')
        report=OUT/'RESULTS.html';text=report.read_text(encoding='utf-8')
        text=text.replace('<h2>2. Are gradients conflicting or too weak?</h2>','<h2>2. Are gradients conflicting or too weak?</h2>'+note)
        report.write_text(text,encoding='utf-8')
        done=read(OUT/'complete.json');done.update(report_sha256=sha(report),numerical_amendment_sha256=sha(OUT/'numerical_amendment.json'),
                                                 bf16_precision_flags=len(flags),fp32_references=len(flags))
        write(OUT/'complete.json',done)
        write(OUT/'status.json',dict(state='complete',pid=os.getpid(),receipt=done))
    except BaseException as exc:
        write(OUT/'status.json',dict(state='failed',message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
