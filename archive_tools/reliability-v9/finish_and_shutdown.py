"""Finite, user-authorized V9 completion guard. Never shuts down on partial work.

Run after attaching to the existing pipeline PID. This is not a scheduler.
No cloud upload, forced application termination, protocol changes, or retries.
"""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(r'E:\Codex\2026-09-27\yo')
REPO = ROOT / 'outputs/marigold-depth'
OUT = REPO / 'results/reliability_training_v9'
WORK = ROOT / 'work/marigold-local/reliability_training_v9'
DELIVERY = ROOT / 'output/training_v9_delivery'
PLANS = ROOT / 'work/.planning/marigold-local'
MODES = ['depth', 'normal', 'joint', 'uniform', 'weaker_uniform', 'weighted', 'shuffled']
EXPECTED = dict(independently_recomputed_predictions=1536, exact_restored_tasks=36,
                checkpoints=33, data_pairs=160, history_steps=26880)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    assert read(path) == value


def status(state, **extra):
    row = dict(state=state, updated_utc=now(), pid=os.getpid(), **extra)
    save(WORK / 'completion_guard.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def wait_process(pid):
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.OpenProcess.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.WaitForSingleObject.restype = ctypes.c_uint32
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise RuntimeError('Cannot attach to pipeline; completion is not assumed')
    status('waiting_for_pipeline', pipeline_pid=pid)
    try:
        while True:
            code = k.WaitForSingleObject(handle, 30000)
            if code == 0:
                break
            if code != 258:
                raise RuntimeError(f'Windows process wait failed: {code}')
        exit_code = ctypes.c_uint32()
        if not k.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if exit_code.value:
            raise RuntimeError(f'Pipeline exited with code {exit_code.value}')
    finally:
        k.CloseHandle(handle)


def validate_summary(pipeline, analysis, gate):
    assert pipeline['state'] == 'complete', 'Pipeline did not complete'
    assert pipeline['completed_trainings'] == 21
    assert gate['passed'] is True
    for key, value in EXPECTED.items():
        assert analysis['verification'][key] == value, key
    assert analysis['verification']['historical_files_unchanged'] > 0
    assert len(analysis['comparisons']) == 6
    assert set(analysis['summary']) == set(MODES)


def verify_completion():
    sys.path.insert(0, str(REPO))
    from training_v9.protocol import verify
    for stage in ['b', 'c']:
        verify(stage)
    pipeline = read(OUT / 'pipeline_status.json')
    analysis = read(OUT / 'analysis_c.json')
    validate_summary(pipeline, analysis, read(OUT / 'baseline_gate.json'))
    assert sha(OUT / 'analysis_c.json') == pipeline['analysis_sha256']
    expected_names = {f'{mode}_seed{seed}' for mode in MODES for seed in [17, 29, 43]}
    assert {p.parent.name for p in (OUT / 'runs').glob('*/complete.json')} == expected_names
    for name in sorted(expected_names):
        folder = OUT / 'runs' / name
        done = read(folder / 'complete.json')
        stage = 'b' if name.split('_seed')[0] in ['depth', 'normal'] else 'c'
        assert done['protocol_sha256'] == sha(OUT / f'protocol_{stage}.json')
        assert done['training_sha256'] == sha(folder / 'training.json')
        assert read(folder / 'training.json')['steps'] == 1280
        assert done['final_checkpoint_sha256'] == sha(WORK / 'checkpoints' / name / 'step1280/adapter.pt')
        resume = WORK / 'checkpoints' / name / 'resume.pt'
        meta = read(resume.with_suffix('.json'))
        assert meta['step'] == 1280 and meta['sha256'] == sha(resume)
        for step, digest in done['metrics_sha256'].items():
            assert digest == sha(folder / f'validation_{step}.json')
    assert len(list((WORK / 'predictions').rglob('*.npy'))) == 1536
    assert len(list((WORK / 'checkpoints').glob('*/step*/adapter.pt'))) == 33
    from PIL import Image
    for name in ['learning_curves.png', 'final_metrics.png']:
        with Image.open(OUT / 'figures' / name) as im:
            im.verify()
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ['href', 'src'] and value and '://' not in value:
                    assert (OUT / value.split('#')[0]).is_file(), value
    for stage, name in [('b', 'RESULTS_b.html'), ('c', 'RESULTS.html')]:
        checks = read(OUT / f'report_checks_{stage}.json')
        assert checks['report_sha256'] == sha(OUT / name)
        assert checks['analysis_sha256'] == sha(OUT / f'analysis_{stage}.json')
        Links().feed((OUT / name).read_text(encoding='utf-8'))
    return analysis


def backup(sources, destination):
    """Archive every selected file and read it back against a SHA256 manifest."""
    entries = []
    for label, folder in sources:
        for path in sorted(folder.rglob('*')):
            relative = path.relative_to(folder)
            if not path.is_file() or any(p in ['.git', '__pycache__', '.venv', '.cache'] for p in relative.parts):
                continue
            if path.name in ['completion_guard.json', 'completion_guard.lock'] or path.name.startswith('.env'):
                continue
            if path.suffix in ['.partial', '.pyc', '.lock']:
                continue
            entries.append(dict(source=str(path), archive=f'{label}/{relative.as_posix()}',
                                size=path.stat().st_size, sha256=sha(path)))
    assert entries and len({r['archive'] for r in entries}) == len(entries)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for row in entries:
            z.write(row['source'], row['archive'])
        z.writestr('MANIFEST.json', json.dumps(entries, ensure_ascii=False, indent=2))
    with partial.open('rb+') as f:
        os.fsync(f.fileno())
    with zipfile.ZipFile(partial) as z:
        assert len(z.infolist()) == len(entries) + 1
        for row in entries:
            h = hashlib.sha256()
            with z.open(row['archive']) as f:
                for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    h.update(b)
            assert h.hexdigest() == row['sha256'], row['archive']
            assert sha(row['source']) == row['sha256'], 'Source changed during backup'
    os.replace(partial, destination)
    result = dict(path=str(destination), sha256=sha(destination), bytes=destination.stat().st_size,
                  files=len(entries), source_bytes=sum(r['size'] for r in entries))
    save(destination.with_suffix('.manifest.json'), dict(archive=result, files=entries))
    return result


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--pipeline-pid', type=int)
    selection.add_argument('--verify-finished', action='store_true')
    parser.add_argument('--shutdown-after-success', action='store_true')
    args = parser.parse_args()
    lock = WORK / 'completion_guard.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.pipeline_pid:
            wait_process(args.pipeline_pid)
        status('verifying_completion')
        analysis = verify_completion()
        # Preserve a receipt independent of the pipeline before creating the archive.
        stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        folder = DELIVERY / stamp
        folder.mkdir(parents=True, exist_ok=False)
        with (PLANS / 'progress.md').open('a', encoding='utf-8') as f:
            f.write(f'\nV9 completion guard {now()}: all21trainings completed and verified;1536predictions,33checkpoints,36restoredtasks,26880updates. Source/protocol/report hashes pass. Saving full local experiment archive. Shutdown requested for this invocation: {args.shutdown_after_success}. No new independent-data claim or GitHub publication.\n')
            f.flush(); os.fsync(f.fileno())
        status('backing_up', destination=str(folder))
        archive = backup([
            ('repository', REPO),
            ('v8_raw_weights_and_evidence', WORK.parent / 'reliability_v8'),
            ('v9_checkpoints_predictions_and_logs', WORK),
            ('planning', PLANS),
            ('execution_tools', ROOT / 'work/reliability-v9'),
        ], folder / 'experiment_files.zip')
        verify_completion()
        receipt = dict(state='all_experiments_verified_and_saved', updated_utc=now(),
                       verification=analysis['verification'], archive=archive,
                       report=str(OUT / 'RESULTS.html'), analysis_sha256=sha(OUT / 'analysis_c.json'),
                       original_assets=str(WORK.parent / 'assets'),
                       notes=['Archive is local; no upload was performed.',
                              'Previously saved original model/dataset assets stay in their original directory.',
                              'All scheduled experiments completed; this does not imply a positive scientific result.'])
        save(folder / 'COMPLETION.json', receipt)
        save(DELIVERY / 'LATEST.json', receipt)
        if not args.shutdown_after_success or (WORK / 'CANCEL_SHUTDOWN').exists():
            status('saved_without_shutdown', receipt=str(folder / 'COMPLETION.json'))
            return
        status('shutdown_requested', receipt=str(folder / 'COMPLETION.json'),
               command='shutdown.exe /s /t 0; no force flag')
        # Normal Windows shutdown only. A rejection is logged; no alternate bypass.
        command = str(Path(os.environ['SystemRoot']) / 'System32/shutdown.exe')
        done = subprocess.run([command, '/s', '/t', '0'], capture_output=True, text=True)
        save(folder / 'shutdown_command.json', dict(returncode=done.returncode, stdout=done.stdout,
                                                   stderr=done.stderr, updated_utc=now()))
        if done.returncode:
            raise RuntimeError(f'Windows shutdown command failed: {done.returncode}: {done.stderr}')
    except BaseException as exc:
        status('failed_no_shutdown', error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
