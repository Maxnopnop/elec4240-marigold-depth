"""Use the skill's PNG renderer with Word as the Windows PDF conversion backend.

The runtime has no bundled LibreOffice for Windows. Do not use desktop LibreOffice.
Word exports the authored DOCX itself; no separately authored PDF substitutes for it.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
renderer_path = Path(r'C:\Users\A\.codex\plugins\cache\openai-primary-runtime\documents\26.909.11814\skills\documents\render_docx.py')
poppler_path = r'C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
os.environ['PATH'] = poppler_path + os.pathsep + os.environ.get('PATH', '')
spec = importlib.util.spec_from_file_location('canonical_docx_renderer', renderer_path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

def word_convert(doc_path, user_profile, convert_tmp_dir, stem, verbose=False):
    pdf = str(Path(convert_tmp_dir) / (stem + '.pdf'))
    result = subprocess.run([
        'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', str(ROOT / 'export_word.ps1'), '-InputDocx', doc_path, '-OutputPdf', pdf
    ], capture_output=True, text=True, timeout=60)
    log = result.stdout + result.stderr
    print(log)
    if result.returncode or not Path(pdf).exists():
        raise RuntimeError('Word PDF conversion failed: ' + log)
    return pdf, log

renderer.convert_to_pdf = word_convert
renderer.main()
