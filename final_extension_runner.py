"""Path-configurable launcher without modifying the archived experiment sources.

Run from this repository's root. Earlier manifests/results remain under results/;
ROOT redirects excluded assets and predictions, OUT redirects new v5 results.
"""
import argparse, importlib, sys
from pathlib import Path

MODULES = {'control':'run_final_control', 'cost':'benchmark_final_v5',
           'analysis':'analyze_final_v5', 'failure':'failure_analysis_v5',
           'summary':'summarize_final_v5', 'audit':'check_final_delivery'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=MODULES)
    p.add_argument('--root', type=Path, required=True,
                   help='Storage root containing assets/, robustness_v3/ and prospective_v4/.')
    p.add_argument('--out', type=Path, required=True, help='Directory for v5 result records.')
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--check-paths', action='store_true', help='Read-only preflight; no models are loaded.')
    a = p.parse_args()
    repo = Path(__file__).resolve().parent
    assert Path.cwd().resolve() == repo, 'Run from the repository root.'
    assert (repo/'results/robustness_v3/split_manifest.json').is_file()
    assert (repo/'results/prospective_v4/manifest.json').is_file()
    a.root = a.root.resolve(); a.out = a.out.resolve()
    for name in ['assets/model_path.json','assets/subset','assets/external_sun3d_v4']:
        assert (a.root/name).exists(), str(a.root/name)
    for old in ['prospective_v4','robustness_v3','scaleup_v2']:
        assert a.out != (repo/'results'/old).resolve(), 'Do not overwrite historical results.'
    if a.check_paths:
        print('PATHS_OK', a.root, a.out); return
    if a.freeze:
        assert a.action in ['control','cost'], '--freeze applies to control or cost only.'
        target = a.out if a.action == 'control' else a.out/'cost'
        assert not (target/'protocol.json').exists(), 'Existing protocol: resume without --freeze.'
    assert not a.audit_only or a.action == 'control'
    import run_final_control as control
    control.ROOT = a.root; control.ASSETS = a.root/'assets'
    control.WORK = a.root/'final_extension_v5'; control.OUT = a.out
    sys.argv = [MODULES[a.action]] + (['--freeze'] if a.freeze else []) + (['--audit-only'] if a.audit_only else [])
    module = importlib.import_module(MODULES[a.action])
    module.main()


if __name__ == '__main__': main()
