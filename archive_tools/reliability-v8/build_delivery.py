from pathlib import Path
import json, hashlib, zipfile, html, ast, platform, importlib.metadata

ROOT=Path('E:/Codex/2026-09-27/yo/outputs/marigold-depth')
OUT=ROOT/'results/reliability_v8'
DEST=ROOT/'reliability_v8/colab'
DEST.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

protocol=read(OUT/'component_protocol.json')
members=sorted(set(list(protocol['source_sha256'])+list(protocol['v7_source_sha256'])+[
    'results/metric_multitask_v7/protocol.json','results/metric_multitask_v7/manifest.json',
    'results/metric_multitask_v7/camera.json','results/reliability_v8/component_protocol.json',
    'reliability_v8/EXPERIMENT_PLAN.md']))
bundle=DEST/'reliability_v8_colab_bundle.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(ROOT/name,name)
bundle_hash=sha(bundle)

cells=[]
def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
def code(text):
    ast.parse(text)
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(True)})

md('''# Reliability-weighted geometry: component validation

This notebook reruns the frozen **192-case synthetic component experiment**, not model training. CPU is sufficient. It includes heldout generator seeds and stable-but-wrong counterexamples. Weighted label error must not be described as improved model accuracy.

Upload this notebook to Colab using **File > Upload notebook**, then run the cells and select the accompanying `reliability_v8_colab_bundle.zip` when requested. The bundle contains code, protocol and scene metadata; no raw RGB-D images, checkpoints or credentials. Local source/bundle checks were performed; this notebook has not been executed on a Google-managed runtime.

The real 128-scene audit and GPU integration check require the separate local V7 assets and are not run here. Colab resources are variable; see the [official FAQ](https://research.google.com/colaboratory/faq.html).
''')
code('''import sys, subprocess, importlib.util, importlib.metadata, platform
for module, package in [('numpy','numpy'),('scipy','scipy'),('matplotlib','matplotlib'),('torch','torch')]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
print('Python', platform.python_version())
for package in ['numpy','scipy','matplotlib','torch']:
    print(package, importlib.metadata.version(package))
''')
code(f'''from google.colab import files
from pathlib import Path
import io, zipfile, hashlib, os, json
uploaded = files.upload()
expected = '{bundle.name}'
if expected not in uploaded:
    raise ValueError('Select the accompanying ' + expected)
payload = uploaded[expected]
assert hashlib.sha256(payload).hexdigest() == '{bundle_hash}', 'Bundle differs from delivered source'
project = Path('/content/elec4240_reliability/workspace/repo')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        target = (project / member.filename).resolve()
        assert target.is_relative_to(project.resolve()), 'Unsafe archive path'
    archive.extractall(project)
os.chdir(project)
sys.path.insert(0, str(project))
from reliability_v8.study import verify_protocol
print('Frozen source verified:', verify_protocol()['version'])
''')
code('''subprocess.run([sys.executable, '-m', 'reliability_v8.test_core'], check=True)
subprocess.run([sys.executable, '-m', 'reliability_v8.study', 'synthetic'], check=True)
''')
code('''result_dir = project / 'results/reliability_v8'
data = json.loads((result_dir / 'synthetic.json').read_text())
held = [r for r in data['records'] if r['split'] == 'heldout']
print('Cases:', len(data['records']), '| Heldout-generator cases:', len(held))
for condition in ['clean','iid_noise','heterogeneous_noise','outliers','holes','smooth_bias']:
    rows = [r['metrics']['all'] for r in held if r['corruption'] == condition]
    means = {k: sum(r[k] for r in rows)/len(rows) for k in
             ['uniform_error_deg','weighted_error_deg','shuffled_error_deg']}
    print(condition, means)
print('These are supervision-selection errors, not trained-model accuracy. No real-data gate is evaluated here.')
from IPython.display import display, Image
display(Image(filename=str(result_dir / 'figures/synthetic_examples.png')))
environment = {'python':platform.python_version(),
               'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','torch']}}
(result_dir / 'colab_environment.json').write_text(json.dumps(environment, indent=2))
''')
code('''archive_path = Path('/content/reliability_v8_colab_results.zip')
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in result_dir.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(result_dir))
files.download(str(archive_path))
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'reliability_v8_component.ipynb'}},'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(cells):cell['id']=f'cell-{i:02d}'
nb=DEST/'reliability_v8_component.ipynb';nb.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')

result=read(OUT/'analysis.json');gpu=read(OUT/'gpu_check.json')
rows=''.join(f'<tr><td>{html.escape(name.replace("_"," "))}</td><td>{v["uniform_error_deg"]:.4f}</td><td>{v["weighted_error_deg"]:.4f}</td><td>{v["shuffled_error_deg"]:.4f}</td></tr>' for name,v in result['heldout_macro_by_corruption'].items())
gaterows=''.join(f'<tr><td>{html.escape(k.replace("_"," "))}</td><td>{v["value"]:.4f}</td><td>{html.escape(v["required"])}</td><td>{"PASS" if v["pass"] else "FAIL"}</td></tr>' for k,v in result['checks'].items())
syn=read(OUT/'synthetic.json');real=read(OUT/'training_audit.json')
boundary={}
for kind in ['crease','step']:
    for condition in ['clean','smooth_bias']:
        selected=[r['metrics']['boundary'] for r in syn['records'] if r['split']=='heldout' and r['surface']==kind and r['corruption']==condition and r['metrics']['boundary']]
        boundary[kind+'/'+condition]={k:sum(r[k] for r in selected)/len(selected) for k in ['uniform_error_deg','weighted_error_deg','mean_weight']}
boundary_rows=''.join(f'<tr><td>{k}</td><td>{v["uniform_error_deg"]:.4f}</td><td>{v["weighted_error_deg"]:.4f}</td><td>{v["mean_weight"]:.4f}</td></tr>' for k,v in boundary.items())
report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>V8: reliability proxy component study</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#233747;background:#f3f7fa;margin:0}}main{{max-width:1050px;margin:30px auto;padding:30px 42px;background:white;border-radius:12px}}h1,h2{{color:#16324f;line-height:1.2}}h2{{margin-top:32px}}a{{color:#087f8c}}.box{{padding:16px;background:#edf5f7;border-left:4px solid #087f8c}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d5e2e7;text-align:left}}th{{background:#16324f;color:white}}img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}small{{color:#566975}}@media(max-width:650px){{main{{margin:0;padding:18px}}table{{font-size:12px}}}}
</style><main>
<small>ELEC4240 | 2 October 2026 | Component study, not a model-training result</small>
<h1>Does construction-scale sensitivity identify unreliable normal supervision?</h1>
<div class="box"><b>Outcome:</b> the frozen engineering screen passed. In the predeclared noisy synthetic conditions, weighting reduced selected-supervision error by {100*result['checks']['noise_relative_reduction']['value']:.2f}% relative to uniform weights. It did not meaningfully detect smooth systematic bias. No adapted-model accuracy improvement has been demonstrated.</div>
<p>The rule computes normals from the same depth at sigma 1, 2 and 4 pixels, averages their pairwise angular differences, and assigns <code>w = exp(-disagreement_deg / 10)</code>. Only the extra geometric penalty receives these weights. Existing supervised losses and evaluation masks remain unchanged.</p>
<h2>Completed work and evidence</h2><ul>
<li>192 analytic synthetic cases, including 96 cases from four heldout generator seeds. Generator families are shared; this is not independent real-world validation.</li>
<li>All 128 frozen V7 training scenes audited. The sigma-1 normal targets were reproduced exactly and native geometry masks were preserved.</li>
<li>Six unit tests passed. Three training images and four conditions produced twelve finite forward/backward checks, with zero optimizer updates.</li>
<li>Uniform-weight replacement loss differs from the frozen V7 objective by {gpu['uniform_vs_v7_loss_abs_difference']:.8g}. Peak allocated GPU memory in the integration checks: {max(r['peak_allocated_mib'] for r in gpu['records']):.1f} MiB. This is not a full optimizer-memory benchmark.</li>
<li>{result['historical_files_unchanged']} historical report/result files and all pinned V7 sources remain unchanged.</li></ul>
<h2>Heldout synthetic label-selection error</h2>
<p>All values are macro-averaged normal errors in degrees over the same retained support. Lower is better. The predicted labels themselves are not corrected: the weighting changes which labels contribute most strongly to a prospective loss. Shuffling preserves the weight distribution.</p>
<table><tr><th>Condition</th><th>Uniform</th><th>Proposed weighting</th><th>Shuffled weights</th></tr>{rows}</table>
<p>Noise, heterogeneous noise and outliers show useful discrimination under the synthetic assumptions. Filled-hole residual error is already near zero after the original mask; its small numerical reduction is not a substantial practical gain. Smooth bias retains about 5.36 degrees of error despite almost unit weights: a clear failure case.</p>
<img src="figures/proxy_comparison.png" alt="Uniform, proposed and shuffled errors across six synthetic conditions">
<h2>Predeclared engineering gate</h2><table><tr><th>Check</th><th>Observed</th><th>Threshold</th><th>Outcome</th></tr>{gaterows}</table>
<p>These are screening criteria, not p-values or Holm-adjusted claims. Pixels were not treated as independent trials. Passing permits further integration and baseline work; it does not establish a causal explanation of V7 negative transfer.</p>
<h2>Boundary and failure inspection</h2><p>The following boundary summaries use the already declared boundary band and report clean or smoothly biased cases. They are descriptive and add no hypothesis tests.</p>
<table><tr><th>Surface / condition</th><th>Uniform error</th><th>Weighted error</th><th>Mean weight</th></tr>{boundary_rows}</table>
<img src="figures/synthetic_examples.png" alt="Noise, a clean crease and smooth bias show the limits of the sensitivity proxy">
<h2>Real training-scene coverage</h2>
<p>Mean native weight: {result['real_native_mean_weight']:.4f}. Median effective pixel fraction: {result['real_native_median_effective_fraction']:.4f}. Effective fraction measures concentration of weights, not the fraction of pixels deleted; the original validity support is unchanged. Real-world normal correctness is not known independently.</p>
<img src="figures/real_weight_coverage.png" alt="Weight concentration and high-depth-gradient coverage across training scenes">
<h2>Next experiment</h2><p>First inspect single-task learning curves. Then freeze the seven-condition, three-seed model comparison: depth-only, normal-only, joint-only, uniform geometry, weaker uniform geometry, proposed weighting and shuffled weighting. Report metric depth without ground-truth alignment, normal accuracy, task tradeoffs and compute. The six primary development contrasts compare weighted versus uniform/weaker/shuffled for both task metrics, with Holm adjustment. No full ablation or new independent cohort has been run.</p>
<p><a href="../../reliability_v8/EXPERIMENT_PLAN.md">Complete experiment plan</a> | <a href="component_protocol.json">Frozen component protocol</a> | <a href="analysis.json">Analysis</a> | <a href="gpu_check.json">GPU checks</a> | <a href="../../reliability_v8/colab/reliability_v8_component.ipynb">Colab notebook</a> | <a href="../../reliability_v8/colab/reliability_v8_colab_bundle.zip">Portable source bundle</a></p>
<p>The notebook reruns the synthetic component only and has not been run in a Google-managed runtime. It needs neither an account token in code nor raw project image uploads. Follow the <a href="https://research.google.com/colaboratory/faq.html">official Colab upload instructions</a>.</p>
<h2>Claim boundary and related work</h2><p><a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03265.pdf">GeoWizard</a> already studies joint diffusion-based depth and normal estimation; <a href="https://papers.miccai.org/miccai-2025/paper/1214_paper.pdf">Li et al., MICCAI 2025</a> study distance-based uncertainty in geometric consistency. A future novelty claim must compare the proposed construction-scale proxy against relevant reliability rules. Current evidence supports only a bounded component feasibility result.</p>
<small>Protocol SHA-256: <code>{result['protocol_sha256']}</code><br>Raw weights remain local, outside Git. Existing final-report PDF unchanged.</small></main></html>'''
(OUT/'RESULTS.html').write_text(report,encoding='utf-8')
checks={'notebook_code_cells_compile':True,'bundle_bytes':bundle.stat().st_size,'bundle_sha256':bundle_hash,'bundle_members':members,
        'notebook_sha256':sha(nb),'report_sha256':sha(OUT/'RESULTS.html'),
        'local_environment':{'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','torch']}},
        'colab_execution':'not executed; local frozen-source unit and component checks completed',
        'boundary_diagnostics':boundary}
(OUT/'delivery_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'bundle':str(bundle),'bytes':bundle.stat().st_size,'notebook':str(nb),'report':str(OUT/'RESULTS.html')},indent=2))
