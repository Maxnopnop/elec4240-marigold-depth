from pathlib import Path
import sys,json,hashlib,zipfile,ast,re
import numpy as np
ROOT=Path('E:/Codex/2026-09-27/yo/outputs/marigold-depth')
sys.path.insert(0,str(ROOT))
from reliability_v8.study import verify_protocol, OUT, WORK, SURFACES, CORRUPTIONS, DEV, HOLD
from multitask_v7.common import read,sha
p=verify_protocol();a=read(OUT/'analysis.json');s=read(OUT/'synthetic.json');r=read(OUT/'training_audit.json');g=read(OUT/'gpu_check.json')
expected={(seed,surf,c) for seed in DEV+HOLD for surf in SURFACES for c in CORRUPTIONS}
actual={(x['seed'],x['surface'],x['corruption']) for x in s['records']}
assert expected==actual and len(s['records'])==192
assert sum(x['split']=='heldout' for x in s['records'])==96
for x in s['records']:
    for value in x['metrics'].values():
        if value:
            for k,v in value.items():
                if isinstance(v,(float,int)):assert np.isfinite(v),(x,k)
    assert x['metrics']['all']['pixels']>100
held=[x for x in s['records'] if x['split']=='heldout']
for c in CORRUPTIONS:
    group=[x['metrics']['all'] for x in held if x['corruption']==c]
    for key,value in a['heldout_macro_by_corruption'][c].items():
        assert abs(value-np.mean([x[key] for x in group]))<1e-12
v7=read(ROOT/'results/metric_multitask_v7/manifest.json')
assert {x['id'] for x in r['records']}=={x['id'] for x in v7['training']}
assert not {x['scene'] for x in r['records']}&{x['scene'] for x in v7['validation']}
rowbyid={x['id']:x for x in v7['training']}
for x in r['records']:
    path=WORK/'weights'/rowbyid[x['id']]['file']
    assert sha(path)==x['weight_sha256']
    with np.load(path) as z:
        assert z['weights'].shape==z['mask'].shape==(1,1,192,256)
        assert np.isfinite(z['weights']).all()
        assert (z['weights'][~z['mask']]==0).all()
        assert (z['weights'][z['mask']]>0).all()
        assert (z['weights']<=1.000001).all()
assert len(g['records'])==12 and g['uniform_vs_v7_loss_abs_difference']==0
assert len({(x['id'],x['mode']) for x in g['records']})==12
assert all(x['gradient_norm']>0 and np.isfinite(x['gradient_norm']) for x in g['records'])
historical=read(WORK/'historical_hashes.json')
assert all(sha(ROOT/k)==v for k,v in historical.items())
checks=read(OUT/'delivery_checks.json')
bundle=ROOT/'reliability_v8/colab/reliability_v8_colab_bundle.zip'
assert sha(bundle)==checks['bundle_sha256']
with zipfile.ZipFile(bundle) as z:
    assert sorted(z.namelist())==checks['bundle_members']
    assert not any(Path(n).suffix in ['.npz','.npy','.pt','.safetensors','.env'] for n in z.namelist())
    for name,value in {**p['source_sha256'],**p['v7_source_sha256']}.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==value
notebook=ROOT/'reliability_v8/colab/reliability_v8_component.ipynb'
assert sha(notebook)==checks['notebook_sha256']
nb=read(notebook)
for cell in nb['cells']:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
text=(OUT/'RESULTS.html').read_text(encoding='utf-8')
for url in re.findall(r'(?:href|src)="([^"]+)"',text):
    if '://' not in url:assert (OUT/url).resolve().is_file(),url
assert sha(OUT/'RESULTS.html')==checks['report_sha256']
verification={'frozen_sources_verified':len(p['source_sha256'])+len(p['v7_source_sha256']),
    'synthetic_case_matrix_verified':192,'heldout_cases':96,'training_caches_verified':128,
    'validation_scene_overlap':0,'gpu_forward_backward_checks':12,'optimizer_updates':0,
    'historical_files_unchanged':len(historical),'portable_source_hashes_verified':True,
    'notebook_code_cells_compile':True,'report_local_links_valid':True,
    'colab_runtime_executed':False,'all_checks_passed':True}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
print(json.dumps(verification,indent=2))
