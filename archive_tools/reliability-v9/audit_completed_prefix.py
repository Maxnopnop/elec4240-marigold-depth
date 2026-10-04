import sys,json
from pathlib import Path
import numpy as np
ROOT=Path('E:/Codex/2026-09-27/yo/outputs/marigold-depth');sys.path.insert(0,str(ROOT))
from training_v9.analyze import depth_metrics,normal_metrics
from multitask_v7.common import read,sha,sample,ASSETS,WORK as V7WORK,OUT as V7OUT
from training_v9.protocol import OUT,WORK,verify
verify('b')
manifest=read(V7OUT/'manifest.json');rows={r['id']:r for r in manifest['validation']}
count=0;means={}
for step in [320,640,1280]:
    records=read(OUT/'runs/depth_seed17'/f'validation_{step}.json')
    means[str(step)]=float(np.mean([r['tasks']['depth']['abs_rel'] for r in records]))
    for rec in records:
        row=rows[rec['id']];original=sample(ASSETS/'subset'/row['file']);derived=sample(V7WORK/'derived'/row['file'])
        path=WORK/'predictions/depth_seed17'/f'step{step}'/f"{rec['id']:04d}_depth.npy"
        assert sha(path)==rec['tasks']['depth']['prediction_sha256']
        metrics=depth_metrics(np.load(path)[0],original['depth'],derived['depth_valid'])
        for k,v in metrics.items():assert np.isclose(v,rec['tasks']['depth'][k],rtol=1e-10,atol=1e-9)
        count+=1
# Validate the new independent normal implementation on retained old arrays,
# without running new inference or treating old metrics as current controls.
old=read(V7OUT/'runs/normal_seed17/validation.json')
for rec in old:
    row=rows[rec['id']];derived=sample(V7WORK/'derived'/row['file'])
    pred=np.load(V7WORK/'predictions/normal_seed17'/f"{rec['id']:04d}_normal.npy")
    metrics=normal_metrics(pred,derived['normal'],derived['normal_valid'])
    for k,v in metrics.items():assert np.isclose(v,rec['tasks']['normal'][k],rtol=1e-10,atol=1e-9)
result={'new_depth_predictions_checked':count,'historical_normal_metric_crosschecks':len(old),'first_seed_learning_curve':means}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
