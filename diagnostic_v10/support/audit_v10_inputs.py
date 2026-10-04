import sys
from pathlib import Path
sys.path.insert(0,r'E:\Codex\2026-09-27\yo\outputs\marigold-depth')
import numpy as np
import torch
from diagnostic_v10.common import OUT,WORK,CONDITIONS,read,write,sha,verify
from multitask_v7.common import sample
from reliability_v8.core import angles

torch.set_num_threads(2)
p=verify();data=read(OUT/'noise_data.json')
for name,digest in data['files'].items():assert sha(WORK/name)==digest,name
checked=0
for file in sorted((WORK/'train/clean').glob('*.npz')):
    clean=sample(file)
    for condition in CONDITIONS:
        case=sample(WORK/'train'/condition/file.name)
        for key in ['image','depth','normal','mask']:
            assert np.array_equal(case[key],clean[key]),(file.name,condition,key)
        oracle=np.exp(-angles(case['target_normal'],case['normal'])/10).astype(np.float32)
        assert np.array_equal(oracle,case['oracle'])
        checked+=1
    assert np.array_equal(clean['depth'],clean['observed'])
for condition in CONDITIONS:
    cache=torch.load(WORK/f'cache_{condition}.pt',map_location='cpu',weights_only=True)
    for item in cache:
        mask=item['geo_mask'].numpy()
        assert np.array_equal(np.sort(item['weighted'].numpy()[mask]),np.sort(item['shuffled'].numpy()[mask]))
        assert not item['weighted'].requires_grad and not item['oracle'].requires_grad
quality={c:{k:float(np.mean([r[k] for r in data['label_quality'] if r['condition']==c]))
              for k in ['normal_target_error','weighted_target_error','oracle_target_error']} for c in CONDITIONS}
result=dict(verified_hashed_files=len(data['files']),training_condition_images=checked,
            rgb_truth_and_mask_equal_across_conditions=True,oracle_recomputed=True,shuffled_native_distribution_identical=True,
            label_quality_degrees=quality,scope='Input/label audit only, not trained-model improvement.')
write(OUT/'independent_input_audit.json',result)
print(result)
