import sys
sys.path.insert(0,r'E:\Codex\2026-09-27\yo\outputs\marigold-depth')
import numpy as np
import torch
import torch.nn.functional as F
from diagnostic_v10.common import OUT,WORK,CONDITIONS,read,write,sha,verify
from multitask_v7.common import sample
from reliability_v8.core import angles

torch.set_num_threads(2);verify();data=read(OUT/'noise_data.json')
caches={c:torch.load(WORK/f'cache_{c}.pt',map_location='cpu',weights_only=True) for c in CONDITIONS}
records=[]
for condition in CONDITIONS:
    assert sha(WORK/f'cache_{condition}.pt')==data['files'][f'cache_{condition}.pt']
    for item,clean in zip(caches[condition],caches['clean']):
        assert item['id']==clean['id']
        case=sample(WORK/'train'/condition/f"{item['id']}.npz")
        def normal_resize(n):
            return F.normalize(F.interpolate(torch.from_numpy(n)[None],size=(192,256),mode='area'),dim=1,eps=1e-6)[0].numpy()
        def depth_resize(d):
            return F.interpolate(torch.from_numpy(d)[None,None],size=(192,256),mode='bilinear',align_corners=False,antialias=True)[0,0].numpy()
        native_mask=item['geo_mask'].numpy()[0,0]
        dn=depth_resize(case['observed']);dt=depth_resize(case['depth'])
        records.append(dict(condition=condition,id=item['id'],
            normal_fullres_deg=float(angles(case['target_normal'],case['normal'])[case['mask']].mean()),
            normal_native_deg=float(angles(normal_resize(case['target_normal']),normal_resize(case['normal']))[native_mask].mean()),
            depth_fullres_absrel=float((abs(case['observed']-case['depth'])/case['depth'])[case['mask']].mean()),
            depth_native_absrel=float((abs(dn-dt)/dt)[native_mask].mean()),
            latent_target_mse={task:float(((item[task]-clean[task])**2)[item[task+'_mask'].expand_as(item[task])].mean()) for task in ['depth','normal']}))
keys=['normal_fullres_deg','normal_native_deg','depth_fullres_absrel','depth_native_absrel']
summary={c:{**{k:float(np.mean([r[k] for r in records if r['condition']==c])) for k in keys},
            'latent_target_mse':{t:float(np.mean([r['latent_target_mse'][t] for r in records if r['condition']==c])) for t in ['depth','normal']}} for c in CONDITIONS}
result=dict(records=records,summary=summary,source_sha256=sha(__file__),noise_data_sha256=sha(OUT/'noise_data.json'),
            scope='Post-hoc preprocessing diagnosis; latent MSE compares noisy with clean encoded targets and is not a physical angular or distance error. No new training or tests.')
write(OUT/'representation_audit.json',result);print(summary)
