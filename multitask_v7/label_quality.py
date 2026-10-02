"""Post-hoc training-only label sensitivity prompted by qualitative inspection.

No model inference, optimization, relabeling, validation selection or new tests.
Resolution round trips are target diagnostics, not RGB prediction baselines or
information-theoretic lower bounds. Smoothing sensitivity does not establish
which target is physically correct.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from .common import ASSETS,WORK,OUT,read,write,sha,sample,rays,depth_to_normals,normal_metrics


def main():
    torch.set_num_threads(2)
    manifest=read(OUT/'manifest.json');camera=read(OUT/'camera.json')['intrinsics']
    ray=rays(480,640,camera);records=[]
    with torch.inference_mode():
        for row in manifest['training']:
            a=sample(ASSETS/'subset'/row['file']);b=sample(WORK/'derived'/row['file'])
            assert sha(WORK/'derived'/row['file'])==row['derived_sha256']
            original=torch.from_numpy(b['normal'])[None]
            small=F.normalize(F.interpolate(original,size=(192,256),mode='area'),dim=1,eps=1e-6)
            restored=F.normalize(F.interpolate(small,size=(480,640),mode='bilinear',align_corners=False),dim=1,eps=1e-6)[0].numpy()
            values={'native_resolution_roundtrip_mean_deg':normal_metrics(restored,b['normal'],b['normal_valid'])['mean_deg']}
            for sigma in [2.,4.]:
                depth=gaussian_filter(a['depth'],sigma=sigma,truncate=2.)
                alternate=depth_to_normals(torch.from_numpy(depth)[None,None],ray)[0].numpy()
                values[f'sigma{sigma:g}_vs_frozen_sigma1_mean_deg']=normal_metrics(alternate,b['normal'],b['normal_valid'])['mean_deg']
            records.append({'id':row['id'],**values})
    summary={k:{'mean':float(np.mean([r[k] for r in records])),
                'min':float(min(r[k] for r in records)), 'max':float(max(r[k] for r in records))}
             for k in records[0] if k!='id'}
    result={'scope':'post-hoc training-only descriptive target-quality audit; prompted by visible label noise; no validation data, new significance tests, model selection or target changes',
            'training_images':len(records),'native_resolution':[192,256],
            'mask':'unchanged frozen normal-valid mask',
            'interpretation':'Round-trip errors measure sensitivity to output resolution, not an error lower bound. Smoothing changes do not identify physically correct normals. Obtain better verified normal labels before strong physical-normal accuracy claims.',
            'manifest_sha256':sha(OUT/'manifest.json'),'summary':summary,'records':records}
    write(OUT/'label_quality.json',result)
    print('TRAINING_ONLY_LABEL_QUALITY_AUDIT',summary,flush=True)


if __name__=='__main__':main()
