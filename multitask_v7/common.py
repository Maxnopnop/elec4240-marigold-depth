"""Explicit metric encoding, geometry and evaluation without GT alignment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1] / 'work/marigold-local/metric_multitask_v7'
ASSETS = WORK.parent / 'assets'
OUT = ROOT / 'results/metric_multitask_v7'
PROMPTS = {'depth': 'A metric depth map of the indoor scene, encoded as logarithmic depth.',
           'normal': 'A camera-facing surface normal map of the indoor scene, encoded in RGB.'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def encode_depth(d): return (torch.log(d.clamp(.1,10)/.1)/np.log(100)*2-1)
def decode_depth(v): return .1*torch.exp(((v.clamp(-1,1)+1)/2)*np.log(100))


def rays(height, width, camera, device='cpu'):
    """1-based pixel convention, matching NYUv2 toolbox projection indexing.

    Half-pixel resize matches torch align_corners=False. Provided projected depth
    is treated as camera-z in the RGB pinhole approximation; targets are derived,
    not an independent official normal benchmark.
    """
    ys,xs=torch.meshgrid(torch.arange(height,device=device,dtype=torch.float32),
                         torch.arange(width,device=device,dtype=torch.float32),indexing='ij')
    x=(xs+.5)*(640/width)+.5
    y=(ys+.5)*(480/height)+.5
    return torch.stack([(x-camera['cx_rgb'])/camera['fx_rgb'],
                        (y-camera['cy_rgb'])/camera['fy_rgb'],torch.ones_like(x)],0)[None]


def depth_to_normals(depth, ray):
    points=depth*ray
    dx=points[:,:,1:-1,2:]-points[:,:,1:-1,:-2]
    dy=points[:,:,2:,1:-1]-points[:,:,:-2,1:-1]
    n=torch.linalg.cross(dx,dy,dim=1)
    n=F.normalize(n,dim=1,eps=1e-8)
    # Orient toward the optical center, not simply toward negative Z.
    facing=(n*points[:,:,1:-1,1:-1]).sum(1,keepdim=True)
    n=n*torch.where(facing>0,-1.,1.)
    return F.pad(n,(1,1,1,1),mode='replicate')


def decode_normal(v): return F.normalize(v,dim=1,eps=1e-6)


def depth_metrics(pred, gt, mask):
    p=np.asarray(pred,dtype=np.float64)[mask]
    d=np.asarray(gt,dtype=np.float64)[mask]
    assert p.size>100 and np.isfinite(p).all() and np.all(p>0)
    outside=float(np.mean((p<.1)|(p>10)))
    p=np.clip(p,.1,10.)  # Same declared indoor prediction range for every method.
    ratio=np.maximum(p/d,d/p)
    return {'abs_rel':float(np.mean(abs(p-d)/d)),
            'rmse_m':float(np.sqrt(np.mean((p-d)**2))),
            'delta1':float(np.mean(ratio<1.25)), 'valid_pixels':int(p.size),
            'prediction_outside_range_fraction':outside}


def normal_metrics(pred, gt, mask):
    p=pred[:,mask].astype(np.float64);g=gt[:,mask].astype(np.float64)
    assert p.shape[1]>100 and np.isfinite(p).all()
    p=p/np.maximum(np.linalg.norm(p,axis=0,keepdims=True),1e-8)
    g=g/np.maximum(np.linalg.norm(g,axis=0,keepdims=True),1e-8)
    angles=np.degrees(np.arccos(np.clip(np.sum(p*g,axis=0),-1,1)))
    return {'mean_deg':float(angles.mean()),'median_deg':float(np.median(angles)),
            'within_11_25':float(np.mean(angles<11.25)),
            'within_22_5':float(np.mean(angles<22.5)),
            'within_30':float(np.mean(angles<30)), 'valid_pixels':int(mask.sum())}


def sample(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
