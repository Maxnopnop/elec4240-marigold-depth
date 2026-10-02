"""Frozen staged experiment runner; exact per-task exposure across methods."""
import argparse
import ctypes
import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
from .common import ROOT,WORK,ASSETS,OUT,read,write,sha,sample,depth_metrics,normal_metrics
from .engine import setup,build_cache,train,evaluate,predict


def verify_protocol():
    p=read(OUT/'protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'manifest.json')==p['manifest_sha256']
    assert sha(OUT/'camera.json')==p['camera_sha256']
    assert read(OUT/'smoke_verification.json')['status']=='passed'
    return p


def baselines(manifest):
    path=OUT/'baselines.json'
    if path.exists():return
    means=[]
    for row in manifest['training']:
        a=sample(ASSETS/'subset'/row['file']);b=sample(WORK/'derived'/row['file'])
        means.append(float(a['depth'][b['depth_valid']].mean()))
    constant=float(np.mean(means));records=[]
    proc=AutoImageProcessor.from_pretrained(str(ASSETS/'expert'),local_files_only=True,use_fast=False)
    model=AutoModelForDepthEstimation.from_pretrained(str(ASSETS/'expert'),local_files_only=True).to('cuda').eval()
    predroot=WORK/'predictions'/'expert';predroot.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for row in manifest['validation']:
            a=sample(ASSETS/'subset'/row['file']);b=sample(WORK/'derived'/row['file'])
            inputs=proc(images=a['image'],return_tensors='pt').to('cuda')
            pred=model(**inputs).predicted_depth
            pred=F.interpolate(pred[:,None],size=a['depth'].shape,mode='bicubic',align_corners=False)[0,0].cpu().numpy()
            dest=predroot/f"{row['id']:04d}_depth.npy";np.save(dest,pred)
            front=np.zeros((3,*a['depth'].shape),np.float32);front[2]=-1
            records.append({'id':row['id'],'scene':row['scene'],
                            'constant_depth':depth_metrics(np.full_like(a['depth'],constant),a['depth'],b['depth_valid']),
                            'constant_normal':normal_metrics(front,b['normal'],b['normal_valid']),
                            'expert_native_metric':depth_metrics(pred,a['depth'],b['depth_valid']),
                            'expert_input_shape':list(inputs.pixel_values.shape[-2:]),'prediction_sha256':sha(dest)})
    write(path,{'constant_depth_m':constant,'calibration':'training-only equal-image mean depth',
                'expert_note':'Pinned Depth Anything V2 Metric Indoor Small; Hypersim fine-tuning; official default processor; different prior supervision and input budget; no GT alignment',
                'records':records})
    del model,proc;gc.collect();torch.cuda.empty_cache()


def one_run(mode,seed,protocol,manifest):
    name=f'{mode}_seed{seed}';dest=OUT/'runs'/name;local=WORK/'checkpoints'/name
    if (dest/'complete.json').exists():
        complete=read(dest/'complete.json')
        assert complete['protocol_sha256']==sha(OUT/'protocol.json')
        assert sha(local/'adapter.pt')==complete['checkpoint_sha256']
        assert sha(dest/'validation.json')==complete['metrics_sha256']
        print('V7_SKIP_COMPLETE',name,flush=True);return
    write(OUT/'status.json',{'state':'running','run':name})
    print('V7_START',name,flush=True)
    pipe,params=setup(seed)
    cache_path=WORK/'train_cache.pt';cache_meta=WORK/'train_cache.json'
    fingerprint={'manifest_sha256':sha(OUT/'manifest.json'),'engine_sha256':sha(ROOT/'multitask_v7/engine.py'),
                 'common_sha256':sha(ROOT/'multitask_v7/common.py')}
    if cache_path.exists():
        assert read(cache_meta)['fingerprint']==fingerprint
        assert read(cache_meta)['sha256']==sha(cache_path)
        cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    else:
        start=time.perf_counter();cache=build_cache(pipe,manifest['training'])
        torch.save(cache,cache_path);write(cache_meta,{'fingerprint':fingerprint,'sha256':sha(cache_path),
                                                   'cache_seconds':time.perf_counter()-start})
    stats=train(pipe,params,cache,mode,seed,protocol['steps'],local)
    stats['protocol_sha256']=sha(OUT/'protocol.json')
    write(dest/'training.json',stats)
    records=evaluate(pipe,manifest['validation'],stats['tasks'],WORK/'predictions'/name)
    write(dest/'validation.json',records)
    first=manifest['validation'][0];image=sample(ASSETS/'subset'/first['file'])['image']
    del pipe,params,cache;gc.collect();torch.cuda.empty_cache()
    restored,restored_params=setup(seed,local/'adapter.pt')
    predictions,_=predict(restored,image,stats['tasks'],73000+first['id'])
    for task,pred in predictions.items():
        saved=np.load(WORK/'predictions'/name/f"{first['id']:04d}_{task}.npy")
        assert np.array_equal(saved,pred),('Checkpoint prediction mismatch',name,task,float(abs(saved-pred).max()))
    write(dest/'complete.json',{'protocol_sha256':sha(OUT/'protocol.json'),
          'checkpoint_sha256':sha(local/'adapter.pt'),'metrics_sha256':sha(dest/'validation.json'),
          'checkpoint_restored_exact_tasks':stats['tasks'],'validation_images':len(records)})
    del restored,restored_params;gc.collect();torch.cuda.empty_cache()
    print('V7_COMPLETE',name,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['all','depth','normal','joint','joint_geometry'],default='all');args=a.parse_args()
    protocol=verify_protocol();manifest=read(OUT/'manifest.json')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        baselines(manifest)
        for mode in protocol['modes']:
            if args.stage not in ['all',mode]:continue
            for seed in protocol['seeds']:one_run(mode,seed,protocol,manifest)
        write(OUT/'status.json',{'state':'stage_finished','stage':args.stage})
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
