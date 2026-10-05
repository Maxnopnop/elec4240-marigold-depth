"""Test the production cache/update/state path, exclusively on training images.

Parent runs 17 updates; a fresh child restores durable step16 and reproduces17.
Inference timing uses the same training pair, never the fresh holdout.
"""
from scale_runtime import *
import argparse
import subprocess
import sys
import hashlib


def tensor_hash(params):
    h = hashlib.sha256()
    for name,p in sorted(params.items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def core(arm, child=False):
    deterministic()
    dest = DRIVE/'scale_preflight'/arm
    ph = sha(ROOT/'scale_runtime.py')
    rr = rows()
    gg = groups(rr)
    order = schedule(read(ROOT/'scale_plan.json')['small_train_images'])
    cache = load_cache()
    pipe,params = setup(17, ROOT/'initial_seed17.pt')
    pipe.unet.train()
    opt = torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01)
    history=[]
    if child:
        pointer=read(dest/'latest.json')
        assert sha(dest/pointer['file']) == pointer['sha256']
        step0,history=restore(dest/pointer['file'],params,opt,ph)
        assert step0==16
    else:step0=0
    for step in range(step0+1,18):
        pair=[g[(step+g[0]['ann_id'])%len(g)] for g in gg[order[step-1]]]
        row=update(pipe,params,opt,cache,pair,arm,17,step)
        history.append(row)
        if step==16:save_state(dest,params,opt,step,history,ph)
    result={'row':{k:v for k,v in row.items() if k!='seconds'},'adapter_sha256':tensor_hash(params)}
    if not child:
        result['warm_step_seconds']=[r['seconds'] for r in history[8:]]
        pipe.unet.eval()
        with torch.inference_mode():
            rgb=cache['rgb'][pair[0]['image_id']].cuda()
            emb=cache['text'][pair[0]['sent_id']].cuda()
            noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(700000+pair[0]['image_id']))
            latent_prediction(pipe,rgb,emb,noise)
            torch.cuda.synchronize();begin=time.monotonic()
            for _ in range(6): latent_prediction(pipe,rgb,emb,noise)
            torch.cuda.synchronize()
        result['inference_seconds_per_expression']=(time.monotonic()-begin)/6
    write(dest/('child.json' if child else 'parent.json'),result)


def main():
    assert sys.platform=='linux' and ROOT.is_dir(), 'Cloud only'
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.arm:
        core(args.arm,args.child)
        return
    dest=DRIVE/'scale_preflight'
    if (dest/'passed.json').exists():
        receipt=read(dest/'passed.json')
        assert receipt['sources']=={n:sha(ROOT/n) for n in receipt['sources']}
        print('Existing matching production preflight passed; not rerunning',flush=True)
        return
    begin=time.monotonic()
    budget=Budget(dest/'engineering_budget.json',0,cap=7200)
    for arm in ARMS:
        for child in [False,True]:
            command=[sys.executable,__file__,'--arm',arm]+(['--child'] if child else [])
            budget.reserve(660,arm+(':restore' if child else ':parent'))
            subprocess.run(command,check=True,timeout=600)
            budget.settle()
        a,b=read(dest/arm/'parent.json'),read(dest/arm/'child.json')
        assert a['row']==b['row'] and a['adapter_sha256']==b['adapter_sha256'],arm
    names=['scale_runtime.py','scale_train.py','scale_evaluate.py','scale_report.py','scale_preflight.py','scale_run.py','scale_queue.py','frozen_functions.py','prepare_scale_cache.py']
    write(dest/'passed.json',{'status':'passed','actual_fresh_process_resume':'durable16 ->17 all three arms, production cache/update/state code',
          'sources':{n:sha(ROOT/n) for n in names},'seconds':budget.state['charged_seconds'],
          'results':{arm:read(dest/arm/'parent.json') for arm in ARMS},'heldout_inference':False})
    print('PRODUCTION PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
