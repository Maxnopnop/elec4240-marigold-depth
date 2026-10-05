"""Test the production cache/update/state path, exclusively on training images.

Parent runs 17 updates; a fresh child restores durable step16 and reproduces17.
Inference timing uses the same training pair, never the fresh holdout.
"""
from runtime import *
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
    dest = DRIVE/'preflight'/arm
    ph = sha(ROOT/'runtime.py')
    rr = rows()
    gg = groups(rr)
    order = schedule(read(ROOT/'scale_plan.json')['small_train_images'])
    cache = load_cache()
    pipe,params = setup(17, arm=arm)
    # Directly verify shared encoders against reused cache on training data only.
    if not child:
        r=gg[order[0]][0][0]
        with torch.no_grad():
            a=np.array(Image.open(r['image']).convert('RGB').resize((256,192),Image.Resampling.BILINEAR),copy=True)
            z=encode(pipe,torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/127.5-1).float().cpu()
            assert torch.equal(z,cache['rgb'][r['image_id']]),'RGB cache encoder mismatch'
            a=np.array(Image.open(r['mask']).resize((256,192),Image.Resampling.NEAREST),copy=True)
            z=encode(pipe,(torch.from_numpy(a)[None,None].float().cuda()/127.5-1).repeat(1,3,1,1)).float().cpu()
            assert torch.equal(z,cache['mask'][r['ann_id']]),'Mask cache encoder mismatch'
            tokens=pipe.tokenizer(text_prompt(r['text']),padding='max_length',max_length=77,return_tensors='pt').input_ids.cuda()
            assert torch.equal(pipe.text_encoder(tokens)[0].cpu(),cache['text'][r['sent_id']]),'Text cache mismatch'
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
    result={'initialization_audit':pipe.initialization_audit,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'row':{k:v for k,v in row.items() if k!='seconds'},'adapter_sha256':tensor_hash(params)}
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
    dest=DRIVE/'preflight'
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
            budget.reserve(900,arm+(':restore' if child else ':parent'))
            subprocess.run(command,check=True,timeout=840)
            budget.settle()
        a,b=read(dest/arm/'parent.json'),read(dest/arm/'child.json')
        assert a['row']==b['row'] and a['adapter_sha256']==b['adapter_sha256'],arm
    names=['runtime.py','train.py','evaluate.py','report.py','preflight.py','run.py','model_setup.py','prepare.py','frozen_functions.py','io_utils.py']
    assert len({read(dest/a/'parent.json')['initialization_audit']['lora_sha256'] for a in ARMS})==1
    write(dest/'passed.json',{'status':'passed','actual_fresh_process_resume':'durable16 ->17 all three arms, production cache/update/state code',
          'sources':{n:sha(ROOT/n) for n in names},'seconds':budget.state['charged_seconds'],
          'results':{arm:read(dest/arm/'parent.json') for arm in ARMS},'heldout_inference':False})
    print('PRODUCTION PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
