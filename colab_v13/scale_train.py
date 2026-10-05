"""Matched finite 12-run training with durable optimizer and RNG recovery."""
from scale_runtime import *


def train(arm, size, seed, rr, cache, budget):
    name = label(arm,size,seed)
    dest = WORK/'runs'/name
    ph = sha(WORK/'protocol.json')
    if (dest/'trained.json').exists():
        receipt = read(dest/'trained.json')
        assert receipt['protocol_sha256'] == ph and receipt['updates'] == STEPS
        for step,digest in receipt['checkpoints'].items(): assert sha(dest/f'adapter_{step}.pt') == digest
        return
    budget.reserve(600, name+':setup')
    gg = groups(rr)
    plan = read(ROOT/'scale_plan.json')
    order = schedule(plan['small_train_images' if size == 2048 else 'large_train_images'])
    pipe,params = setup(seed, ROOT/f'initial_seed{seed}.pt')
    pipe.unet.train()
    opt = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=.01)
    step0, history = 0, []
    if (dest/'latest.json').exists():
        pointer = read(dest/'latest.json')
        assert pointer['protocol_sha256'] == ph and sha(dest/pointer['file']) == pointer['sha256']
        step0,history = restore(dest/pointer['file'],params,opt,ph)
        assert step0 == pointer['step']
    assert [r['image_id'] for r in history] == order[:step0]
    for milestone in [2048,4096]:
        if step0 >= milestone: assert (dest/f'adapter_{milestone}.pt').exists()
    torch.cuda.reset_peak_memory_stats()
    budget.settle()
    for first in range(step0+1,STEPS+1,128):
        budget.reserve(1200, name+':training')
        for step in range(first, min(first+128,STEPS+1)):
            pair = [g[(step+g[0]['ann_id'])%len(g)] for g in gg[order[step-1]]]
            row = update(pipe,params,opt,cache,pair,arm,seed,step)
            history.append(row)
            if step % 16 == 0:
                write(WORK/'status.json', {'stage':'training','run':name,'step':step,'steps_per_run':STEPS,'budget_charged_hours':budget.state['charged_seconds']/3600})
            if step in [2048,4096]:
                temp = ROOT/f'adapter_{step}.pt'
                torch.save({k:p.detach().cpu().clone() for k,p in params.items()}, temp)
                copy_verified(temp,dest/f'adapter_{step}.pt')
        save_state(dest,params,opt,step,history,ph)
        write(dest/'history.json',history)
        budget.settle()
        print(name,step,row['loss'],flush=True)
    assert len(history) == STEPS and len(set(r['image_id'] for r in history)) == size
    assert len(set(r['image_id'] for r in history[:2048])) == 2048
    write(dest/'trained.json', {'status':'trained','updates':STEPS,'unique_images':size,
          'optimizer_seconds':sum(r['seconds'] for r in history),'peak_mib':torch.cuda.max_memory_allocated()/2**20,
          'protocol_sha256':ph,'checkpoints':{str(s):sha(dest/f'adapter_{s}.pt') for s in [2048,4096]}})
    del pipe,params,opt
    gc.collect()
    torch.cuda.empty_cache()
