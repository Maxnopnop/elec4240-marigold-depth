"""Training-only implementation/profile gate before fixing the experiment matrix."""
import ctypes
import gc
import numpy as np
import torch
from .common import ASSETS,WORK,OUT,read,write,sample
from .engine import setup,build_cache,paired_loss,train,evaluate


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    manifest=read(OUT/'manifest.json')
    rows=sorted(manifest['training'],key=lambda r:r['id'])[:8]
    records={}
    try:
        for task in ['depth','normal']:
            pipe,params=setup(17)
            cache=build_cache(pipe,rows)
            before=evaluate(pipe,rows,[task],WORK/'smoke'/task/'before')
            result=train(pipe,params,cache,task,17,120,WORK/'smoke'/task)
            after=evaluate(pipe,rows,[task],WORK/'smoke'/task/'after')
            key='abs_rel' if task=='depth' else 'mean_deg'
            initial=float(np.mean([r['tasks'][task][key] for r in before]))
            final=float(np.mean([r['tasks'][task][key] for r in after]))
            first=float(np.mean([r['loss'] for r in result['history'][:10]]))
            last=float(np.mean([r['loss'] for r in result['history'][-10:]]))
            records[task]={'before_training_error':initial,'after_training_error':final,
                           'metric':key,'first10_loss':first,'last10_loss':last,
                           'peak_allocated_mib':result['peak_allocated_mib'],
                           'training_seconds':result['training_seconds']}
            write(OUT/f'smoke_{task}.json',{'training':result,'before':before,'after':after})
            assert last<first and final<initial, ('Training-only overfit gate failed',task,records[task])
            del pipe,params,cache;gc.collect();torch.cuda.empty_cache()
        pipe,params=setup(17);cache=build_cache(pipe,rows[:1]);pipe.unet.train()
        torch.cuda.reset_peak_memory_stats()
        loss,parts,geo=paired_loss(pipe,cache[0],['depth','normal'],17,.1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(geo)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        peak=torch.cuda.max_memory_allocated()/2**20
        assert peak<7500,('Geometry exceeds bounded memory target',peak)
        write(OUT/'smoke_verification.json',{'status':'passed','training_ids':[r['id'] for r in rows],
              'overfit':records,'geometry_profile':{'loss':float(loss.detach()),'geometry_loss':float(geo.detach()),
              'peak_allocated_mib':peak,'all_adapter_gradients_finite':True},
              'scope':'implementation-only training subset; no validation or test outcomes consulted'})
        print('V7_SMOKE_PASSED',records,'GEOMETRY_MIB',peak,flush=True)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
