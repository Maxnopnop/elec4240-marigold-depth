"""Check actual adapter/AdamW/sampler/RNG serialization against uninterrupted updates."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from .run import save_state,restore_random


def main():
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    model=torch.nn.Linear(4,2).cuda();params=dict(model.named_parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(19);order=[];history=[]
    def update(model,optimizer,rng,order,history,count):
        for _ in range(count):
            if not order:order.extend(rng.permutation(5).tolist())
            index=order.pop();x=torch.randn(2,4,device='cuda')+index
            optimizer.zero_grad(set_to_none=True);loss=model(x).square().mean();loss.backward();optimizer.step()
            history.append({'step':len(history)+1,'id':index,'loss':float(loss.detach())})
    update(model,optimizer,rng,order,history,3)
    with tempfile.TemporaryDirectory(prefix='v9_resume_check_') as temp:
        path=Path(temp)/'resume.pt'
        save_state(path,params,optimizer,rng,order,history,1.,2.,'test-protocol')
        update(model,optimizer,rng,order,history,4)
        expected={k:v.detach().clone() for k,v in params.items()};expected_history=list(history)
        resumed=torch.nn.Linear(4,2).cuda();opt2=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        resumed.load_state_dict(saved['adapter']);opt2.load_state_dict(saved['optimizer'])
        rng2=np.random.default_rng();rng2.bit_generator.state=saved['sampler'];order2=saved['order'];history2=saved['history']
        restore_random(saved['random']);update(resumed,opt2,rng2,order2,history2,4)
        assert expected_history==history2
        for name,value in resumed.named_parameters():assert torch.equal(value,expected[name]),name
    print('RESUME_TEST_PASSED: exact parameters, losses, data order and RNG after interruption',flush=True)


if __name__=='__main__':main()
