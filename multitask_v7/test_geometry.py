"""Analytic geometry/metric checks; these are not model accuracy tests."""
import torch
import torch.nn.functional as F
from .common import encode_depth,decode_depth,rays,depth_to_normals,normal_metrics,depth_metrics,write,OUT


def main():
    camera={'fx_rgb':518.8579011745,'fy_rgb':519.4696111213,'cx_rgb':325.5824494112,'cy_rgb':253.736166334}
    ray=rays(48,64,camera)
    d=torch.full((1,1,48,64),2.)
    n=depth_to_normals(d,ray)
    assert torch.allclose(n[:,2,2:-2,2:-2],-torch.ones_like(n[:,2,2:-2,2:-2]),atol=1e-5)
    # Analytic tilted plane n.X=constant; derivative normals recover its normal.
    target=F.normalize(torch.tensor([.2,-.3,-1.]),dim=0)
    plane=-2./(ray*target[None,:,None,None]).sum(1,keepdim=True)
    got=depth_to_normals(plane,ray)
    assert (got[:,:,2:-2,2:-2]*target[None,:,None,None]).sum(1).min()>.99999
    assert torch.allclose(got,depth_to_normals(plane*3,ray),atol=1e-4)
    values=torch.logspace(-1,1,1001)
    assert torch.allclose(decode_depth(encode_depth(values)),values,rtol=1e-5,atol=1e-6)
    assert not torch.allclose(encode_depth(torch.tensor([1.,2.])),encode_depth(torch.tensor([2.,4.])))
    mask=torch.ones((48,64),dtype=torch.bool).numpy()
    dm=depth_metrics(d[0,0].numpy(),d[0,0].numpy(),mask)
    nm=normal_metrics(n[0].numpy(),n[0].numpy(),mask)
    assert dm['abs_rel']==0 and dm['delta1']==1 and nm['mean_deg']<1e-4
    train_d=plane.clone().requires_grad_()
    train_n=(got+.05*torch.randn_like(got)).detach().requires_grad_()
    loss=(1-(depth_to_normals(train_d,ray)*F.normalize(train_n,dim=1)).sum(1))[:,2:-2,2:-2].mean()
    loss.backward()
    assert torch.isfinite(train_d.grad).all() and torch.isfinite(train_n.grad).all()
    assert train_d.grad.abs().sum()>0 and train_n.grad.abs().sum()>0
    checks=['fronto-parallel plane orientation','tilted-plane analytic normal',
            'normal invariance to global depth scale','metric codec roundtrip',
            'metric encoding distinguishes different absolute scales','identity metrics',
            'finite nonzero geometry gradients for both tasks']
    write(OUT/'geometry_checks.json',{'status':'passed','checks':checks})
    print('GEOMETRY_CHECKS_PASSED',len(checks))


if __name__=='__main__':main()
