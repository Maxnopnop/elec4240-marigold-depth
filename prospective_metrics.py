"""External RGB-D protocol: full-image measured depths, no NYUv2 crop."""
import numpy as np

def valid_mask(gt):
    return np.isfinite(gt)&(gt>.1)&(gt<10.)

def metrics(pred,gt,scale=1.,shift=0.):
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    assert pred.shape==gt.shape and np.isfinite(pred).all()
    mask=valid_mask(gt);assert mask.sum()>=100
    y=gt[mask];z=np.clip(pred[mask]*scale+shift,.1,10.)
    return {'abs_rel':float(np.mean(abs(z-y)/y)), 'rmse_m':float(np.sqrt(np.mean((z-y)**2))),
            'delta1':float(np.mean(np.maximum(z/y,y/z)<1.25)), 'valid_pixels':int(mask.sum())}

def aligned_metrics(pred,gt):
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    mask=valid_mask(gt);x=pred[mask];y=gt[mask]
    scale=float(np.mean((x-x.mean())*(y-y.mean()))/np.var(x)) if np.var(x)>1e-12 else 0.
    shift=float(y.mean()-scale*x.mean())
    return {**metrics(pred,gt,scale,shift),'scale':scale,'shift':shift}
