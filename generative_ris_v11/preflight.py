"""Engineering-only probe: no training, no RIS performance claim."""
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from multitask_v7.engine import setup, encode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/generative_ris_v11'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pipe, params = setup(17)
    torch.cuda.reset_peak_memory_stats()
    # Analytic diagnostic inputs, never presented as real referring data.
    rgb = torch.full((1, 3, 192, 256), -0.7, device='cuda')
    rgb[:, :, 48:144, 24:104] = torch.tensor([1., -1., -1.], device='cuda')[None, :, None, None]
    rgb[:, :, 48:144, 152:232] = torch.tensor([-1., -1., 1.], device='cuda')[None, :, None, None]
    mask = torch.full_like(rgb, -1.)
    mask[:, :, 48:144, 24:104] = 1.
    prompts = ['A binary segmentation mask of the red rectangle, white foreground and black background.',
               'A binary segmentation mask of the blue rectangle, white foreground and black background.']
    pipe.unet.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ids = pipe.tokenizer(prompts, padding='max_length', max_length=77,
                             truncation=True, return_tensors='pt').input_ids.to('cuda')
        emb = pipe.text_encoder(ids)[0].detach()
        zrgb, zmask = encode(pipe, rgb).float(), encode(pipe, mask).float()
        reconstructed = pipe.vae.decode(zmask.to(torch.bfloat16) / pipe.vae.config.scaling_factor).sample.float()
        noise = torch.randn(zrgb.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(91001))
        inputs = torch.cat([zrgb, noise], 1)
        red = pipe.unet(inputs, 999, emb[:1]).sample.float()
        blue = pipe.unet(inputs, 999, emb[1:]).sample.float()
    pred = reconstructed.mean(1) > 0
    truth = mask.mean(1) > 0
    iou = ((pred & truth).sum() / (pred | truth).sum()).item()
    pipe.unet.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        velocity = pipe.unet(inputs, 999, emb[:1]).sample.float()
        loss = F.mse_loss(velocity, -zmask)
    loss.backward()
    grads = [p.grad for p in params.values() if p.grad is not None]
    finite = all(torch.isfinite(g).all().item() for g in grads)
    norm = sum(g.float().square().sum().item() for g in grads) ** .5
    torch.cuda.synchronize()
    result = {
        'status': 'engineering_probe_complete', 'optimizer_updates': 0,
        'scope': 'Single analytic RGB/mask pair. Unadapted Marigold with fresh rank-4 LoRA. No object recognition or selector efficacy measured.',
        'gpu': torch.cuda.get_device_name(), 'trainable_parameters': sum(p.numel() for p in params.values()),
        'prompts': prompts, 'embedding_mean_absolute_difference': (emb[0].float()-emb[1].float()).abs().mean().item(),
        'velocity_mean_absolute_difference_same_noise': (red-blue).abs().mean().item(),
        'mask_vae_roundtrip_iou': iou, 'mask_vae_roundtrip_is_not_prediction': True,
        'latent_supervision_loss': loss.item(), 'gradient_norm': norm,
        'gradients_finite': finite, 'parameters_with_grad': len(grads),
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'seconds_including_load': time.perf_counter()-start,
        'ris_baseline_trained': False, 'budget_policy_validated': False,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'multitask_v7/engine.py', ROOT/'multitask_v7/common.py', ROOT/'experiment.py']}
    }
    assert finite and norm > 0 and torch.isfinite(loss).item()
    (OUT/'preflight.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
