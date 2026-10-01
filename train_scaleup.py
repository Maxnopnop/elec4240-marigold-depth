"""LoRA training at fixed/mixed resolutions with optional velocity preservation."""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig
from experiment import seed_all

MODES = ['low256', 'high512', 'mixed', 'mixed_prior']


def train(pipe, caches, mode, seed, outdir, steps=320):
    assert mode in MODES
    seed_all(seed)
    unet = pipe.unet
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian',
                               target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    assert sum(p.numel() for p in params) == 829952
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing='trailing')
    unet.train()
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    order, history = [], []
    count = len(next(iter(caches.values())))
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        res = 256 if mode == 'low256' else 512 if mode == 'high512' else [256, 512][step % 2]
        if not order:
            order = list(rng.permutation(count))
        idx = int(order.pop())
        rgb, depth, valid = [x.to('cuda') for x in caches[res][idx]]
        valid = valid.expand(-1, 4, -1, -1)
        # Reset per-update RNG so image order and timesteps match across resolution modes.
        generator = torch.Generator(device='cuda').manual_seed(seed + 1000 * step)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device='cuda', generator=generator)
        noise = torch.randn(depth.shape, device='cuda', generator=generator)
        noisy = scheduler.add_noise(depth, noise, timestep)
        target = scheduler.get_velocity(depth, noise, timestep)
        model_input = torch.cat([rgb, noisy], dim=1)
        optimizer.zero_grad(set_to_none=True)
        teacher = None
        if mode == 'mixed_prior':
            # The frozen base weights are shared; disable only the adapter for the teacher.
            unet.eval()
            unet.disable_adapters()
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    teacher = unet(model_input, timestep, pipe.empty_text_embedding).sample.detach().float()
            finally:
                unet.enable_adapters()
                unet.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = unet(model_input, timestep, pipe.empty_text_embedding).sample.float()
            supervised = F.mse_loss(pred[valid], target.float()[valid])
            preservation = F.mse_loss(pred[valid], teacher[valid]) if teacher is not None else pred.new_zeros(())
            loss = supervised + .1 * preservation
        assert torch.isfinite(loss), 'Non-finite objective'
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(grad), 'Non-finite gradient'
        optimizer.step()
        history.append({'step': step+1, 'resolution': res, 'sample_index': idx, 'timestep': int(timestep),
                        'loss': float(loss.detach()), 'supervised_loss': float(supervised.detach()),
                        'prior_loss': float(preservation.detach()), 'gradient_norm': float(grad)})
        if (step+1) % 40 == 0 or step+1 == steps:
            print('TRAIN', mode, seed, step+1, '/', steps, f'loss={loss.item():.5f}', flush=True)
    torch.cuda.synchronize()
    stats = {'mode': mode, 'seed': seed, 'steps': steps, 'train_images': count,
             'training_seconds': time.perf_counter()-start,
             'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
             'trainable_parameters': sum(p.numel() for p in params),
             'teacher_weight': .1 if mode == 'mixed_prior' else 0,
             'history': history}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().float().cpu() for n, p in unet.named_parameters() if p.requires_grad}, outdir/'adapter.pt')
    del optimizer
    unet.eval().to(dtype=torch.bfloat16)
    pipe.vae.to('cuda')
    torch.cuda.empty_cache()
    return stats
