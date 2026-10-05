"""Common architecture, shared encoders, new matched LoRA; no old adapters."""
from runtime import *
import hashlib

def tensors_hash(params):
    h=hashlib.sha256()
    for n,p in sorted(params.items()):
        h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def make_model(arm,seed,checkpoint=None):
    from frozen_functions import load_pipe, seed_all
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file
    from peft import LoraConfig
    paths=read(ROOT/'model_paths.json')
    seed_all(seed)
    pipe=load_pipe(paths['C'])
    config=dict(pipe.unet.config)
    shapes={n:tuple(t.shape) for n,t in pipe.unet.state_dict().items()}
    if arm!='C_depth':
        del pipe.unet;gc.collect();torch.cuda.empty_cache()
        seed_all(seed)
        original_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            unet=UNet2DConditionModel.from_config(config)
        finally:torch.set_default_dtype(original_dtype)
        assert {n:tuple(t.shape) for n,t in unet.state_dict().items()}==shapes
        if arm=='B_generative':
            state=load_file(str(Path(paths['B'])/'unet/diffusion_pytorch_model.fp16.safetensors'))
            assert state['conv_in.weight'].shape[1]==4
            state['conv_in.weight']=state['conv_in.weight'].repeat(1,2,1,1)/2
            assert {n:tuple(t.shape) for n,t in state.items()}==shapes
            unet.load_state_dict(state,strict=True);del state
        else:assert arm=='A_random'
        pipe.unet=unet.to('cuda');del unet
    pipe.unet.requires_grad_(False)
    # Reset after backbone construction, ensuring identical LoRA initialization.
    seed_all(seed)
    pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
                                   target_modules=['to_q','to_k','to_v','to_out.0']))
    params={n:p for n,p in pipe.unet.named_parameters() if p.requires_grad}
    for p in params.values():p.data=p.data.float()
    assert sum(p.numel() for p in params.values())==829952
    assert all('lora_' in n for n in params)
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    pipe.initialization_audit={'arm':arm,'seed':seed,'lora_sha256':tensors_hash(params),
       'trainable_parameters':829952,'architecture_shapes_sha256':hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest(),
       'shared_encoders':'pinned Marigold VAE and text encoder','B_input_expansion':'repeat 4-channel weights twice / 2; original bias retained'}
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert set(saved)==set(params)
        with torch.no_grad():
            for n,p in params.items():p.copy_(saved[n].cuda())
    pipe.unet.enable_gradient_checkpointing();pipe.vae.enable_gradient_checkpointing()
    pipe.scheduler.set_timesteps(1,device='cuda')
    assert pipe.scheduler.timesteps.tolist()==[999] and float(pipe.scheduler.alphas_cumprod[999])==0
    assert pipe.scheduler.config.prediction_type=='v_prediction'
    return pipe,params
