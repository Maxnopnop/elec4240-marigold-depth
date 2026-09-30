# Sources and attribution

- Ke et al., **Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation**, CVPR 2024. https://arxiv.org/abs/2312.02145
- Ke et al., **Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis**, 2025. https://arxiv.org/abs/2505.09358
- Official Marigold repository: https://github.com/prs-eth/Marigold , inspected commit `2bfbdeae5d10a50b71f1ba20c865d46e480f6010`.
- Official model: https://huggingface.co/prs-eth/marigold-depth-v1-1 , pinned revision `9571e7123e258cf052b4e54241f17971c290e9a8`. Model terms: CreativeML Open RAIL++-M. The model weights are not redistributed here.
- Diffusers Marigold pipeline: https://huggingface.co/docs/diffusers/main/en/using-diffusers/marigold_usage . This repository uses the actual pipeline from diffusers 0.35.2.
- Silberman et al., **Indoor Segmentation and Support Inference from RGBD Images**, ECCV 2012. NYU Depth V2: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html . RGB/depth arrays are extracted from the official labeled MAT file by HTTP range reads, without using third-party relabeled depths.
- Conventional NYUv2 train/test split mirror: https://github.com/cleinc/bts/blob/master/utils/splits.mat . SHA256: `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`.
- Yang et al., **Depth Anything V2**, 2024. https://arxiv.org/abs/2406.09414 ; official repository https://github.com/DepthAnything/Depth-Anything-V2 ; metric indoor small model https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf , pinned revision `8078d68a9c75a972131914f6afd0c1723be0da7f`.
- Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, https://arxiv.org/abs/2106.09685 . Adapter implementation uses Hugging Face PEFT 0.17.1.
- Gabeur et al., **Image Generators are Generalist Vision Learners**, 2026. https://arxiv.org/abs/2604.20329v1 . Inspiration only; this project does not reproduce or run Vision Banana.

## Implementation attribution

`experiment.py` reimplements the latent denoising training formulation documented in the Apache-2.0 Marigold `src/trainer/marigold_depth_trainer.py`: frozen VAE and text encoder, concatenated RGB/depth latents, DDPM noise, velocity target, and a masked latent MSE. Depth normalization follows the 2%/98% quantiles described in `src/util/depth_transform.py`. The evaluation crop follows `src/dataset/nyu_dataset.py`.

This is a deliberately smaller adaptation experiment, not the complete original training recipe. It uses a depth-specialized checkpoint, cached latents at 256x192, Gaussian noise, an empty text prompt, fixed learning rate, no image augmentation, and only a short sequence of optimization steps. It omits the original large synthetic-data mixture, multi-resolution noise, large training resolution, full UNet updates, and full training schedule. These differences must be retained in any report.

Third-party package and model licenses remain applicable. No upstream source tree, training dataset, model weights, access tokens or local environment is included in the GitHub upload.
