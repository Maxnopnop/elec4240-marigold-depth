# V14: initialization under matched LoRA adaptation

User-authorized successor to the paused V13 loss study. Cloud-only free T4;
old V13 PAUSE and all original results are retained. No local GPU or paid compute.

## Registered candidate design, frozen only after production preflight

- A: random UNet; B: community SD2 768 v-prediction UNet; C: Marigold depth UNet.
- Shared Marigold VAE, text encoder, tokenizer, zero-terminal-SNR one-step scheduler.
- B input convolution expanded from 4 to 8 channels by repeating weights / 2.
- Identical architecture and fresh rank-4 alpha-4 LoRA (829,952 trainable parameters).
- 512/2048 nested images; seeds 17/29/43; 2048 updates; checkpoints 1024/2048.
- Same latent MSE + balanced pixel BCE/Dice, two referring expressions per update.
- 18 runs, then 36 evaluations on fixed 320-image/640-expression holdout.
- Seven contrasts on two metrics: B-A and C-B at both sizes, plus large-small
  for all three initializers. Holm family = 14. Intermediate milestone descriptive.
- CPU tests exercise exposure, budget recovery, corruption barrier and complete
  report prediction recomputation. Cloud preflight checks exact fresh-process
  step16->17 recovery, shared cache encoders, identical initial LoRA and shapes.
- Budget includes prior V13 cumulative ledger; total cap remains 50h. Training
  estimate has 20% margin and overhead floor8h. No automatic matrix reduction.

## Scope

A retains pretrained encoders and is not an entirely from-scratch model. A frozen
random backbone with LoRA is not a fully optimized random baseline. B is an openly
available community SD2 mirror pinned by revision and tensor-file hashes; original
StabilityAI bytes are currently unavailable for independent identity confirmation.
C versus B includes depth adaptation and recipe changes, not only a pure objective
intervention. No full-parameter training, conventional segmentation baseline,
pretraining-compute accounting, convergence or cross-domain claim in this phase.
Report per-seed effects; image-cluster intervals condition on three tested seeds.

`build_sources.py` records the initial derivation from V13. Generated files have
subsequent reviewed changes; do not rerun the generator over frozen files.
Use `run.py` only in `/content/v14_pretraining_lora` after mounting project Drive,
installing pinned dependencies and extracting original migration inputs. Inspect
process identity before any resume. Checkpoints preserve AdamW, CPU/CUDA RNG and
protocol hashes; the budget journal retains interrupted reservations.
