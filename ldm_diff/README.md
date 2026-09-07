# Conditional Latent DDPM (MRI -> PET)

LDM stage 2. Frozen PET VQGAN provides the target latent; frozen 3D-swin AE
encoder (our best MRI->PET AE) provides the condition. A 3D UNet denoises the
PET latent conditioned on the MRI latent, two ways:

- **concat**: z_mri concatenated to noisy latent (in_channels 8). LDM concat cond.
- **adagn** : z_mri pooled to a vector, injected per ResBlock via AdaGN scale/shift.

## Credit
- Latent DDPM + concat conditioning: CompVis/latent-diffusion.
- UNet (ResBlock + timestep emb) pattern: openaimodel.
- Encoders: PET VQGAN + 3D-swin AE encoder (ours).

## Run
```
git clone <repo> ; cd ldm_diff
sbatch run_ldm_concat.sh
sbatch run_ldm_adagn.sh
```
