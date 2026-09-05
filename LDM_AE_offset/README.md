# LDM Offset-Attention AutoEncoder (MRI -> PET)

Stage 1 of an LDM pipeline: a 2D VQ-VAE that translates MRI slices to PET
slices. The encoder latent will later feed a DDPM/LDM diffusion stage.

## Design
- **2D, slice-by-slice**: each 128^3 volume is processed as 128 axial slices.
- **16x16 patch embedding**: encoder downsamples 128 -> 8 (factor 16), so each
  latent location corresponds to one 16x16 input patch. Attention runs on the
  8x8 = 64 token grid (small map, avoids OOM).
- **PCT offset-attention**: `F_out = LBR(F_in - F_sa) + F_in`, with softmax on
  dim 1 + L1-norm on dim 2 (attention that focuses on high-difference regions).
- **VQ-VAE**: discrete latent (num_embeddings=1024, dim=4).
- **MRI -> PET**: reconstruction target is the paired PET slice; loss is a plain
  L1 (or L2) distance. No perceptual/adversarial terms.

## Sources / credit
- Offset-attention: PCT (Guo et al. 2020, https://arxiv.org/pdf/2012.09688);
  code adapted from https://github.com/qinglew/PointCloudTransformer (module.py).
- VQ-VAE quantizer: van den Oord et al. 2017 (https://arxiv.org/abs/1711.00937),
  taming-transformers pattern, adapted to 2D.
- Conv encoder/decoder: standard VQ-VAE 2D skeleton, written for this task.

## Files
- `models/offset_attention.py` - PCT offset-attention (2D patch tokens)
- `models/vq.py`                - vector quantizer
- `models/vqvae_2d_offset.py`   - encoder + VQ + decoder
- `datasets/dataset_slice_mri2pet.py` - slice-level MRI/PET pairs
- `train_ae_offset.py`          - training (MRI in, PET target, distance loss)
- `configs/ae_offset_2d.yaml`   - config
- `run_train.sh`                - SLURM launcher

## Run on SERAPH
```bash
cd /data/yunmin0111
git clone <YOUR_REPO_URL> LDM-offset-AE
cd LDM-offset-AE
sbatch run_train.sh
```

## Next (later)
- Swap 2D 16x16 patches for 3D 4x4x4 patches once this works.
- Add the DDPM/LDM diffusion stage on top of the trained latent.
