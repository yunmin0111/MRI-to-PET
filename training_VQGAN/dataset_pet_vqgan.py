"""
PET-only dataset for VQGAN training (Cor2Vox MRI->PET extension).

The VQGAN trainer (vqgan3d_v2.py) only uses batch[0][0] as the
reconstruction target and ignores everything else. So we return the PET
volume as target ('img') and a dummy for the 'sdf' slot to keep the
tuple shape compatible.

CRITICAL: the preprocessed PET is ALREADY normalized to [-1,1] with a
global fixed SUVR scale. We must NOT re-normalize (no RescaleIntensity,
no (t*2)-1), otherwise per-volume rescaling would destroy the
quantitative fixed-scale normalization (the classic double-normalization
bug). We read the volume and return it as-is.

Folder layout (created from manifest split):
    <img_folder>Tr / <img_folder>Val / <img_folder>Ts
each containing <PET_ImageID>.nii.gz already at 128^3, [-1,1].
"""

import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from Register import Registers


@Registers.datasets.register_with_name('pet_vqgan')
class PetVqganDataset(Dataset):
    def __init__(self, dataset_config, transform=None,
                 target_transform=None, stage='train'):
        super().__init__()
        suffix = {'train': 'Tr', 'val': 'Val', 'test': 'Ts'}
        if stage not in suffix:
            raise NotImplementedError(f"Stage '{stage}' not supported.")

        self.folder = dataset_config.img_folder + suffix[stage]
        self.files = sorted(glob.glob(os.path.join(self.folder, '*.nii.gz')))
        if len(self.files) == 0:
            raise RuntimeError(f"No .nii.gz found in {self.folder}")
        self.input_size = dataset_config.input_size
        self.depth_size = dataset_config.depth_size
        print(f"[PetVqganDataset] stage={stage}: {len(self.files)} volumes from {self.folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        scan_id = os.path.basename(path).replace('.nii.gz', '')

        # already 128^3 and already [-1,1]; DO NOT re-normalize
        vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        t = torch.from_numpy(vol).float().unsqueeze(0)   # (1, D, H, W)

        # target in slot 0 ('img'); dummy in slot 1 ('sdf') for tuple compat
        return (t, 'img'), (t, 'sdf'), scan_id
