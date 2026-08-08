"""
MRI-only dataset for VQGAN training (Cor2Vox MRI->PET extension, task 2).
Symmetric to dataset_pet_vqgan.py. The trainer uses batch[0][0] as the
reconstruction target only.

CRITICAL: preprocessed MRI is ALREADY [-1,1] (per-volume min-max). Do NOT
re-normalize; return as-is (avoids double-normalization).

Folder: <img_folder>Tr / <img_folder>Val / <img_folder>Ts
  e.g. /data/yunmin0111/dataset_pet_c2v/mriTr
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from Register import Registers


@Registers.datasets.register_with_name('mri_vqgan')
class MriVqganDataset(Dataset):
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
        print(f"[MriVqganDataset] stage={stage}: {len(self.files)} volumes from {self.folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        scan_id = os.path.basename(path).replace('.nii.gz', '')
        vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)  # already [-1,1]
        t = torch.from_numpy(vol).float().unsqueeze(0)
        return (t, 'img'), (t, 'sdf'), scan_id
