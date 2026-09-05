"""Slice-level MRI->PET paired dataset (same as offset-attention version)."""
import os, glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class SliceMri2PetDataset(Dataset):
    def __init__(self, root, stage='train', drop_empty=True, empty_thr=1e-3):
        suffix = {'train': 'Tr', 'val': 'Val', 'test': 'Ts'}[stage]
        self.mri_dir = os.path.join(root, f'mri{suffix}')
        self.pet_dir = os.path.join(root, f'pet{suffix}')
        pet_files = sorted(glob.glob(os.path.join(self.pet_dir, '*.nii.gz')))
        self.index = []; self.pids = []
        for pf in pet_files:
            pid = os.path.basename(pf)[:-7]
            if not os.path.exists(os.path.join(self.mri_dir, f'{pid}.nii.gz')):
                continue
            self.pids.append(pid)
            vol = np.asarray(nib.load(pf).get_fdata(), dtype=np.float32)
            for s in range(vol.shape[2]):
                if drop_empty and np.mean(vol[:, :, s] > -0.99) < empty_thr:
                    continue
                self.index.append((pid, s))
        self._cache = {}
        print(f"[SliceMri2PetDataset] stage={stage}: {len(self.pids)} volumes, {len(self.index)} slices")

    def __len__(self):
        return len(self.index)

    def _load(self, folder, pid):
        key = (folder, pid)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = np.asarray(
                nib.load(os.path.join(folder, f'{pid}.nii.gz')).get_fdata(), dtype=np.float32)
        return self._cache[key]

    def __getitem__(self, i):
        pid, s = self.index[i]
        mri = self._load(self.mri_dir, pid)[:, :, s]
        pet = self._load(self.pet_dir, pid)[:, :, s]
        return (torch.from_numpy(mri).float().unsqueeze(0),
                torch.from_numpy(pet).float().unsqueeze(0), pid, s)
