"""
Volume-level MRI->PET paired dataset (full 3D, NOT slices).
Returns one (MRI volume, PET volume) pair, each (1,128,128,128), already [-1,1].
"""
import os, glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class VolMri2PetDataset(Dataset):
    def __init__(self, root, stage='train'):
        suffix = {'train':'Tr','val':'Val','test':'Ts'}[stage]
        self.mri_dir = os.path.join(root, f'mri{suffix}')
        self.pet_dir = os.path.join(root, f'pet{suffix}')
        pet_files = sorted(glob.glob(os.path.join(self.pet_dir, '*.nii.gz')))
        self.ids = []
        for pf in pet_files:
            pid = os.path.basename(pf)[:-7]
            if os.path.exists(os.path.join(self.mri_dir, f'{pid}.nii.gz')):
                self.ids.append(pid)
        print(f"[VolMri2PetDataset] stage={stage}: {len(self.ids)} volumes")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        mri = np.asarray(nib.load(os.path.join(self.mri_dir, f'{pid}.nii.gz')).get_fdata(), dtype=np.float32)
        pet = np.asarray(nib.load(os.path.join(self.pet_dir, f'{pid}.nii.gz')).get_fdata(), dtype=np.float32)
        return (torch.from_numpy(mri).float().unsqueeze(0),
                torch.from_numpy(pet).float().unsqueeze(0), pid)
