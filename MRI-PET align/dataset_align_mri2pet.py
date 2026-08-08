"""
Paired MRI+PET dataset for encoder-alignment training (task 1).

Returns the SAME tuple shape the align runner expects, mirroring how
C2vDataset fed (target, source) into sdf_encoder_align.py:
    batch[0][0] = target  = PET   (ground truth to reconstruct)
    batch[1][0] = source  = MRI   (fed into the trainable MRI encoder)

Both volumes are ALREADY [-1,1] (per-volume MRI, fixed-scale PET);
NO re-normalization here.

Reads same-id files from:
    <root>/petTr|petVal|petTs   and   <root>/mriTr|mriVal|mriTs
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from Register import Registers


def _load(path):
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


@Registers.datasets.register_with_name('align_mri2pet')
class AlignMri2PetDataset(Dataset):
    def __init__(self, dataset_config, transform=None,
                 target_transform=None, stage='train'):
        super().__init__()
        suffix = {'train': 'Tr', 'val': 'Val', 'test': 'Ts'}
        if stage not in suffix:
            raise NotImplementedError(f"Stage '{stage}' not supported.")
        sf = suffix[stage]
        root = dataset_config.root
        self.pet_dir = os.path.join(root, f'pet{sf}')
        self.mri_dir = os.path.join(root, f'mri{sf}')
        pet_files = sorted(glob.glob(os.path.join(self.pet_dir, '*.nii.gz')))
        # keep only ids that exist in BOTH pet and mri
        self.ids = []
        for pf in pet_files:
            pid = os.path.basename(pf).replace('.nii.gz', '')
            if os.path.exists(os.path.join(self.mri_dir, f'{pid}.nii.gz')):
                self.ids.append(pid)
        if len(self.ids) == 0:
            raise RuntimeError(f"No paired ids in {self.pet_dir} / {self.mri_dir}")
        self.input_size = dataset_config.input_size
        self.depth_size = dataset_config.depth_size
        print(f"[AlignMri2PetDataset] stage={stage}: {len(self.ids)} pairs")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pid = self.ids[index]
        pet = _load(os.path.join(self.pet_dir, f'{pid}.nii.gz'))  # target
        mri = _load(os.path.join(self.mri_dir, f'{pid}.nii.gz'))  # source
        pet_t = torch.from_numpy(pet).float().unsqueeze(0)  # (1,D,H,W)
        mri_t = torch.from_numpy(mri).float().unsqueeze(0)
        # match sdf_encoder_align expectation:
        #   batch[0][0]=target(PET), batch[1][0]=source(MRI)
        # source slot mimics the [:, :1] slicing used before, so give (1,1,D,H,W)-friendly shape:
        return (pet_t, 'img'), (mri_t.unsqueeze(0), 'sdf'), pid
