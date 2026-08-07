#!/usr/bin/env python3
"""
MRI -> PET FINAL preprocessing.

Skull-stripping is done PER MODALITY with the best tool for each:
  MRI : HD-BET brain mask   (mri_brain/<id>_mask.nii.gz)
  PET : SynthStrip mask, b=2 (pet_brain/<id>_mask.nii.gz)
Each mask is applied to its OWN modality in its OWN native space, so the
PET keeps its own (soft) brain shape instead of being cut by the MRI mask.

Steps per pair:
  1. Load original MRI + HD-BET mask; apply -> skull-stripped MRI.
  2. Load original PET + SynthStrip mask; apply -> skull-stripped PET.
  3. Register skull-stripped PET -> skull-stripped MRI
     (Translation init to remove ~100mm affine offset, then Rigid).
  4. Resample both onto one common 128^3 grid (MRI geometry, no rotation).
  5. Normalize:
        MRI -> per-volume 0.5/99.5 percentile min-max -> [-1,1]; bg -> -1
        PET -> global fixed scale [0, 2.5]            -> [-1,1]; bg -> -1
"""

import argparse, os
import numpy as np
import pandas as pd
import ants

PET_MIN, PET_MAX = 0.0, 2.5


def make_ref_grid(img, size):
    extent = np.array(img.shape, float) * np.array(img.spacing, float)
    return ants.make_image(imagesize=(size, size, size),
                           spacing=tuple((extent/size).tolist()),
                           origin=img.origin, direction=img.direction)


def apply_mask(img, mask):
    """Zero-out voxels outside mask, return ANTs image (same geometry)."""
    a = img.numpy(); m = mask.numpy() > 0.5
    a = a * m
    return img.new_image_like(a.astype(np.float32)), m


def register(mri_ss, pet_ss):
    init = ants.registration(fixed=mri_ss, moving=pet_ss, type_of_transform="Translation")
    reg = ants.registration(fixed=mri_ss, moving=pet_ss, type_of_transform="Rigid",
                            initial_transform=init["fwdtransforms"][0])
    return reg


def norm_mri(arr, brain):
    v = arr[brain]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    a = np.clip(arr, lo, hi); a = (a-lo)/(hi-lo+1e-8)*2-1
    a[~brain] = -1.0
    return a.astype(np.float32)


def norm_pet(arr, brain):
    a = np.clip(arr, PET_MIN, PET_MAX); a = (a-PET_MIN)/(PET_MAX-PET_MIN)*2-1
    a[~brain] = -1.0
    return a.astype(np.float32)


def process_one(mri_p, pet_p, mri_mask_p, pet_mask_p, size):
    mri = ants.image_read(mri_p)
    pet = ants.image_read(pet_p)
    mri_mask = ants.image_read(mri_mask_p)
    pet_mask = ants.image_read(pet_mask_p)

    mri_ss, _ = apply_mask(mri, mri_mask)          # skull-stripped MRI
    pet_ss, _ = apply_mask(pet, pet_mask)          # skull-stripped PET (own shape)

    reg = register(mri_ss, pet_ss)
    pet_in_mri = reg["warpedmovout"]
    # bring the PET mask into MRI space too, to define PET brain after resample
    pet_mask_in_mri = ants.apply_transforms(fixed=mri_ss, moving=pet_mask,
                                            transformlist=reg["fwdtransforms"],
                                            interpolator="nearestNeighbor")

    ref = make_ref_grid(mri_ss, size)
    mri_r  = ants.resample_image_to_target(mri_ss, ref, interp_type=4).numpy()
    pet_r  = ants.resample_image_to_target(pet_in_mri, ref, interp_type=4).numpy()
    mmask_r = ants.resample_image_to_target(mri_mask, ref, interp_type=1).numpy() > 0.5
    pmask_r = ants.resample_image_to_target(pet_mask_in_mri, ref, interp_type=1).numpy() > 0.5

    mri_n = norm_mri(mri_r, mmask_r)
    pet_n = norm_pet(pet_r, pmask_r)
    return mri_n, pet_n


def save_nii(arr, path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), path)


def split_subjects(df, seed=42, ratios=(0.8, 0.1, 0.1)):
    subj = df["Subject ID"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(subj); a = int(n*ratios[0]); b = int(n*ratios[1])
    tr, va = set(subj[:a]), set(subj[a:a+b])
    df = df.copy()
    df["split"] = df["Subject ID"].map(lambda s: "train" if s in tr else ("val" if s in va else "test"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mri_mask_dir", required=True)
    ap.add_argument("--pet_mask_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    df = split_subjects(df)
    for s in ["mri", "pet"]:
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    rows = []
    for i, r in df.iterrows():
        pid = str(r["PET_ImageID"])
        mri_mask = os.path.join(args.mri_mask_dir, f"{pid}_mask.nii.gz")
        pet_mask = os.path.join(args.pet_mask_dir, f"{pid}_mask.nii.gz")
        if not (os.path.exists(mri_mask) and os.path.exists(pet_mask)):
            print(f"[{i}] SKIP {pid}: missing mask", flush=True); continue
        try:
            mri, pet = process_one(r["mri_path"], r["pet_path"], mri_mask, pet_mask, args.size)
            save_nii(mri, os.path.join(args.out, "mri", f"{pid}.nii.gz"))
            save_nii(pet, os.path.join(args.out, "pet", f"{pid}.nii.gz"))
            rows.append({**r.to_dict(),
                         "mri_proc": os.path.join(args.out, "mri", f"{pid}.nii.gz"),
                         "pet_proc": os.path.join(args.out, "pet", f"{pid}.nii.gz")})
            if (i+1) % 25 == 0:
                print(f"[{i+1}/{len(df)}] done", flush=True)
        except Exception as e:
            print(f"[{i}] FAILED {pid}: {e}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.out, "manifest_proc.csv"), index=False)
    print("wrote manifest_proc.csv")
    print(out_df["split"].value_counts().to_string())
    print("subjects:", out_df["Subject ID"].nunique(), "| pairs:", len(out_df))


if __name__ == "__main__":
    main()
