#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import KFold

# ======================
# CONFIG
# ======================
images_dir = "/kaggle/input/datasets/mohamednasserhussien/psma-pet-ct-lesions/PSMA-PET-CT-Lesions_v2/imagesTr"
labels_dir = "/kaggle/input/datasets/mohamednasserhussien/psma-pet-ct-lesions/PSMA-PET-CT-Lesions_v2/labelsTr"
out_dir = "/kaggle/working/data_2d"

os.makedirs(out_dir, exist_ok=True)

save_empty_slices = False  
n_splits = 3

CT_IDX = 0   # _0000
PET_IDX = 1  # _0001


# ======================
# HELPERS
# ======================
def load_nifti(path):
    return nib.load(path).get_fdata()


def normalize_pet(pet):
    pet = np.clip(pet, 0, 20)
    return pet / 20.0


def normalize_ct(ct):
    ct = np.clip(ct, -1000, 400)
    return (ct + 1000) / 1400.0


def get_case_ids(images_dir):
    files = os.listdir(images_dir)
    case_ids = set()

    for f in files:
        if f.endswith(".nii.gz"):
            case_id = "_".join(f.split("_")[:-1])  # robust
            case_ids.add(case_id)

    return sorted(list(case_ids))


# ======================
# MAIN
# ======================
case_ids = get_case_ids(images_dir)

# Store slice names per case
case_to_slices = {}

for case_id in tqdm(case_ids, desc="Processing cases"):

    ct_path = os.path.join(images_dir, f"{case_id}_{CT_IDX:04d}.nii.gz")
    pet_path = os.path.join(images_dir, f"{case_id}_{PET_IDX:04d}.nii.gz")
    label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

    if not (os.path.exists(ct_path) and os.path.exists(pet_path) and os.path.exists(label_path)):
        print(f"Skipping {case_id} (missing files)")
        continue

    ct = load_nifti(ct_path)
    pet = load_nifti(pet_path)
    label = load_nifti(label_path)

    if not (ct.shape == pet.shape == label.shape):
        print(f"Shape mismatch in {case_id}, skipping")
        continue

    ct = normalize_ct(ct)
    pet = normalize_pet(pet)

    H, W, Z = ct.shape

    case_out_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_out_dir, exist_ok=True)

    slice_names = []

    for z in tqdm(range(Z), desc=f"{case_id}", leave=False):

        ct_slice = ct[:, :, z]
        pet_slice = pet[:, :, z]
        label_slice = label[:, :, z]

        if not save_empty_slices:
            if np.sum(label_slice) == 0:
                continue

        image = np.stack([pet_slice, ct_slice], axis=0)

        slice_id = f"{case_id}_z{z:03d}"

        img_name = f"{slice_id}_img.npy"
        mask_name = f"{slice_id}_mask.npy"

        np.save(os.path.join(case_out_dir, img_name), image.astype(np.float32))
        np.save(os.path.join(case_out_dir, mask_name), label_slice.astype(np.uint8))

        # Store slice identifier (without suffix like nnU-Net)
        slice_names.append(slice_id)

    case_to_slices[case_id] = slice_names


print("✅ Done slicing. Now creating splits...")

# ======================
# CREATE 3-FOLD SPLITS
# ======================
splits_path = os.path.join(out_dir, "splits_final.json")

if os.path.exists(splits_path):
    print(f"✅ Splits already exist at {splits_path}, skipping.")
else:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    splits = []

    case_ids_array = np.array(case_ids)

    for fold, (train_idx, val_idx) in enumerate(kf.split(case_ids_array)):

        train_cases = case_ids_array[train_idx]
        val_cases = case_ids_array[val_idx]

        train_slices = []
        val_slices = []

        for c in train_cases:
            train_slices.extend(case_to_slices.get(c, []))

        for c in val_cases:
            val_slices.extend(case_to_slices.get(c, []))

        splits.append({
            "train": train_slices,
            "val": val_slices
        })

        print(f"Fold {fold}: {len(train_slices)} train slices, {len(val_slices)} val slices")

    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"✅ Splits saved to {splits_path}")