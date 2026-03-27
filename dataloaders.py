import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

# ======================
# GLOBAL SEED
# ======================
def set_seed(seed=42):
    """Set global seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======================
# DATASET
# ======================
class PETCTSliceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        splits_json,
        fold=0,
        split="train",
        modality="both",
        target_size=(256, 256),
        augment=True,
        seed=42
    ):
        self.data_dir = data_dir
        self.modality = modality
        self.target_size = target_size
        self.augment = augment
        self.seed = seed

        with open(splits_json, "r") as f:
            splits = json.load(f)
        assert split in ["train", "val"]
        self.slice_ids = splits[fold][split]

    def __len__(self):
        return len(self.slice_ids)

    def resize(self, image, mask):
        image = F.interpolate(
            image.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).float(), size=self.target_size, mode="nearest"
        ).squeeze(0).long()
        return image, mask

    def random_augment(self, image, mask, idx):
        # Create a local RNG per sample for reproducibility
        rng = random.Random(self.seed + idx)

        # Horizontal flip
        if rng.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        # Vertical flip
        if rng.random() > 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        # Random 90-degree rotation
        k = rng.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, [1, 2])
            mask = torch.rot90(mask, k, [1, 2])

        # PET intensity jitter
        if self.modality in ["pet", "both"]:
            pet_idx = 0 if self.modality == "pet" else 0
            noise = (torch.rand_like(image[pet_idx]) - 0.5) * 0.1
            image[pet_idx] = torch.clamp(image[pet_idx] + noise, 0.0, 1.0)

        # CT Gaussian noise
        if self.modality in ["ct", "both"]:
            ct_idx = 0 if self.modality == "ct" else 1
            noise = torch.randn_like(image[ct_idx]) * 0.01
            image[ct_idx] = torch.clamp(image[ct_idx] + noise, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        case_id = "_".join(slice_id.split("_")[:-1])

        img_path = os.path.join(self.data_dir, case_id, f"{slice_id}_img.npy")
        mask_path = os.path.join(self.data_dir, case_id, f"{slice_id}_mask.npy")

        image = np.load(img_path)
        mask = np.load(mask_path)

        # Modality selection
        if self.modality == "ct":
            image = image[1:2]
        elif self.modality == "pet":
            image = image[0:1]
        elif self.modality == "both":
            pass
        else:
            raise ValueError("modality must be 'ct', 'pet', or 'both'")

        # To torch
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long().unsqueeze(0)

        # Resize
        image, mask = self.resize(image, mask)

        # Augmentation
        if self.augment:
            image, mask = self.random_augment(image, mask, idx)

        return image, mask

# ======================
# DATALOADER FACTORY
# ======================
def get_dataloaders(
    data_dir,
    splits_json,
    fold=0,
    modality="both",
    batch_size=8,
    num_workers=4,
    seed=42
):
    """
    Returns deterministic train and validation DataLoaders
    """

    # Train dataset with augmentations
    train_dataset = PETCTSliceDataset(
        data_dir=data_dir,
        splits_json=splits_json,
        fold=fold,
        split="train",
        modality=modality,
        augment=True,
        seed=seed
    )

    # Validation dataset without augmentations
    val_dataset = PETCTSliceDataset(
        data_dir=data_dir,
        splits_json=splits_json,
        fold=fold,
        split="val",
        modality=modality,
        augment=False,
        seed=seed
    )

    # Use a Generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # deterministic due to generator
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # always in order
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
