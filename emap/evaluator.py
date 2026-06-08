"""EMAPEvaluator: applies the projection to a trained PET+CT segmentation model.

This file contains all the model- and data-specific assumptions. To adapt EMAP
to a different architecture or dataset, this is the only file you should need
to edit. See ../README.md for the full list of assumptions and how to change
each one.
"""
import json
import random

import torch

from .projection import appendix_g_test


class EMAPEvaluator:
    """Run EMAP on a trained two-modality segmentation model.

    Default assumptions (edit this class to change them):
      - Model forward signature: model(x) where x has shape [B, 2, H, W]
        with channel 0 = modality A (PET), channel 1 = modality B (CT).
        For two-input models like LateFusionUNet, uses model(mod_a, mod_b).
      - Model output: [B, 1, H, W] per-pixel binary-segmentation logits.
        For multi-class, the projection still works element-wise; only the
        thresholding at the end (sigmoid > 0.5) needs replacing with argmax.
      - Checkpoint: a dict with key "model_state_dict".
      - Dataset: PETCTSliceDataset (../dataloaders.py). The dataset must
        return (image, mask) where image is [2, H, W] = [PET, CT] and mask
        is [1, H, W]. Modality A is taken as image[0:1], modality B as
        image[1:2]. To swap which channel is which modality, change the
        slice indices inside `run()`.
      - Augmentation: must be disabled (otherwise f(a_i, b_j) becomes
        non-deterministic and the projection is meaningless).
    """

    def __init__(self, config_path, ckpt_path, n_samples=500, n_repeats=1, seed=100):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
        self.n = n_samples
        self.k = n_repeats
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk = self.cfg["dataloader"].get("batch_size", 8)

        # --- Model ---
        # Mirrors trainer.py's dynamic class lookup. Local import to avoid
        # the top-level torchinfo summary in unet.py firing on package import.
        import unet
        
        # Load checkpoint first to detect model type
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Extract state dict from checkpoint wrapper if present
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt if isinstance(ckpt, dict) else {}
        
        # Detect model type from state dict keys
        has_unet1_keys = any("unet1." in k for k in state_dict.keys())
        has_unet2_keys = any("unet2." in k for k in state_dict.keys())
        has_ct_down_early = any("ct_down_early." in k for k in state_dict.keys())
        has_pet_down_early = any("pet_down_early." in k for k in state_dict.keys())
        has_early_fuser = any("early_fuser." in k for k in state_dict.keys())
        has_shared_down = any("shared_down." in k for k in state_dict.keys())
        has_ct_down = any("ct_down." in k for k in state_dict.keys())
        has_pet_down = any("pet_down." in k for k in state_dict.keys())
        has_bottleneck_fuser = any("bottleneck_fuser." in k for k in state_dict.keys())
        
        detected_late_fusion = has_unet1_keys and has_unet2_keys
        detected_early_fusion = has_ct_down_early and has_pet_down_early and has_early_fuser and has_shared_down
        detected_intermediate_fusion = has_ct_down and has_pet_down and has_bottleneck_fuser
        
        # Use detected type or fallback to config
        config_model_name = self.cfg["model"]["name"]
        if detected_late_fusion:
            self.model_name = "LateFusionUNet"
        elif detected_early_fusion:
            self.model_name = "EarlyIntermediateFusionUNet"
        elif detected_intermediate_fusion:
            self.model_name = "IntermediateFusionUNet"
        else:
            self.model_name = config_model_name
        
        Model = getattr(unet, self.model_name)
        
        if self.model_name == "LateFusionUNet":
            # LateFusionUNet requires two separate UNet instances
            unet1 = unet.Unet(
                in_chans=1,  # Single modality input
                out_chans=self.cfg["model"]["out_chans"],
                chans=self.cfg["model"]["chans"],
                num_pool_layers=self.cfg["model"]["num_pool_layers"],
                drop_prob=self.cfg["model"]["drop_prob"],
                use_att=self.cfg["model"]["use_att"],
                use_res=self.cfg["model"]["use_res"],
            )
            unet2 = unet.Unet(
                in_chans=1,  # Single modality input
                out_chans=self.cfg["model"]["out_chans"],
                chans=self.cfg["model"]["chans"],
                num_pool_layers=self.cfg["model"]["num_pool_layers"],
                drop_prob=self.cfg["model"]["drop_prob"],
                use_att=self.cfg["model"]["use_att"],
                use_res=self.cfg["model"]["use_res"],
            )
            self.model = Model(
                unet1=unet1,
                unet2=unet2,
                out_channels=self.cfg["model"]["out_chans"],
                fusion_mode=self.cfg["model"].get("fusion_mode", "concat"),
            ).to(self.device)
        elif self.model_name == "IntermediateFusionUNet":
            # IntermediateFusionUNet with separate CT and PET inputs
            self.model = Model(
                in_chans_ct=self.cfg["model"].get("in_chans_ct", 1),
                in_chans_pet=self.cfg["model"].get("in_chans_pet", 1),
                out_chans=self.cfg["model"]["out_chans"],
                chans=self.cfg["model"]["chans"],
                num_pool_layers=self.cfg["model"]["num_pool_layers"],
                drop_prob=self.cfg["model"]["drop_prob"],
                use_att=self.cfg["model"]["use_att"],
                use_res=self.cfg["model"]["use_res"],
                fusion_mode=self.cfg["model"].get("fusion_mode", "concat"),
                fuse_skips=self.cfg["model"].get("fuse_skips", True),
            ).to(self.device)
        elif self.model_name == "EarlyIntermediateFusionUNet":
            # EarlyIntermediateFusionUNet with separate CT and PET inputs
            self.model = Model(
                in_chans_ct=self.cfg["model"].get("in_chans_ct", 1),
                in_chans_pet=self.cfg["model"].get("in_chans_pet", 1),
                out_chans=self.cfg["model"]["out_chans"],
                chans=self.cfg["model"]["chans"],
                num_pool_layers=self.cfg["model"]["num_pool_layers"],
                fusion_depth=self.cfg["model"].get("fusion_depth", 2),
                drop_prob=self.cfg["model"]["drop_prob"],
                use_att=self.cfg["model"]["use_att"],
                use_res=self.cfg["model"]["use_res"],
                fusion_mode=self.cfg["model"].get("fusion_mode", "concat"),
                fuse_skips=self.cfg["model"].get("fuse_skips", True),
            ).to(self.device)
        else:
            # Standard Unet with 2 input channels
            self.model = Model(
                in_chans=self.cfg["model"].get("in_chans", 2),
                out_chans=self.cfg["model"]["out_chans"],
                chans=self.cfg["model"]["chans"],
                num_pool_layers=self.cfg["model"]["num_pool_layers"],
                drop_prob=self.cfg["model"]["drop_prob"],
                use_att=self.cfg["model"]["use_att"],
                use_res=self.cfg["model"]["use_res"],
            ).to(self.device)
        
        # Load state dict
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Error loading checkpoint. Available keys: {list(state_dict.keys())[:5]}...")
            raise e
        self.model.eval()

        # --- Dataset ---
        from dataloaders import PETCTSliceDataset
        d = self.cfg["dataset"]
        self.ds = PETCTSliceDataset(
            data_dir=d["data_dir"],
            splits_json=d["splits_json"],
            fold=d.get("fold", 0),
            split="val",
            modality="both",
            target_size=tuple(d.get("target_size", [256, 256])),
            augment=False,
            seed=seed,
        )

    @torch.no_grad()
    def run(self):
        from metrics import BatchSegmentationMetrics
        appendix_g_test()  # cheap sanity check before expensive forwards

        rng = random.Random(self.seed)
        metrics = BatchSegmentationMetrics()
        results = []

        for r in range(self.k):
            indices = rng.sample(range(len(self.ds)), self.n)
            imgs = torch.stack([self.ds[i][0] for i in indices]).to(self.device)
            masks = torch.stack([self.ds[i][1] for i in indices]).to(self.device)
            mod_a = imgs[:, 0:1]   # PET — change to imgs[:, 1:2] to swap roles
            mod_b = imgs[:, 1:2]   # CT
            N, _, H, W = mod_a.shape

            # Streaming N x N projection: never materialize the full
            # [N, N, 1, H, W] cache (~131 GB at N=500, H=W=256, fp32).
            row_sum = torch.zeros(N, 1, H, W, device=self.device)
            col_sum = torch.zeros(N, 1, H, W, device=self.device)
            total_sum = torch.zeros(1, H, W, device=self.device)
            diag = torch.zeros(N, 1, H, W, device=self.device)

            for i in range(N):
                a_i = mod_a[i:i+1]
                for j0 in range(0, N, self.chunk):
                    j1 = min(j0 + self.chunk, N)
                    a_b = a_i.expand(j1 - j0, -1, -1, -1)
                    # --- Model forward ---
                    if self.model_name == "LateFusionUNet":
                        # Two-input model: pass modalities separately
                        logits = self.model(a_b, mod_b[j0:j1])  # [k, 1, H, W]
                    elif self.model_name in ("IntermediateFusionUNet", "EarlyIntermediateFusionUNet"):
                        # Two-input fusion models: pass CT and PET separately
                        # Note: a_b is PET, mod_b is CT
                        logits = self.model(mod_b[j0:j1], a_b)  # model(ct, pet)
                    else:
                        # Single-input model: concatenate modalities
                        x = torch.cat([a_b, mod_b[j0:j1]], dim=1)   # [k, 2, H, W]
                        logits = self.model(x)                       # [k, 1, H, W]
                    row_sum[i] += logits.sum(dim=0)
                    col_sum[j0:j1] += logits
                    total_sum += logits.sum(dim=0)
                    if j0 <= i < j1:
                        diag[i] = logits[i - j0]

            emap_logits = row_sum / N + col_sum / N - total_sum / (N * N)

            # --- Thresholding ---
            # Binary: sigmoid > 0.5. For multi-class change to argmax along
            # the channel dim, and use a multi-class metric.
            pred_full = (torch.sigmoid(diag) > 0.5).long().cpu()
            pred_emap = (torch.sigmoid(emap_logits) > 0.5).long().cpu()
            mk = masks.cpu()

            full_m = metrics.compute_all(pred_full, mk)
            emap_m = metrics.compute_all(pred_emap, mk)
            results.append({"full": full_m, "emap": emap_m})
            print(
                f"[repeat {r+1}/{self.k}] "
                f"full dice={full_m['dice']:.4f}  emap dice={emap_m['dice']:.4f}"
            )

        avg = {"full": {}, "emap": {}}
        for key in ("dice", "precision", "recall", "hd95"):
            avg["full"][key] = sum(x["full"][key] for x in results) / self.k
            avg["emap"][key] = sum(x["emap"][key] for x in results) / self.k
        return {"per_repeat": results, "avg": avg, "n_samples": self.n, "n_repeats": self.k}
