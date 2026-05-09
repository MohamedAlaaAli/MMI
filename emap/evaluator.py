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
        Other model shapes (e.g. two-input model(a, b)) require editing the
        forward call inside `run()`.
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
        Model = getattr(unet, self.cfg["model"]["name"])
        self.model = Model(
            in_chans=self.cfg["model"]["in_chans"],
            out_chans=self.cfg["model"]["out_chans"],
            chans=self.cfg["model"]["chans"],
            num_pool_layers=self.cfg["model"]["num_pool_layers"],
            drop_prob=self.cfg["model"]["drop_prob"],
            use_att=self.cfg["model"]["use_att"],
            use_res=self.cfg["model"]["use_res"],
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
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
                    # For a two-input model, replace the next two lines with:
                    #     logits = self.model(a_b, mod_b[j0:j1])
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
