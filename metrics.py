import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score

class BatchSegmentationMetrics:
    """
    Robust metrics for batch segmentation.
    Handles cases with all-normal slices.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def dice_coef(self, pred, target):
        # Flatten
        pred_flat = pred.contiguous().view(pred.shape[0], -1).float()
        target_flat = target.contiguous().view(target.shape[0], -1).float()

        intersection = (pred_flat * target_flat).sum(dim=1)
        denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps

        # Dice = 1 for empty slices
        dice = torch.where(
            (target_flat.sum(dim=1) == 0) & (pred_flat.sum(dim=1) == 0),
            torch.ones_like(intersection),
            2.0 * intersection / denominator
        )
        return dice.mean().item()

    def precision(self, pred, target):
        batch_size = pred.shape[0]
        prec_list = []
        for i in range(batch_size):
            p = pred[i].flatten().cpu().numpy()
            t = target[i].flatten().cpu().numpy()
            if t.sum() == 0 and p.sum() == 0:
                prec_list.append(1.0)
            else:
                prec_list.append(precision_score(t, p, zero_division=1))
        return np.mean(prec_list)

    def recall(self, pred, target):
        batch_size = pred.shape[0]
        rec_list = []
        for i in range(batch_size):
            p = pred[i].flatten().cpu().numpy()
            t = target[i].flatten().cpu().numpy()
            if t.sum() == 0 and p.sum() == 0:
                rec_list.append(1.0)
            else:
                rec_list.append(recall_score(t, p, zero_division=1))
        return np.mean(rec_list)

    def hd95(self, pred, target):
        batch_size = pred.shape[0]
        hd_list = []
        for i in range(batch_size):
            p = pred[i].cpu().numpy().astype(bool)
            t = target[i].cpu().numpy().astype(bool)
            if p.sum() == 0 and t.sum() == 0:
                hd_list.append(0.0)
            elif p.sum() == 0 or t.sum() == 0:
                continue  # skip slices with one empty mask
            else:
                pred_coords = np.argwhere(p)
                target_coords = np.argwhere(t)
                forward = directed_hausdorff(pred_coords, target_coords)[0]
                backward = directed_hausdorff(target_coords, pred_coords)[0]
                hd_list.append(np.percentile([forward, backward], 95))
        if len(hd_list) == 0:
            return 0.0
        return float(np.mean(hd_list))

    def compute_all(self, pred, target):
        return {
            "dice": self.dice_coef(pred, target),
            "precision": self.precision(pred, target),
            "recall": self.recall(pred, target),
            "hd95": self.hd95(pred, target)
        }