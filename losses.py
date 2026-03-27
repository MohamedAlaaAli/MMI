import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# nnU-Net style Dice Loss with optional mask-empty weighting
# ======================
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Returns negative dice (nnU-Net convention).
    Optionally ignores images with empty ground truth masks.
    """
    def __init__(self, smooth=1e-5, ignore_empty=False):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            ignore_empty (bool): If True, slices with empty masks are ignored in the loss.
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_empty = ignore_empty

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: Predicted logits (N, 1, H, W) or (N, 1, D, H, W)
            target: Ground truth mask (same shape, 0 or 1)
        Returns:
            Negative Dice score (scalar)
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.ignore_empty:
            # Only keep slices with any foreground
            mask = target_flat.sum(1) > 0
            if mask.sum() == 0:
                # If all slices are empty, return zero loss
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            dice_score = dice_score[mask]

        loss = -dice_score.mean()
        return loss


# ======================
# Dice + BCE loss (nnU-Net style)
# ======================
class DiceBCELoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and nnU-Net style Dice loss.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-5, ignore_empty=False):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth, ignore_empty=ignore_empty)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target.float())
        return self.dice_weight * dice + self.bce_weight * bce
    
