"""Segmentation backbones M_theta, Dice loss and evaluation metrics."""

import torch
from torch import nn

ARCHS = {"unet": "Unet", "unetplusplus": "UnetPlusPlus", "linknet": "Linknet", "deeplabv3plus": "DeepLabV3Plus"}


def build_model(arch, encoder):
    """M_theta: images in [0, 1] -> logits (B, 1, H, W). The encoder is randomly initialized."""
    import segmentation_models_pytorch as smp

    return getattr(smp, ARCHS[arch])(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)


def dice_loss(prob, target, eps=1.0, reduction="mean"):
    """Soft Dice loss L(sigma(M(x)), y) per sample; targets may be soft (Eq. 4)."""
    inter = (prob * target).flatten(1).sum(1)
    denom = prob.flatten(1).sum(1) + target.flatten(1).sum(1)
    loss = 1 - (2 * inter + eps) / (denom + eps)
    return loss.mean() if reduction == "mean" else loss


class SegMetrics:
    """Per-image foreground IoU / Dice (threshold 0.5), averaged over the images."""

    def __init__(self):
        self.iou, self.dice = [], []

    @torch.no_grad()
    def update(self, logits, target):
        pred = (logits.sigmoid() > 0.5).flatten(1).float()
        gt = (target > 0.5).flatten(1).float()
        inter, ps, gs = (pred * gt).sum(1), pred.sum(1), gt.sum(1)
        union = ps + gs - inter
        empty = (ps + gs) == 0  # both empty: perfect prediction
        self.iou += torch.where(empty, torch.ones_like(inter), inter / union.clamp(min=1)).tolist()
        self.dice += torch.where(empty, torch.ones_like(inter), 2 * inter / (ps + gs).clamp(min=1)).tolist()

    def compute(self):
        return {"mIoU": 100 * sum(self.iou) / len(self.iou), "mDice": 100 * sum(self.dice) / len(self.dice)}


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    m = SegMetrics()
    for batch in loader:
        m.update(model(batch["image"].to(device)), batch["mask"].to(device))
    return m.compute()
