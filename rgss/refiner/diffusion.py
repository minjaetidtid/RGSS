"""Discrete diffusion mask refiner R_phi of SegRefiner (Wang et al., NeurIPS 2023), Sec. 2 of the paper.

Every pixel has a latent state s_t in {fine, coarse}. The forward process is unidirectional
(fine -> coarse): a pixel of the fine mask y_0 is still fine at step t with probability abar_t, which
decreases linearly from `start` (t = 0) to `stop` (t = K - 1); the mask at step t is
y_t = s_t * y_0 + (1 - s_t) * y_T with y_T the coarse mask (Eq. 1).

A denoising network p_phi(x, y_t, t) predicts the fine mask. The reverse process starts from a coarse
mask and, at every step, turns coarse pixels into fine ones with the posterior probability of the
state transition, using the predicted fine mask for them.

R_phi(x, y, T) places the given mask at step T (1 <= T <= K) and runs the reverse process from step T
down to step 1, so that T controls the refinement strength (Eq. 3).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .guided_diffusion.unet import UNetModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sobel_magnitude(x, eps=1e-4):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx**2 + gy**2 + eps)


class MaskRefiner(nn.Module):
    def __init__(self, num_steps, start, stop, image_size, model_channels, num_res_blocks, num_heads,
                 attention_strides, channel_mult, dropout):
        super().__init__()
        self.num_steps, self.image_size = num_steps, image_size
        self.register_buffer("abar", torch.tensor(np.linspace(start, stop, num_steps), dtype=torch.float32))
        self.register_buffer("abar_prev", torch.cat([torch.ones(1), self.abar[:-1]]))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.net = UNetModel(image_size=image_size, in_channels=4, model_channels=model_channels, out_channels=1,
                             num_res_blocks=num_res_blocks, attention_resolutions=tuple(attention_strides),
                             dropout=dropout, channel_mult=tuple(channel_mult), num_heads=num_heads,
                             use_scale_shift_norm=True, resblock_updown=True)

    def _prepare(self, img, mask):
        size = (self.image_size, self.image_size)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        mask = (F.interpolate(mask, size=size, mode="nearest") > 0.5).float()
        return (img - self.mean) / self.std, mask

    def denoise(self, img_n, y_t, t):
        return self.net(torch.cat([img_n, y_t], dim=1), t)

    # ------------------------------------------------------------------ training
    def q_sample(self, y0, y_coarse, t):
        keep_fine = (torch.rand_like(y0) < self.abar[t].view(-1, 1, 1, 1)).float()
        return keep_fine * y0 + (1 - keep_fine) * y_coarse

    def training_loss(self, img, y0, y_coarse, texture_weight):
        """BCE + Sobel-texture L1 between the predicted and the fine mask (SegRefiner)."""
        img_n, y0 = self._prepare(img, y0)
        _, y_coarse = self._prepare(img, y_coarse)
        t = torch.randint(0, self.num_steps, (y0.shape[0],), device=y0.device)
        logits = self.denoise(img_n, self.q_sample(y0, y_coarse, t), t)
        bce = F.binary_cross_entropy_with_logits(logits, y0)
        texture = (sobel_magnitude(logits.sigmoid()) - sobel_magnitude(y0)).abs().mean()
        return bce + texture_weight * texture, logits

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def refine(self, img, coarse, steps):
        """R_phi(x, y, T). img (B,3,H,W) in [0,1], coarse (B,1,H,W) in {0,1}, steps (B,) ints in [1, K].

        Returns the refined mask (B,1,H,W) in [0,1] at the input resolution.
        """
        h, w = img.shape[-2:]
        img_n, y_last = self._prepare(img, coarse)
        steps = torch.as_tensor(steps, device=img.device).long().clamp(1, self.num_steps)
        x = y_last.clone()
        fine_prob = torch.zeros_like(y_last)
        out = torch.zeros_like(y_last)
        for t in range(int(steps.max()) - 1, -1, -1):
            active = (steps - 1 >= t).view(-1, 1, 1, 1).float()  # sample b runs steps T_b - 1, ..., 0
            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            logits = self.denoise(img_n, x, tt)
            if t == 0:
                out = logits.sigmoid()
                break
            # posterior probability that a coarse pixel becomes fine between t and t - 1
            p_fine = 2 * (logits.sigmoid() - 0.5).abs()
            p_c2f = p_fine * (self.abar_prev[t] - self.abar[t]) / (1 - p_fine * self.abar[t]).clamp(min=1e-6)
            new_fine_prob = fine_prob + (1 - fine_prob) * p_c2f
            fine_map = (torch.rand_like(x) < new_fine_prob).float()
            new_x = (logits >= 0).float() * fine_map + y_last * (1 - fine_map)
            x = active * new_x + (1 - active) * x
            fine_prob = active * new_fine_prob + (1 - active) * fine_prob
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def build_refiner(cfg):
    model = MaskRefiner(**cfg)
    model._config = dict(cfg)
    return model


def save_refiner(path, model, extra=None):
    torch.save({"config": model._config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_refiner(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu")
    model = build_refiner(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval().requires_grad_(False)
