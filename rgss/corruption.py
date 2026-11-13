"""Mask corruptions.

- `corrupt_mask`: structured annotation noise of Sec. 4.1 (used to build the mixed-quality training sets
  and to synthesize coarse masks for refiner pre-training): region-wise dilation / erosion with random
  rectangular, circular or elliptical structuring elements of size 3 or 5 inside random spatial windows,
  plus stochastic pixel flips, repeated for up to `max_iters` iterations and stopped before the IoU with
  the original mask falls below `min_iou`. The procedure follows the perturbation of SegRefiner
  (Wang et al., NeurIPS 2023).
"""

import cv2
import numpy as np

KERNEL_SIZES = (3, 5)


def random_kernel(rng):
    size = int(rng.choice(KERNEL_SIZES))
    kind = rng.integers(4)
    if kind == 0:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if kind == 1:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))  # circular
    if kind == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, max(1, size // 2)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, size // 2), size))


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def corrupt_mask(mask, rng, min_iou, max_iters, windows_per_iter=4, flip_prob=0.25):
    """mask: (H, W) {0, 1} uint8. Returns a corrupted copy with IoU >= min_iou to the input."""
    gt = (mask > 0).astype(np.uint8)
    seg = gt.copy()
    h, w = seg.shape
    if h <= 2 or w <= 2 or gt.sum() == 0:
        return seg
    for _ in range(max_iters):
        prev = seg.copy()
        for _ in range(windows_per_iter):
            x0, y0 = rng.integers(w), rng.integers(h)
            x1, y1 = rng.integers(x0 + 1, w + 1), rng.integers(y0 + 1, h + 1)
            if rng.random() < flip_prob:  # stochastic pixel flip, grown into holes / blobs by the next op
                seg[(y0 + y1) // 2, (x0 + x1) // 2] = rng.integers(2)
            op = cv2.dilate if rng.random() < 0.5 else cv2.erode
            seg[y0:y1, x0:x1] = op(seg[y0:y1, x0:x1], random_kernel(rng), iterations=1)
        if iou(seg, gt) < min_iou:
            return prev
    return seg
