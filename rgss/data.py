"""Datasets. Every dataset is stored as <root>/images/<id>.png and <root>/masks/<id>.png (see
prepare_data.py); a split file lists one id per line. Noisy annotations produced by
make_noisy_labels.py live in another mask directory with the same file names."""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]


class SegDataset(Dataset):
    """Returns dict(image (3,S,S) in [0,1], mask (1,S,S) in {0,1}, index).

    `targets` (optional, (N,1,S,S) tensor) replaces the masks, e.g. the constructed soft targets of RGSS.
    `flip`: random horizontal / vertical flips applied jointly to image and mask / target.
    """

    def __init__(self, root, ids, size, mask_dir="masks", flip=False):
        self.root, self.ids, self.size, self.mask_dir, self.flip = root, list(ids), size, mask_dir, flip
        self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = load_image(os.path.join(self.root, "images", name + ".png"), self.size)
        if self.targets is not None:
            mask = self.targets[i]
        else:
            mask = load_mask(os.path.join(self.root, self.mask_dir, name + ".png"), self.size)
        if self.flip:
            if random.random() < 0.5:
                img, mask = img.flip(-1), mask.flip(-1)
            if random.random() < 0.5:
                img, mask = img.flip(-2), mask.flip(-2)
        return {"image": img, "mask": mask, "index": i}
