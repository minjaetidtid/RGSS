"""Mixed-quality annotations of Sec. 4.1: a fraction `--ratio` of the training masks is corrupted with
structured morphological noise (rgss/corruption.py); the remaining masks are copied unchanged.

Example:
    python make_noisy_labels.py --data_root data/kvasir --split splits/kvasir_train.txt --ratio 0.5 \
        --min_iou 0.5 --max_iters 200 --out_dir masks_noise50
"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from rgss.corruption import corrupt_mask, iou
from rgss.data import read_split


def main(args):
    rng = np.random.default_rng(args.seed)
    ids = read_split(args.split)
    noisy = set(rng.choice(len(ids), size=int(round(args.ratio * len(ids))), replace=False).tolist())
    out = os.path.join(args.data_root, args.out_dir)
    os.makedirs(out, exist_ok=True)
    ious = []
    for i, name in enumerate(tqdm(ids, desc="masks")):
        mask = (np.asarray(Image.open(os.path.join(args.data_root, "masks", name + ".png")).convert("L")) > 127).astype(np.uint8)
        if i in noisy:
            new = corrupt_mask(mask, rng, args.min_iou, args.max_iters)
            ious.append(iou(new, mask))
            mask = new
        Image.fromarray(mask * 255).save(os.path.join(out, name + ".png"))
    with open(os.path.join(out, "noisy_ids.txt"), "w") as f:
        f.write("\n".join(ids[i] for i in sorted(noisy)) + ("\n" if noisy else ""))
    msg = f"mean IoU of corrupted masks {np.mean(ious):.3f}" if ious else "no corrupted masks"
    print(f"Corrupted {len(noisy)}/{len(ids)} masks ({msg}). Saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGSS: synthesize mixed-quality annotations")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", required=True, help="training split; only these masks are written")
    p.add_argument("--ratio", type=float, required=True, help="fraction of corrupted masks (0.25 / 0.5 in the paper)")
    p.add_argument("--min_iou", type=float, required=True, help="corruption stops before the IoU falls below this")
    p.add_argument("--max_iters", type=int, required=True, help="maximum number of corruption iterations")
    p.add_argument("--out_dir", required=True, help="output mask directory under data_root")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
