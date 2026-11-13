"""Convert Kvasir-SEG / ISIC 2017 to the layout used by this repository:
    <out>/images/<id>.png, <out>/masks/<id>.png
Kvasir-SEG uses the 8:1:1 split in splits/kvasir_*.txt; for ISIC 2017 the official training / validation /
test sets are written to splits/isic2017_*.txt.

Examples:
    python prepare_data.py --dataset kvasir --src /path/to/Kvasir-SEG --out data/kvasir
    python prepare_data.py --dataset isic2017 --src /path/to/ISIC-2017 --out data/isic2017
"""

import argparse
import glob
import os

from PIL import Image
from tqdm import tqdm

ISIC_SETS = {"train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
             "val": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
             "test": ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth")}


def save_pair(img_path, mask_path, out, name):
    Image.open(img_path).convert("RGB").save(os.path.join(out, "images", name + ".png"))
    Image.open(mask_path).convert("L").save(os.path.join(out, "masks", name + ".png"))


def main(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    if args.dataset == "kvasir":
        for p in tqdm(sorted(glob.glob(os.path.join(args.src, "images", "*.jpg"))), desc="kvasir"):
            name = os.path.splitext(os.path.basename(p))[0]
            save_pair(p, os.path.join(args.src, "masks", name + ".jpg"), args.out, name)
    else:
        os.makedirs(args.split_dir, exist_ok=True)
        for split, (img_dir, gt_dir) in ISIC_SETS.items():
            names = []
            # the official archives may unpack into a nested folder of the same name
            masks = {os.path.basename(m): m for m in glob.glob(os.path.join(args.src, gt_dir, "**", "*_segmentation.png"), recursive=True)}
            for p in tqdm(sorted(glob.glob(os.path.join(args.src, img_dir, "**", "ISIC_*.jpg"), recursive=True)), desc=f"isic {split}"):
                name = os.path.splitext(os.path.basename(p))[0]
                save_pair(p, masks[name + "_segmentation.png"], args.out, name)
                names.append(name)
            with open(os.path.join(args.split_dir, f"isic2017_{split}.txt"), "w") as f:
                f.write("\n".join(names) + "\n")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGSS: dataset preparation")
    p.add_argument("--dataset", choices=["kvasir", "isic2017"], required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split_dir", default="splits")
    main(p.parse_args())
