"""Evaluate a trained segmentation network: per-image mIoU / mDice (foreground, threshold 0.5).

Example:
    python eval_seg.py --checkpoint runs/kvasir_noise50/unet_rgss/best.pth --data_root data/kvasir --split splits/kvasir_test.txt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from rgss.data import SegDataset, read_split
from rgss.segmentation import build_model, evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGSS: evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], ckpt["encoder"]).cuda()
    model.load_state_dict(ckpt["state_dict"])
    data = SegDataset(args.data_root, read_split(args.split), ckpt["image_size"])
    res = evaluate(model, DataLoader(data, batch_size=args.batch_size, num_workers=4))
    print(f"mIoU {res['mIoU']:.2f} | mDice {res['mDice']:.2f}")
