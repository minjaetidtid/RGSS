"""Pre-train the diffusion-based mask refiner R_phi (Sec. 2, following SegRefiner) on the given masks.

For every training mask y, a coarse mask y_T is synthesized with the structured corruption of
rgss/corruption.py; R_phi learns to recover y from y_t = s_t y + (1 - s_t) y_T (Eq. 1) and the image.

Example:
    python train_refiner.py --data_root data/kvasir --split splits/kvasir_train.txt --mask_dir masks_noise50 \
        --num_steps <K> --beta_start <B0> --beta_stop <B1> --image_size <S> --model_channels <C> --num_res_blocks <R> \
        --num_heads <H> --attention_strides <A ...> --channel_mult <M ...> --dropout <P> \
        --coarse_min_iou <IOU> --coarse_max_iters <N> --lr <LR> --batch_size <B> --iterations <IT> --texture_weight <W> \
        --out checkpoints/refiner/kvasir/refiner.pth
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from rgss.corruption import corrupt_mask
from rgss.data import SegDataset, read_split
from rgss.refiner.diffusion import build_refiner, save_refiner
from rgss.seed import seed_everything


class CoarsePairs(Dataset):
    def __init__(self, base, min_iou, max_iters, seed):
        self.base, self.min_iou, self.max_iters, self.seed = base, min_iou, max_iters, seed
        self.epoch = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        coarse = corrupt_mask(b["mask"][0].numpy().astype(np.uint8), rng, self.min_iou, self.max_iters)
        return b["image"], b["mask"], torch.from_numpy(coarse.astype(np.float32))[None]


def main(args):
    seed_everything(args.seed)
    cfg = dict(num_steps=args.num_steps, start=args.beta_start, stop=args.beta_stop, image_size=args.image_size,
               model_channels=args.model_channels, num_res_blocks=args.num_res_blocks, num_heads=args.num_heads,
               attention_strides=args.attention_strides, channel_mult=args.channel_mult, dropout=args.dropout)
    model = build_refiner(cfg).cuda().train()
    print(f"R_phi: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    base = SegDataset(args.data_root, read_split(args.split), args.image_size, mask_dir=args.mask_dir, flip=True)
    data = CoarsePairs(base, args.coarse_min_iou, args.coarse_max_iters, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def batches():
        while True:
            for b in DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers):
                yield b
            data.epoch += 1

    it = batches()
    pbar = trange(1, args.iterations + 1, desc="refiner")
    for step in pbar:
        img, y0, coarse = (x.cuda() for x in next(it))
        loss, _ = model.training_loss(img, y0, coarse, args.texture_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"refiner | loss {loss.item():.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_refiner(args.out, model, {"args": vars(args)})
    print(f"Saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGSS: pre-train the diffusion mask refiner")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--mask_dir", required=True, help="masks used for pre-training (the given annotations)")
    p.add_argument("--out", required=True)
    g = p.add_argument_group("discrete diffusion")
    g.add_argument("--num_steps", type=int, required=True, help="K, number of diffusion steps")
    g.add_argument("--beta_start", type=float, required=True, help="probability of a fine state at the first step")
    g.add_argument("--beta_stop", type=float, required=True, help="probability of a fine state at the last step")
    g = p.add_argument_group("denoising UNet")
    g.add_argument("--image_size", type=int, required=True)
    g.add_argument("--model_channels", type=int, required=True)
    g.add_argument("--num_res_blocks", type=int, required=True)
    g.add_argument("--num_heads", type=int, required=True)
    g.add_argument("--attention_strides", type=int, nargs="+", required=True, help="downsampling rates with attention, e.g. 16 32")
    g.add_argument("--channel_mult", type=int, nargs="+", required=True)
    g.add_argument("--dropout", type=float, required=True)
    g = p.add_argument_group("coarse masks and optimization")
    g.add_argument("--coarse_min_iou", type=float, required=True, help="lower bound of the IoU of a synthesized coarse mask")
    g.add_argument("--coarse_max_iters", type=int, required=True)
    g.add_argument("--lr", type=float, required=True)
    g.add_argument("--batch_size", type=int, required=True)
    g.add_argument("--iterations", type=int, required=True)
    g.add_argument("--texture_weight", type=float, required=True, help="weight of the Sobel texture L1 loss")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
