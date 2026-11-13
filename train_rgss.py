"""Train two segmentation networks with RGSS (Sec. 3) on mixed-quality annotations.

Epochs [0, e_0): both models are trained on the given annotations.
Epochs [e_0, e_max): every epoch, per-sample losses -> GMM reliabilities (from the peer) -> targets
(original / refined by R_phi with sample-dependent steps, blended with the peer prediction) -> one
epoch of Dice training of each model on its targets.
With --e0 equal to --e_max this reduces to plain training of the backbone on the given annotations.

Example:
    python train_rgss.py --data_root data/kvasir --split_prefix splits/kvasir --train_mask_dir masks_noise50 \
        --refiner checkpoints/refiner/kvasir/refiner.pth --arch unet --encoder <ENCODER> --image_size <S> \
        --batch_size <B> --lr <LR> --e0 <E0> --e_max <EMAX> --t_min <TMIN> --t_max <TMAX> --tau <TAU> --flip \
        --out_dir runs/kvasir_noise50/unet_rgss
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rgss.data import SegDataset, read_split
from rgss.method import clean_probability, construct_targets, per_sample_losses
from rgss.refiner.diffusion import load_refiner
from rgss.segmentation import ARCHS, build_model, dice_loss, evaluate
from rgss.seed import seed_everything


def train_epoch(model, optimizer, dataset, args, desc):
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    total = 0.0
    for b in tqdm(loader, desc=desc, leave=False):
        loss = dice_loss(model(b["image"].cuda()).sigmoid(), b["mask"].cuda())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() / len(loader)
    return total


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    ids = {s: read_split(f"{args.split_prefix}_{s}.txt") for s in ("train", "val", "test")}
    S = args.image_size
    train_set = SegDataset(args.data_root, ids["train"], S, mask_dir=args.train_mask_dir, flip=args.flip)  # training
    plain_set = SegDataset(args.data_root, ids["train"], S, mask_dir=args.train_mask_dir)  # losses / targets
    val_loader = DataLoader(SegDataset(args.data_root, ids["val"], S), batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(SegDataset(args.data_root, ids["test"], S), batch_size=args.batch_size, num_workers=args.num_workers)

    models = []
    for k in range(2):  # two networks with different initializations
        torch.manual_seed(args.seed + k)
        models.append(build_model(args.arch, args.encoder).cuda())
    optims = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in models]
    refiner = load_refiner(args.refiner) if args.e0 < args.e_max else None

    best, best_path, log = -1.0, os.path.join(args.out_dir, "best.pth"), []
    for epoch in range(args.e_max):
        rec = {"epoch": epoch}
        if epoch < args.e0:  # training on the given annotations
            train_set.targets = None
            for k in range(2):
                rec[f"loss_{k}"] = train_epoch(models[k], optims[k], train_set, args, f"epoch {epoch} M{k + 1}")
        else:
            losses = [per_sample_losses(m, plain_set, args.batch_size, args.num_workers) for m in models]
            alpha = (args.e_max - epoch) / (args.e_max - args.e0)
            for k in range(2):
                peer = 1 - k
                r = clean_probability(losses[peer], args.seed)  # reliability for M_k from the peer's losses
                train_set.targets = construct_targets(plain_set, r, models[peer], refiner, args.tau, args.t_min, args.t_max,
                                                      alpha, args.batch_size, args.num_workers)
                rec[f"loss_{k}"] = train_epoch(models[k], optims[k], train_set, args, f"epoch {epoch} M{k + 1}")
                rec[f"clean_ratio_{k}"] = float((r >= args.tau).mean())
            train_set.targets = None
        for k in range(2):
            val = evaluate(models[k], val_loader)
            rec[f"val_mIoU_{k}"] = val["mIoU"]
            if val["mIoU"] > best:
                best = val["mIoU"]
                torch.save({"arch": args.arch, "encoder": args.encoder, "image_size": S, "model": k + 1, "epoch": epoch,
                            "state_dict": models[k].state_dict()}, best_path)
        log.append(rec)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))

    ckpt = torch.load(best_path, map_location="cpu")
    model = build_model(ckpt["arch"], ckpt["encoder"]).cuda()
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader)
    print(f"Test (M{ckpt['model']}, epoch {ckpt['epoch']}): mIoU {test['mIoU']:.2f} | mDice {test['mDice']:.2f}")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"best_val_mIoU": best, "best_model": ckpt["model"], "best_epoch": ckpt["epoch"], "test": test, "log": log}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGSS training")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_prefix", required=True, help="e.g. splits/kvasir -> splits/kvasir_{train,val,test}.txt")
    p.add_argument("--train_mask_dir", required=True, help="directory (under data_root) of the training annotations, e.g. masks_noise50")
    p.add_argument("--refiner", default=None, help="pre-trained R_phi from train_refiner.py (required if e0 < e_max)")
    p.add_argument("--out_dir", required=True)
    g = p.add_argument_group("segmentation network")
    g.add_argument("--arch", choices=list(ARCHS), required=True)
    g.add_argument("--encoder", required=True, help="segmentation_models_pytorch encoder (randomly initialized)")
    g.add_argument("--image_size", type=int, required=True)
    g = p.add_argument_group("optimization (no defaults)")
    g.add_argument("--batch_size", type=int, required=True)
    g.add_argument("--lr", type=float, required=True)
    g.add_argument("--e0", type=int, required=True, help="epochs on the given annotations before RGSS")
    g.add_argument("--e_max", type=int, required=True, help="total epochs")
    g.add_argument("--t_min", type=int, required=True, help="T_min of Eq. (3)")
    g.add_argument("--t_max", type=int, required=True, help="T_max of Eq. (3), at most the refiner's number of steps")
    g.add_argument("--tau", type=float, required=True, help="reliability threshold of J_clean")
    g.add_argument("--flip", action="store_true", help="random horizontal / vertical flips")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.e0 < a.e_max and not a.refiner:
        p.error("--refiner is required when e0 < e_max")
    main(a)
