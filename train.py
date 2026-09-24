"""Treino de U-Net / DeepLabV3+ com segmentation_models_pytorch.

Uso:
    python train.py --data data --epochs 50
    python train.py --arch DeepLabV3Plus --encoder efficientnet-b3
"""
import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import seg_config as C
from dataset import MeatDataset, get_train_transform, get_val_transform


def build_model(arch, encoder, weights, n_classes):
    return smp.create_model(arch=arch, encoder_name=encoder,
                            encoder_weights=weights, in_channels=3,
                            classes=n_classes)


def run_epoch(model, loader, loss_fn, device, n_classes, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    tp = fp = fn = torch.zeros(n_classes, dtype=torch.long)
    for images, masks in tqdm(loader, leave=False, desc="treino" if train else "val"):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(train), \
             torch.autocast(device.type, enabled=scaler is not None):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if train:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        s_tp, s_fp, s_fn, _ = smp.metrics.get_stats(
            preds, masks, mode="multiclass", num_classes=n_classes)
        tp, fp, fn = tp + s_tp.sum(0).cpu(), fp + s_fp.sum(0).cpu(), fn + s_fn.sum(0).cpu()
    iou = tp / (tp + fp + fn).clamp(min=1)
    return total_loss / len(loader.dataset), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", default=C.ARCH)
    ap.add_argument("--encoder", default=C.ENCODER)
    ap.add_argument("--encoder-weights", default=C.ENCODER_WEIGHTS,
                    help="'imagenet' ou 'none'")
    ap.add_argument("--size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(C.CLASSES)
    weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    print(f"Dispositivo: {device} | {args.arch} + {args.encoder} | classes: {C.CLASSES}")

    train_ds = MeatDataset(Path(args.data) / "train", get_train_transform(args.size))
    val_ds = MeatDataset(Path(args.data) / "val", get_val_transform(args.size))
    train_dl = DataLoader(train_ds, args.batch, shuffle=True,
                          num_workers=C.NUM_WORKERS, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, args.batch, num_workers=C.NUM_WORKERS)

    model = build_model(args.arch, args.encoder, weights, n).to(device)

    dice = smp.losses.DiceLoss(mode="multiclass")
    ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda logits, y: dice(logits, y) + ce(logits, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(model, train_dl, loss_fn, device, n, optimizer, scaler)
        va_loss, iou = run_epoch(model, val_dl, loss_fn, device, n)
        scheduler.step()
        miou = iou[1:].mean().item()  # média sem o fundo
        per_class = " ".join(f"{c}={v:.3f}" for c, v in zip(C.CLASSES, iou.tolist()))
        print(f"[{epoch:03d}] loss treino={tr_loss:.4f} val={va_loss:.4f} "
              f"mIoU={miou:.3f} | {per_class}")

        ckpt = {"state_dict": model.state_dict(), "arch": args.arch,
                "encoder": args.encoder, "classes": C.CLASSES,
                "img_size": args.size, "epoch": epoch, "miou": miou}
        torch.save(ckpt, out / "last.pt")
        if miou > best:
            best = miou
            torch.save(ckpt, out / "best.pt")
            print(f"      novo melhor modelo salvo (mIoU={best:.3f})")


if __name__ == "__main__":
    main()
