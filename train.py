# train.py
import argparse, torch, math
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from collections import defaultdict
from typing import Optional, Tuple

from config import (
    EPOCHS, LR, WEIGHT_DECAY, CKPT_PATH, CLASSES, SEED,
    LABEL_SMOOTH, USE_AMP, MAX_GRAD_NORM
)
from data import make_dataloaders
from models import CNNLSTM
from utils import set_seed, accuracy, confusion_matrix, measure_latency

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k].copy_(v)

def unpack_batch(batch):
    """
    Accepts (seq, target) or (seq, target, lengths) or (seq, target, mask).
    Returns: seq, target, lengths, mask
    """
    if len(batch) == 2:
        return batch[0], batch[1], None, None
    if len(batch) == 3:
        # Heuristic: if third is 1D -> lengths; if 2D -> mask
        third = batch[2]
        if third.ndim == 1:
            return batch[0], batch[1], third, None
        else:
            return batch[0], batch[1], None, third
    if len(batch) == 4:
        return batch
    raise ValueError("Unexpected batch format")

def epoch_loop(model, loader, criterion, optimizer=None, device="cpu",
               scaler=None, scheduler=None, ema: EMA|None=None, train=True):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_logits, all_targets = [], []

    for batch in loader:
        seq, target, lengths, mask = unpack_batch(batch)
        seq, target = seq.to(device), target.to(device)
        lengths = lengths.to(device) if lengths is not None else None
        mask    = mask.to(device)    if mask    is not None else None

        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=USE_AMP and device.startswith("cuda")):
                logits = model(seq, lengths=lengths, mask=mask)
                loss = criterion(logits, target)

            if USE_AMP and device.startswith("cuda"):
                scaler.scale(loss).backward()
                if MAX_GRAD_NORM is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if MAX_GRAD_NORM is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            if scheduler is not None: scheduler.step()
            if ema is not None: ema.update(model)
        else:
            with torch.no_grad():
                logits = model(seq, lengths=lengths, mask=mask)
                loss = criterion(logits, target)

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), target)
        n_batches  += 1
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())

    avg_loss = total_loss / max(1, n_batches)
    avg_acc  = total_acc  / max(1, n_batches)
    return avg_loss, avg_acc, torch.cat(all_logits), torch.cat(all_targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--save", type=str, default=CKPT_PATH)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    args = parser.parse_args()

    set_seed(SEED)
    train_loader, val_loader, test_loader, (train_idx, _, _), (norm_mean, norm_std) = make_dataloaders()
    print(f"Using dataset normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # class weights (inverse frequency on TRAIN split)
    class_counts = defaultdict(int)
    for ci, _ in train_idx: class_counts[ci] += 1
    total = sum(class_counts.values())
    weights = [total / max(1, class_counts[c]) for c in range(len(CLASSES))]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = CNNLSTM(num_classes=len(CLASSES), in_ch=1).to(device)

    # optional torch.compile for speed (PyTorch 2.x)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile.")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler(enabled=USE_AMP and device.startswith("cuda"))
    ema = EMA(model, decay=0.999)

    best_val = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc, _, _ = epoch_loop(model, train_loader, criterion,
                                           optimizer, device, scaler, scheduler, ema, train=True)
        va_loss, va_acc, _, _ = epoch_loop(model, val_loader, criterion,
                                           optimizer=None, device=device, scaler=None, scheduler=None, ema=None, train=False)

        if va_acc > best_val:
            best_val = va_acc
            ema_model = CNNLSTM(num_classes=len(CLASSES), in_ch=1).to(device)
            ema.copy_to(ema_model)
            torch.save({"model": ema_model.state_dict(),
                        "classes": CLASSES,
                        "norm": {"mean": norm_mean, "std": norm_std},
                        "class_weights": weights},
                       args.save)

        print(f"[{ep:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | best_val={best_val:.4f}")

    # Load best EMA and test
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc, te_logits, te_targets = epoch_loop(model, test_loader, criterion,
                                                        optimizer=None, device=device, train=False)
    print(f"\nTest: loss={te_loss:.4f} acc={te_acc:.4f}")

    cm = confusion_matrix(te_logits, te_targets, num_classes=len(CLASSES))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{c[:12]:>12}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:16]:>16} | " + " ".join(f"{x:12d}" for x in row))

    # Latency
    test_ds = test_loader.dataset
    if len(test_ds) > 0:
        sample_seq, *rest = test_ds[0]          # support optional (seq, y, len/mask)
        sample_seq = sample_seq.unsqueeze(0)    # (1,T,1,H,W)
        ms = measure_latency(model, sample_seq, device=device, runs=50)
        print(f"\nCPU forward latency: {ms:.2f} ms / sample")

if __name__ == "__main__":
    main()
