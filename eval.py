# eval.py
import argparse, torch
import torch.nn as nn

from config import CKPT_PATH, CLASSES, SEQ_LEN, TTA_SHIFTS
from data import make_dataloaders
from models import CNNLSTM
from utils import accuracy, confusion_matrix, circular_time_shift, slice_time_axis

def tta_logits(model, seq_batch, tta_shifts=TTA_SHIFTS):
    device = seq_batch.device
    B, T, C, Hs, Ws = seq_batch.shape
    # (B,T,C,Hs,Ws) -> (B,C,Hs,T*Ws)
    wide = seq_batch.permute(0,2,3,1,4).contiguous().view(B, C, Hs, T*Ws)
    all_logits = []
    for s in tta_shifts:
        shifted = torch.stack([circular_time_shift(wide[i], s) for i in range(B)], dim=0)
        seqs = []
        for i in range(B):
            seq = slice_time_axis(shifted[i], SEQ_LEN, (Hs, Ws))  # (T,1,Hs,Ws)
            seqs.append(seq.unsqueeze(0))
        seqs = torch.cat(seqs, dim=0).to(device)  # (B,T,1,Hs,Ws)
        with torch.no_grad():
            logits = model(seqs)
        all_logits.append(logits)
    return torch.stack(all_logits, dim=0).mean(dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--tta", type=int, default=1)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, _, _ = make_dataloaders()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model = CNNLSTM(num_classes=len(CLASSES), in_ch=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    def run(loader, name):
        total_loss, total_acc, n = 0.0, 0.0, 0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for seq, target in loader:
                seq, target = seq.to(device), target.to(device)
                logits = tta_logits(model, seq) if args.tta else model(seq)
                loss = criterion(logits, target)
                total_loss += loss.item()
                total_acc  += accuracy(logits, target)
                n += 1
                all_logits.append(logits.cpu())
                all_targets.append(target.cpu())
        avg_loss = total_loss / max(n, 1)
        avg_acc  = total_acc  / max(n, 1)
        print(f"{name}: loss={avg_loss:.4f} acc={avg_acc:.4f}")
        return torch.cat(all_logits), torch.cat(all_targets)

    tr_logits, tr_targets = run(train_loader, "Train")
    va_logits, va_targets = run(val_loader,   "Val")
    te_logits, te_targets = run(test_loader,  "Test")

    cm = confusion_matrix(te_logits, te_targets, num_classes=len(CLASSES))
    print("\nTest Confusion Matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{c[:12]:>12}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:16]:>16} | " + " ".join(f"{x:12d}" for x in row))

if __name__ == "__main__":
    main()
