import argparse, csv, os, glob, torch, torch.nn.functional as F
from PIL import Image
import numpy as np

from config import CLASSES, CKPT_PATH, SEQ_LEN, IMG_SEG_SIZE, TTA_SHIFTS
from models import CNNLSTM
from utils import normalize_img, slice_time_axis, circular_time_shift

def load_gray(p): return Image.open(p).convert("L")

def build_seq(img_path: str, mean: float, std: float) -> torch.Tensor:
    arr = np.array(load_gray(img_path), dtype=np.float32) / 255.0  # (H,W)
    t = torch.from_numpy(arr[None, ...])  # (1,H,W)
    t = normalize_img(t, mean=mean, std=std)
    seq = slice_time_axis(t, SEQ_LEN, (IMG_SEG_SIZE, IMG_SEG_SIZE))  # (T,1,Hs,Ws)
    return seq  # (T,1,Hs,Ws)

@torch.no_grad()
def tta_logits(seq: torch.Tensor, model: torch.nn.Module, shifts=TTA_SHIFTS) -> torch.Tensor:
    # seq: (T,1,Hs,Ws) -> logits (num_classes,)
    T, C, Hs, Ws = seq.shape
    wide = seq.permute(1,2,0,3).contiguous().view(C, Hs, T*Ws)  # (1,Hs, WT)
    outs = []
    dev = next(model.parameters()).device
    for s in shifts:
        shifted = circular_time_shift(wide, s)                   # (1,Hs,WT)
        seq_s = slice_time_axis(shifted, SEQ_LEN, (Hs, Ws))      # (T,1,Hs,Ws)
        logits = model(seq_s.unsqueeze(0).to(dev)).cpu().squeeze(0)
        outs.append(logits)
    return torch.stack(outs, 0).mean(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_PATH, type=str)
    ap.add_argument("--glob", required=True, help="Glob pattern for images, e.g. \"E:\\data\\**\\*.png\"")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    norm = ckpt.get("norm", {"mean": 0.5, "std": 0.5})
    mean, std = float(norm["mean"]), float(norm["std"])

    model = CNNLSTM(num_classes=len(CLASSES), in_ch=1).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        print(f"[ERROR] No files matched: {args.glob}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["path", "pred_idx", "pred_class", "pred_prob"] + \
                 [f"p_{c}" for c in CLASSES[:args.topk]]
        w.writerow(header)

        for i, p in enumerate(files, 1):
            try:
                seq = build_seq(p, mean, std)
                logits = tta_logits(seq, model) if args.tta else model(seq.unsqueeze(0).to(device)).cpu().squeeze(0)
                probs = F.softmax(logits, dim=0).numpy()

                top_idx = probs.argsort()[::-1][:args.topk]
                pred_idx = int(top_idx[0])
                pred_cls = CLASSES[pred_idx]
                pred_prob = float(probs[pred_idx])

                row = [p, pred_idx, pred_cls, f"{pred_prob:.4f}"] + [f"{probs[j]:.4f}" for j in top_idx]
                w.writerow(row)
                if i % 50 == 0:
                    print(f"Processed {i}/{len(files)}")
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
