# infer.py
import argparse, torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from config import CLASSES, CKPT_PATH, SEQ_LEN, IMG_SEG_SIZE, TTA_SHIFTS
from models import CNNLSTM
from utils import normalize_img, slice_time_axis, circular_time_shift

def load_gray(p): return Image.open(p).convert("L")

def build_seq_single(img_path: str, mean: float, std: float) -> torch.Tensor:
    arr = np.array(load_gray(img_path), dtype=np.float32) / 255.0  # (H,W)
    t = torch.from_numpy(arr[None, ...])  # (1,H,W)
    t = normalize_img(t, mean=mean, std=std)
    seq = slice_time_axis(t, SEQ_LEN, (IMG_SEG_SIZE, IMG_SEG_SIZE))  # (T,1,Hs,Ws)
    return seq

def tta_single(seq: torch.Tensor, model: torch.nn.Module, shifts=TTA_SHIFTS) -> torch.Tensor:
    T, C, Hs, Ws = seq.shape
    wide = seq.permute(1,2,0,3).contiguous().view(1, Hs, T*Ws)  # (1,H,WT)
    logits_list = []
    device = next(model.parameters()).device
    for s in shifts:
        shifted = circular_time_shift(wide, s)  # (1,H,WT)
        seq_s = slice_time_axis(shifted, SEQ_LEN, (Hs, Ws))  # shifted is (C,H,WT); returns (T,1,Hs,Ws)

        with torch.no_grad():
            logits = model(seq_s.unsqueeze(0).to(device))  # (1,num_classes)
        logits_list.append(logits.cpu())
    return torch.stack(logits_list, dim=0).mean(dim=0).squeeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH)
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    norm = ckpt.get("norm", {"mean": 0.5, "std": 0.5})
    mean, std = float(norm["mean"]), float(norm["std"])

    model = CNNLSTM(num_classes=len(CLASSES), in_ch=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seq = build_seq_single(args.img, mean, std)
    if args.tta:
        logits = tta_single(seq, model)
    else:
        with torch.no_grad():
            logits = model(seq.unsqueeze(0).to(device)).cpu().squeeze(0)
    probs = F.softmax(logits, dim=0).numpy()

    top_idx = probs.argsort()[::-1][:args.topk]
    print("\nPrediction:")
    for i in top_idx:
        print(f"{CLASSES[i]:>22}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
