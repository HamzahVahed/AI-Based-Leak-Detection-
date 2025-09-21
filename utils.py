# utils.py
import random, time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Union
from torchvision.transforms import functional as TF
import torch.nn.functional as F

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()

def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    return cm

def measure_latency(model: nn.Module, sample_seq: torch.Tensor, device="cpu", warmup=5, runs=50) -> float:
    model.eval(); sample_seq = sample_seq.to(device)
    with torch.no_grad():
        for _ in range(warmup): _ = model(sample_seq)
        if device.startswith("cuda"): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs): _ = model(sample_seq)
        if device.startswith("cuda"): torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (dt / runs) * 1000.0

# ---- normalization helpers ----
def _broadcast_params(param: Union[float, Sequence[float]], C: int, name: str):
    if isinstance(param, (int, float)): return (float(param),) * C
    try: seq = tuple(param)
    except TypeError: raise TypeError(f"{name} must be float or a sequence (len 1 or {C}).")
    if len(seq) == 1: return (float(seq[0]),) * C
    if len(seq) != C: raise ValueError(f"{name} length {len(seq)} != channels {C}.")
    return tuple(float(x) for x in seq)

def normalize_img(t: torch.Tensor, mean=0.5, std=0.5) -> torch.Tensor:
    # t: (C,H,W)
    if t.ndim != 3: raise ValueError(f"normalize_img expects (C,H,W), got {tuple(t.shape)}")
    C = t.shape[0]
    mean_t = _broadcast_params(mean, C, "mean")
    std_t  = _broadcast_params(std,  C, "std")
    return TF.normalize(t, mean=mean_t, std=std_t)

# ---- sequence slicing along time (width) ----
def slice_time_axis(img_t: torch.Tensor, seq_len: int, out_hw: Tuple[int,int]) -> torch.Tensor:
    # img_t: (C,H,W) -> (T,C,out_h,out_w)
    if img_t.ndim != 3: raise ValueError(f"slice_time_axis expects (C,H,W), got {tuple(img_t.shape)}")
    C, H, W = img_t.shape
    if W < seq_len:
        img_t = F.interpolate(img_t.unsqueeze(0), size=(H, seq_len),
                              mode="bilinear", align_corners=False, antialias=True).squeeze(0)
        W = seq_len
    base_w = W // seq_len; extra = W % seq_len
    widths = [base_w + (1 if i < extra else 0) for i in range(seq_len)]
    slices, x = [], 0
    for w_i in widths:
        seg = img_t[:, :, x:x+w_i]; x += w_i
        seg = TF.resize(seg, out_hw, antialias=True)
        slices.append(seg.unsqueeze(0))
    return torch.cat(slices, dim=0)

# ---- SpecAugment-like masking on (C,H,W) tensor ----
def rand_mask_1d(length: int, max_frac: float):
    w = max(1, int(round(length * max_frac)))
    if w >= length: return 0, length
    s = random.randint(0, length - w)
    return s, s + w

def spec_augment_(img_chw: torch.Tensor, time_masks: int, time_w: float,
                  freq_masks: int, freq_h: float):
    # img_chw: (C,H,W), mask all channels for the selected regions (we use C=1)
    assert img_chw.ndim == 3, "Expect (C,H,W)"
    _, H, W = img_chw.shape
    for _ in range(time_masks):
        xs, xe = rand_mask_1d(W, time_w); img_chw[:, :, xs:xe] = 0.0
    for _ in range(freq_masks):
        ys, ye = rand_mask_1d(H, freq_h); img_chw[:, ys:ye, :] = 0.0

# ---- circular time shift (for TTA) ----
def circular_time_shift(img_chw: torch.Tensor, shift: int) -> torch.Tensor:
    assert img_chw.ndim == 3, f"expected (C,H,W), got {tuple(img_chw.shape)}"
    C, H, W = img_chw.shape
    if W == 0 or shift % W == 0:
        return img_chw
    s = shift % W
    return torch.cat([img_chw[:, :, -s:], img_chw[:, :, :-s]], dim=2)
