# models.py
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from config import DROPOUT, LSTM_HIDDEN, LSTM_BIDIR, CLASSES, FEAT_DIM, USE_ATTENTION_POOL

# ---- Small helpers ----
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (B,T,H), mask: (B,T) with 1 for valid, 0 for pad
    returns: (B,H)
    """
    if mask is None:
        return x.mean(dim=1)
    w = mask.to(x.dtype).unsqueeze(-1)               # (B,T,1)
    s = (x * w).sum(dim=1)                           # (B,H)
    d = w.sum(dim=1).clamp_min(1e-6)                 # (B,1)
    return s / d

# ---- Feature extractor ----
class ConvBNAct(nn.Module):
    """Conv2d -> BN -> GELU (bias=False since BN has affine)"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CNNFeatureExtractor(nn.Module):
    """(B,in_ch,H,W) -> (B,feat_dim)"""
    def __init__(self, in_ch: int = 1, feat_dim: int = FEAT_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, 32),   ConvBNAct(32, 32),   nn.MaxPool2d(2),
            ConvBNAct(32, 64),      ConvBNAct(64, 64),   nn.MaxPool2d(2),
            ConvBNAct(64, 128),     ConvBNAct(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, feat_dim, bias=True),
            nn.GELU(),  # keep negative information (GELU > ReLU for features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---- Temporal attention with optional mask ----
class TemporalAttentionPool(nn.Module):
    """
    Additive attention with optional mask and temperature scaling.
    x: (B,T,H) -> (B,H)
    mask: (B,T) with 1 for valid, 0 for pad (optional)
    """
    def __init__(self, hid: int):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=True)
        self.v = nn.Linear(hid, 1,  bias=False)
        # learnable temperature (starts near 1.0)
        self.log_tau = nn.Parameter(torch.zeros(1))  # tau = exp(log_tau)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = torch.tanh(self.W(x))        # (B,T,H)
        e = self.v(s).squeeze(-1)        # (B,T)
        tau = self.log_tau.exp().clamp_min(1e-2)
        e = e / tau
        if mask is not None:
            # set attention logits to -inf for padded steps
            e = e.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(e, dim=1)      # (B,T)
        return (x * a.unsqueeze(-1)).sum(dim=1)  # (B,H)

# ---- Main model ----
class CNNLSTM(nn.Module):
    """
    CNN over slices + LSTM over time + attention (or masked mean) pooling.
    Input:  x  (B,T,C,H,W) [C=1]
            lengths (optional) tensor of shape (B,) with valid lengths per sequence
            OR mask (B,T) with 1 valid, 0 pad
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=len(CLASSES), feat_dim=FEAT_DIM, hidden=LSTM_HIDDEN,
                 bidir=LSTM_BIDIR, dropout=DROPOUT, in_ch=1, use_attention: bool = None):
        super().__init__()
        if use_attention is None:
            use_attention = USE_ATTENTION_POOL
        self.use_attention = use_attention

        self.cnn = CNNFeatureExtractor(in_ch=in_ch, feat_dim=feat_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=bidir)

        lstm_out_dim = hidden * (2 if bidir else 1)
        self.norm = nn.LayerNorm(lstm_out_dim)  # stabilizes sequence features
        self.attn = TemporalAttentionPool(lstm_out_dim) if self.use_attention else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_out_dim, num_classes))

        self._init_lstm()

    def _init_lstm(self):
        # Orthogonal init for recurrent matrices, kaiming for input projections
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # small forget gate bias helps (optional)
        for names in ["bias_ih_l0", "bias_hh_l0"]:
            if hasattr(self.lstm, names):
                b = getattr(self.lstm, names)
                n = b.numel() // 4
                b.data[n:2*n].fill_(1.0)  # forget gate

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B,T,C,H,W)
        lengths: (B,) optional; if provided we will pack the sequence for LSTM
        mask: (B,T) optional (1=valid, 0=pad). If lengths is given and mask is None, mask is derived.
        """
        import math  # lazy import to keep header light
        B, T, C, H, W = x.shape
        # safer than view for non-contiguous tensors
        x = x.reshape(B * T, C, H, W)
        feats = self.cnn(x)                    # (B*T, feat_dim)
        feats = feats.reshape(B, T, -1)        # (B, T, feat_dim)

        # derive mask from lengths if not provided
        if lengths is not None and mask is None:
            device = lengths.device
            rng = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
            mask = (rng < lengths.unsqueeze(1)).to(feats.dtype)  # (B,T)

        # pack for LSTM if lengths available (efficiency + correctness)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=T)  # (B,T,H*)
        else:
            lstm_out, _ = self.lstm(feats)  # (B,T,H*)

        lstm_out = self.norm(lstm_out)

        if self.use_attention:
            pooled = self.attn(lstm_out, mask=mask)  # (B,H*)
        else:
            pooled = masked_mean(lstm_out, mask=mask)  # (B,H*)

        logits = self.head(pooled)  # (B,num_classes)
        return logits
