# plot_eval_artifacts.py
"""
Generates:
- Confusion matrix (raw + normalized)
- ROC curves (one-vs-rest, micro & macro AUC)
- t-SNE scatter of LSTM features (test set by default)
- Training & validation curves (if outputs/history.json exists)

Usage:
  python plot_eval_artifacts.py --ckpt cnn_lstm_cwt.pt --split test
Options:
  --split  {train,val,test,all}  (default: test)
  --tsne_split {test,all}        (default: test)
  --perplexity 30                (t-SNE perplexity)
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F

from config import CLASSES, CKPT_PATH
from data import make_dataloaders
from models import CNNLSTM

def ensure_out():
    os.makedirs("outputs", exist_ok=True)

def get_loader(split):
    train_loader, val_loader, test_loader, _ = make_dataloaders()
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    if split == "all":
        # concat by iterating; we'll just consume each to arrays
        return None, train_loader, val_loader, test_loader
    raise ValueError("split must be one of train/val/test/all")

def collect_logits_targets_features(model, loader, device):
    all_logits, all_targets, all_feats = [], [], []
    model.eval()
    with torch.no_grad():
        for seq, target in loader:
            seq = seq.to(device)
            target = target.to(device)
            # Rebuild penultimate features: CNN -> LSTM -> last step
            B, T, C, H, W = seq.shape
            feats = model.cnn(seq.view(B*T, C, H, W)).view(B, T, -1)
            lstm_out, _ = model.lstm(feats)
            last = lstm_out[:, -1, :]                 # (B, D)
            logits = model.head(last)                 # (B, num_classes)
            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
            all_feats.append(last.cpu())
    return (torch.cat(all_logits), torch.cat(all_targets), torch.cat(all_feats))

def softmax_np(logits):
    return F.softmax(logits, dim=1).cpu().numpy()

def plot_confusion(cm, classes, title_suffix=""):
    # raw
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    plt.tight_layout(); plt.ylabel("True label"); plt.xlabel("Predicted label")
    out = f"outputs/confusion_matrix{title_suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # normalized (row-wise)
    cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    plt.figure()
    plt.imshow(cmn, interpolation="nearest")
    plt.title(f"Confusion Matrix (Normalized) {title_suffix}")
    plt.colorbar()
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    plt.tight_layout(); plt.ylabel("True label"); plt.xlabel("Predicted label")
    outn = f"outputs/confusion_matrix_normalized{title_suffix}.png"
    plt.savefig(outn, bbox_inches="tight"); plt.close()
    print(f"Saved: {outn}")

def plot_roc_curves(probs, y_true, classes, title_suffix=""):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))  # shape (N, C)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC = {roc_auc['macro']:.3f})")
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves {title_suffix}")
    plt.legend(loc="lower right", fontsize="small")
    out = f"outputs/roc_curves{title_suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    with open(f"outputs/roc_auc{title_suffix}.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k,v in roc_auc.items()}, f, indent=2)
    print(f"Saved: outputs/roc_auc{title_suffix}.json")

def plot_tsne(features, labels, classes, title_suffix="", perplexity=30):
    # features: (N, D) numpy; labels: (N,)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    emb = tsne.fit_transform(features)
    plt.figure()
    for ci, cls in enumerate(classes):
        idx = (labels == ci)
        plt.scatter(emb[idx,0], emb[idx,1], s=14, alpha=0.8, label=cls)
    plt.legend(markerscale=1.5, fontsize="small")
    plt.title(f"t-SNE of LSTM Features {title_suffix}")
    plt.tight_layout()
    out = f"outputs/tsne{title_suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

def try_plot_history():
    hist_path = Path("outputs/history.json")
    if not hist_path.exists():
        print("No outputs/history.json found (training curves). "
              "Patch train.py to save history or use the log parser (see below).")
        return
    with hist_path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    epochs = np.arange(1, len(hist["train_loss"])+1)
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/loss_curve.png", bbox_inches="tight"); plt.close()
    print("Saved: outputs/loss_curve.png")

    plt.figure()
    plt.plot(epochs, hist["train_acc"], label="train_acc")
    plt.plot(epochs, hist["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training/Validation Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/accuracy_curve.png", bbox_inches="tight"); plt.close()
    print("Saved: outputs/accuracy_curve.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test","all"])
    ap.add_argument("--tsne_split", type=str, default="test", choices=["test","all"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    ensure_out()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 1) Confusion matrix + ROC on chosen split
    if args.split != "all":
        loader = get_loader(args.split)
        logits, targets, feats = collect_logits_targets_features(model, loader, device)
        probs = softmax_np(logits)
        y_true = targets.numpy()
        cm = confusion_matrix(y_true, probs.argmax(axis=1))
        plot_confusion(cm, CLASSES, title_suffix=f" ({args.split})")
        plot_roc_curves(probs, y_true, CLASSES, title_suffix=f" ({args.split})")
    else:
        # all: aggregate train+val+test
        _, tr, va, te = get_loader("all")
        all_logits, all_targets = [], []
        for loader in (tr, va, te):
            logits, targets, _ = collect_logits_targets_features(model, loader, device)
            all_logits.append(logits); all_targets.append(targets)
        logits = torch.cat(all_logits); targets = torch.cat(all_targets)
        probs = softmax_np(logits)
        y_true = targets.numpy()
        cm = confusion_matrix(y_true, probs.argmax(axis=1))
        plot_confusion(cm, CLASSES, title_suffix=" (all)")
        plot_roc_curves(probs, y_true, CLASSES, title_suffix=" (all)")

    # 2) t-SNE on features (test or all)
    if args.tsne_split == "test":
        loader = get_loader("test")
        _, targets, feats = collect_logits_targets_features(model, loader, device)
        plot_tsne(feats.numpy(), targets.numpy(), CLASSES, title_suffix=" (test)", perplexity=args.perplexity)
    else:
        _, tr, va, te = get_loader("all")
        feats_list, labels_list = [], []
        for loader in (tr, va, te):
            _, t, f = collect_logits_targets_features(model, loader, device)
            feats_list.append(f.numpy()); labels_list.append(t.numpy())
        feats = np.concatenate(feats_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        plot_tsne(feats, labels, CLASSES, title_suffix=" (all)", perplexity=args.perplexity)

    # 3) Training curves (if history.json exists)
    try_plot_history()

if __name__ == "__main__":
    main()
