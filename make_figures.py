# make_figures.py
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from config import CLASSES  # uses your canonical class order


# ---------- utils ----------
def ensure_outdir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def infer_true_from_path(p: str, classes):
    parts = Path(p).parts
    # try exact folder match (case-insensitive) walking upward
    for name in reversed(parts[:-1]):
        for c in classes:
            if name.lower() == c.lower():
                return c
    # fallback to parent folder name
    return parts[-2]

def normalize_class_name(name: str):
    # unifies spacing/case; keeps hyphens/words as in your class names
    return name.strip().lower()

def build_probs_matrix(df: pd.DataFrame, classes):
    """
    Returns (probs NxK, y_pred_str list).
    Accepts:
      - probability columns named 'p_<ClassName>' for any subset of classes
      - 'pred_prob' as confidence for predicted class
      - predicted class in 'pred' or 'pred_class' or 'pred_idx'
    Ensures each row sums ~1.0.
    """
    K = len(classes)
    # Figure out predicted class
    if "pred" in df.columns:
        y_pred = df["pred"].astype(str).tolist()
    elif "pred_class" in df.columns:
        y_pred = df["pred_class"].astype(str).tolist()
    elif "pred_idx" in df.columns:
        idx = df["pred_idx"].astype(int).tolist()
        y_pred = [classes[i] if 0 <= i < K else classes[0] for i in idx]
    else:
        raise ValueError("CSV must include one of: pred, pred_class, or pred_idx")

    # Map available probability columns
    colmap = {}  # class_name (canonical) -> column name
    norm_classes = [normalize_class_name(c) for c in classes]

    for col in df.columns:
        if not col.startswith("p_"):
            continue
        raw = col[2:]  # after p_
        # best-effort match to a canonical class
        raw_norm = normalize_class_name(raw)
        if raw_norm in norm_classes:
            j = norm_classes.index(raw_norm)
            colmap[classes[j]] = col
        else:
            # try exact (case-insensitive) match without normalization quirks
            for c in classes:
                if raw.lower() == c.lower():
                    colmap[c] = col
                    break

    probs = np.zeros((len(df), K), dtype=np.float64)

    # Fill from available p_<class> columns
    for j, c in enumerate(classes):
        if c in colmap:
            try:
                probs[:, j] = df[colmap[c]].astype(float).values
            except Exception:
                pass  # leave zeros if parsing fails

    # Handle missing columns/classes
    missing = [c for c in classes if c not in colmap]
    if missing:
        # if we have 'pred_prob', use it for the predicted class, distribute remainder
        if "pred_prob" in df.columns:
            conf = df["pred_prob"].astype(float).clip(0.0, 1.0).values
            for i, pc in enumerate(y_pred):
                if pc in classes:
                    j = classes.index(pc)
                    # if class already has a prob, keep the max of the two
                    probs[i, j] = max(probs[i, j], conf[i])
            # distribute the remaining mass uniformly over others not already filled
            row_sums = probs.sum(axis=1, keepdims=True)
            rem = np.maximum(0.0, 1.0 - row_sums.squeeze())
            # spread over non-pred classes (or all others)
            for i in range(len(df)):
                j_pred = classes.index(y_pred[i]) if y_pred[i] in classes else 0
                denom = K - 1
                add = (rem[i] / denom) if denom > 0 else 0.0
                for j in range(K):
                    if j != j_pred:
                        probs[i, j] += add
        else:
            # fallback: one-hot on predicted class
            for i, pc in enumerate(y_pred):
                if pc in classes:
                    probs[i, classes.index(pc)] = 1.0

    # Final normalize each row to sum ~1 (avoid div-by-zero)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums
    return probs, y_pred


# ---------- plots (paper style) ----------
def save_confusion_matrix(figpath, cm, labels):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size": 20, "fontweight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=14, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()

def save_roc_curves(figpath, y_true, probs, labels):
    y_bin = label_binarize(y_true, classes=labels)
    plt.figure(figsize=(10, 8))
    for i, c in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{c} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()

def save_tsne(figpath, embeds, y_true, labels):
    N = embeds.shape[0]
    perplexity = max(5, min(30, (N - 1) // 3))
    tsne = TSNE(n_components=2, init="pca", random_state=42,
                perplexity=perplexity, learning_rate="auto")
    xy = tsne.fit_transform(embeds)

    marker_list = ['o', 's', '^', 'v', 'D', '*', 'X']
    color_list = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']

    label_to_idx = {c: i for i, c in enumerate(labels)}
    idxs = np.array([label_to_idx[s] for s in y_true])

    plt.figure(figsize=(12, 10))
    for i, c in enumerate(labels):
        m = marker_list[i % len(marker_list)]
        col = color_list[i % len(color_list)]
        sel = (idxs == i)
        plt.scatter(xy[sel, 0], xy[sel, 1], marker=m, color=col, alpha=0.7, label=c, s=30)

    plt.legend(title="Classes", loc='upper right',
               prop={'weight': 'bold', 'size': 12}, title_fontsize=13)
    plt.title('t-SNE of Hybrid Model Features (2D)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Make paper-style figures from an existing predictions CSV.")
    ap.add_argument("--csv", required=True, help="Path to predictions CSV (from batch_infer.py)")
    ap.add_argument("--outdir", default="figures", help="Directory to save figures/metrics")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    # Path column name
    path_col = "path" if "path" in df.columns else None
    if path_col is None:
        # try to guess a plausible column
        for c in df.columns:
            if "path" in c.lower() or "file" in c.lower():
                path_col = c
                break
    if path_col is None:
        raise ValueError("Could not find a path column in CSV. Expected a column named 'path'.")

    # true labels
    if "true" in df.columns:
        y_true = df["true"].astype(str).tolist()
    else:
        y_true = [infer_true_from_path(p, CLASSES) for p in df[path_col].astype(str).tolist()]

    # probs & predicted labels
    probs, y_pred = build_probs_matrix(df, CLASSES)

    # Metrics
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    overall_acc = float(np.trace(cm)) / max(1, float(np.sum(cm)))

    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        f.write("Classification Report (precision/recall/F1 per class):\n")
        f.write(report)

    # Figures
    save_confusion_matrix(os.path.join(args.outdir, "confusion_matrix.png"), cm, CLASSES)
    save_roc_curves(os.path.join(args.outdir, "roc_curves.png"), y_true, probs, CLASSES)
    save_tsne(os.path.join(args.outdir, "tsne.png"), probs, y_true, CLASSES)

    print("\n=== Summary ===")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print("Saved:", os.path.join(args.outdir, "metrics.txt"))
    print("Saved:", os.path.join(args.outdir, "confusion_matrix.png"))
    print("Saved:", os.path.join(args.outdir, "roc_curves.png"))
    print("Saved:", os.path.join(args.outdir, "tsne.png"))


if __name__ == "__main__":
    main()
