# plot_history_from_log.py
import re, os
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "outputs/train_log.txt"
OUT_DIR = "outputs"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ep_pat = re.compile(
        r"\[(\d+)/(?:\d+)\]\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
    )
    epochs, tr_loss, tr_acc, va_loss, va_acc = [], [], [], [], []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            m = ep_pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_acc.append(float(m.group(3)))
                va_loss.append(float(m.group(4)))
                va_acc.append(float(m.group(5)))

    if not epochs:
        print("No epoch lines found. Make sure outputs/train_log.txt contains the training prints.")
        return

    epochs = np.array(epochs)
    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve_from_log.png"), bbox_inches="tight"); plt.close()
    print("Saved: outputs/loss_curve_from_log.png")

    plt.figure()
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training/Validation Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_curve_from_log.png"), bbox_inches="tight"); plt.close()
    print("Saved: outputs/accuracy_curve_from_log.png")

if __name__ == "__main__":
    main()
