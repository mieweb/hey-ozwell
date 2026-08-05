#!/usr/bin/env python3
"""
Generate evaluation visuals for the wake-word models:
  1. score distributions (positives vs negatives) — shows separation / bimodality
  2. precision-recall curves with max-F1 marked
  3. confusion matrices at threshold 0.5

Reuses WakeWordEvaluator (correct resampling included). SYNTHETIC test audio —
numbers are an optimistic ceiling; per-clip FPR != FP/hour.

Usage (after fetching pretrained models + extracting clips, see README):
  python plot_eval.py --pretrained-dir /tmp/hey-eval --outdir figures \
      --phrase "hey ozwell"      ../../prod/js/models/hey-ozwell.onnx       /tmp/hey-eval/pos      /tmp/hey-eval/neg \
      --phrase "ozwell i'm done" "../../prod/js/models/ozwell-i'm-done.onnx" /tmp/hey-eval/done_pos /tmp/hey-eval/done_neg
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_wakeword import WakeWordEvaluator


def pr_curve(pos, neg, n=200):
    ts = np.linspace(0, 1, n)
    P, R, F = [], [], []
    for t in ts:
        tp = (pos >= t).sum(); fn = (pos < t).sum(); fp = (neg >= t).sum()
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        P.append(prec); R.append(rec); F.append(f1)
    return ts, np.array(P), np.array(R), np.array(F)


def confusion(pos, neg, t=0.5):
    tp = int((pos >= t).sum()); fn = int((pos < t).sum())
    fp = int((neg >= t).sum()); tn = int((neg < t).sum())
    return np.array([[tp, fn], [fp, tn]])  # rows: actual phrase / not; cols: detected / not


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-dir", default="/tmp/hey-eval")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--phrase", action="append", nargs=4, metavar=("LABEL", "MODEL", "POS", "NEG"),
                    required=True, help="repeatable: LABEL MODEL_ONNX POS_DIR NEG_DIR")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = []
    for label, model, posd, negd in args.phrase:
        ev = WakeWordEvaluator(model, args.pretrained_dir)
        pos, neg = ev.score_folder(posd), ev.score_folder(negd)
        data.append((label, pos, neg))
        ts, P, R, F = pr_curve(pos, neg)
        bi = ts[np.argmax(F)]
        print(f"[{label}] recall@0.5={(pos>=0.5).mean()*100:.0f}%  per-clip FPR@0.5={(neg>=0.5).mean()*100:.0f}%  "
              f"max-F1={F.max():.2f}@thr={bi:.2f}")

    n = len(data)
    # ---- 1. score distributions ----
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    bins = np.linspace(0, 1, 26)
    for ax, (label, pos, neg) in zip(axes[0], data):
        ax.hist(neg, bins=bins, alpha=0.6, label=f"negatives (n={len(neg)})", color="#d1495b")
        ax.hist(pos, bins=bins, alpha=0.6, label=f"positives (n={len(pos)})", color="#2e8b57")
        ax.axvline(0.5, ls="--", c="gray", lw=1, label="threshold 0.5")
        ax.set_title(label); ax.set_xlabel("max wake-word probability"); ax.set_ylabel("# clips"); ax.legend(fontsize=8)
    fig.suptitle("Score distributions — SYNTHETIC test audio (optimistic)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/score_distributions.png", dpi=130); plt.close(fig)

    # ---- 2. PR curves ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, pos, neg in data:
        ts, P, R, F = pr_curve(pos, neg)
        i = int(np.argmax(F))
        ax.plot(R, P, lw=2, label=f"{label} (max-F1={F[i]:.2f})")
        ax.scatter([R[i]], [P[i]], s=40, zorder=5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall (dots = max-F1)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/pr_curves.png", dpi=130); plt.close(fig)

    # ---- 3. confusion matrices @0.5 ----
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    names = [["True Positives", "False Negatives"], ["False Positives", "True Negatives"]]
    for ax, (label, pos, neg) in zip(axes[0], data):
        cm = confusion(pos, neg)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{names[i][j]}\n{v}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["detected", "not detected"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["phrase", "not phrase"], fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{label}  (thr 0.5)")
    fig.suptitle("Confusion matrices @0.5 — per-clip, synthetic audio", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/confusion_matrices.png", dpi=130); plt.close(fig)

    print(f"\nSaved 3 figures to {args.outdir}/")


if __name__ == "__main__":
    main()
