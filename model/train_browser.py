#!/usr/bin/env python3
"""
Browser-faithful precision retrain + A/B eval.

Trains a tiny wake MLP over browser-faithful [16,96] embeddings, TWICE per phrase:
  baseline  = positives + bulk negatives
  treatment = positives + bulk negatives + MINED HARD negatives (oversampled)
Then, at the threshold giving equal recall on HELD-OUT positives, compares the false-fire rate on a
HELD-OUT negative corpus. Lower FP at equal recall => the mined hard negatives helped precision.

MLP only (Gemm/Relu/Sigmoid) so it runs in onnxruntime-web (no ai.onnx.ml ops). Export with --export.
"""
import os, sys, argparse
import numpy as np
import torch, torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)


class WakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(1536, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):           # x: [N,16,96]
        return torch.sigmoid(self.net(x))


def load(*paths):
    arrs = [np.load(p) for p in paths if os.path.exists(p) and np.load(p).shape[0] > 0]
    return np.concatenate(arrs, 0).astype("float32") if arrs else np.zeros((0, 16, 96), "float32")


def train(pos, neg, epochs=40):
    X = np.concatenate([pos, neg], 0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype("float32")
    Xt = torch.tensor(X); yt = torch.tensor(y)[:, None]
    m = WakeMLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor([len(neg) / max(len(pos), 1)])
    lossf = nn.BCELoss(reduction="none")
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            p = m(Xt[idx])
            w = torch.where(yt[idx] > 0.5, pw, torch.tensor([1.0]))
            (lossf(p, yt[idx]) * w).mean().backward()
            opt.step()
    return m


@torch.no_grad()
def scores(m, X):
    if len(X) == 0:
        return np.array([])
    return m(torch.tensor(X.astype("float32"))).numpy().reshape(-1)


def recall_at(s_pos, thr):
    return (s_pos >= thr).mean() if len(s_pos) else 0.0


def fp_rate_at(s_neg, thr):
    return (s_neg >= thr).mean() if len(s_neg) else 0.0


def thr_for_recall(s_pos, target):
    # highest threshold that still keeps >= target recall
    for thr in np.arange(0.99, 0.0, -0.01):
        if recall_at(s_pos, thr) >= target:
            return float(thr)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", choices=["done", "hey"], required=True)
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    ph = args.phrase
    PC = "precalculated"

    pos_tr = load(f"{PC}/nbf_{ph}_train/pos_clean.npy", f"{PC}/nbf_{ph}_train/pos_noise.npy", f"{PC}/nbf_{ph}_train/pos_reverb.npy")
    pos_te = load(f"{PC}/nbf_{ph}_test/pos_clean.npy")          # held-out, CLEAN recall test
    bulk = load(f"{PC}/browser_negs_train.npy")
    hard = load(f"{PC}/mined_ff_train_{ph}.npy")
    evneg = load(f"{PC}/browser_negs_eval.npy")                 # held-out FP test
    print(f"[{ph}] pos_train {pos_tr.shape[0]} pos_test {pos_te.shape[0]} bulk {bulk.shape[0]} hard {hard.shape[0]} eval_neg {evneg.shape[0]}")

    hard_os = np.repeat(hard, 20, axis=0) if len(hard) else hard   # oversample the ~230 hard negs
    models = {
        "baseline":  train(pos_tr, bulk),
        "treatment": train(pos_tr, np.concatenate([bulk, hard_os], 0)),
    }

    print(f"\n=== {ph}: A/B at equal HELD-OUT recall (FP = % of held-out neg windows firing; ~/hr ≈ x30k) ===")
    print(f"{'model':10s} {'recall-target':13s} {'thr':5s} {'recall':7s} {'FP/win':8s} {'~FP/hr':7s}")
    for tgt in (0.95, 0.90):
        for name, m in models.items():
            sp = scores(m, pos_te); sn = scores(m, evneg)
            thr = thr_for_recall(sp, tgt)
            print(f"{name:10s} {tgt*100:11.0f}%  {thr:.2f}  {recall_at(sp,thr)*100:5.1f}%  {fp_rate_at(sn,thr)*100:6.3f}%  {fp_rate_at(sn,thr)*30000:6.0f}")
        print()

    if args.export:
        m = models["treatment"]; m.eval()
        out = f"checkpoints/scratch-onnx/{ph}_browser_precision.onnx"
        torch.onnx.export(m, torch.zeros(1, 16, 96), out, input_names=["input"], output_names=["prob"],
                          dynamic_axes={"input": {0: "batch"}}, opset_version=13)
        print("exported ->", out)


if __name__ == "__main__":
    main()
