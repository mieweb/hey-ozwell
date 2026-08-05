#!/usr/bin/env python3
"""
Training-objective A/B for the wake model: plain cross-entropy (current) vs FOCAL loss vs a MARGIN
variant. Focal down-weights easy examples so training concentrates on the hard, near-boundary cases,
which is the research-flagged lever we hadn't tried. Same browser-faithful data as the other A/Bs;
compares at equal held-out recall on data neither model trained on. Honest expectation: may be
marginal (won't fix the synthetic-vs-real gap), but it's a clean controlled test, same as mining.
"""
import numpy as np, torch, torch.nn as nn
import train_browser as tb

torch.manual_seed(0); np.random.seed(0)
PC = "precalculated"


def train_loss(pos, neg, loss="bce", gamma=2.0, epochs=40):
    X = np.concatenate([pos, neg], 0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype("float32")
    Xt = torch.tensor(X.astype("float32")); yt = torch.tensor(y)[:, None]
    m = tb.WakeMLP(); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = len(neg) / max(len(pos), 1); n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]; opt.zero_grad()
            p = m(Xt[idx]).clamp(1e-6, 1 - 1e-6); t = yt[idx]
            w = torch.where(t > 0.5, torch.tensor(pw), torch.tensor(1.0))   # class-imbalance weight
            if loss == "bce":
                l = (nn.functional.binary_cross_entropy(p, t, reduction="none") * w).mean()
            else:  # focal: down-weight easy (well-classified) examples
                pt = torch.where(t > 0.5, p, 1 - p)
                l = (w * (1 - pt) ** gamma * -torch.log(pt)).mean()
            l.backward(); opt.step()
    return m


def run(ph):
    pos_tr = tb.load(f"{PC}/nbf_{ph}_train/pos_clean.npy", f"{PC}/nbf_{ph}_train/pos_noise.npy", f"{PC}/nbf_{ph}_train/pos_reverb.npy")
    pos_te = tb.load(f"{PC}/nbf_{ph}_test/pos_clean.npy")
    bulk = tb.load(f"{PC}/browser_negs_train.npy")
    evneg = tb.load(f"{PC}/browser_negs_eval.npy")
    models = {"bce (current)": train_loss(pos_tr, bulk, "bce"),
              "focal g=2": train_loss(pos_tr, bulk, "focal", 2.0),
              "focal g=3": train_loss(pos_tr, bulk, "focal", 3.0)}
    print(f"\n=== {ph}: loss A/B @ equal held-out recall (FP = % of held-out neg windows firing) ===")
    print(f"{'loss':16s} {'thr':5s} {'recall':7s} {'FP/win':8s} {'~FP/hr':7s}")
    for tgt in (0.97, 0.95):
        print(f"-- recall target {int(tgt*100)}% --")
        for name, m in models.items():
            sp = tb.scores(m, pos_te); sn = tb.scores(m, evneg)
            thr = tb.thr_for_recall(sp, tgt)
            print(f"{name:16s} {thr:.2f}  {tb.recall_at(sp,thr)*100:5.1f}%  {tb.fp_rate_at(sn,thr)*100:6.3f}%  {tb.fp_rate_at(sn,thr)*30000:6.0f}")
        print()


if __name__ == "__main__":
    for ph in ("hey", "done"):
        run(ph)
