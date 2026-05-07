#!/usr/bin/env python3
"""
Train a pairwise ranker (MLP) on warmstart_samples.jsonl.

Run:
  python train_pairwise_ranker_mlp.py \
      --in warmstart_samples.jsonl \
      --out struct_ranker_mlp.json \
      --min_speedup 1.0 \
      --pairs_per_task 64 \
      --epochs 50 \
      --lr 3e-4 \
      --hidden 64,32 \
      --dropout 0.10
"""

import json, math, random, argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_FEATURES = [
    "coal", "ai", "occ", "div", "xfer",
    "atomics", "syncthreads", "kernels", "tpb",
    "gmem_access", "ops",
]

# -----------------------
# IO + feature utils
# -----------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def vec_from_features(feat: Dict[str, float], names: List[str]) -> List[float]:
    return [float(feat.get(k, 0.0)) for k in names]

def compute_norm_stats(rows: List[Dict], feat_names: List[str]) -> Tuple[List[float], List[float]]:
    X = [vec_from_features(r["features"], feat_names) for r in rows]
    d = len(feat_names)

    mean = [0.0] * d
    for x in X:
        for j in range(d):
            mean[j] += x[j]
    n = max(1, len(X))
    mean = [m / n for m in mean]

    var = [0.0] * d
    for x in X:
        for j in range(d):
            var[j] += (x[j] - mean[j]) ** 2
    var = [v / n for v in var]

    std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, std

def zscore(x: List[float], mean: List[float], std: List[float]) -> List[float]:
    return [(x[i] - mean[i]) / std[i] for i in range(len(x))]

def build_pairs_by_task(
    items_by_task: Dict[str, List[Dict]],
    feat_names: List[str],
    mean: List[float],
    std: List[float],
    pairs_per_task: int,
    min_speedup: float,
) -> List[Tuple[List[float], List[float]]]:
    """
    Returns list of (x_good, x_bad) feature vectors (already normalized).
    """
    pairs = []
    for tid, items in items_by_task.items():
        goodset = [r for r in items if r.get("compile_ok") and float(r.get("pass_rate", 0.0)) >= 1.0]
        if len(goodset) < 2:
            continue

        goodset.sort(key=lambda r: float(r.get("speedup", 0.0)))
        speeds = [float(r.get("speedup", 0.0)) for r in goodset]
        if max(speeds) < min_speedup:
            continue

        q = max(1, len(goodset) // 4)
        bottom = goodset[:q]
        top = goodset[-q:]
        if not bottom or not top:
            continue

        for _ in range(pairs_per_task):
            g = random.choice(top)
            b = random.choice(bottom)
            if float(g.get("speedup", 0.0)) <= float(b.get("speedup", 0.0)):
                continue

            xg = zscore(vec_from_features(g["features"], feat_names), mean, std)
            xb = zscore(vec_from_features(b["features"], feat_names), mean, std)
            pairs.append((xg, xb))
    return pairs

# -----------------------
# MLP ranker
# -----------------------
class MLPRanker(nn.Module):
    """
    score(x) = MLP(x) -> scalar
    """
    def __init__(self, d: int, hidden: List[int], dropout: float):
        super().__init__()
        dims = [d] + hidden + [1]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, d] -> [B]
        return self.net(X).squeeze(-1)

def parse_int_list(csv: str) -> List[int]:
    csv = csv.strip()
    if not csv:
        return []
    out = []
    for p in csv.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out

# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--min_speedup", type=float, default=1.0)
    ap.add_argument("--pairs_per_task", type=int, default=64)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # model
    ap.add_argument("--hidden", type=str, default="64,32",
                    help="Comma-separated hidden layer sizes, e.g. 128,64,32")
    ap.add_argument("--dropout", type=float, default=0.10)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    feat_names = [x.strip() for x in args.features.split(",") if x.strip()]
    d = len(feat_names)
    hidden = parse_int_list(args.hidden)
    if not hidden:
        raise SystemExit("--hidden must specify at least one layer, e.g. --hidden 64,32")

    print(f"[INFO] feature dims={d} hidden={hidden} dropout={args.dropout}")

    # group by task (only correct+compiled)
    by_task = defaultdict(list)
    all_train_rows = []
    for r in read_jsonl(args.inp):
        if "features" not in r:
            continue
        if r.get("compile_ok") and float(r.get("pass_rate", 0.0)) >= 1.0:
            all_train_rows.append(r)
            by_task[r["task_id"]].append(r)

    print(f"[INFO] tasks_with_correct={len(by_task)} rows_correct={len(all_train_rows)}")
    if len(all_train_rows) < 2:
        raise SystemExit("Not enough correct samples to train a ranker. Collect more warmstart samples.")

    mean, std = compute_norm_stats(all_train_rows, feat_names)

    pairs = build_pairs_by_task(
        items_by_task=by_task,
        feat_names=feat_names,
        mean=mean,
        std=std,
        pairs_per_task=args.pairs_per_task,
        min_speedup=args.min_speedup,
    )
    print(f"[INFO] total_pairs={len(pairs)}")
    if len(pairs) < 200:
        raise SystemExit("Not enough training pairs. Increase --pairs_per_task or lower filters.")

    # tensors
    Xg = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    Xb = torch.tensor([p[1] for p in pairs], dtype=torch.float32)

    model = MLPRanker(d=d, hidden=hidden, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    batch = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(Xg.shape[0])
        Xg_sh = Xg[perm]
        Xb_sh = Xb[perm]

        total_loss = 0.0
        for i in range(0, Xg.shape[0], batch):
            g = Xg_sh[i:i+batch]
            b = Xb_sh[i:i+batch]

            sg = model.score(g)
            sb = model.score(b)

            # pairwise logistic ranking loss
            loss = F.softplus(-(sg - sb)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.detach().item()) * g.shape[0]

        avg_loss = total_loss / Xg.shape[0]

        with torch.no_grad():
            sg = model.score(Xg[:2048])
            sb = model.score(Xb[:2048])
            acc = float((sg > sb).float().mean().item())
        print(f"[ep {ep+1:03d}] loss={avg_loss:.4f} pair_acc@2k={acc:.3f}")

    # save model weights + architecture
    out = {
        "model_type": "mlp_ranker",
        "feature_names": feat_names,
        "norm_mean": mean,
        "norm_std": std,
        "mlp": {
            "hidden": hidden,
            "dropout": args.dropout,
            "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        },
        "meta": {
            "min_speedup": args.min_speedup,
            "pairs_per_task": args.pairs_per_task,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch": args.batch,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] saved ranker to {args.out}")

if __name__ == "__main__":
    main()
