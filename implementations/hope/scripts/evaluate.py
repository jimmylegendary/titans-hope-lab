"""Evaluate a trained HOPE checkpoint.

Two evaluations, matching the released results:
  1. Held-out validation bpb at training length (2048).
  2. Length extrapolation: stream an 8K-byte sequence (4x the training
     length) and report bpb per 1K chunk. HOPE has no positional
     embedding, so — unlike models whose positional signal saturates —
     the curve should stay flat (within a few tenths of a bpb of the
     training window) instead of diverging; see the README table.

Usage:
    python scripts/evaluate.py --ckpt runs/hope-byte/ckpt.pt --data data/fineweb-edu
"""

import argparse
import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hope import HOPE, HOPEConfig  # noqa: E402
from train_byte import load_bytes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/fineweb-edu")
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--extrap-len", type=int, default=8192)
    ap.add_argument("--extrap-batches", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LN2 = math.log(2)

    import numpy as np
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    tr = ck["config"]
    cfg = HOPEConfig(vocab_size=256, block_size=args.extrap_len,
                     n_layer=tr["n_layer"], n_embd=tr["n_embd"],
                     cms_second_order=False, norm_out=True,
                     deep_mem_learnable=tr.get("deep_mem_learnable", False),
)
    model = HOPE(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded step-{ck['step']} checkpoint", flush=True)

    # same held-out tail the trainer used for validation (the batch arg only
    # pads the read-stop threshold; verified to yield a byte-identical 2M tail
    # for any small batch value)
    _, val_data = load_bytes(args.data, tr.get("token_budget", 300_000_000),
                             2_000_000, tr["seq_len"], 8)

    def crops(seq, n, seed):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            i = rng.integers(0, len(val_data) - seq - 1)
            yield torch.from_numpy(val_data[i:i+seq+1].astype(np.int64))[None].to(dev)

    # 1) validation bpb at training length — the trainer's exact protocol
    #    (same crop RNG stream, batch shape and bf16 autocast), so this number
    #    reproduces the "final held-out validation" the README table reports.
    seq, B = tr["seq_len"], 4
    vrng = np.random.default_rng(1)
    tot = 0.0
    for _ in range(args.val_batches):
        ix = vrng.integers(0, len(val_data) - seq - 1, size=B)
        xb = torch.from_numpy(np.stack([val_data[i:i+seq] for i in ix]).astype(np.int64)).to(dev)
        yb = torch.from_numpy(np.stack([val_data[i+1:i+seq+1] for i in ix]).astype(np.int64)).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=dev == "cuda"):
            _, l = model(xb, yb)
        tot += l.item()
    print(f"val bpb @ {seq}: {tot / args.val_batches / LN2:.3f}", flush=True)

    # 2) 8K extrapolation, bpb per 1K chunk (unguarded: max_updates=None)
    chunks = args.extrap_len // 1024
    sums = [0.0] * chunks
    for b in crops(args.extrap_len, args.extrap_batches, seed=2):
        with torch.no_grad():
            logits, _ = model(b[:, :-1], b[:, 1:])
            per_tok = torch.nn.functional.cross_entropy(
                logits[0], b[0, 1:], reduction="none")
        for c in range(chunks):
            sums[c] += per_tok[c * 1024:(c + 1) * 1024].mean().item()
    bpbs = [s / args.extrap_batches / LN2 for s in sums]
    print("extrapolation bpb per 1K chunk (train window = first "
          f"{tr['seq_len']//1024}):", " ".join(f"{x:.2f}" for x in bpbs), flush=True)


if __name__ == "__main__":
    main()
