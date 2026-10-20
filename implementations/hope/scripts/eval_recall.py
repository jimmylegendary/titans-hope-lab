"""Needle-in-a-haystack recall: does the memory actually hold what it read?

Mean loss is a poor instrument for memory. Most bytes of natural text are
predictable without remembering anything specific, so a model that memorizes
nothing and a model that memorizes well differ by very little on average — the
signal is diluted by roughly an order of magnitude (we measured 18x in a
controlled version of this test).

This script measures the thing directly. A sentence containing a high-entropy
code is planted early in a stream of held-out bytes, and the identical sentence
reappears `distance` bytes later:

    <bytes> The secret access code is XQ-73914-ZETA. <distance bytes> The secret access code is| XQ-73914-ZETA.
            ^ planted at --first-pos (default 600)                                             ^ payload scored from here

The control arm is byte-identical except that the first occurrence is absent, so
the difference isolates one thing: whether the model still holds what it read
`distance` bytes ago. `recall gain` = payload bits/byte in the control minus
bits/byte in the treatment; a model with no long-range memory scores ~0.

Both arms share the same filler bytes (paired comparison), and the reported
error is the standard error over trials.

Usage:
    python scripts/eval_recall.py --ckpt runs/hope-byte/ckpt.pt --data data/fineweb-edu
    python scripts/eval_recall.py --ckpt ... --distances 2048 4096 6600 12000
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hope import HOPE, HOPEConfig  # noqa: E402
from train_byte import load_bytes  # noqa: E402

# A high-entropy alphanumeric payload: digits alone cap the achievable gain at
# ~3.3 bits/byte, which compresses the measurement. Mixed letters/digits/hyphens
# leave room for the effect to show.
NEEDLE = b" The secret access code is XQ-73914-ZETA."
PAYLOAD_OFF = NEEDLE.index(b"XQ")      # payload starts here inside the needle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/fineweb-edu")
    ap.add_argument("--distances", type=int, nargs="+", default=[6600],
                    help="bytes between the planted fact and its repetition")
    ap.add_argument("--first-pos", type=int, default=600,
                    help="byte offset of the planted fact")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LN2 = math.log(2)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    tr = ck["config"]
    max_len = args.first_pos + max(args.distances) + 512
    cfg = HOPEConfig(vocab_size=256, block_size=max_len,
                     n_layer=tr["n_layer"], n_embd=tr["n_embd"],
                     cms_second_order=False, norm_out=True,
                     deep_mem_learnable=tr.get("deep_mem_learnable", False),
)
    model = HOPE(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model = model.eval().float()      # fp32 eval: the state recurrence is sensitive
    print(f"loaded step-{ck['step']} checkpoint "
          f"(deep_mem_learnable={cfg.deep_mem_learnable})", flush=True)

    _, val_data = load_bytes(args.data, tr.get("token_budget", 300_000_000),
                             2_000_000, tr["seq_len"], 8)
    rng = np.random.default_rng(args.seed)

    def build(with_first, pos2, total_len, start):
        """Held-out bytes with the needle planted at pos2 and — in the treatment
        condition — the identical sentence also at first_pos. Both arms use the
        same filler bytes (paired comparison)."""
        s = bytearray(val_data[start:start + total_len + 100].tobytes())
        if with_first:
            s[args.first_pos:args.first_pos + len(NEEDLE)] = NEEDLE
        s[pos2:pos2 + len(NEEDLE)] = NEEDLE
        arr = np.frombuffer(bytes(s[:total_len + 1]), dtype=np.uint8)
        return torch.from_numpy(arr.astype(np.int64)).unsqueeze(0).to(dev)

    def payload_bpb(x, pos2, total_len):
        with torch.no_grad():
            logits, _ = model(x[:, :total_len], x[:, 1:total_len + 1])
        # logits at position t predict byte t+1, so the payload is scored
        # starting one position earlier
        lo = pos2 + PAYLOAD_OFF - 1
        hi = pos2 + len(NEEDLE) - 2
        return F.cross_entropy(logits[0, lo:hi], x[0, lo + 1:hi + 1]).item() / LN2

    for dist in args.distances:
        pos2 = args.first_pos + dist
        total_len = pos2 + len(NEEDLE) + 64
        gains = []
        for _ in range(args.trials):
            start = int(rng.integers(0, len(val_data) - total_len - 200))
            w = payload_bpb(build(True, pos2, total_len, start), pos2, total_len)
            c = payload_bpb(build(False, pos2, total_len, start), pos2, total_len)
            gains.append(c - w)
        m = float(np.mean(gains))
        se = float(np.std(gains, ddof=1) / math.sqrt(len(gains)))
        print(f"distance {dist:>6d} bytes | recall gain {m:+.3f} +/- {se:.3f} bpb "
              f"(mean +/- s.e., n={len(gains)})", flush=True)


if __name__ == "__main__":
    main()
