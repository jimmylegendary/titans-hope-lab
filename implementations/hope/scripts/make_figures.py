"""Regenerate assets/loss_curves.png from the committed training logs.

Reads metrics.log files (format written by scripts/train_byte.py) and plots
train bpb (line) + held-out validation bpb (markers) against tokens.
"""

import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# byte-trainer format (scripts/train_byte.py)
STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+[\d.]+\s+bpb\s+([\d.]+)")
VAL_RE = re.compile(r"step\s+(\d+)\s+VAL loss\s+[\d.]+\s+bpb\s+([\d.]+)")
# bpe-trainer format (scripts/train_bpe.py): loss in nats
BPE_STEP_RE = re.compile(r"step\s+(\d+) \| loss ([\d.]+) \| ppl")
BPE_VAL_RE = re.compile(r"step\s+(\d+) \| val_loss ([\d.]+)")


def parse(path):
    """Auto-detects the trainer format; returns train/val series.
    BPE logs report nats (no bpb) — plotted as-is with a nats axis."""
    steps, bpb, vsteps, vbpb = [], [], [], []
    seen = {}
    vals = {}
    for line in open(path, errors="ignore"):
        if m := VAL_RE.search(line):
            vsteps.append(int(m.group(1))), vbpb.append(float(m.group(2)))
        elif m := STEP_RE.search(line):
            steps.append(int(m.group(1))), bpb.append(float(m.group(2)))
        elif m := BPE_VAL_RE.search(line):
            vals[int(m.group(1))] = float(m.group(2))
        elif m := BPE_STEP_RE.search(line):
            seen[int(m.group(1))] = float(m.group(2))   # dedupe resumed segments
    for s in sorted(seen):
        steps.append(s), bpb.append(seen[s])
    for s in sorted(vals):
        vsteps.append(s), vbpb.append(vals[s])
    return steps, bpb, vsteps, vbpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="label=path pairs, e.g. HOPE=runs/hope-byte/metrics.log")
    ap.add_argument("--tokens-per-step", type=float, default=16384)
    ap.add_argument("--ylabel", default="bits/byte (lower is better)")
    ap.add_argument("--ymin", type=float, default=1.0)
    ap.add_argument("--xunit", type=float, default=1e6, help="1e6 = M, 1e9 = B")
    ap.add_argument("--xlabel", default="training tokens (M)")
    ap.add_argument("--annotate", action="store_true",
                    help="label the final validation point with value and step count")
    ap.add_argument("--out", default="assets/loss_curves.png")
    args = ap.parse_args()

    colors = ["#199e70", "#3987e5", "#d95926", "#9085e9", "#c98500"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for i, spec in enumerate(args.logs):
        label, path = spec.split("=", 1)
        steps, bpb, vs, vb = parse(path)
        tok = [s * args.tokens_per_step / args.xunit for s in steps]
        vtok = [s * args.tokens_per_step / args.xunit for s in vs]
        c = colors[i % len(colors)]
        ax.plot(tok, bpb, color=c, lw=1.6, label=label)
        ax.scatter(vtok, vb, color=c, s=22, zorder=3, edgecolors="white", lw=1)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_ylim(bottom=args.ymin)
    if args.annotate:
        # final value + total steps, so the x-axis unit can't be misread
        label, path = args.logs[0].split("=", 1)
        steps, bpb, vs, vb = parse(path)
        if vb:
            ax.annotate(f"{vb[-1]:.3f}\n({steps[-1]:,} steps)",
                        xy=(steps[-1] * args.tokens_per_step / args.xunit, vb[-1]),
                        xytext=(-12, 26), textcoords="offset points",
                        ha="right", fontsize=9, color="#199e70",
                        arrowprops=dict(arrowstyle="-", color="#199e70", lw=1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
