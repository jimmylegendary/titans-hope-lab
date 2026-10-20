"""Sample text from a byte-level HOPE checkpoint.

Byte-level generation is the most direct sanity check that the model learned
language rather than just a low loss: every byte is sampled autoregressively,
and the model's memories update as its own output streams through them.

Usage:
    python scripts/generate.py --ckpt runs/hope-byte/ckpt.pt \
        --prompt "The history of the Roman Empire" --max-bytes 300
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hope import HOPE, HOPEConfig  # noqa: E402


def load(ckpt_path, dev, max_len):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    tr = ck["config"]
    cfg = HOPEConfig(vocab_size=256, block_size=max_len,
                     n_layer=tr["n_layer"], n_embd=tr["n_embd"],
                     cms_second_order=False, norm_out=True,
                     deep_mem_learnable=tr.get("deep_mem_learnable", False),)
    model = HOPE(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.skip_logits = False
    return model.eval().float(), ck["step"]


@torch.no_grad()
def generate(model, prompt, max_bytes, temperature, top_k, dev, seed):
    torch.manual_seed(seed)
    ids = list(prompt.encode("utf-8"))
    for _ in range(max_bytes):
        x = torch.tensor([ids], device=dev, dtype=torch.long)
        # Proper next-byte targets (shifted; last position padded with the
        # ignore index): the CMS test-time updates are driven by the gradient
        # of this loss, so an unshifted target would corrupt the memory
        # dynamics of any generation longer than one chunk.
        y = torch.tensor([ids[1:] + [-1]], device=dev, dtype=torch.long)
        logits, _ = model(x, y)          # full-prefix forward (states rebuilt)
        logits = logits[0, -1] / temperature
        if top_k:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = -float("inf")
        ids.append(int(torch.multinomial(F.softmax(logits, -1), 1)))
    return bytes(ids).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="The history of the Roman Empire")
    ap.add_argument("--max-bytes", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, step = load(args.ckpt, dev, len(args.prompt.encode()) + args.max_bytes + 8)
    print(f"[{args.ckpt} @ step {step}]")
    out = generate(model, args.prompt, args.max_bytes, args.temperature,
                   args.top_k, dev, args.seed)
    print(out)


if __name__ == "__main__":
    main()
