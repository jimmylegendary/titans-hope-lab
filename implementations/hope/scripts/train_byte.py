"""Byte-level pretraining of HOPE on FineWeb-Edu — the released reference run.

This is the exact recipe that produced assets/reference_run.log (final
held-out validation 1.721 bits/byte; the original research run scored
1.722) and assets/loss_curves.png:

  - 30.6M parameters (4 layers, d=512, chunk 512, CMS periods (1,2,4))
  - byte tokenization (vocab 256), sequence length 2048
  - 0.3B training bytes taken from the FIRST FineWeb-Edu sample shard,
    loaded in memory; batches are random crops (seeded), the last 2M bytes
    are held out for validation
  - micro-batch 4 x grad-accum 2 = 16,384 tokens/step, 18,310 steps
  - AdamW lr 6e-4, weight decay 0.1, grad clip 1.0 (non-finite steps are
    skipped, not applied), warmup 300 then cosine to zero
  - bf16 autocast forward/backward; fp32 master weights
  - held-out validation every 2000 steps, reported in bits/byte (bpb)

With identical seeds this reproduces the reference trajectory closely
(bpb ~5.08 at step 100, ~3.67 at step 200; see tests/test_smoke.py).

Usage (single GPU, ~14 GB peak, several hours):
    python scripts/train_byte.py --data data/fineweb-edu --out runs/hope-byte
"""

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hope import HOPE, HOPEConfig  # noqa: E402


def load_bytes(data_dir, budget, val_holdout, seq, batch):
    """Concatenate utf-8 bytes (NUL-separated documents) from the first
    parquet shard until the budget is covered; split off a validation tail."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    assert files, f"no parquet shards under {data_dir} (run scripts/download_data.py)"
    chunks, total = [], 0
    pf = pq.ParquetFile(files[0])
    for rb in pf.iter_batches(batch_size=1024, columns=["text"]):
        for t in rb.column("text").to_pylist():
            b = t.encode("utf-8", errors="ignore") + b"\x00"
            chunks.append(b)
            total += len(b)
        if total > budget + val_holdout + seq * batch:
            break
    data = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    return data[: len(data) - val_holdout], data[len(data) - val_holdout:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fineweb-edu")
    ap.add_argument("--out", default="runs/hope-byte")
    ap.add_argument("--token-budget", type=int, default=300_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: token-budget / (seq*batch*accum) = 18310")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--deep-mem-learnable", action="store_true",
                    help="use the paper's Eq. 91 two-layer main memory instead of "
                         "the default matrix memory (see docs/DEVIATIONS.md D1)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="activation checkpointing (lower peak memory, ~30%% slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="path to ckpt.pt — restores model, optimizer moments, "
                         "and RNG states for a faithful continuation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    SEQ, BATCH, ACCUM = args.seq_len, args.batch, args.accum
    steps = args.max_steps or args.token_budget // (SEQ * BATCH * ACCUM)
    train_data, val_data = load_bytes(args.data, args.token_budget,
                                      2_000_000, SEQ, BATCH)
    print(f"corpus: {(len(train_data)+len(val_data))/1e6:.0f}M bytes", flush=True)

    def get_batch(src, rng):
        ix = rng.integers(0, len(src) - SEQ - 1, size=BATCH)
        x = torch.from_numpy(np.stack([src[i:i+SEQ] for i in ix]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([src[i+1:i+SEQ+1] for i in ix]).astype(np.int64)).to(dev)
        return x, y

    # norm_out=True and first-order CMS consumption are part of the released
    # configuration (see docs/DEVIATIONS.md D6/D7 and the module docstring).
    cfg = HOPEConfig(vocab_size=256, block_size=SEQ, n_layer=args.n_layer,
                     n_embd=args.n_embd, cms_second_order=False, norm_out=True,
                     deep_mem_learnable=args.deep_mem_learnable,
)
    model = HOPE(cfg).to(dev)
    if args.grad_ckpt:
        model.grad_ckpt = True
    n_par = sum(p.numel() for p in model.parameters())
    print(f"# hope  params={n_par/1e6:.1f}M  steps={steps}  "
          f"batch_tokens={SEQ*BATCH*ACCUM}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    # warmup, then cosine annealed all the way to zero
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (s - args.warmup)
                                                    / max(1, steps - args.warmup)))))))
    log = open(os.path.join(args.out, "metrics.log"), "a", buffering=1)
    log.write(f"# hope  params={n_par/1e6:.1f}M  steps={steps}  "
              f"batch_tokens={SEQ*BATCH*ACCUM}\n")
    LN2 = math.log(2)

    @torch.no_grad()
    def evaluate():
        model.eval()
        rng = np.random.default_rng(1)     # fixed val sample
        losses = []
        for _ in range(args.val_batches):
            xv, yv = get_batch(val_data, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                losses.append(model(xv, yv)[1].item())
        model.train()
        return sum(losses) / len(losses)

    rng = np.random.default_rng(args.seed)
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(rk["model"])
        if "optimizer" in rk:
            opt.load_state_dict(rk["optimizer"])
        else:
            print("[resume] WARNING: checkpoint has no optimizer moments — "
                  "continuation will not match an uninterrupted run", flush=True)
        start_step = rk["step"]
        for _ in range(start_step):
            sched.step()                       # cosine position (stateless in step)
        if "np_rng" in rk:
            rng.bit_generator.state = rk["np_rng"]
            torch.set_rng_state(rk["torch_rng"].cpu())
            if dev == "cuda" and "cuda_rng" in rk:
                torch.cuda.set_rng_state(rk["cuda_rng"].cpu())
        else:
            # older checkpoint: replay the data-stream draws deterministically
            for _ in range(start_step * ACCUM):
                rng.integers(0, len(train_data) - SEQ - 1, size=BATCH)
            print("[resume] no RNG state in checkpoint; data stream replayed, "
                  "torch RNG fresh", flush=True)
        print(f"[resume] continuing from step {start_step}", flush=True)
    model.train()
    t0, run_loss = time.time(), 0.0
    for step in range(start_step + 1, steps + 1):
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(ACCUM):
            xb, yb = get_batch(train_data, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                _, loss = model(xb, yb)
            (loss / ACCUM).backward()
            step_loss += loss.item() / ACCUM
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        else:
            # discard non-finite steps: one bad batch must not poison weights
            print(f"[skip] step {step}: non-finite grad norm", flush=True)
            opt.zero_grad()
        sched.step()
        run_loss += step_loss

        if step % args.log_every == 0:
            el = time.time() - t0
            tps = step * SEQ * BATCH * ACCUM / el
            eta_h = (steps - step) * SEQ * BATCH * ACCUM / tps / 3600
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            line = (f"step {step:6d}  loss {run_loss/args.log_every:.4f}  "
                    f"bpb {run_loss/args.log_every/LN2:.3f}  {tps/1e3:.1f}k tok/s  "
                    f"ETA {eta_h:.1f}h  peak {peak:.1f}GB")
            print(line, flush=True)
            log.write(line + "\n")
            run_loss = 0.0

        if step % args.val_every == 0 or step == steps:
            vl = evaluate()
            line = f"step {step:6d}  VAL loss {vl:.4f}  bpb {vl/LN2:.3f}"
            print(line, flush=True)
            log.write(line + "\n")
            ck = {"model": model.state_dict(), "step": step,
                  "config": vars(args), "optimizer": opt.state_dict(),
                  "np_rng": rng.bit_generator.state,
                  "torch_rng": torch.get_rng_state()}
            if dev == "cuda":
                ck["cuda_rng"] = torch.cuda.get_rng_state()
            torch.save(ck, os.path.join(args.out, "ckpt.pt"))

    vl = evaluate()
    print(f"DONE hope: {n_par/1e6:.1f}M params, final val loss {vl:.4f} "
          f"(bpb {vl/LN2:.3f}), {(time.time()-t0)/3600:.1f}h", flush=True)
    log.write(f"DONE final val bpb {vl/LN2:.3f}\n")
    log.close()


if __name__ == "__main__":
    main()
