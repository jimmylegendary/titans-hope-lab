"""Scale-up: BPE pretraining of HOPE with multi-GPU DDP (accelerate).

This is the exact recipe (ported comments only) that trained our 161.5M-param
HOPE to 1B tokens on 2 GPUs — at equal tokens (0.52B) it tied a Titans-MAC
baseline trained identically (4.557 vs 4.553 nats), reproducing the paper's
competitive-with-hybrids claim in the BPE regime. Differences from the byte
recipe in train_byte.py:

  - Llama-2 32K tokenizer (set TOKENIZER env to swap), sequence 4096
  - 8 layers, d=768, chunk 1024 -> 161.5M params
  - batch 4/GPU x grad-accum 16 x 2 GPUs = 0.52M tokens/step (the Gated
    DeltaNet convention), cosine to 1B tokens
  - activation checkpointing + skip_logits (logits are recomputed only for
    the loss; the full logit tensor at 32K vocab dominates memory otherwise)

Launch (2 GPUs):
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
        scripts/train_bpe.py --data data/fineweb-edu

Operational notes from our runs, so you do not rediscover them:
  - Streaming validation must not use collectives inside the val loop: ranks
    can see different batch counts and a mid-loop gather deadlocks DDP.
  - Full resume: ckpt.pt carries weights + optimizer moments; each rank writes
    its own RNG file (rng_rank{i}.pt) and the streaming dataloader is
    fast-forwarded past consumed batches, so a resumed run replays the exact
    data stream. The save path uses NO collectives: ranks diverge freely around
    the local-only validation loop, and any collective there deadlocks DDP.
  - On resume, rebuild the LR scheduler and fast-forward it to the resumed
    step; a stale scheduler counter re-runs warmup at peak LR and diverges.
  - Set a generous process-group timeout (checkpoint save + val can exceed
    the 10-minute NCCL default at larger scales).
"""

import argparse
import math
import os
import sys
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hope import HOPE, HOPEConfig  # noqa: E402


class ParquetTokenDataset(IterableDataset):
    """Stream parquet text -> tokenizer -> fixed-length (T+1) windows.
    Documents are concatenated with an EOS separator. Each DDP rank/worker
    reads a disjoint stride of files."""

    def __init__(self, files, tokenizer_name, seq_len):
        self.files, self.seq = files, seq_len
        self.tokenizer_name = tokenizer_name

    def __iter__(self):
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        shard, nshard = rank * nw + wid, world * nw
        buf = []
        while True:
            for fi, f in enumerate(self.files):
                if fi % nshard != shard:
                    continue
                pf = pq.ParquetFile(f)
                for rb in pf.iter_batches(batch_size=64, columns=["text"]):
                    for t in rb.column("text").to_pylist():
                        buf += tok(t, add_special_tokens=False)["input_ids"]
                        buf.append(tok.eos_token_id or 0)
                        while len(buf) >= self.seq + 1:
                            yield torch.tensor(buf[: self.seq + 1])
                            del buf[: self.seq + 1]


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fineweb-edu")
    p.add_argument("--max-steps", type=int, default=1907)        # ~1B tokens at 0.52M/step
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--peak-lr", type=float, default=4e-4)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tokenizer", default=os.environ.get("TOKENIZER",
                                                         "NousResearch/Llama-2-7b-hf"))
    p.add_argument("--output-dir", default="runs/hope-bpe")
    p.add_argument("--resume", default=None, help="path to ckpt.pt (weights + step)")
    args = p.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])
    torch.manual_seed(42 + accelerator.process_index)

    import glob
    files = sorted(glob.glob(os.path.join(args.data, "**/*.parquet"), recursive=True))
    assert len(files) >= 2, "need >= 2 parquet shards (scripts/download_data.py)"
    train_ds = ParquetTokenDataset(files[:-1], args.tokenizer, args.seq_len)
    val_ds = ParquetTokenDataset(files[-1:], args.tokenizer, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_batch_size,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_batch_size)

    from transformers import AutoTokenizer
    vocab = len(AutoTokenizer.from_pretrained(args.tokenizer))
    cfg = HOPEConfig(vocab_size=vocab, block_size=args.seq_len, n_layer=8,
                     n_embd=768, chunk_len=1024, cms_periods=(1, 2, 4),
                     update_clip=0.5, cms_second_order=False, norm_out=True)
    model = HOPE(cfg)
    model.grad_ckpt = True
    model.skip_logits = True
    n_params = sum(p_.numel() for p_ in model.parameters())
    accelerator.print(f"hope-bpe: {n_params/1e6:.1f}M params | chunk {cfg.chunk_len} | "
                      f"GPUs={accelerator.num_processes} batch/dev={args.per_device_batch_size} "
                      f"accum={args.grad_accum}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    model, opt, train_loader = accelerator.prepare(model, opt, train_loader)

    batch_tokens = (accelerator.num_processes * args.per_device_batch_size
                    * args.seq_len * args.grad_accum)
    inner = model.module if hasattr(model, "module") else model
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        inner.load_state_dict(ck["model"])
        step = ck["step"]
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            accelerator.print("[resume] WARNING: no optimizer moments in ckpt "
                              "(pre-full-resume checkpoint) — Adam re-warms.")
        rng_path = os.path.join(os.path.dirname(args.resume),
                                f"rng_rank{accelerator.process_index}.pt")
        if os.path.exists(rng_path):
            r = torch.load(rng_path, map_location="cpu", weights_only=False)
            torch.set_rng_state(r["torch"])
            torch.cuda.set_rng_state(r["cuda"], accelerator.device)
        else:
            accelerator.print(f"[resume] WARNING: no rng file for rank "
                              f"{accelerator.process_index} — stream replays from skip only.")
        if ck.get("world", accelerator.num_processes) != accelerator.num_processes:
            accelerator.print("[resume] WARNING: world size changed — "
                              "data sharding will not replay identically.")
        for _ in range(step):
            sched.step()   # restore LR schedule position (see module docstring)
        accelerator.print(f"Resumed from step {step} ({args.resume})")
    t0, run_loss, cnt = time.time(), 0.0, 0
    it = iter(train_loader)
    if args.resume and step > 0:
        # Fast-forward the streaming dataloader past data already consumed, so
        # the resumed run continues the same stream instead of re-reading the
        # beginning (tokenization-bound; ~10-50x faster than training).
        skip = step * args.grad_accum
        accelerator.print(f"[resume] skipping {skip} per-device batches to restore stream position...")
        for _i in range(skip):
            try:
                next(it)
            except StopIteration:
                it = iter(train_loader)
        accelerator.print("[resume] stream position restored")
    os.makedirs(args.output_dir, exist_ok=True)

    while step < args.max_steps:
        with accelerator.accumulate(model):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.autocast("cuda", torch.bfloat16):
                _, loss = model(x, y)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                gn = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    accelerator.print(f"[skip] step {step}: non-finite grad")
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                run_loss += loss.item(); cnt += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    tps = (step - (ck["step"] if args.resume else 0)) * batch_tokens / el if el > 0 else 0
                    avg = run_loss / max(1, cnt); run_loss, cnt = 0.0, 0
                    accelerator.print(
                        f"step {step:>7d} | loss {avg:.4f} | ppl {math.exp(min(avg,20)):.1f} | "
                        f"lr {sched.get_last_lr()[0]:.2e} | {tps/1e3:.1f}k tok/s | "
                        f"{step*batch_tokens/1e9:.3f}B")
                if step % args.validate_every == 0:
                    # local-only validation: NO collectives in this loop (see
                    # module docstring — mismatched batch counts deadlock DDP)
                    inner.eval()
                    vls = []
                    with torch.no_grad():
                        for i, vb in enumerate(val_loader):
                            if i >= args.val_batches:
                                break
                            vb = vb.to(accelerator.device)
                            _, vl = inner(vb[:, :-1], vb[:, 1:])
                            vls.append(vl.item())
                    inner.train()
                    if vls:
                        av = sum(vls) / len(vls)
                        accelerator.print(f"step {step:>7d} | val_loss {av:.4f} | "
                                          f"val_ppl {math.exp(min(av,20)):.1f}")
                if step % args.save_every == 0:
                    # Full-resume checkpoint with ZERO collectives (the val
                    # loop above is local-only by design; any collective here
                    # deadlocks if ranks diverge — learned 2026-09-01).
                    # Each rank writes its own RNG file; main writes the ckpt.
                    torch.save({"torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state(accelerator.device)},
                               os.path.join(args.output_dir,
                                            f"rng_rank{accelerator.process_index}.pt"))
                    if accelerator.is_main_process:
                        torch.save({"model": inner.state_dict(), "step": step,
                                    "opt": opt.state_dict(),
                                    "world": accelerator.num_processes},
                                   os.path.join(args.output_dir, "ckpt.pt"))
                        accelerator.print(f"Checkpoint saved -> {args.output_dir}/ckpt.pt")
    if accelerator.is_main_process:
        torch.save({"model": inner.state_dict(), "step": step},
                   os.path.join(args.output_dir, "final.pt"))
    accelerator.print(f"DONE hope-bpe {step} steps")


if __name__ == "__main__":
    main()
