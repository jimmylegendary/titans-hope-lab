"""Self-contained tiny byte training, evaluation, and autoregressive generation."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from .model import HOPE, HOPEConfig


DEFAULT_CORPUS = (
    b"the tiny memory keeps bytes in order.\n"
    b"when tokens arrive, context becomes memory.\n"
    b"a learner predicts the next byte from the stream.\n"
    b"small tests check state updates and held out loss.\n"
    b"memory learns which bytes should follow.\n"
) * 64

DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_STEPS = 12
TRAIN_FRACTION = 0.8


def _configure_cpu_threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows setting inter-op threads only before its work pool is
        # initialized. Repeated in-process CLI calls keep the existing value.
        pass


def load_byte_corpus(path=None):
    """Read a raw byte corpus or use deterministic built-in demonstration text."""
    data = DEFAULT_CORPUS if path is None else Path(path).read_bytes()
    if len(data) < 2:
        raise ValueError("byte corpus must contain at least two bytes")
    return data


def split_byte_corpus(data, block_size):
    """Create a contiguous train/held-out split with one full sample each."""
    minimum = 2 * (block_size + 1)
    if len(data) < minimum:
        raise ValueError(
            f"corpus has {len(data)} bytes; need at least {minimum} for "
            f"train and held-out sequences of length {block_size}"
        )
    split = int(len(data) * TRAIN_FRACTION)
    split = max(block_size + 1, min(split, len(data) - block_size - 1))
    return data[:split], data[split:]


def sample_batch(data, block_size, batch_size, rng):
    """Sample next-byte training pairs from one contiguous byte split."""
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("split is too short for the configured block size")
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([
        np.frombuffer(data[start:start + block_size], dtype=np.uint8)
        for start in starts
    ]).astype(np.int64)
    y = np.stack([
        np.frombuffer(data[start + 1:start + block_size + 1], dtype=np.uint8)
        for start in starts
    ]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def evaluate_loss(model, data, *, block_size, batch_size, batches, seed):
    """Average causal next-byte loss over deterministic held-out crops."""
    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(data, block_size, batch_size, rng)
            _, loss = model(x, y)
            losses.append(float(loss))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _fixed_batch_loss(model, x, y):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    if was_training:
        model.train()
    return float(loss)


def _config_from_args(deep_mem_learnable=False):
    return HOPEConfig(
        vocab_size=256,
        block_size=32,
        n_layer=1,
        n_embd=32,
        chunk_len=4,
        cms_periods=(1, 2),
        deep_mem_learnable=deep_mem_learnable,
    )


def train(args):
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    corpus = load_byte_corpus(args.corpus)
    cfg = None
    previous = None
    if args.resume:
        previous = read_checkpoint(args.resume)
        cfg = HOPEConfig(**previous["config"])
        if (args.deep_mem_learnable is not None
                and args.deep_mem_learnable != cfg.deep_mem_learnable):
            raise ValueError("resume must use the checkpoint's memory mode")
        old_metadata = previous.get("metadata", {})
        fingerprint = hashlib.sha256(corpus).hexdigest()
        if fingerprint != old_metadata.get("corpus_sha256"):
            raise ValueError("resume corpus differs from the checkpoint corpus")
        batch_size = old_metadata["batch_size"]
        learning_rate = old_metadata["learning_rate"]
        seed = old_metadata["seed"]
        for name, supplied, stored in (
            ("batch size", args.batch_size, batch_size),
            ("learning rate", args.learning_rate, learning_rate),
            ("seed", args.seed, seed),
        ):
            if supplied is not None and supplied != stored:
                raise ValueError(f"resume must use the checkpoint's {name}")
    else:
        cfg = _config_from_args(bool(args.deep_mem_learnable))
        batch_size = (DEFAULT_BATCH_SIZE if args.batch_size is None
                      else args.batch_size)
        learning_rate = (DEFAULT_LEARNING_RATE if args.learning_rate is None
                         else args.learning_rate)
        seed = args.seed if args.seed is not None else 0

    if batch_size <= 0 or learning_rate <= 0:
        raise ValueError("batch size and learning rate must be positive")
    train_data, heldout_data = split_byte_corpus(corpus, cfg.block_size)
    fingerprint = hashlib.sha256(corpus).hexdigest()
    torch.manual_seed(seed)
    random.seed(seed)
    data_rng = np.random.default_rng(seed)

    model = HOPE(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    start_step = 0
    if args.resume:
        previous = load_checkpoint(
            args.resume, model, optimizer, data_rng=data_rng, restore_rng=True,
        )
        start_step = int(previous["step"])

    diagnostic_x, diagnostic_y = sample_batch(
        train_data, cfg.block_size, batch_size, np.random.default_rng(seed + 1)
    )
    initial_train_loss = _fixed_batch_loss(model, diagnostic_x, diagnostic_y)
    model.train()
    final_step = start_step
    for final_step in range(start_step + 1, start_step + args.steps + 1):
        x, y = sample_batch(train_data, cfg.block_size, batch_size, data_rng)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at optimizer step {final_step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at optimizer step {final_step}")
        optimizer.step()

    final_train_loss = _fixed_batch_loss(model, diagnostic_x, diagnostic_y)
    heldout_loss = evaluate_loss(
        model, heldout_data, block_size=cfg.block_size, batch_size=batch_size,
        batches=4, seed=seed + 2,
    )
    out_path = Path(args.out) if args.out else (
        Path(args.resume) if args.resume else Path("runs/hope-tiny.pt")
    )
    metadata = {
        "corpus_sha256": fingerprint,
        "corpus_bytes": len(corpus),
        "corpus_source": "builtin" if args.corpus is None else "file",
        "train_fraction": TRAIN_FRACTION,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    save_checkpoint(
        out_path, model, optimizer, step=final_step, data_rng=data_rng,
        metadata=metadata,
    )
    mode = "deep" if cfg.deep_mem_learnable else "matrix"
    source = "builtin" if args.corpus is None else "file"
    print(f"mode={mode} corpus={source} bytes={len(corpus)} block_size={cfg.block_size}")
    print(
        f"train_loss_initial={initial_train_loss:.6f} "
        f"train_loss_final={final_train_loss:.6f} heldout_loss={heldout_loss:.6f}"
    )
    print(f"step={final_step} checkpoint={out_path}")


def _load_model(checkpoint_path):
    checkpoint = read_checkpoint(checkpoint_path)
    model = HOPE(HOPEConfig(**checkpoint["config"])).cpu()
    load_checkpoint(checkpoint_path, model, restore_rng=False)
    model.eval()
    return model, checkpoint


def evaluate(args):
    checkpoint = read_checkpoint(args.checkpoint)
    model = HOPE(HOPEConfig(**checkpoint["config"])).cpu()
    load_checkpoint(args.checkpoint, model, restore_rng=False)
    corpus = load_byte_corpus(args.corpus)
    metadata = checkpoint.get("metadata", {})
    if hashlib.sha256(corpus).hexdigest() != metadata.get("corpus_sha256"):
        raise ValueError("evaluation corpus differs from the checkpoint corpus")
    _, heldout_data = split_byte_corpus(corpus, model.config.block_size)
    loss = evaluate_loss(
        model, heldout_data, block_size=model.config.block_size,
        batch_size=metadata["batch_size"], batches=args.batches, seed=args.seed,
    )
    print(
        f"heldout_loss={loss:.6f} batches={args.batches} "
        f"corpus_bytes={len(corpus)}"
    )


@torch.no_grad()
def generate_bytes(model, prompt, *, max_bytes, temperature, top_k, seed):
    if max_bytes < 0:
        raise ValueError("max-bytes must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = list(prompt)
    if not ids:
        raise ValueError("prompt must contain at least one byte")
    if top_k < 0 or top_k > model.config.vocab_size:
        raise ValueError("top-k must be between 0 and the vocabulary size")
    torch.manual_seed(seed)
    model.eval()
    for _ in range(max_bytes):
        x = torch.tensor([ids], dtype=torch.long)
        # For an observed prefix, byte t's next-byte target is the observed
        # byte t+1. The current final byte has no target and is ignored. Each
        # prefix is replayed from a fresh state, so a partial final chunk is
        # finalized for that prefix and recomputed after the next sample.
        targets = torch.tensor([ids[1:] + [-1]], dtype=torch.long)
        logits, _ = model(x, targets)
        next_logits = logits[0, -1] / temperature
        if top_k:
            threshold = torch.topk(next_logits, top_k).values[-1]
            next_logits[next_logits < threshold] = -float("inf")
        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        ids.append(int(next_id))
    return bytes(ids)


def generate(args):
    model, checkpoint = _load_model(args.checkpoint)
    prompt = args.prompt.encode("utf-8")
    result = generate_bytes(
        model, prompt, max_bytes=args.max_bytes, temperature=args.temperature,
        top_k=args.top_k, seed=args.seed,
    )
    print(
        f"step={checkpoint['step']} generated_bytes={args.max_bytes} "
        f"memory_mode={'deep' if model.config.deep_mem_learnable else 'matrix'}"
    )
    print(result.decode("utf-8", errors="replace"))


def build_parser():
    parser = argparse.ArgumentParser(prog="hope-tiny")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train on built-in or local byte text")
    train_parser.add_argument("--corpus", type=Path, help="optional local raw-byte corpus")
    train_parser.add_argument("--out", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                              help="additional optimizer steps; default: %(default)s")
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--deep-mem-learnable", action="store_true", default=None,
                              help="use the Eq. 91 deep-memory option")
    train_parser.set_defaults(handler=train)

    eval_parser = commands.add_parser("eval", help="evaluate held-out byte loss")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--corpus", type=Path,
                             help="same corpus used for training; defaults to built-in text")
    eval_parser.add_argument("--batches", type=int, default=8)
    eval_parser.add_argument("--seed", type=int, default=101)
    eval_parser.set_defaults(handler=evaluate)

    gen_parser = commands.add_parser("generate", help="generate bytes from a checkpoint")
    gen_parser.add_argument("--checkpoint", type=Path, required=True)
    gen_parser.add_argument("--prompt", default="memory")
    gen_parser.add_argument("--max-bytes", type=int, default=32)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=40)
    gen_parser.add_argument("--seed", type=int, default=0)
    gen_parser.set_defaults(handler=generate)
    return parser


def main(argv=None):
    _configure_cpu_threads()
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"hope-tiny: {exc}") from exc


if __name__ == "__main__":
    main()
