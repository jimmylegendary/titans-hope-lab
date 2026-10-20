"""Atomic checkpoints for optimizer-step boundaries of independent byte streams."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch


CHECKPOINT_VERSION = 1


def save_checkpoint(path, model, optimizer, *, step, data_rng, metadata=None):
    """Save model, optimizer, and CPU RNG state atomically.

    Runtime autograd graphs and in-flight memory states are intentionally not
    serialized. Callers save after an optimizer step and at a stream boundary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "step": int(step),
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy_data": data_rng.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
        },
        "metadata": dict(metadata or {}),
    }
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temp_file:
            temp_name = temp_file.name
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        raise
    return path


def read_checkpoint(path):
    """Read and validate a checkpoint without loading weights or changing RNG."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {checkpoint.get('format_version')!r}"
        )
    return checkpoint


def load_checkpoint(path, model, optimizer=None, *, data_rng=None, restore_rng=True):
    """Restore model and optional optimizer; restore CPU/data RNG for resume.

    Set ``restore_rng=False`` for read-only evaluation or generation, where a
    caller may establish a separate sampling seed.
    """
    checkpoint = read_checkpoint(path)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError("checkpoint has no optimizer state for resume")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if data_rng is not None:
            data_rng.bit_generator.state = rng["numpy_data"]
    return checkpoint
