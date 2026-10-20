#!/usr/bin/env python3
"""Run a tiny synthetic Titan-Llama train/eval/generate/checkpoint cycle on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_titan_llama import (  # noqa: E402
    TrainingConfig,
    create_model_and_optimizer,
    load_checkpoint,
    save_checkpoint,
)


def _tiny_config(*, steps: int, freeze_backbone: bool) -> TrainingConfig:
    return TrainingConfig(
        model_name="titan-llama-tiny-cpu",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        segment_len=4,
        neural_memory_layers=(0,),
        neural_memory_segment_len=4,
        neural_memory_batch_size=4,
        neural_memory_depth=3,
        history_blend=True,
        use_flex_attn=False,
        use_flash_attn=False,
        total_tokens=steps * 8,
        batch_size=1,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        model_dtype="float32",
        use_pretrained_backbone=False,
        base_model_name=None,
        freeze_backbone=freeze_backbone,
    )


def _assert_state_equal(expected, actual, name: str):
    if isinstance(expected, torch.Tensor):
        equal = isinstance(actual, torch.Tensor) and torch.equal(expected, actual)
    elif isinstance(expected, dict):
        equal = expected.keys() == actual.keys() and all(
            _assert_state_equal(expected[key], actual[key], f"{name}.{key}")
            for key in expected
        )
    elif isinstance(expected, (tuple, list)):
        equal = (
            isinstance(actual, type(expected))
            and len(expected) == len(actual)
            and all(
                _assert_state_equal(left, right, f"{name}[{index}]")
                for index, (left, right) in enumerate(zip(expected, actual))
            )
        )
    else:
        equal = expected == actual
    if not equal:
        raise RuntimeError(f"checkpoint restore mismatch in {name}")
    return True


def run_smoke(output_dir, *, steps: int = 1, seed: int = 42, freeze_backbone: bool = False):
    """Run a self-contained synthetic training and inference cycle; return its metrics."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _tiny_config(steps=steps, freeze_backbone=freeze_backbone)
    config.output_dir = str(output_dir)
    model, optimizer, scheduler = create_model_and_optimizer(config, torch.device("cpu"))
    input_ids = torch.tensor([[1, 5, 7, 9, 11, 13, 15, 17]], dtype=torch.long)

    last_loss = None
    model.train()
    for _ in range(steps):
        model.reset_memory_states()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        last_loss = output["loss"]
        if not bool(torch.isfinite(last_loss)):
            raise FloatingPointError("non-finite loss in synthetic CPU training run")
        last_loss.backward()
        optimizer.step()
        scheduler.step()

    checkpoint_path = save_checkpoint(
        model, optimizer, scheduler, config, steps, float(last_loss.detach())
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model.eval()
    model.reset_memory_states()
    with torch.no_grad():
        eval_output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    eval_loss = float(eval_output["loss"])

    prompt = input_ids[:, :4]
    generated = model.generate_with_titan_memory(
        input_ids=prompt,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    generated_tokens = int(generated.shape[1] - prompt.shape[1])
    if generated_tokens < 1:
        raise RuntimeError("generation returned no new tokens")

    torch.manual_seed(seed + 1)
    restored, restored_optimizer, restored_scheduler = create_model_and_optimizer(
        config, torch.device("cpu")
    )
    resume_step = load_checkpoint(
        restored, restored_optimizer, restored_scheduler, checkpoint_path
    )
    original_state = model.state_dict()
    restored_state = restored.state_dict()
    if original_state.keys() != restored_state.keys():
        raise RuntimeError("checkpoint restore changed the model state-dict keys")
    for name, value in original_state.items():
        if not torch.equal(value, restored_state[name]):
            raise RuntimeError(f"checkpoint restore mismatch in {name}")
    _assert_state_equal(optimizer.state_dict(), restored_optimizer.state_dict(), "optimizer")
    _assert_state_equal(scheduler.state_dict(), restored_scheduler.state_dict(), "scheduler")

    return {
        "device": "cpu",
        "train_steps": steps,
        "train_loss": float(last_loss.detach()),
        "eval_loss": eval_loss,
        "generated_tokens": generated_tokens,
        "resume_step": resume_step,
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tiny-cpu")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    result = run_smoke(
        args.output_dir,
        steps=args.steps,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
