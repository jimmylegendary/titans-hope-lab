#!/usr/bin/env python3
"""Optional CPU execution check for the pinned Llama-3.2-1B backbone.

This checks loading, frozen-backbone gradients, an NMM update, and cached decode.
It is not a quality benchmark and is never run by the CPU CI workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
TITAN_ROOT = PUBLIC_ROOT / "implementations" / "titan-llama"
MODEL = "unsloth/Llama-3.2-1B-Instruct"
REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
SOURCE_GIT_HEAD = "9ea21d6d86367f581f656a3886c8baedaf5697d8"
sys.path.insert(0, str(TITAN_ROOT))


def emit(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download the pinned model files without loading or executing them",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="use an existing cached snapshot without network access",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PUBLIC_ROOT / ".cache" / "huggingface"))
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from huggingface_hub import snapshot_download

    emit("snapshot", model=MODEL, revision=REVISION, local_only=args.local_only)
    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        token=False,
        max_workers=1,
        local_files_only=args.local_only,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    )
    emit("snapshot_ready")
    if args.download_only:
        return 0

    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    available_gib = int(
        next(line.split()[1] for line in meminfo if line.startswith("MemAvailable:"))
    ) / 1024**2
    if available_gib < 14:
        emit(
            "deferred",
            reason="Less than 14 GiB available for the 1B CPU smoke",
            available_gib=round(available_gib, 2),
        )
        return 2

    import torch
    from transformers import AutoConfig, AutoTokenizer

    from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(17)
    start = time.monotonic()
    base_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    titan_config = TitanLLaMAConfig.from_llama_config(
        base_config,
        segment_len=4,
        neural_memory_segment_len=4,
        neural_memory_layers=(0, 1, 2, 3),
        neural_memory_batch_size=4,
        neural_memory_depth=3,
        history_blend=True,
        use_flex_attn=False,
        use_flash_attn=False,
    )
    model = TitanLLaMAForCausalLM.from_pretrained_llama(
        snapshot,
        titan_config=titan_config,
        freeze_backbone=True,
        dtype=torch.float32,
        device_map=None,
        local_files_only=True,
    )
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    is_base = lambda name: "neural_memory" not in name and "to_learned_v_mix" not in name
    assert not any(is_base(name) and param.numel() for name, param in trainable)
    emit(
        "loaded",
        layers=base_config.num_hidden_layers,
        converted_layers=[0, 1, 2, 3],
        trainable=sum(param.numel() for _, param in trainable),
        seconds=round(time.monotonic() - start, 3),
    )

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    text = "Memory lets a language model retain information across several pieces of text. " * 4
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :16]
    assert input_ids.shape[1] == 16
    model.train()
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = output["loss"] if isinstance(output, dict) else output.loss
    assert torch.isfinite(loss).item()
    loss.backward()
    nmm_nonzero = [
        name
        for name, param in trainable
        if param.grad is not None
        and torch.isfinite(param.grad).all().item()
        and param.grad.abs().sum().item() > 0
        and "neural_memory" in name
    ]
    assert nmm_nonzero
    assert all(param.grad is None for name, param in model.named_parameters() if is_base(name))
    emit(
        "backward",
        loss=loss.item(),
        nmm_nonzero_grad_tensors=len(nmm_nonzero),
        base_gradients=False,
        tokens=int(input_ids.numel()),
    )
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=1e-5)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del output, loss

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
        cache = output["past_key_values"] if isinstance(output, dict) else output.past_key_values
        logits = output["logits"] if isinstance(output, dict) else output.logits
        token = logits[:, -1].argmax(-1, keepdim=True)
        del output, logits
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        assert tuple(logits.shape) == (1, 1, base_config.vocab_size)
        assert torch.isfinite(logits).all().item()
        emit("cached_decode", logits_shape=list(logits.shape), cache_type=type(cache).__name__)

    emit(
        "complete",
        status="PASS",
        source_git_head=SOURCE_GIT_HEAD,
        model=MODEL,
        revision=REVISION,
        device="cpu",
        dtype="float32",
        segment_length=4,
        input_tokens=16,
        elapsed_seconds=round(time.monotonic() - start, 3),
        peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        paper_quality_reproduced=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
