"""CPU-only runtime regressions for the Titan-Llama training factory."""

import importlib
import importlib.util

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from train_titan_llama import (
    TrainingConfig,
    create_model_and_optimizer,
    load_checkpoint,
    save_checkpoint,
)


def _tiny_config(*, use_pretrained_backbone, base_model_name=None, freeze_backbone=True):
    return TrainingConfig(
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
        total_tokens=8,
        batch_size=1,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        model_dtype="float32",
        use_pretrained_backbone=use_pretrained_backbone,
        base_model_name=base_model_name,
        freeze_backbone=freeze_backbone,
    )


def _save_tiny_llama(directory):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(directory, safe_serialization=False)
    return str(directory)


def test_pretrained_factory_freezes_backbone_and_trains_neural_memory(tmp_path):
    torch.set_num_threads(1)
    base_model = _save_tiny_llama(tmp_path / "base")
    config = _tiny_config(use_pretrained_backbone=True, base_model_name=base_model)

    model, optimizer, _ = create_model_and_optimizer(config, torch.device("cpu"))
    layer = model.backbone.model.layers[0]
    frozen = (
        model.backbone.model.embed_tokens.weight,
        layer.mlp.gate_proj.weight,
        layer.input_layernorm.weight,
        layer.segmented_attn.to_q.weight,
    )
    assert all(not parameter.requires_grad for parameter in frozen)

    trainable_memory = [
        parameter
        for name, parameter in model.named_parameters()
        if "neural_memory.memory_model_parameters" in name and parameter.requires_grad
    ]
    assert trainable_memory, "the learned neural-memory initial weights must remain trainable"

    input_ids = torch.tensor([[1, 5, 7, 9, 11, 13, 15, 17]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False)["loss"]
    loss.backward()

    assert all(parameter.grad is None for parameter in frozen)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in trainable_memory
    )
    updated = next(
        parameter for parameter in trainable_memory
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
    )
    before = updated.detach().clone()
    optimizer.step()
    assert not torch.equal(before, updated.detach())


def test_pretrained_factory_keeps_backbone_trainable_when_freezing_is_disabled(tmp_path):
    torch.set_num_threads(1)
    base_model = _save_tiny_llama(tmp_path / "base")
    config = _tiny_config(
        use_pretrained_backbone=True,
        base_model_name=base_model,
        freeze_backbone=False,
    )

    model, optimizer, _ = create_model_and_optimizer(config, torch.device("cpu"))
    embedding = model.backbone.model.embed_tokens.weight
    assert embedding.requires_grad
    assert any(embedding is parameter for group in optimizer.param_groups for parameter in group["params"])


@pytest.mark.parametrize("base_model_name", (None, ""))
def test_pretrained_factory_requires_a_nonempty_model_path(base_model_name):
    config = _tiny_config(use_pretrained_backbone=True, base_model_name=base_model_name)

    with pytest.raises(ValueError, match="base_model_name"):
        create_model_and_optimizer(config, torch.device("cpu"))


def test_scratch_factory_builds_and_trains_a_random_hf_backbone():
    torch.set_num_threads(1)
    config = _tiny_config(use_pretrained_backbone=False, freeze_backbone=False)

    model, optimizer, _ = create_model_and_optimizer(config, torch.device("cpu"))
    assert isinstance(model.backbone, LlamaForCausalLM)
    assert model.backbone.config.vocab_size == 256
    assert model.backbone.config.hidden_size == 64

    input_ids = torch.tensor([[1, 5, 7, 9, 11, 13, 15, 17]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False)["loss"]
    loss.backward()
    assert model.backbone.model.embed_tokens.weight.grad is not None
    assert torch.count_nonzero(model.backbone.model.embed_tokens.weight.grad).item() > 0
    optimizer.step()


def test_tiny_cpu_entrypoint_trains_evaluates_generates_and_restores_checkpoint(tmp_path):
    spec = importlib.util.find_spec("scripts.titan_llama_smoke")
    assert spec is not None, "the standalone synthetic CPU smoke entrypoint must be packaged in the repo"
    smoke = importlib.import_module("scripts.titan_llama_smoke")

    result = smoke.run_smoke(output_dir=tmp_path, steps=1, seed=11)

    assert result["train_steps"] == 1
    assert result["resume_step"] == 1
    assert torch.isfinite(torch.tensor(result["eval_loss"]))
    assert result["generated_tokens"] >= 1
    assert (tmp_path / "latest_checkpoint.pt").is_file()


def _assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


def _one_tiny_train_step(model, optimizer, scheduler, input_ids):
    model.train()
    model.reset_memory_states()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False)["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_partial_checkpoint_resume_restores_optimizer_scheduler_and_next_step(tmp_path):
    torch.set_num_threads(1)
    config = _tiny_config(use_pretrained_backbone=False, freeze_backbone=False)
    config.total_tokens = 16
    config.total_steps = 2
    config.output_dir = str(tmp_path)
    torch.manual_seed(21)
    continuous, optimizer, scheduler = create_model_and_optimizer(config, torch.device("cpu"))
    input_ids = torch.tensor([[1, 5, 7, 9, 11, 13, 15, 17]])

    loss = _one_tiny_train_step(continuous, optimizer, scheduler, input_ids)
    checkpoint_path = save_checkpoint(continuous, optimizer, scheduler, config, step=0, loss=loss)

    torch.manual_seed(22)
    resumed, resumed_optimizer, resumed_scheduler = create_model_and_optimizer(
        config, torch.device("cpu")
    )
    next_step = load_checkpoint(resumed, resumed_optimizer, resumed_scheduler, checkpoint_path)
    assert next_step == 1
    _assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(scheduler.state_dict(), resumed_scheduler.state_dict())

    # The training loop resets Titan sequence state at each batch boundary.
    torch.manual_seed(99)
    _one_tiny_train_step(continuous, optimizer, scheduler, input_ids)
    torch.manual_seed(99)
    _one_tiny_train_step(resumed, resumed_optimizer, resumed_scheduler, input_ids)
    _assert_nested_equal(continuous.state_dict(), resumed.state_dict())
    _assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(scheduler.state_dict(), resumed_scheduler.state_dict())
