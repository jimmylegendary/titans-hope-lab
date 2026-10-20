import torch
import pytest
import random
import os
import re
import subprocess
import sys
import numpy as np

from hope import HOPE, HOPEConfig


def tiny_config(*, deep_mem_learnable=False):
    return HOPEConfig(
        vocab_size=256,
        block_size=32,
        n_layer=1,
        n_embd=32,
        chunk_len=4,
        cms_periods=(1, 2),
        deep_mem_learnable=deep_mem_learnable,
    )


def test_partial_target_chunk_is_finite_and_generation_updates_only_smt():
    torch.manual_seed(0)
    model = HOPE(tiny_config()).eval()
    # Five inputs make one full four-token chunk and one one-token tail. The
    # last input has no known next-byte target yet, as in autoregressive use.
    x = torch.tensor([[3, 5, 7, 11, 13]])
    targets = torch.tensor([[5, 7, 11, 13, -1]])

    logits, loss, trained_states = model(x, targets, return_states=True)
    assert logits.shape == (1, 5, 256)
    assert torch.isfinite(loss), f"ignored tail poisoned the objective: {loss}"
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name

    with torch.no_grad():
        last_logits, no_loss, generation_states = model(x, return_states=True)

    assert trained_states[0]["cms"]["step"].item() == 2
    assert trained_states[0]["smt"]["n_upd"].item() == 2

    # targets=None is the generation contract: only the final logit is
    # returned, self-modifying memory writes still occur, and task-loss CMS
    # has no supervision signal to consume.
    assert last_logits.shape == (1, 1, 256)
    assert no_loss is None
    assert generation_states[0]["cms"]["step"].item() == 0
    assert generation_states[0]["smt"]["n_upd"].item() == 2


@pytest.mark.parametrize("deep_mem_learnable", [False, True])
def test_checkpointed_all_ignored_tail_keeps_side_cms_state_and_gradients_finite(
    deep_mem_learnable, monkeypatch,
):
    import hope.model as model_module

    torch.manual_seed(19)
    model = HOPE(tiny_config(deep_mem_learnable=deep_mem_learnable))
    model.grad_ckpt = True
    x = torch.tensor([[3, 5, 7, 11, 13]])
    targets = torch.tensor([[5, 7, 11, 13, -1]])
    cross_entropy = model_module.F.cross_entropy
    chunk_losses = []

    def record_chunk_loss(*args, **kwargs):
        value = cross_entropy(*args, **kwargs)
        chunk_losses.append(value)
        return value

    monkeypatch.setattr(model_module.F, "cross_entropy", record_chunk_loss)

    logits, loss, states = model(x, targets, return_states=True)
    assert logits.shape == (1, 5, 256)
    assert torch.isfinite(loss)
    assert len(chunk_losses) == 4  # main and CMS side graph for each chunk
    for index, chunk_loss in enumerate(chunk_losses):
        assert torch.isfinite(chunk_loss), f"non-finite chunk loss at call {index}: {chunk_loss}"
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    cms_state = states[0]["cms"]
    assert cms_state["step"].item() == 2
    for name, value in cms_state.items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), name


@pytest.mark.parametrize("deep_mem_learnable", [False, True])
def test_matrix_and_deep_paths_train_finite_gradients_and_update_both_memories(
    deep_mem_learnable,
):
    torch.manual_seed(1)
    model = HOPE(tiny_config(deep_mem_learnable=deep_mem_learnable))
    x = torch.arange(64).reshape(2, 32) % 256
    targets = torch.roll(x, shifts=-1, dims=1)
    initial = [blk.init_state(2) for blk in model.blocks]

    logits, loss, states = model(x, targets, return_states=True)
    assert logits.shape == (2, 32, 256)
    assert torch.isfinite(loss)
    loss.backward()

    smt = model.blocks[0].smt
    cms = model.blocks[0].cms
    for name, grad in (
        ("eta memory", smt.inits["eta_w2"].grad),
        ("CMS level weight", cms.inits["0_w1"].grad),
        ("CMS learning rate", cms.log_lr[0].grad),
    ):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name

    smt_after, cms_after = states[0]["smt"], states[0]["cms"]
    assert smt_after["n_upd"].item() == 8
    assert cms_after["step"].item() == 8
    memory_name = "mem_w1" if deep_mem_learnable else "memM"
    assert not torch.equal(initial[0]["smt"][memory_name], smt_after[memory_name])
    assert not torch.equal(initial[0]["cms"]["0_w1"], cms_after["0_w1"])


def _assert_nested_close(left, right, *, path="state"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_nested_close(left[key], right[key], path=f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_close(left_item, right_item, path=f"{path}[{index}]")
    elif torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7, msg=path)
    else:
        assert left == right, path


@pytest.mark.parametrize("deep_mem_learnable", [False, True])
def test_aligned_streaming_matches_whole_call_for_both_memory_modes(deep_mem_learnable):
    torch.manual_seed(2)
    model = HOPE(tiny_config(deep_mem_learnable=deep_mem_learnable)).eval()
    x = torch.randint(0, 256, (2, 32))
    targets = torch.roll(x, shifts=-1, dims=1)

    with torch.no_grad():
        whole_logits, whole_loss, whole_states = model(x, targets, return_states=True)
        pieces, piece_losses, streamed_states = [], [], None
        for start in range(0, x.shape[1], model.config.chunk_len):
            logits, loss, streamed_states = model(
                x[:, start:start + 4], targets[:, start:start + 4],
                states=streamed_states, return_states=True,
            )
            pieces.append(logits)
            piece_losses.append(loss)

    torch.testing.assert_close(whole_logits, torch.cat(pieces, dim=1))
    torch.testing.assert_close(whole_loss, torch.stack(piece_losses).mean())
    _assert_nested_close(whole_states, streamed_states)


def _checkpoint_train_step(model, optimizer, data_rng):
    offset = int(data_rng.integers(0, 256)) + random.randrange(256)
    x = (torch.randint(0, 256, (2, 32)) + offset) % 256
    targets = torch.roll(x, shifts=-1, dims=1)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, targets)[1]
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_checkpoint_restores_model_optimizer_and_all_rng_at_step_boundary(tmp_path):
    from hope.checkpoint import load_checkpoint, save_checkpoint

    torch.manual_seed(17)
    random.seed(23)
    data_rng = np.random.default_rng(31)
    cfg = tiny_config()
    model = HOPE(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _checkpoint_train_step(model, optimizer, data_rng)
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(checkpoint, model, optimizer, step=1, data_rng=data_rng)

    expected_loss = _checkpoint_train_step(model, optimizer, data_rng)
    restored_model = HOPE(cfg)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(
        checkpoint, restored_model, restored_optimizer, data_rng=data_rng,
    )
    restored_loss = _checkpoint_train_step(restored_model, restored_optimizer, data_rng)

    assert loaded["step"] == 1
    torch.testing.assert_close(expected_loss, restored_loss, rtol=0, atol=0)
    _assert_nested_close(model.state_dict(), restored_model.state_dict())
    _assert_nested_close(optimizer.state_dict(), restored_optimizer.state_dict())


def test_tiny_cli_trains_evaluates_generates_and_resumes_without_downloads(tmp_path):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })

    def run_cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "hope.cli", *args],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        return result.stdout

    checkpoint = tmp_path / "tiny.pt"
    trained = run_cli("train", "--steps", "12", "--seed", "7", "--out", str(checkpoint))
    assert "corpus=builtin" in trained
    losses = re.search(r"train_loss_initial=([0-9.]+) train_loss_final=([0-9.]+)", trained)
    assert losses, trained
    assert float(losses.group(2)) < float(losses.group(1)), trained
    assert checkpoint.is_file()

    evaluated = run_cli("eval", "--checkpoint", str(checkpoint), "--batches", "2")
    assert re.search(r"heldout_loss=[0-9.]+", evaluated), evaluated

    generated = run_cli(
        "generate", "--checkpoint", str(checkpoint), "--prompt", "memory", "--max-bytes", "6", "--seed", "3",
    )
    assert "generated_bytes=6" in generated

    resumed = tmp_path / "resumed.pt"
    resumed_output = run_cli(
        "train", "--resume", str(checkpoint), "--steps", "2", "--out", str(resumed),
    )
    assert "step=14" in resumed_output
    assert resumed.is_file()

    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes((b"a small local byte corpus for self contained tests.\n" * 96))
    external = tmp_path / "external.pt"
    external_output = run_cli(
        "train", "--corpus", str(corpus), "--steps", "1", "--out", str(external),
    )
    assert "corpus=file" in external_output
    assert external.is_file()
