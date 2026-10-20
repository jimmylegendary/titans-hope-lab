"""Exactness of the chunk-parallel matrix-memory against a sequential
reference implementation of Eq. 88-90.

The sequential reference applies, token by token:
    M_t = M_{t-1} (alpha_t I - eta_t k_t k_t^T) + eta_t (-e_t) k_t^T
with e_t the mse-gradient of the inner loss evaluated at the *chunk-start*
state (paper §8: "all the gradients ... can be computed in parallel before
starting the processing of the current chunk"), while the retention factor
acts on the running state. Retrieval at token t reads M_{t-1} (zero lag).

The parallel path (WY representation + decayed lower-triangular solve, the
"UT transform" of Gated DeltaNet) must match this recurrence to numerical
precision. In fp64 the match is ~1e-15; we assert 1e-10 to be safe across
BLAS builds. Run on CPU so results are deterministic everywhere:

    pytest tests/test_exactness.py -q
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def sequential_reference(W0, k, v_hat, q, eta, alpha):
    """Token-by-token Eq. 88-90 for a matrix memory M(x) = x + W x."""
    B, T, d = k.shape
    W = W0.clone()
    outs = []
    for t in range(T):
        kt, qt = k[:, t], q[:, t]
        # retrieval BEFORE this token's write (M_{t-1})
        outs.append(qt + torch.einsum("bij,bj->bi", W, qt))
        # inner-loss gradient at the CHUNK-START state W0 (paper §8)
        pred0 = kt + torch.einsum("bij,bj->bi", W0, kt)
        e = (pred0 - v_hat[:, t]) * (2.0 / d)
        a, n = alpha[:, t, None, None], eta[:, t, None, None]
        kk = torch.einsum("bi,bj->bij", kt, kt)
        W = a * W - n * torch.einsum("bij,bjk->bik", W, kk) \
            + n * torch.einsum("bi,bj->bij", -e, kt)
    return torch.stack(outs, dim=1), W


def parallel_path(W0, k, v_hat, q, eta, alpha):
    """The exact dual-form retrieval + chunk-end state, as in hope/model.py
    (extracted with beta = eta/alpha and gates given)."""
    B, T, d = k.shape
    logc = torch.log(alpha).cumsum(1)
    cT = logc[:, -1:].exp()
    d_t = (logc[:, -1:] - logc).exp()
    c_t = logc.exp()
    c_prev = torch.cat([torch.ones_like(c_t[:, :1]), c_t[:, :-1]], dim=1)
    logc_prev = torch.cat([torch.zeros_like(logc[:, :1]), logc[:, :-1]], dim=1)

    bf = eta / alpha
    KK = torch.einsum("bti,bsi->bts", k, k)
    L = torch.tril(KK, diagonal=-1) * bf.unsqueeze(-1)
    eye = torch.eye(T, dtype=k.dtype).expand_as(KK)
    U = torch.linalg.solve_triangular(eye + L, bf.unsqueeze(-1) * k, upper=False)

    mask = torch.tril(torch.ones(T, T, dtype=k.dtype), diagonal=-1)
    kq = torch.einsum("bsi,bti->bts", k, q) * mask
    M0q = torch.einsum("bij,btj->bti", W0, q)
    M0u = torch.einsum("bij,bsj->bsi", W0, U)
    o_hom = c_prev.unsqueeze(-1) * (M0q - torch.einsum("bts,bsi->bti", kq, M0u))

    pred0 = k + torch.einsum("bij,btj->bti", W0, k)
    e0 = (pred0 - v_hat) * (2.0 / d)
    w_col = -eta.unsqueeze(-1) * e0
    decay_ts = torch.exp((logc_prev.unsqueeze(-1) - logc.unsqueeze(1)).clamp(max=0)) * mask
    D_ts = torch.exp((logc.unsqueeze(-1) - logc.unsqueeze(1)).clamp(max=0)) * mask
    A_sys = eye + bf.unsqueeze(-1) * D_ts * KK * mask
    H_hat = torch.linalg.solve_triangular(A_sys, w_col, upper=False)
    o_intra = torch.einsum("bts,bts,bsi->bti", kq, decay_ts, H_hat)

    o = q + o_hom + o_intra
    inhom = torch.einsum("bt,bti,btj->bij", d_t, H_hat, k)
    WU = torch.einsum("bij,bsj->bsi", W0, U)
    W_end = cT.unsqueeze(-1) * (W0 - torch.einsum("bsi,bsj->bij", WU, k)) + inhom
    return o, W_end


def test_matrix_memory_exactness():
    torch.manual_seed(0)
    B, T, d = 2, 64, 32
    dt = torch.float64
    k = torch.nn.functional.normalize(torch.randn(B, T, d, dtype=dt), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, T, d, dtype=dt), dim=-1)
    v_hat = torch.randn(B, T, d, dtype=dt)
    W0 = torch.randn(B, d, d, dtype=dt) * 0.1
    alpha = 0.25 + 0.74 * torch.rand(B, T, dtype=dt)     # in [0.25, 0.99]
    eta = alpha * torch.rand(B, T, dtype=dt)             # beta = eta/alpha <= 1

    o_ref, W_ref = sequential_reference(W0, k, v_hat, q, eta, alpha)
    o_par, W_par = parallel_path(W0, k, v_hat, q, eta, alpha)

    err_o = (o_ref - o_par).abs().max().item()
    err_w = (W_ref - W_par).abs().max().item()
    print(f"retrieval err {err_o:.2e}, state err {err_w:.2e}")
    assert err_o < 1e-10, f"retrieval mismatch: {err_o}"
    assert err_w < 1e-10, f"state mismatch: {err_w}"


def test_causality():
    """Perturbing future tokens must not change past logits."""
    from hope import HOPE, HOPEConfig
    torch.manual_seed(0)
    cfg = HOPEConfig(vocab_size=256, block_size=512, n_layer=2, n_embd=128,
                     chunk_len=128)
    m = HOPE(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 512))
    with torch.no_grad():
        a, _ = m(x, x)
    x2 = x.clone()
    x2[0, 300:] = torch.randint(0, 256, (212,))
    with torch.no_grad():
        b, _ = m(x2, x2)
    diff = (a[0, :299] - b[0, :299]).abs().max().item()
    assert diff == 0.0, f"future perturbation leaked into the past: {diff}"


def test_gradients_reach_gate_memories():
    """The outer loss must train the self-modifying gate memories (STAB-9)."""
    from hope import HOPE, HOPEConfig
    torch.manual_seed(0)
    cfg = HOPEConfig(vocab_size=256, block_size=256, n_layer=1, n_embd=64,
                     chunk_len=64)
    m = HOPE(cfg)
    x = torch.randint(0, 256, (2, 256))
    _, loss = m(x, x)
    loss.backward()
    g = m.blocks[0].smt.inits["eta_w2"].grad
    assert g is not None and g.abs().sum() > 0, "eta memory receives no gradient"


if __name__ == "__main__":
    test_matrix_memory_exactness()
    test_causality()
    test_gradients_reach_gate_memories()
    print("all tests passed")
