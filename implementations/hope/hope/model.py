"""HOPE — a faithful, chunk-parallel PyTorch implementation of
"Nested Learning: The Illusion of Deep Learning Architectures" (arXiv:2512.24695).

HOPE is an attention-free language model built from two components, applied in
every block (paper §7-§8):

  1. Self-Modifying Titans (Eq. 83-93): an associative memory whose *own
     projections* (keys, values, learning rates, forget gates) are themselves
     associative memories, updated online at test time by gradient descent on
     an inner objective. "The model learns how to modify itself."
  2. Continuum Memory System, CMS (Eq. 70-71): a chain of MLP levels updated
     every C^(l) chunks with the *accumulated task-loss gradient* — a spectrum
     of update frequencies between "context" (every token) and "weights"
     (never), inspired by multi-frequency consolidation in the brain.

No official implementation exists at the time of writing.
This file was validated by (i) exactness tests against a sequential reference
(state updates match to ~1e-15 in fp64, see tests/test_exactness.py), and
(ii) full byte-level pretraining runs whose loss trajectories are committed
under assets/ (see README).

Parallelization follows the paper's own prescription (§8, below Eq. 90):
  "all the gradients as well as new keys, values, learning-rates, and weight
   decays can be computed in parallel before starting the processing of the
   current chunk"
Retrieval and generation use the chunk-start state for all tokens in a chunk;
updates take effect from the next chunk (causality preserved). Within a chunk,
the matrix memory additionally uses an *exact* dual form (per-token effective
state) via a lower-triangular solve — see the WY-representation notes inline.

External references used for design choices the paper leaves unspecified:
  [Titans]    Behrouz et al., 2024, arXiv:2501.00663 (surprise gates, momentum)
  [DeltaNet]  Yang et al., 2024, arXiv:2406.06484 (WY representation /
              triangular-solve parallelization of the delta rule; beta <= 1)
  [GatedDN]   Yang et al., 2025, arXiv:2412.06464 (gated delta rule, UT
              transform for decayed triangular systems)
  [TTT]       Sun et al., 2024, arXiv:2407.04620 (dual form of test-time
              gradient states)
  [RMSNorm]   Zhang & Sennrich, 2019, arXiv:1910.07467
Stability devices that appear in no paper (empirically necessary; each one
documented with symptom + diagnosis) are collected in docs/STABILITY.md and
marked "STAB-n" inline below.
"""

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HOPEConfig:
    vocab_size: int = 256           # byte-level by default
    block_size: int = 4096          # data-pipeline convenience only: the model is
                                    # purely recurrent and reads no context limit
                                    # from this field (no positional embedding).
    n_layer: int = 4
    n_embd: int = 512
    chunk_len: int = 512            # parallel processing unit (paper §8: "chunk")
    mem_hidden_mult: int = 2        # hidden width multiplier of the 2-layer memories
    cms_periods: tuple = (1, 2, 4)  # CMS update periods C^(l), in chunks (Eq. 71)
    cms_hidden_mult: int = 2
    bias: bool = False
    # --- Stabilized configuration. Defaults below are the paper-faithful
    #     setting; the released training scripts override two of them
    #     (norm_out=True, cms_second_order=False — see the README recipe table). ---
    delta_retention: bool = True    # Eq. 90 matrix retention (alpha*I - eta*k k^T)
                                    # on the input-side weights (non-expansive).
    norm_target: bool = True        # RMS-normalize the self-generated target v_hat.
                                    # STAB-2: removes the g ~ ||W||^3 feedback loop.
    norm_out: bool = False          # optional LayerNorm on retrieval output.
    update_clip: float = 0.5        # relative update clip ||dW|| <= c*max(||W||,1).
                                    # STAB-1/8: trust-ratio style (cf. LARS,
                                    # You et al. 2017); the max(.,1) floor prevents
                                    # the zero-init lockout described in STABILITY.md.
    max_updates: int = None         # freeze state updates after N chunks (eval guard
                                    # for checkpoints trained with short horizons;
                                    # None = unguarded, our released model needs none).
    deep_mem_learnable: bool = False  # Eq. 91 main memory exactly as written in the
                                    # paper: two-layer residual MLP, learned initial
                                    # weights, updated with the other memories.
                                    # Retrieval reads the chunk-start state (no exact
                                    # dual form exists for this case). Use this to test
                                    # matrix-vs-deep rather than take our word for it.
    cms_second_order: bool = True   # False: skip create_graph for CMS task gradients
                                    # (faster; outer training then cannot shape CMS
                                    # updates through second-order terms).


def mem_forward(w2, w1, x):
    """Eq. 91: M(x) = x + W1 sigma(W2 x). The paper leaves sigma and the hidden
    width unspecified; we use silu and mult 2 (DEVIATIONS.md D10).
    w2: (B, d, h), w1: (B, h, d), x: (B, T, d)."""
    return x + F.silu(x @ w2) @ w1


def make_init(d, h):
    """Learnable initial weights of one two-layer memory.

    W1 is zero-initialized so each memory starts as the identity map
    (M(x) = x): the model's function at t=0 does not depend on random
    memory content. Zero-init of the *output side* is the convention used
    by adapters/residual branches (cf. LoRA, Hu et al. 2021); the resulting
    "first update can never pass a relative clip" hazard is exactly STAB-1,
    handled in the clip itself."""
    return nn.Parameter(torch.randn(d, h) * 0.02), nn.Parameter(torch.zeros(h, d))


# The four self-modifying memories of Eq. 83-86: key, value, learning-rate
# (eta) and forget-gate (alpha) projections. The fifth memory of the paper —
# the main memory M_mem — is kept as a separate *matrix* state ("memM"),
# updated by the exact delta rule (Eq. 88-90); see DEVIATIONS.md D1 for why.
BOXES = ("k", "v", "eta", "alpha")


class CausalConv(nn.Module):
    """Depthwise causal convolution, window 4.

    Paper (following Titans §4.4): "local convolutions with window size of 4"
    on the token stream. In a purely recurrent model this carries local order
    within a chunk (there is no positional embedding anywhere in HOPE — order
    is represented by recurrence across chunks plus this conv within them).
    Zero-initialized so it contributes nothing at t=0 (cf. zero-init residual
    branches, e.g. ReZero, Bachlechner et al. 2020)."""

    def __init__(self, dim, k=4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(dim, dim, k, groups=dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):                  # (B, T, d)
        y = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(y).transpose(1, 2)


class SelfModifyingTitans(nn.Module):
    """Self-Modifying Titans (Eq. 83-93), chunk-parallel with ONE autograd call
    per chunk for all four self-modifying memories.

    Inner objective (Eq. 84-85): each memory M_box learns the association
    k -> v_hat where the target v_hat = M_box(v) is generated by the memory
    itself ("self-referential": the model produces its own learning signal;
    v_hat is treated as a constant via stop-gradient, standard for
    bootstrapped targets).

    Update rule (Eq. 88-90), per token t:
        W <- W (alpha_t I - eta_t k_t k_t^T) - eta_t grad_W l_t
    with token-wise gates alpha_t (retention) and eta_t (write strength),
    themselves produced by the alpha/eta memories — the "surprise"
    formulation of [Titans]: the write is proportional to the gradient of
    the inner loss, so unexpected tokens write more."""

    def __init__(self, cfg: HOPEConfig):
        super().__init__()
        d, h = cfg.n_embd, cfg.n_embd * cfg.mem_hidden_mult
        # The paper's Eq. 91 main memory: a two-layer residual MLP with learned
        # initial weights, updated online like the projection memories. It has
        # no closed form for the intra-chunk state, so retrieval reads the
        # chunk-start state (one-chunk lag) — the honest cost of the deep form.
        # Provided so the matrix-vs-deep question can be tested, not assumed.
        self.boxes = BOXES + ("mem",) if cfg.deep_mem_learnable else BOXES
        self.inits = nn.ParameterDict()
        for box in self.boxes:
            w2, w1 = make_init(d, h)
            self.inits[f"{box}_w2"], self.inits[f"{box}_w1"] = w2, w1
        # Queries are a plain linear projection. The paper is self-contradictory
        # here: the prose says q is "the only non-adaptive projection", while the
        # box set of Eq. 85/88/90 includes q and Table 6 ablates "w/o
        # inner-projection q". We follow the prose (DEVIATIONS.md D11).
        self.W_q = nn.Linear(d, d, bias=False)
        self.conv = CausalConv(d)
        # Gate biases. eta starts small (sigmoid(-3) ~ 0.05): the memory begins
        # as a gentle writer and the outer loop learns to open it. alpha_bias 9:
        # with token-wise alpha the *per-chunk* retention is the product
        # sigmoid(bias)^chunk_len, so the bias must be recalibrated when moving
        # from per-chunk to per-token gates (STAB-7): sigmoid(9)^512 ~ 0.94.
        self.eta_bias = nn.Parameter(torch.tensor(-3.0))
        self.alpha_bias = nn.Parameter(torch.tensor(9.0))
        # Momentum coefficient (paper Table 6 lists momentum as a component but
        # never specifies its value -> we make it a learned scalar; cf. [Titans]
        # "past surprise" S_t = eta S_{t-1} - theta grad l).
        self.mu_logit = nn.Parameter(torch.tensor(-1.5))
        self.update_clip = cfg.update_clip
        self.max_updates = cfg.max_updates
        self.cfg = cfg
        self.out_norm = nn.LayerNorm(d) if cfg.norm_out else nn.Identity()

    def init_state(self, B):
        st = {k: p.unsqueeze(0).expand(B, -1, -1) for k, p in self.inits.items()}
        d = self.cfg.n_embd
        dev_ = self.inits["k_w1"].device
        # Momentum buffers created eagerly so the state dict has a fixed
        # structure regardless of chunk position (needed for checkpoint
        # flattening / activation checkpointing).
        for k, p in list(self.inits.items()):
            st[f"S_{k}"] = torch.zeros(B, *p.shape, device=dev_, dtype=p.dtype)
        if not self.cfg.deep_mem_learnable:
            # Main memory starts at zero: its content is *exactly what was
            # written during this sequence* — nothing else (DEVIATIONS.md D1).
            # In deep mode the main memory lives in the "mem" box instead and
            # these matrix states would be dead weight, so they are not created.
            st["memM"] = torch.zeros(B, d, d, device=dev_,
                                     dtype=self.inits["k_w1"].dtype)
            st["S_memM"] = torch.zeros(B, d, d, device=dev_,
                                       dtype=self.inits["k_w1"].dtype)
        st["n_upd"] = torch.zeros((), dtype=torch.long, device=dev_)
        return st

    def forward(self, x, state):
        """Process one chunk x: (B, T, d) fully in parallel. Returns (o, new_state)."""
        B = x.shape[0]
        # Explicit outer-gradient mode. The chunk loop runs under enable_grad
        # even in eval (test-time learning needs per-chunk graphs), so
        # torch.is_grad_enabled() alone cannot distinguish train from eval here;
        # trusting it corrupts the outer graph (mode-confusion bug).
        outer_grad = getattr(self, "_outer", torch.is_grad_enabled())
        x = x + self.conv(x)   # local context (paper's conv-4)

        # Past max_updates -> frozen state: retrieval only (a guard for
        # evaluating checkpoints beyond their trained update horizon).
        if self.max_updates is not None and int(state["n_upd"]) >= self.max_updates:
            q = F.normalize(self.W_q(x), dim=-1)
            if self.cfg.deep_mem_learnable:
                # Deep mode keeps the main memory in the "mem" box weights.
                o = mem_forward(state["mem_w2"], state["mem_w1"], q)
            else:
                o = q + torch.einsum("bij,btj->bti", state["memM"], q)
            o = self.out_norm(o)
            new_state = dict(state)
            new_state["n_upd"] = state["n_upd"] + 1
            return o, new_state

        with torch.enable_grad():
            keys = [f"{b}_{p}" for b in self.boxes for p in ("w2", "w1")]
            ws = [state[k] for k in keys]
            if (not outer_grad) or (not ws[0].requires_grad):
                ws = [w.detach().requires_grad_(True) for w in ws]
            w = dict(zip(keys, ws))

            # --- Generation + retrieval: all tokens in parallel from the
            #     chunk-start state (Eq. 83, 94). Keys and queries are
            #     l2-normalized (paper, following Titans §4.4). ---
            k = F.normalize(mem_forward(w["k_w2"], w["k_w1"], x), dim=-1)
            v = mem_forward(w["v_w2"], w["v_w1"], x)
            eta_raw = mem_forward(w["eta_w2"], w["eta_w1"], x)
            alpha_raw = mem_forward(w["alpha_w2"], w["alpha_w1"], x)
            q = F.normalize(self.W_q(x), dim=-1)

        T = x.shape[1]
        # --- Token-wise gates (Eq. 86/90: alpha_t, eta_t are per-token) ---
        alpha_t = torch.sigmoid(alpha_raw.mean(-1) + self.alpha_bias)          # (B,T)
        # STAB-4 (alpha floor): alpha < 0.25 is semantically "erase almost
        # everything for this token", but b = eta/alpha below explodes as
        # alpha -> 0 (measured: b ~ 1e6 -> fp32 overflow inside the triangular
        # solve -> NaN, appearing only after ~200 training steps once the gate
        # memories grow confident). The floor is compatible with the paper's
        # alpha in (0,1).
        alpha_t = alpha_t.clamp(min=0.25)
        # Bounded write strength: eta_t <= 1 + alpha_t builds the stability
        # condition of the delta rule into the parameterization.
        eta_t = (1 + alpha_t) * torch.sigmoid(eta_raw.mean(-1) + self.eta_bias)

        # Suffix decay d_t = prod_{s>t} alpha_s — the per-timestep weight each
        # token's gradient contribution carries to the end of the chunk (the
        # product form of applying Eq. 90 token by token).
        logc = torch.log(alpha_t.clamp_min(1e-6)).cumsum(1)                    # (B,T)
        cT = logc[:, -1:].exp()                                                # prod alpha (B,1)
        d_t = (logc[:, -1:] - logc).exp()                                      # (B,T)

        # One autograd call for all four memories: differentiate the weighted
        # sum sum_t (eta_t d_t) l_t, which yields exactly sum_t d_t eta_t g_t.
        og = (lambda t: t) if outer_grad else (lambda t: t.detach())
        with torch.enable_grad():
            # When outer_grad, gates stay differentiable so the outer loop can
            # learn eta/alpha through their effect on the writes (STAB-9 in
            # STABILITY.md: consuming eta only through a detached path leaves
            # the gate untrainable and it silently saturates).
            wsum = og(eta_t * d_t)
            inner_w = 0.0
            for box in self.boxes:
                v_hat = mem_forward(w[f"{box}_w2"], w[f"{box}_w1"], v)
                if self.cfg.norm_target:
                    # STAB-2: RMS-normalize the self-generated target. Without
                    # this, grad ~ ||W||^3: the target grows with the memory,
                    # the gradient grows with the target — a scale feedback
                    # loop no outer loss ever penalizes.
                    d_ = v_hat.shape[-1]
                    v_hat = v_hat * (d_ ** 0.5 / v_hat.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                v_hat = v_hat.detach()      # bootstrapped target (stop-gradient)
                pred = mem_forward(w[f"{box}_w2"], w[f"{box}_w1"], k)
                per_tok = (pred - v_hat).pow(2).mean(-1)                       # (B,T)
                # Eq. 90 sums per-token contributions: sum_t eta_t g_t.
                # Averaging over T instead (a habitual reduction="mean") shrinks
                # every write by 1/T and silently disables the memory (STAB-10).
                inner_w = inner_w + (wsum * per_tok).sum(1).sum()
            # retain_graph is required: the CMS task-gradient (Eq. 71) will
            # traverse this same chunk graph afterwards.
            grads_w = torch.autograd.grad(inner_w, ws, create_graph=outer_grad,
                                          retain_graph=True)

        # --- Homogeneous part of Eq. 90: P = A_T ... A_1, with
        #     A_t = alpha_t I - eta_t k_t k_t^T.
        # WY representation [DeltaNet]: prod_t (I - b_t k_t k_t^T) =
        #     I - sum_t k_t m_t^T, where the rows m_t solve the
        #     lower-triangular linear system
        #     m_t = b_t (k_t - sum_{s<t} (k_t^T k_s) m_s),  b_t = eta_t/alpha_t.
        # Verified against a sequential reference (tests/test_exactness.py). ---
        with torch.autocast(device_type="cuda", enabled=False) if k.is_cuda else contextlib.nullcontext():
            kd = k.detach().float()                                            # fp32: STAB-12,
            # the triangular solve neither supports nor survives bf16.
            # STAB-5 (beta <= 1, the DeltaNet standard): b > 1 "over-erases"
            # beyond full deletion. Trained key maps can collapse entropy-tail
            # inputs (binary blobs in web data) onto a narrow cone of keys;
            # with b > 1 the triangular system then amplifies exponentially
            # (NaN at training step ~2595, reproduced 10/10 with dumped
            # weights + random bytes; 10/10 clean after the cap).
            b = (eta_t / alpha_t).detach().float().clamp(max=1.0)              # (B,T)
            KK = torch.einsum("bti,bsi->bts", kd, kd)                          # (B,T,T)
            L = torch.tril(KK, diagonal=-1) * b.unsqueeze(-1)
            eye = torch.eye(T, device=kd.device).expand_as(KK)
            Mrows = torch.linalg.solve_triangular(eye + L, b.unsqueeze(-1) * kd,
                                                  upper=False)                 # (B,T,d)
        kd, Mrows = kd.to(k.dtype), Mrows.to(k.dtype)

        def apply_P(W):                                                        # W: (B,d,h)
            return cT.unsqueeze(-1) * (W - torch.einsum("bti,bth->bih",
                                                        kd, torch.einsum("btj,bjh->bth", Mrows, W)))

        # --- Momentum (paper Table 6; coefficient unspecified -> learned). ---
        mu = torch.sigmoid(self.mu_logit)

        new_state = {}
        for kk_, W, g in zip(keys, ws, grads_w):
            delta = g                                        # already eta_t d_t-weighted
            if self.update_clip is not None:
                with torch.no_grad():
                    dn = delta.norm(dim=(1, 2), keepdim=True)
                    # STAB-1: floor ||W|| at 1.0. A purely relative clip with
                    # zero-init weights caps the first update at 0 forever —
                    # the memory silently freezes for the whole of training
                    # (diagnosis: finite-difference perturbation of the weight
                    # changes the loss by exactly nothing).
                    wn = W.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
                # STAB-3: the clip factor is a non-differentiable constant.
                # Differentiating through norm at ||g|| = 0 backpropagates 0/0.
                delta = delta * (self.update_clip * wn / dn.clamp_min(1e-12)).clamp(max=1.0).detach()
            S_prev = state.get(f"S_{kk_}")
            if S_prev is None:
                S_prev = torch.zeros_like(W)
            S = mu * S_prev + delta
            new_state[f"S_{kk_}"] = S
            if self.cfg.delta_retention and kk_.endswith("_w2"):
                new_state[kk_] = apply_P(W) - S              # input side: full Eq. 90
            else:
                # Hidden-side / non-delta weights: token-wise scalar decay
                # prod alpha_t. The right-multiplication of Eq. 88 is only
                # type-correct for the layer whose input is k; how to
                # distribute the matrix retention across the layers of a deep
                # memory is undefined in the paper (DEVIATIONS.md D2).
                new_state[kk_] = cT.unsqueeze(-1) * W - S

        # --- deep_mem_learnable: the paper's Eq. 91 memory. It was updated
        #     above as the fifth box; retrieval uses the chunk-start weights,
        #     so token t reads a state that excludes this chunk's writes
        #     (one-chunk lag). No exact dual form exists for this case. ---
        if self.cfg.deep_mem_learnable:
            o = mem_forward(w["mem_w2"], w["mem_w1"], q)
            new_state["n_upd"] = state["n_upd"] + 1
            return (self.out_norm(o) if not isinstance(self.out_norm, nn.Identity)
                    else o), new_state

        # --- Main memory M_mem: a matrix memory M(x) = x + W x, the exact
        #     setting in which Eq. 88-90 are derived. Retrieval below is
        #     *per-token exact* (dual form, cf. [TTT]; decayed triangular
        #     system as in [GatedDN]'s UT transform): token t reads the state
        #     that includes all writes s < t of the same chunk — zero lag. ---
        Wm = state["memM"]                                   # (B,d,d)
        Sm = state.get("S_memM")
        if Sm is None:
            Sm = torch.zeros_like(Wm)
        # Self-generated target v_hat = M(v) = v + W v (chunk-start state).
        vt = (v + torch.einsum("bij,btj->bti", Wm, v))
        if self.cfg.norm_target:
            vt = vt * (vt.shape[-1] ** 0.5 / vt.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        vt = vt.detach()
        # Prefix products R_t = A_1 ... A_t in WY form: R_t = c_t (I - sum_{s<=t} u_s k_s^T),
        # where u solves the same lower-triangular system in the prefix direction.
        with torch.autocast(device_type="cuda", enabled=False) if k.is_cuda else contextlib.nullcontext():
            kf = k.detach().float()
            bf = (eta_t / alpha_t).detach().float().clamp(max=1.0)   # STAB-5 again
            KKf = torch.einsum("bti,bsi->bts", kf, kf)
            Lf = torch.tril(KKf, diagonal=-1) * bf.unsqueeze(-1)
            eyef = torch.eye(T, device=kf.device).expand_as(KKf)
            U = torch.linalg.solve_triangular(eyef + Lf, bf.unsqueeze(-1) * kf, upper=False)
        kf, U = kf.to(k.dtype), U.to(k.dtype)
        # Gate accumulation scalars, prefix form: c_t = prod_{s<=t} alpha_s.
        c_t = logc.exp()                                     # (B,T)
        c_prev = torch.cat([torch.ones_like(c_t[:, :1]), c_t[:, :-1]], dim=1)
        # Retrieval part 1: M0 applied through the decayed prefix operator,
        # M0(R_{t-1} q_t) = c_{t-1} (M0 q_t - sum_{s<t} (k_s^T q_t) M0 u_s).
        kq = torch.einsum("bsi,bti->bts", kf, q)             # (t,s) = k_s . q_t
        mask = torch.tril(torch.ones(T, T, device=q.device), diagonal=-1)
        kq_m = kq * mask
        M0q = torch.einsum("bij,btj->bti", Wm, q)
        M0u = torch.einsum("bij,bsj->bsi", Wm, U)
        o_hom = c_prev.unsqueeze(-1) * (M0q - torch.einsum("bts,bsi->bti", kq_m, M0u))
        # Retrieval part 2 (intra-chunk writes) — the *exact* dual form:
        # if the write columns h_hat solve the lower-triangular system
        #   (I + tril(b_t (c_t/c_s) k_s^T k_t)) h_hat = w,
        # then M_{t-1} = c_{t-1} M0 R~_{t-1} + sum_{s<t} (c_{t-1}/c_s) h_hat_s k_s^T
        # matches the sequential recurrence exactly (cf. [GatedDN] UT transform).
        pred0 = k + torch.einsum("bij,btj->bti", Wm, k)
        e0 = (pred0 - vt) * (2.0 / vt.shape[-1])             # grad of the mse inner loss
        w_col = -og(eta_t).unsqueeze(-1) * e0                                    # (B,T,d)
        logc_prev = torch.cat([torch.zeros_like(logc[:, :1]), logc[:, :-1]], dim=1)
        # Pairwise decay ratios as exp of log-differences (dividing the raw
        # cumulative products underflows). STAB-6: clamp the exponent at 0
        # *before* exp. Valid entries (s < t) are mathematically <= 0, so the
        # clamp is lossless; the masked-out region (s >= t) can reach +hundreds
        # once alpha sits at its floor -> exp = Inf, and IEEE Inf * 0 = NaN:
        # multiplicative masking requires the masked values to be finite too.
        decay_ts = torch.exp((logc_prev.unsqueeze(-1) - logc.unsqueeze(1)).clamp(max=0)) * mask
        with torch.autocast(device_type="cuda", enabled=False) if k.is_cuda else contextlib.nullcontext():
            D_ts = torch.exp((logc.unsqueeze(-1) - logc.unsqueeze(1)).clamp(max=0)).float() * mask
            A_sys = eyef + bf.unsqueeze(-1) * D_ts * KKf * mask
            H_hat = torch.linalg.solve_triangular(A_sys, w_col.float(), upper=False)
        H_hat = H_hat.to(k.dtype)
        o_intra = torch.einsum("bts,bts,bsi->bti", kq_m, decay_ts, H_hat)
        o = q + o_hom + o_intra                              # M_{t-1}(q_t) = q + (...) q_t
        # Chunk-end state: M = c_T M0 R~_T + sum_s (c_T/c_s) h_hat_s k_s^T
        # (+ momentum and clip).
        inhom = torch.einsum("bt,bti,btj->bij", d_t, H_hat, kf)
        if self.update_clip is not None:
            with torch.no_grad():
                fG = (self.update_clip * Wm.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
                      / inhom.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
            inhom = inhom * fG
        Sm = mu * Sm + inhom
        WU = torch.einsum("bij,bsj->bsi", Wm, U)             # W u_s
        W_new = cT.unsqueeze(-1) * (Wm - torch.einsum("bsi,bsj->bij", WU, kf)) + Sm
        new_state["memM"], new_state["S_memM"] = W_new, Sm
        new_state["n_upd"] = state["n_upd"] + 1
        return self.out_norm(o) if not isinstance(self.out_norm, nn.Identity) else o, new_state


class CMS(nn.Module):
    """Continuum Memory System (Eq. 70-71).

    A chain of two-layer MLP levels; level l accumulates the gradient of the
    *task loss* and applies the accumulated update every C^(l) chunks:

        Eq. 71:  theta^(l) <- theta^(l) - eta^(l) sum_{i in chunk window} grad L(theta; x_i)

    The paper is explicit that the teaching signal is the task objective, not
    a local reconstruction loss: "the objective of L(.) is the objective of
    choice for the task at hand, e.g., for language modeling it is next token
    prediction objective." (An easy mechanism to get wrong: a local
    reconstruction or Hebbian objective also trains fine and looks healthy.)

    The paper writes the chain composition (Eq. 70) with the note "we
    disregard normalizations for the sake of clarity" — normalization exists
    but its form is unspecified. We place an RMS normalization on each level's
    output ([RMSNorm]); without it the chain amplifies multiplicatively
    (STAB-11, re-confirmed three times in our experiments)."""

    def __init__(self, cfg: HOPEConfig):
        super().__init__()
        d, h = cfg.n_embd, cfg.n_embd * cfg.cms_hidden_mult
        self.periods = cfg.cms_periods
        self.inits = nn.ParameterDict()
        self.log_lr = nn.ParameterList()
        for ell in range(len(self.periods)):
            w2, w1 = make_init(d, h)
            self.inits[f"{ell}_w2"], self.inits[f"{ell}_w1"] = w2, w1
            # eta^(l) of Eq. 71: the paper does not define or give values ->
            # minimal interpretation: one learned scalar per level.
            self.log_lr.append(nn.Parameter(torch.tensor(-3.0)))
        self.update_clip = cfg.update_clip
        self.max_updates = cfg.max_updates

    def init_state(self, B):
        st = {}
        for ell in range(len(self.periods)):
            for p in ("w2", "w1"):
                w = self.inits[f"{ell}_{p}"].unsqueeze(0).expand(B, -1, -1)
                st[f"{ell}_{p}"] = w
                st[f"{ell}_acc_{p}"] = torch.zeros_like(w)
        st["step"] = torch.zeros((), dtype=torch.long, device=self.inits["0_w1"].device)
        return st

    def forward(self, x, state):
        """Chain forward only; updates happen in apply_grads (the task gradient
        can only be computed once the chunk's loss exists)."""
        outer_grad = getattr(self, "_outer", torch.is_grad_enabled())
        params = []
        for ell in range(len(self.periods)):
            w2, w1 = state[f"{ell}_w2"], state[f"{ell}_w1"]
            if (not outer_grad) or (not w2.requires_grad):
                w2 = w2.detach().requires_grad_(True)
                w1 = w1.detach().requires_grad_(True)
            params += [w2, w1]
            y = mem_forward(w2, w1, x)
            # STAB-11 / paper's elided normalization (see class docstring).
            y = y * (y.shape[-1] ** 0.5 / y.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            x = y
        self._last_params = params   # exposed for the task-gradient call
        return x, state

    def apply_grads(self, state, grads, from_state=False):
        """Eq. 71: acc += eta grad L(task); at each period boundary apply
        theta <- theta - acc and reset the accumulator.

        The period phase is derived from the state's own step counter, so it is
        global across forward calls: streaming a sequence chunk-by-chunk through
        `states=` produces bit-identical results to a single call (Eq. 71's
        index i is a global step, not a per-call one).

        from_state=True reads theta from state tensors instead of
        _last_params (activation-checkpointing mode: no references into the
        checkpointed region are allowed; consumption-learning gradients flow
        through the state tensors of the main graph)."""
        chunk_idx = int(state["step"])
        new_state = dict(state)
        new_state["step"] = state["step"] + 1
        if self.max_updates is not None and chunk_idx >= self.max_updates:
            return new_state
        for ell, period in enumerate(self.periods):
            if from_state:
                w2, w1 = state[f"{ell}_w2"], state[f"{ell}_w1"]
            else:
                w2, w1 = self._last_params[ell * 2], self._last_params[ell * 2 + 1]
            g2, g1 = grads[ell * 2], grads[ell * 2 + 1]
            lr = F.softplus(self.log_lr[ell])
            acc2 = state[f"{ell}_acc_w2"] + lr * g2
            acc1 = state[f"{ell}_acc_w1"] + lr * g1
            # "+2": fire one chunk before the period boundary so the applied
            # update is consumed (and thus supervised) within the training
            # sequence — otherwise the slowest level's updates are never seen
            # by the loss and receive no learning signal (STAB-13).
            if (chunk_idx + 2) % period == 0:
                if self.update_clip is not None:
                    with torch.no_grad():   # STAB-3: clip factor non-differentiable
                        f2 = (self.update_clip * w2.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
                              / acc2.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                        f1 = (self.update_clip * w1.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
                              / acc1.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                    acc2, acc1 = acc2 * f2, acc1 * f1
                new_state[f"{ell}_w2"] = w2 - acc2
                new_state[f"{ell}_w1"] = w1 - acc1
                new_state[f"{ell}_acc_w2"] = torch.zeros_like(acc2)
                new_state[f"{ell}_acc_w1"] = torch.zeros_like(acc1)
            else:
                new_state[f"{ell}_w2"], new_state[f"{ell}_w1"] = w2, w1
                new_state[f"{ell}_acc_w2"], new_state[f"{ell}_acc_w1"] = acc2, acc1
        return new_state


class HOPEBlock(nn.Module):
    """One HOPE block: pre-norm self-modifying Titans, then pre-norm CMS,
    both residual (paper Fig. 5, with the normalizations the paper elides
    "for the sake of clarity" made explicit)."""

    def __init__(self, cfg: HOPEConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.smt = SelfModifyingTitans(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.cms = CMS(cfg)

    def init_state(self, B):
        return {"smt": self.smt.init_state(B), "cms": self.cms.init_state(B)}

    def forward(self, x, state):
        o, smt_state = self.smt(self.ln_1(x), state["smt"])
        x = x + o
        y, cms_state = self.cms(self.ln_2(x), state["cms"])
        x = x + y
        return x, {"smt": smt_state, "cms": cms_state}


class HOPE(nn.Module):
    """HOPE language model: embedding -> N HOPE blocks -> tied LM head.

    Purely recurrent: there is NO positional embedding of any kind. Order is
    carried by the recurrence across chunks and the causal conv within them.
    This is what lets the model stream past its training length (see the
    README extrapolation chart)."""

    def __init__(self, cfg: HOPEConfig):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(HOPEBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, states=None, return_states=False):
        B, T = idx.shape
        x = self.wte(idx)

        if states is None:
            states = [blk.init_state(B) for blk in self.blocks]

        ck = self.config.chunk_len
        outer_grad = torch.is_grad_enabled()
        for blk in self.blocks:
            blk.smt._outer = outer_grad
            blk.cms._outer = outer_grad
        logits_l, loss_sum, ntok = [], 0.0, 0
        use_ckpt = getattr(self, "grad_ckpt", False) and outer_grad

        def flat(st):
            ks = sorted(k for k in st if torch.is_tensor(st[k]))
            return ks, [st[k] for k in ks]

        def unflat(ks, ts):
            return dict(zip(ks, ts))

        for s in range(0, T, ck):
            # The CMS task gradient (Eq. 71) needs a per-chunk graph even in
            # eval — that is the intrinsic cost of test-time learning. We
            # compute under enable_grad and detach at chunk boundaries.
            states_in = [dict(st) for st in states] if use_ckpt else None
            with torch.enable_grad():
                xs = x[:, s : s + ck]
                if not outer_grad:
                    xs = xs.detach()
                if use_ckpt:
                    # Per-block activation checkpointing (peak activation ~ one
                    # chunk); CMS gradients are produced by the side graph below.
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    for li, blk in enumerate(self.blocks):
                        sk, sv = flat(states[li]["smt"])
                        ckk, cv = flat(states[li]["cms"])

                        def blk_fn(xs_, *ts, _blk=blk, _sk=sk, _ck=ckk, _ns=len(sv)):
                            st = {"smt": unflat(_sk, ts[:_ns]), "cms": unflat(_ck, ts[_ns:])}
                            y, st2 = _blk(xs_, st)
                            return (y, *[st2["smt"][k] for k in _sk], *[st2["cms"][k] for k in _ck])

                        out = _ckpt(blk_fn, xs, *sv, *cv, use_reentrant=False)
                        xs = out[0]
                        states[li] = {"smt": unflat(sk, out[1:1+len(sv)]),
                                      "cms": unflat(ckk, out[1+len(sv):])}
                else:
                    for li, blk in enumerate(self.blocks):
                        xs, states[li] = blk(xs, states[li])
                lg = self.lm_head(self.ln_f(xs))
                if targets is not None:
                    yt = targets[:, s : s + ck]
                    nt = (yt != -1).sum()
                    # `mean` cross-entropy is NaN when every target in a
                    # partial generation chunk is ignored. Sum then divide by
                    # the clamped valid-token count so an unsupervised tail
                    # contributes a finite zero loss and zero CMS gradient.
                    lc = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yt.reshape(-1),
                                         ignore_index=-1, reduction="sum") / nt.clamp_min(1)
                    # --- CMS update (Eq. 71): accumulate the task-loss gradient.
                    #     This chunk's loss affects state from the next chunk on
                    #     (causality). ---
                    if use_ckpt:
                        # Side graph: re-forward with detached states to obtain
                        # the CMS gradients (they are consumed detached anyway);
                        # the new theta is built from main-graph state tensors
                        # (from_state=True) so consumption learning is preserved
                        # without referencing into the checkpointed region.
                        with torch.enable_grad():
                            xs_d = x[:, s : s + ck].detach()
                            side_p = []
                            for li, blk in enumerate(self.blocks):
                                st_d = {"smt": {k: (v.detach() if torch.is_tensor(v) else v)
                                                for k, v in states_in[li]["smt"].items()},
                                        "cms": {k: (v.detach().requires_grad_(True)
                                                    if not k.startswith(("S_",)) and "acc" not in k and (k.endswith("_w2") or k.endswith("_w1"))
                                                    else (v.detach() if torch.is_tensor(v) else v))
                                                for k, v in states_in[li]["cms"].items()}}
                                xs_d, _ = blk(xs_d, st_d)
                                side_p += blk.cms._last_params
                            lg_d = self.lm_head(self.ln_f(xs_d))
                            lc_d = F.cross_entropy(
                                lg_d.reshape(-1, lg_d.size(-1)), yt.reshape(-1),
                                ignore_index=-1, reduction="sum",
                            ) / nt.clamp_min(1)
                            gs_d = torch.autograd.grad(lc_d, side_p, allow_unused=True)
                        gs_d = [torch.zeros_like(p) if g is None else g.detach()
                                for p, g in zip(side_p, gs_d)]
                        np_ = len(self.config.cms_periods) * 2
                        for li, blk in enumerate(self.blocks):
                            states[li] = dict(states[li])
                            states[li]["cms"] = blk.cms.apply_grads(
                                states[li]["cms"], gs_d[li * np_:(li + 1) * np_],
                                from_state=True)
                        loss_sum = loss_sum + lc * nt
                        ntok = ntok + nt
                    else:
                        all_p = [p for blk in self.blocks for p in blk.cms._last_params]
                        so = outer_grad and self.config.cms_second_order
                        gs = torch.autograd.grad(lc, all_p, create_graph=so,
                                                 retain_graph=outer_grad, allow_unused=True)
                        if not so:
                            gs = [None if g is None else g.detach() for g in gs]
                        gs = [torch.zeros_like(p) if g is None else g for p, g in zip(all_p, gs)]
                        np_ = len(self.config.cms_periods) * 2
                        for li, blk in enumerate(self.blocks):
                            states[li] = dict(states[li])
                            states[li]["cms"] = blk.cms.apply_grads(
                                states[li]["cms"], gs[li * np_:(li + 1) * np_])
                        loss_sum = loss_sum + (lc if outer_grad else lc.detach()) * nt
                        ntok = ntok + nt
            if not getattr(self, "skip_logits", False):
                logits_l.append(lg if outer_grad else lg.detach())
            if not outer_grad:
                # Eval: cut the state graph at chunk boundaries. The blocks run
                # under enable_grad and would otherwise keep extending one
                # graph across the whole stream (unbounded memory).
                for li in range(len(self.blocks)):
                    states[li] = {kk: ({k2: (v2.detach() if torch.is_tensor(v2) else v2)
                                        for k2, v2 in vv.items()} if isinstance(vv, dict)
                                       else (vv.detach() if torch.is_tensor(vv) else vv))
                                  for kk, vv in states[li].items()}
        logits = torch.cat(logits_l, dim=1) if logits_l else None
        loss = (loss_sum / ntok.clamp_min(1)) if targets is not None else None
        if targets is None:
            logits = logits[:, [-1]]
        if return_states:
            return logits, loss, states
        return logits, loss
