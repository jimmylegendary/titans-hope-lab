# Deviations and under-specified choices

Our faithfulness policy: **everything except scale and data follows the paper;
every place where the paper under-specifies is documented here**, with the
choice we made, the evidence behind it, and the prior work the choice comes
from. If you believe any entry contradicts the authors' intent, please open an
issue — this file is the contract.

Paper: "Nested Learning: The Illusion of Deep Learning Architectures"
(arXiv:2512.24695). No official code exists at the time of writing.

---

## D1. Main memory: matrix memory instead of Eq. 91 deep memory (a trade-off, measured)

**Paper**: Eq. 91 defines the memory as a two-layer residual MLP
`M(x) = x + W1 sigma(W2 x)`; Eq. 88-90 derive the delta-rule update in the
matrix-memory setting.

**We ship**: a plain matrix memory `M(x) = x + W x`, zero-initialized, as the
main memory (`memM`), with the four self-modifying projection memories kept as
two-layer MLPs per Eq. 91. `HOPEConfig(deep_mem_learnable=True)` provides the
paper's Eq. 91 deep memory exactly as written.

**Why we default to the matrix form**

1. *Exactness.* Eq. 88-90 are derived in the matrix-memory setting, and only
   there does our chunk-parallel path reproduce the token-by-token recurrence
   exactly (D5, `tests/test_exactness.py`). A learnable two-layer memory has no
   closed form for the intra-chunk state; `deep_mem_learnable=True` therefore
   reads the chunk-start state (one-chunk retrieval lag).
2. *Legibility.* A zero-initialized matrix memory contains, at any point in a
   sequence, exactly what was written during that sequence and nothing else,
   which makes it possible to reason about what the memory holds. This is the
   fast-weights / linear-attention convention (state starts at zero; cf. TTT,
   arXiv:2407.04620), and it echoes Titans' stated design principle that the
   long-term memory is a *contextual* memory whose "output is fully depend on
   the context", with data-independent knowledge assigned to a separate
   persistent-memory component (arXiv:2501.00663 §3.1, §3.3). Neither paper
   states an initialization for the memory state; widely used Titans
   implementations do learn it. **Note that we predicted this property would
   also improve recall, and measured that it does not — see below.**

**What we measured (fair test, 2026-08).** We implemented the paper's Eq. 91
memory exactly — two-layer residual MLP, learned initial weights, updated with
the other memories (`deep_mem_learnable=True`) — and trained it under conditions
identical to the released matrix run: same data, seed, schedule, 0.3B bytes,
18,310 steps. The deep arm carries 14% more parameters (34.8M vs 30.6M), an
advantage it keeps.

| | matrix (default) | deep, Eq. 91 |
|---|---|---|
| held-out bits/byte | **1.721** | 1.743 |
| needle recall @ 1K bytes | +2.586 +/- 0.043 | **+2.944 +/- 0.063** |
| needle recall @ 6.6K bytes | +2.102 +/- 0.065 | **+2.477 +/- 0.083** |

(recall: `scripts/eval_recall.py`, n=40 paired trials, mean +/- s.e.)

**This refuted our previous reasoning, and we are keeping the record straight.**
We had argued that learned initial weights would swallow test-time writes and
therefore hurt recall. The opposite is true: the paper's deep memory recalls
*better* at both distances, by ~0.36-0.38 bits/byte (roughly 4 and 3.6 standard
errors). The matrix memory is ahead only on mean loss, and only by 0.022 while
using fewer parameters — i.e. it is more parameter-efficient, not more
memory-capable. An earlier A/B of ours that showed matrix ahead by 0.062 does
not reproduce; that comparison used a frozen-feature variant which is not part
of this release.

So the honest statement of the trade-off is:

- **matrix**: exact chunk-parallel semantics (Eq. 88-90 are derived in this
  setting; `tests/test_exactness.py` holds only here), better parameter
  efficiency, and a state whose content is exactly the current sequence's
  writes — which is what makes the memory legible for analysis.
- **deep (paper)**: more expressive, measurably better recall, at the cost of
  an approximate intra-chunk state and more parameters.

We keep matrix as the default because the exactness guarantee is the load-bearing
claim of this repository, not because it is the better memory. If you care about
recall more than about exactness, use `deep_mem_learnable=True`; the paper's
design wins there and Titans §5.5 predicted it.

Caveats that still apply: one seed, one scale (30.6M / 0.3B / 2048 context), and
a synthetic recall probe. Multi-seed replication is open.

## D2. Distributing matrix retention across deep-memory layers (undefined in paper)

Eq. 88's right-multiplication `W (alpha I - eta k k^T)` is type-correct only
for the layer whose input is `k`. For a two-layer memory the paper never says
how retention distributes across `W1`/`W2`. We apply the full matrix retention
to the input-side weights (`w2`, where the types match) and token-wise scalar
decay `prod_t alpha_t` to the hidden-side weights (`w1`). Configurable via
`delta_retention`.

## D3. CMS level learning rates (form and values undefined in paper)

Eq. 71 writes a time-subscripted `eta_t^(l)` ("learning rates" per level), but
the paper neither defines its functional form nor gives values. We simplify to
one learned scalar per level (`softplus`-parameterized) — this drops the time
dependence the subscript implies, which we note explicitly as a simplification
rather than a reading of the paper.

## D4. Momentum coefficient (listed, never specified)

Paper Table 6 lists momentum among the update components (inherited from
Titans, arXiv:2501.00663, "past surprise"), but no coefficient or
parameterization is given. We use one learned scalar (`sigmoid(mu_logit)`,
initialized to ~0.18). Titans makes it data-dependent; we found a learned
scalar sufficient at our scale and note the difference.

## D5. Chunk-parallel realization (paper gives the strategy, not the algebra)

§8 prescribes computing all gradients and gate elements in parallel at the
chunk start. The concrete parallel algebra is ours, assembled from published
primitives:
- WY representation of products of `(I - b k k^T)` factors and the
  lower-triangular solve: DeltaNet parallelization (arXiv:2406.06484).
- Decay-weighted triangular system ("UT transform") for the gated case, and
  the exact per-token dual form of the intra-chunk state: Gated DeltaNet
  (arXiv:2412.06464); dual-form framing from TTT (arXiv:2407.04620).
- Gradients for the four projection memories are taken at the chunk-start
  state (exactly as §8 specifies), applied with the per-token *scalar* decay
  weighting `sum_t d_t eta_t g_t` via a single weighted-loss autograd call.
  Note the approximation this contains: unrolling Eq. 90 token-by-token, each
  gradient contribution should also be right-multiplied by the remaining
  `(I - b k k^T)` matrix factors; for the projection memories we keep only the
  scalar `prod alpha` part of that suffix product (the matrix factor is applied
  to the homogeneous term). The main matrix memory (`memM`) path carries the
  full cross terms and is exact.
- The paper's §8.2 additionally says two *different* chunk sizes were used
  (one for the main memory, one for the other memories). We use a single
  `chunk_len` for everything, including the CMS period unit — a simplification,
  not a reading of the paper.

`tests/test_exactness.py` pins the matrix-memory semantics: the parallel path
equals the sequential recurrence to fp64 machine precision (~1e-16); the
projection-memory scalar-decay approximation above is outside its scope.

## D6. Normalizations (paper: "we disregard normalizations for the sake of clarity")

The paper states normalizations exist but elides them from Eq. 70 and Fig. 5.
We chose: pre-LayerNorm around both block components, RMS normalization
(arXiv:1910.07467) on each CMS level output, optional RMS on the
self-generated target (see STABILITY.md STAB-2), and an optional LayerNorm on
the retrieval output (`norm_out` — off by default, enabled in both released
training recipes). The inter-level normalization is not decoration: without it
the CMS chain amplifies multiplicatively (measured three separate times,
including a loss-24 collapse in a grafting experiment).

## D7. Relative update clip (not in paper)

`||delta W|| <= clip * max(||W||, 1)` on all online updates (default 0.5).
Trust-ratio-style scaling in the spirit of LARS (arXiv:1708.03888). The
`max(., 1)` floor is load-bearing — see STABILITY.md STAB-1 (zero-init
lockout). The paper's unclipped updates are recoverable with
`update_clip=None`; expect the failure modes in STABILITY.md at scale.

## D8. Gate parameterizations and biases (form unspecified)

Eq. 86/90 make `alpha_t, eta_t` token-wise but do not give their
parameterization. Ours: `alpha_t = sigmoid(mean(M_alpha(x)) + b_alpha)`,
`eta_t = (1 + alpha_t) * sigmoid(mean(M_eta(x)) + b_eta)` — the `(1+alpha)`
bound builds the delta-rule stability condition into the parameterization;
biases start the model as a gentle writer (see model.py comments, including
the `sigmoid(b)^chunk_len` recalibration required by token-wise retention).

## D9. Scale and data (declared global deviation)

Paper-scale experiments (760M-1.3B params, 15-100B tokens) exceed our
compute. Released recipe: 30.6M params / 0.3B bytes; sanity-checked at 161.5M
params / 1B BPE tokens. All claims in the README are scoped to these scales.

## D10. Memory activation, hidden width, and initialization (unspecified)

Eq. 89/91 write `M(x) = x + W1 sigma(W2 x)` with a generic `sigma` and no
hidden dimension or initialization. We use silu, hidden width `2d` for both
the Titans memories (`mem_hidden_mult`) and the CMS levels
(`cms_hidden_mult`), and initialize `W2 ~ N(0, 0.02^2)` with `W1` zero — the
LoRA-style output-side zero-init, so every memory starts as the identity map
and the model's function at t=0 does not depend on random memory content
(the resulting clip-lockout hazard is STAB-1). All our choices, not the
paper's.

## D11. Query adaptivity (the paper contradicts itself; we follow the prose)

The paper's prose says `q_t = x_t W_q` is "the only non-adaptive projection",
but the update-set notation of Eq. 85/88/90/93 includes `q` in the box set
`{k, v, q, eta, alpha, memory}`, and Table 6 carries a "w/o inner-projection q"
ablation — i.e. the full HOPE of the equations adapts q too. We implement the
prose reading (static `W_q`), which corresponds to the paper's own ablation row
(which, for what it is worth, slightly outperforms the full model in the
paper's Table 6 ppl). A material ambiguity, resolved in favor of the simpler
prose statement.

## D12. Inner-loss scaling: feature-mean instead of L2 sum

§8.1 defines the inner objective as `||M(k) - v||_2^2` (a sum over feature
dimensions). We use the feature-*mean* (MSE), which scales every inner
gradient by `1/d` relative to the paper. The learned write-strength gate
`eta_t` can absorb most but not all of this (it is bounded by `1 + alpha_t`),
so the effective write-strength range differs from the paper's by a constant
factor. Interacts with the update clip (D7), which bounds update magnitude
anyway.

## D13. Deviations from specified math adopted for stability (cross-references)

Two devices documented in STABILITY.md alter math the paper *does* specify,
so they belong in this contract as well:

- **CMS firing phase** (STAB-13): Eq. 71 applies the accumulated update when
  `i = 0 (mod C^(l))`; we fire one chunk *before* the period boundary
  (`(i + 2) mod C^(l) = 0`) so the slowest level's updates are consumed —
  and therefore supervised — within the training sequence. The phase index
  `i` is global across `states=` streaming, matching Eq. 71's global step
  (streamed and single-call execution are bit-identical).
- **Write-strength cap** (STAB-5): the delta-rule coefficient
  `b = eta/alpha` is clamped to `<= 1` (the DeltaNet convention), altering
  Eq. 90's coefficients whenever the learned gates would exceed full
  erasure. Without it: reproducible NaN (see STABILITY.md).
