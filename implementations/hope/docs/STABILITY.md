# Stability catalog — 14 devices no paper mentions

Papers state the math over ideal real arithmetic and well-behaved data. This
architecture fails at three places those assumptions break: **near zero**
(initialization), **near infinity** (finite precision), and **in the
distribution tail** (dirty web data). Every entry below was found the hard
way — a frozen memory, a NaN at step 2595, a silently useless write — and
carries its symptom, mechanism, fix, and how to diagnose it. Inline code
markers `STAB-n` in `hope/model.py` point back here.

A meta-lesson up front: the worst bugs here do **not** crash. They produce a
model that trains, converges, and quietly underperforms — which is then easy
to misread as "the architecture doesn't work". Our development gate (equal
data, equal steps, compare against a baseline; a gap means an implementation
bug until proven otherwise) is what surfaced most of these.

---

### STAB-1 - Zero-init clip lockout (the silent killer)
- **Symptom**: memory contributes nothing after full training; loss curve looks fine.
- **Mechanism**: a relative update clip `||dW|| <= c * ||W||` with a
  zero-initialized `W` caps the first update at zero — forever. With `W1 = 0`,
  the chain rule also zeroes `W2`'s gradient: the entire memory freezes.
- **Fix**: floor the reference norm, `c * max(||W||, 1.0)`.
- **Diagnosis**: finite-difference test — perturb the weight by hand; if the
  loss does not move, the parameter is dead.

### STAB-2 - Self-generated target scale feedback
- **Symptom**: memory norms grow without bound; no outer loss penalizes it.
- **Mechanism**: the target `v_hat = M(v)` grows with `||W||`, the gradient
  grows with the target: `g ~ ||W||^3`. A positive feedback loop invisible to
  the task loss.
- **Fix**: RMS-normalize `v_hat` (`norm_target=True`).

### STAB-3 - Clip factor must be non-differentiable
- **Symptom**: NaN in backward only, forward is clean.
- **Mechanism**: differentiating `c * ||W|| / ||g||` at `||g|| = 0`
  backpropagates 0/0.
- **Fix**: compute the clip factor under `no_grad` and `.detach()` it — "how
  much to clip" is a decision, not a differentiable quantity.

### STAB-4 - Alpha floor (delayed bomb)
- **Symptom**: NaN appearing only after ~200 training steps.
- **Mechanism**: the gate memory `M_alpha` grows confident and emits large
  negative values on some tokens; `sigmoid` underflows to `alpha = 0`;
  `beta = eta / alpha` explodes to ~1e6 and overflows the fp32 triangular
  solve. Never triggers in short smoke tests.
- **Fix**: `alpha_t = alpha_t.clamp(min=0.25)` — "no single token may erase
  more than 75% of memory". Compatible with the paper's `alpha in (0,1)`.

### STAB-5 - beta <= 1 (the DeltaNet standard) on entropy-tail inputs
- **Symptom**: NaN mid-training (our incident: step 2595), only on specific
  batches.
- **Mechanism**: `beta > 1` means erasing *more* than fully — an expansive
  update. Trained key maps can collapse entropy-tail inputs (binary blobs,
  quasi-random bytes in web corpora) onto a narrow cone of keys; overlapping
  keys with expansive updates make the triangular system amplify
  exponentially.
- **Fix**: `beta = (eta / alpha).clamp(max=1.0)` — full erasure at most, as in
  DeltaNet (arXiv:2406.06484). All WY factors are then non-expansive.
- **Diagnosis**: dump weights at the NaN step, replay the triggering batch;
  we reproduced 10/10 pre-fix and 0/10 post-fix.

### STAB-6 - Masked values must be finite (IEEE: Inf * 0 = NaN)
- **Symptom**: NaN born inside a masked region that is "thrown away anyway".
- **Mechanism**: pairwise decay `exp(logc_s - logc_t)` on the masked side
  (s >= t) can reach exp(+hundreds) = Inf once alpha sits at its floor;
  `Inf * 0` under the multiplicative mask is NaN, and one NaN poisons the
  whole backward.
- **Fix**: `clamp(max=0)` on the exponent *before* `exp` — lossless, because
  valid entries (s < t) are mathematically <= 0.
- **Lesson**: multiplicative masking obliges you to keep the masked-out
  values finite too.

### STAB-7 - Gate bias recalibration under token-wise retention
- **Symptom**: intended memory half-life off by orders of magnitude after
  switching from per-chunk to per-token gates.
- **Mechanism**: per-chunk retention becomes the product
  `sigmoid(bias)^chunk_len`.
- **Fix**: recalibrate (`alpha_bias = 9`: `sigmoid(9)^512 ~ 0.94` per chunk).

### STAB-8 - Relative (trust-ratio) update clip
- Not a bug fix but the safety envelope for online updates (default 0.5),
  in the spirit of LARS (arXiv:1708.03888). Interacts with STAB-1 (the floor)
  and STAB-3 (non-differentiability).

### STAB-9 - Gates consumed only through a detached path never learn
- **Symptom**: `eta` saturates early and stays there; outer training cannot
  move it.
- **Mechanism**: if the gate's only consumer is inside a detached inner
  computation, the outer loss has no gradient path to the gate.
- **Fix**: keep one differentiable consumption path when outer gradients are
  enabled (the `og()` switch in the code).
- **Diagnosis**: `tests/test_exactness.py::test_gradients_reach_gate_memories`.

### STAB-10 - Eq. 90 sums over tokens; do not average
- **Symptom**: model trains, converges, memory writes are ~1e-4 and useless.
  No error anywhere.
- **Mechanism**: a habitual `reduction="mean"` divides every write by the
  chunk length (1/512).
- **Fix**: per-token weighted **sum**, as the equation states.

### STAB-11 - Inter-level normalization in the CMS chain
- **Symptom**: activation explosion down the level chain (in a grafting
  experiment: loss 24 vs 3 with/without).
- **Mechanism**: stacking un-normalized MLP outputs multiplies scales.
- **Fix**: RMS normalization on each level output. Status in the paper:
  *explicitly elided* — "we disregard normalizations for the sake of
  clarity" (Eq. 70 preamble; Fig. 5's caption likewise removes them) —
  form and placement are ours.

### STAB-12 - Triangular solves in fp32
- `torch.linalg.solve_triangular` neither supports nor numerically survives
  bf16. The solve runs in an fp32 autocast-disabled island; everything else
  stays in the ambient dtype.

### STAB-13 - Update firing must be consumed inside the training horizon
- **Symptom**: the slowest CMS level receives no learning signal; its updates
  are never supervised.
- **Mechanism**: if a level fires its accumulated update at the sequence
  boundary, the updated weights are never used by any loss inside the
  sequence — the outer loop cannot learn what those updates should be. The
  same logic limits self-modifying updates: supervision covers only as many
  recursive updates as the training sequence actually consumes; beyond that
  horizon the recursion is unsupervised (see `max_updates` for evaluating
  short-horizon checkpoints).
- **Fix**: shift the firing point one chunk before the period boundary
  (`(chunk_idx + 2) % period` in the code) and train with sequences that
  consume several updates.

### STAB-14 - Do not clear a module-attribute hook inside the forward pass under activation checkpointing

Symptom: `CheckpointError: recomputed values ... different metadata` at the
first backward, with the saved/recomputed tensor lists offset by a few
positions.

Cause (empirical, isolated by bisection): a feature-map hook
(historical: an experimental `smt.mem_feat_fn` map, since removed) set at block-forward entry and cleared in a
`finally:` block dies under `use_reentrant=False` checkpointing; the same
hook left permanently assigned (overwritten on every call, never cleared)
recomputes deterministically. Verified: checkpoint-on/off gradients agree to
bf16 recompute noise (max 2.4e-4) with the overwrite-only pattern.

Rule: hooks consumed inside a checkpointed region must be *overwritten*, never
*cleared* — the forward path must re-assign them before every use so the
recomputation binds the correct per-chunk closure.

---

## Where the difficulty actually lives

The equations in this paper transcribe easily; they are written out in full and
there are only a dozen of them that matter. Our experience is that the hard
part is everything around them — the devices above, the exactness test that
tells you whether your parallelization is the recurrence you think it is, and a
comparison baseline that tells you whether the number you got is good.

We think this generalizes beyond us. Reading the public reproductions of this
paper surfaced the same clustering: the mechanisms
most often degraded are exactly the ones that fail silently — a retention term
that collapses to scalar decay, a teaching signal from the wrong objective, a
memory update applied outside `forward()`. None of these crash. All of them
produce a descending loss curve. Which is the argument for the two artifacts
this repository leads with: an exactness test against a sequential reference,
and a training curve that our own code produced end to end.
