# VALIDATION — AbsTopK SAE port

This run delivered the AbsTopK variant (code + CPU correctness tests + the scoped
eval fix). The GPU training-parity experiment below is **deferred to a human with
a GPU/Colab budget** — finalizing the variant does not gate on it.

## Already validated here (CPU, in CI)

- Mechanism: `AbsTopK` selects exactly k features per sample by pre-activation
  magnitude and preserves their signs (a large negative outranks a smaller
  positive); verified on hand-computed inputs and on 500 random samples
  (`tests/saes/test_abstopk_sae.py`).
- Aux loss: dead latents are selected by magnitude with signed values.
- Eval correctness: L0 / feature-density counting now uses `!= 0`, so
  negative-but-active features are counted; the dedicated test fails (l0 = 0)
  under the previous `> 0` counting, and all 58 existing
  `tests/test_evals.py` tests still pass (no-op for ReLU-based variants).
- Integration: a real (tiny) `LanguageModelSAETrainingRunner` run for
  architecture `abstopk` trains, checkpoints, saves an `AbsTopKSAE` inference
  model, and reloads it (`test_LanguageModelSAETrainingRunner_runs_and_saves_all_architectures[abstopk]`).

## Deferred: W&B training-parity run (AbsTopK vs TopK / JumpReLU)

Goal: show AbsTopK matches or beats TopK/JumpReLU at matched sparsity on
L0 / CE / MSE, per the paper (arXiv:2510.00404).

Setup (control/test dashboards, as the SAELens PR template expects):

1. Cache activations once for a small model (e.g. `gpt2-small` or
   `pythia-160m`, `blocks.*.hook_resid_post`), then train three SAEs on the same
   cache so only the architecture varies:
   - control A: `topk` (`TopKTrainingSAEConfig`, `k` = target L0)
   - control B: `jumprelu` (`JumpReLUTrainingSAEConfig`, `target_l0` = target L0)
   - test: `abstopk` (`AbsTopKTrainingSAEConfig`, `k` = target L0)
2. Match everything else: `d_sae` (e.g. 8192), lr, batch size, training tokens
   (e.g. 500M–1B), `aux_loss_coefficient`, seed, dataset.
3. Sweep the target L0 (e.g. k ∈ {16, 64, 256}) so the comparison is a
   sparsity–fidelity frontier, not a single point.
4. Log to W&B per run: `mse_loss`, `auxiliary_reconstruction_loss`, CE loss
   delta (`tests/evals` / `run_evals` outputs: `l0`, `mse`, `ce_loss_score`,
   `explained_variance`), plus the trainer's dead-feature count.
5. Acceptance: at matched L0, AbsTopK should show MSE / CE no worse than TopK,
   with the paper's claimed gains concentrated at low L0; feature-density
   histograms should show roughly balanced positive/negative firing features
   (a sanity check that bidirectionality is being used).

Suggested runner entry point (CPU-runnable skeleton, GPU for the real thing):

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner
from sae_lens.saes.abstopk_sae import AbsTopKTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig(
    sae=AbsTopKTrainingSAEConfig(k=64, d_in=768, d_sae=8192),
    model_name="gpt2",
    hook_name="blocks.6.hook_resid_post",
    dataset_path="monology/pile-uncopyrighted",
    streaming=True,
    training_tokens=500_000_000,
    train_batch_size_tokens=4096,
    context_size=256,
    device="cuda",
)
LanguageModelSAETrainingRunner(cfg).run()
```

## Self-review

- **Call sites targeted**: `sae_lens/saes/topk_sae.py` (the template mirrored by
  the new `sae_lens/saes/abstopk_sae.py`), the registration block at the foot of
  `sae_lens/__init__.py`, and the two L0 / feature-density counting expressions
  in `sae_lens/evals.py::get_sparsity_and_variance_metrics` (~L575, ~L616 in this
  checkout).
- **What was ported (MIT, with attribution)**: only the selection rule — top-k on
  `hidden_pre.abs()`, then keep the signed values (no ReLU) — from
  arXiv:2510.00404 / `GoXzascc/AbsTopK-SAE` (credited in the module docstring).
  Everything else is SAELens's own TopK scaffolding, reused by subclassing
  (`AbsTopKSAE(TopKSAE)`, `AbsTopKTrainingSAE(TopKTrainingSAE)`), so encoder /
  decoder, `encode_with_hidden_pre`, decoder-norm folding and inference export
  are inherited rather than copied. The per-sample-k variant is implemented (the
  reference's batch-level variant is not), matching the brief.
- **Scoped eval fix**: both `> 0` counts became `!= 0`. No-op for every existing
  ReLU-based variant because their activations are ≥ 0 and exact zeros are
  excluded by both predicates (all 58 `tests/test_evals.py` tests pass
  unchanged). A dedicated test fails with l0 = 0 under the old counting when all
  active features are negative.
- **Base-class dead-neuron tracking**: unchanged, as the brief predicted —
  `sae_lens/training/sae_trainer.py` tracks firing via
  `feature_acts.bool()` (≡ `!= 0`), so negative activations already count as
  firing.
- **Deviations from the touch-list** (both required by the registration the
  brief mandates; guardrails permit them):
  - `tests/saes/test_sae.py`: added `"abstopk"` to the two topk-family
    architecture sets. `ALL_TRAINING_ARCHITECTURES` is derived from the
    registry, so registering `"abstopk"` automatically runs these parametrized
    tests against it; it behaves like the topk family there
    (`rescale_acts_by_decoder_norm=True` default). Without this, two existing
    tests fail for the new architecture.
  - `sae_lens/config.py`: NOT modified — `LanguageModelSAERunnerConfig` is
    registry-driven, so no wiring was needed (the brief said "if required").
- **Intentionally out of scope / stubbed**: the W&B training-parity run above
  (deferred to a human, per the brief — not a stub of code shipped here);
  sparse-COO activation output (inherited `use_sparse_activations` config field
  is documented as ignored, matching the BatchTopK precedent); the reference's
  batch-level k variant; `docs/training_saes.md` architecture section (outside
  the touch-list — `docs/custom_saes.md` was updated instead).
