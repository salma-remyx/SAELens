# Description

This PR adds the AbsTopK SAE architecture from [AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features](https://arxiv.org/abs/2510.00404), a TopK variant that selects the k largest-magnitude pre-activations while preserving their signs.

TopK-style SAEs keep the k largest pre-activations and ReLU them, which forces every feature activation to be non-negative. This fragments bidirectional concepts (e.g. male vs. female) into separate features. AbsTopK instead applies the L0 hard-thresholding on the largest-magnitude pre-activations while preserving their signs, so a single feature can encode both directions of a concept.

- New module `sae_lens/saes/abstopk_sae.py` adds `AbsTopKSAE` / `AbsTopKTrainingSAE` (plus their configs and the `AbsTopK` activation). Both subclass the existing TopK classes and override only `get_activation_fn` and the dead-neuron selection used by the aux loss; they are re-exported from `sae_lens` and registered under `"abstopk"` (usable via `--architecture abstopk`).
- Behavioral contrast, demonstrated in `tests/saes/test_abstopk_sae.py`: with a single feature aligned to `e_0`, an `AbsTopKSAE` reconstructs an input of `-3*e_0` exactly, while an equivalent `TopKSAE` reconstructs zeros (`test_AbsTopKSAE_reconstructs_bidirectional_inputs_where_TopKSAE_cannot`). On non-negative pre-activations the two architectures agree exactly.
- The training variant keeps the TopK dead-neuron auxiliary reconstruction loss (controlled by `aux_loss_coefficient`), except dead latents are selected by activation magnitude with their signs kept. Plain TopK behavior is unchanged: the selection lives in the new `_select_dead_aux_acts` hook on `TopKTrainingSAE`, which defaults to the original top-k-by-value selection, so existing architectures' losses are untouched.
- `save_inference_model` / `load` route through the existing TopK save/load machinery (including decoder-norm folding when `rescale_acts_by_decoder_norm=True`), and a saved SAE loads back as an `AbsTopKSAE`.

Out of scope for this PR: registering any pretrained AbsTopK releases, changes to eval metrics or normalization options, and tutorial notebook updates.

## Type of change

- [x] New feature (non-breaking change which adds functionality)

# Checklist:

- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing tests pass locally with my changes
- [x] I have not rewritten tests relating to key interfaces which would affect backward compatibility

The `tests/saes/test_sae.py` edits only extend existing parametrizations by adding `"abstopk"` to their architecture sets; they do not rewrite them. `tests/helpers.py` only adds AbsTopK config builders and `"abstopk"` registry entries.

### You have tested formatting, typing and tests

- [ ] I have run `make check-ci` to check format and linting. (you can run `make format` to format code if needed.)

`make check-ci` could not be run in this environment: importing the package fails with `ModuleNotFoundError: No module named 'transformer_lens.HookedTransformer'` (the installed transformer_lens build exposes a different module layout), and poetry / ruff / pyright are not installed in this runner. The scoped checks for this change are:

- `ruff check sae_lens/saes/abstopk_sae.py sae_lens/__init__.py sae_lens/saes/__init__.py tests/helpers.py tests/saes/test_abstopk_sae.py tests/saes/test_sae.py`
- `ruff format --check` on the same files
- `pyright sae_lens/saes/abstopk_sae.py`
- `pytest tests/saes/test_abstopk_sae.py tests/saes/test_sae.py`

No CI success or local test pass from this environment is claimed.

### Performance Check.

Not a training change: existing architectures' loss and parameter updates are identical; the new abstopk classes only add an activation-variant path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
