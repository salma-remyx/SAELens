"""Typical analysis workflow: measuring and transferring bilingual SAE features.

Reproduces the qualitative findings of "How a Bilingual LM Becomes Bilingual:
Tracing Internal Representations with Sparse Autoencoders"
(https://arxiv.org/abs/2503.06394) on synthetic data. Two languages share
semantics (token j of language A means the same as token j + 4 of language B)
and each SAE feature group is hand-built, so that the paper's claims can be
checked exactly rather than statistically:

- language-specific features come first and bilingual (mixed) features only
  form later, while monosemanticity stays high, with the alignment score
  peaking in a mid layer;
- the bilingual feature mask, and not the monolingual ones, recovers the
  bilingual component of a donor model's residual stream, so injecting it adds
  exactly that component to a recipient (mid-training) residual stream.
"""

import pytest
import torch

from sae_lens.analysis import (
    BilingualAlignment,
    bilingual_alignment,
    collect_token_attributions,
    compute_monosemanticity,
    feature_mask,
    language_selectivity,
    make_feature_injection_hook,
    peak_alignment_layer,
    transfer_decomposed_features,
)
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig

N_LANGUAGE_FEATURES = 4
N_BILINGUAL_FEATURES = 4
D_SAE = N_LANGUAGE_FEATURES * 2 + N_BILINGUAL_FEATURES
D_IN = 16
N_POSITIONS_PER_LANGUAGE = 40
TOLERANCE = 1e-6


def synthetic_sae() -> StandardSAE:
    """An SAE whose encoder recovers feature activations exactly.

    Decoder directions are orthonormal and biases are zero, so decoding a
    feature activation vector and re-encoding it is the identity.
    """
    sae = StandardSAE(StandardSAEConfig(d_in=D_IN, d_sae=D_SAE))
    orthogonal, _ = torch.linalg.qr(torch.randn(D_IN, D_IN))
    sae.requires_grad_(False)
    with torch.no_grad():
        sae.W_dec.data = orthogonal[:D_SAE]
        sae.W_enc.data = orthogonal[:D_SAE].T
        sae.b_enc.data = torch.zeros_like(sae.b_enc.data)
        sae.b_dec.data = torch.zeros_like(sae.b_dec.data)
    return sae


def token_embeddings() -> torch.Tensor:
    """(2 * N_LANGUAGE_FEATURES, N_LANGUAGE_FEATURES) embeddings.

    Token j and token j + N_LANGUAGE_FEATURES are cross-language synonyms and
    therefore share an embedding; tokens within a language are orthogonal.
    """
    embeddings = torch.zeros(2 * N_LANGUAGE_FEATURES, N_LANGUAGE_FEATURES)
    for token in range(2 * N_LANGUAGE_FEATURES):
        embeddings[token, token % N_LANGUAGE_FEATURES] = 1.0
    return embeddings


def layer_acts(
    bilingual: bool, spread: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Feature activations and tokens for both monolingual corpora at one layer.

    Language A feature j fires on language A token j, language B feature j on
    language B token j. Bilingual feature j fires on the synonym pair (token j,
    token j + N_LANGUAGE_FEATURES); with `spread` it also fires on the next
    synonym pair, which keeps it bilingual but splits it over two meanings.
    """
    acts_a = torch.zeros(N_POSITIONS_PER_LANGUAGE, D_SAE)
    acts_b = torch.zeros(N_POSITIONS_PER_LANGUAGE, D_SAE)
    tokens_a = torch.zeros(N_POSITIONS_PER_LANGUAGE, dtype=torch.int64)
    tokens_b = torch.zeros(N_POSITIONS_PER_LANGUAGE, dtype=torch.int64)
    bilingual_offset = 2 * N_LANGUAGE_FEATURES

    for position in range(N_POSITIONS_PER_LANGUAGE):
        token = position % N_LANGUAGE_FEATURES
        tokens_a[position] = token
        acts_a[position, token] = 1.0
        tokens_b[position] = N_LANGUAGE_FEATURES + token
        acts_b[position, N_LANGUAGE_FEATURES + token] = 1.0
        if bilingual:
            fired = [token]
            if spread:
                fired.append((token - 1) % N_LANGUAGE_FEATURES)
            for feature in fired:
                acts_a[position, bilingual_offset + feature] = 1.0
                acts_b[position, bilingual_offset + feature] = 1.0

    return acts_a, acts_b, tokens_a, tokens_b


def layer_profile(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
) -> BilingualAlignment:
    selectivity = language_selectivity(acts_a, acts_b)
    attributions = collect_token_attributions(
        torch.cat([acts_a, acts_b]), torch.cat([tokens_a, tokens_b])
    )
    return bilingual_alignment(
        selectivity, compute_monosemanticity(attributions, token_embeddings())
    )


def test_typical_bilingual_alignment_workflow():
    sae = synthetic_sae()
    early = layer_acts(bilingual=False, spread=False)
    mid = layer_acts(bilingual=True, spread=False)
    upper = layer_acts(bilingual=True, spread=True)
    profiles = [layer_profile(*early), layer_profile(*mid), layer_profile(*upper)]

    # Languages are learned separately first, bilingual features form later,
    # and they stay semantically coherent (each one is a synonym pair).
    assert profiles[0].mixed_proportion == pytest.approx(0.0)
    assert profiles[1].mixed_proportion == pytest.approx(1 / 3)
    assert profiles[1].mean_monosemanticity == pytest.approx(1.0)
    assert peak_alignment_layer(profiles) == 1

    # Transfer: decompose the donor (fully trained) residual stream with an SAE
    # trained on it and keep only one feature kind at a time.
    acts_a, acts_b, _, _ = mid
    acts = torch.cat([acts_a, acts_b])
    donor_residual = acts @ sae.W_dec
    bilingual_features = slice(2 * N_LANGUAGE_FEATURES, None)
    bilingual_component = acts[:, bilingual_features] @ sae.W_dec[bilingual_features]
    bilingual_energy = float(bilingual_component.pow(2).sum())
    selectivity = language_selectivity(acts_a, acts_b)
    mixed_mask = feature_mask(selectivity, "mixed")
    assert mixed_mask.tolist() == [False] * 8 + [True] * N_BILINGUAL_FEATURES

    bilingual_update = transfer_decomposed_features(
        sae, donor_residual, mixed_mask, alpha=0.1
    )
    torch.testing.assert_close(
        bilingual_update, 0.1 * bilingual_component, rtol=TOLERANCE, atol=TOLERANCE
    )

    def overlap(residual: torch.Tensor) -> float:
        return (
            torch.dot(residual.flatten(), bilingual_component.flatten()).item()
            / bilingual_energy
        )

    language_a_update = transfer_decomposed_features(
        sae, donor_residual, feature_mask(selectivity, "language_a"), alpha=0.1
    )
    assert overlap(language_a_update) == pytest.approx(0.0, abs=1e-6)

    # Inject into a recipient (mid-training) residual stream, which has no
    # bilingual content of its own.
    recipient_residual = torch.cat([early[0], early[1]]) @ sae.W_dec
    assert overlap(recipient_residual) == pytest.approx(0.0, abs=1e-6)

    hook = make_feature_injection_hook(sae, donor_residual, mixed_mask, alpha=0.1)
    assert overlap(hook(recipient_residual, None)) == pytest.approx(0.1, rel=1e-5)
