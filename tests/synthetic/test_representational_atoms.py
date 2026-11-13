import math

import pytest
import torch

from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    eval_sae_on_synthetic_data,
    init_sae_to_match_feature_dict,
)
from sae_lens.synthetic.representational_atoms import (
    AtomStabilityCalculator,
    AtomStabilityMetrics,
    eval_sae_atom_stability,
    normalized_atom_inner_products,
)
from sae_lens.util import cosine_similarities
from tests.helpers import build_jumprelu_sae_cfg

# Three unit vectors in the plane at 120 degrees to each other. Their atomic
# inner products are exactly -1/3 off-diagonal and 2/3 on the diagonal, since
# DD^T = 1.5 * I, so S~ = (DD^T)^-1 = (2/3) I.
THREE_ATOMS_AT_120_DEGREES = torch.tensor(
    [
        [1.0, 0.0],
        [-0.5, math.sqrt(3) / 2],
        [-0.5, -math.sqrt(3) / 2],
    ],
    dtype=torch.float64,
)


def latents_with_l0(
    num_atoms: int, num_samples: int, active: list[int]
) -> torch.Tensor:
    """Codes where every sample activates exactly the given atom indices."""
    codes = torch.zeros(num_samples, num_atoms)
    codes[:, active] = 1.0
    return codes


class TestNormalizedAtomInnerProducts:
    def test_orthonormal_dictionary_gives_identity(self) -> None:
        naip = normalized_atom_inner_products(torch.eye(8, dtype=torch.float64))
        assert torch.allclose(naip, torch.eye(8, dtype=torch.float64), atol=1e-12)

    def test_matches_the_direct_inverse_formula(self) -> None:
        decoder = torch.randn(12, 5, dtype=torch.float64)
        dictionary = decoder.T
        direct = dictionary.T @ torch.linalg.inv(dictionary @ dictionary.T) @ dictionary
        assert torch.allclose(
            normalized_atom_inner_products(decoder), direct, atol=1e-10
        )

    def test_two_identical_atoms_in_one_dimension_share_half_the_space(self) -> None:
        decoder = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        naip = normalized_atom_inner_products(decoder)
        assert torch.allclose(
            naip, torch.full((2, 2), 0.5, dtype=torch.float64), atol=1e-12
        )

    def test_is_invariant_to_hidden_space_reparameterization(self) -> None:
        decoder = torch.randn(24, 8, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        scaling = torch.diag(torch.linspace(0.5, 2.0, 8, dtype=torch.float64))
        reparameterized = decoder @ (scaling @ rotation).T

        assert torch.allclose(
            normalized_atom_inner_products(decoder),
            normalized_atom_inner_products(reparameterized),
            atol=1e-10,
        )
        # Plain cosine similarity between decoder rows is *not* invariant to the
        # same change of basis, which is the property that distinguishes the
        # atomic inner product from an ambient-space coherence measure.
        assert not torch.allclose(
            cosine_similarities(decoder, decoder),
            cosine_similarities(reparameterized, reparameterized),
            atol=1e-3,
        )

    def test_undercomplete_dictionary_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="overcomplete"):
            normalized_atom_inner_products(torch.randn(4, 8))

    def test_three_atoms_at_120_degrees(self) -> None:
        naip = normalized_atom_inner_products(THREE_ATOMS_AT_120_DEGREES)
        expected = torch.tensor(
            [
                [2 / 3, -1 / 3, -1 / 3],
                [-1 / 3, 2 / 3, -1 / 3],
                [-1 / 3, -1 / 3, 2 / 3],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(naip, expected, atol=1e-12)


class TestAtomStabilityCalculator:
    def test_coherence_is_the_largest_off_diagonal_naip(self) -> None:
        decoder = torch.randn(40, 12, dtype=torch.float64)
        calculator = AtomStabilityCalculator(decoder)
        calculator.add_batch(latents_with_l0(40, 100, [0, 1]))

        naip = normalized_atom_inner_products(decoder)
        naip.fill_diagonal_(0.0)
        assert calculator.compute().coherence == pytest.approx(
            naip.abs().max().item(), abs=1e-12
        )

    def test_single_atom_codes_are_fully_stable(self) -> None:
        calculator = AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES)
        calculator.add_batch(latents_with_l0(3, 1000, [1]))

        metrics = calculator.compute()
        # K_q = 1 for every quantile, so the bound is 1 / (2 - 1) = 1 and the
        # coherence of 1/3 clears it everywhere.
        assert metrics.stability == pytest.approx(1.0, abs=1e-6)
        assert metrics.retention == pytest.approx(1.0, abs=1e-6)
        assert metrics.coherence == pytest.approx(1 / 3, abs=1e-12)
        assert metrics.mean_atom_sparsity == pytest.approx(1.0)

    def test_three_atom_codes_fail_distinguishability_but_keep_retention(self) -> None:
        calculator = AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES)
        calculator.add_batch(latents_with_l0(3, 1000, [0, 1, 2]))

        metrics = calculator.compute()
        # K_q = 3 gives the bound 1 / (6 - 1) = 0.2, which the coherence of 1/3
        # fails, while the retention bound 1 / (3 - 1) = 0.5 is still cleared.
        assert metrics.stability == pytest.approx(0.0, abs=1e-6)
        assert metrics.retention == pytest.approx(1.0, abs=1e-6)
        assert metrics.mean_atom_sparsity == pytest.approx(3.0)

    def test_stability_lands_on_the_largest_quantile_with_one_atom_codes(self) -> None:
        calculator = AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES)
        # 90% of codes use a single atom (bound 1), 10% use all three (bound 1/5).
        calculator.add_batch(latents_with_l0(3, 900, [1]))
        calculator.add_batch(latents_with_l0(3, 100, [0, 1, 2]))

        metrics = calculator.compute()
        assert metrics.stability == pytest.approx(0.9, abs=2e-4)
        assert metrics.retention == pytest.approx(1.0, abs=1e-6)
        assert metrics.mean_atom_sparsity == pytest.approx(1.2)

    def test_batches_are_pooled(self) -> None:
        pooled = AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES)
        pooled.add_batch(latents_with_l0(3, 900, [1]))
        pooled.add_batch(latents_with_l0(3, 100, [0, 1, 2]))

        single = AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES)
        pooled_codes = torch.cat(
            [latents_with_l0(3, 900, [1]), latents_with_l0(3, 100, [0, 1, 2])]
        )
        single.add_batch(pooled_codes)

        assert pooled.compute() == single.compute()

    def test_compute_without_batches_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no batches"):
            AtomStabilityCalculator(THREE_ATOMS_AT_120_DEGREES).compute()

    def test_to_log_dict(self) -> None:
        metrics = AtomStabilityMetrics(
            stability=0.5, retention=0.75, coherence=0.25, mean_atom_sparsity=2.0
        )
        assert metrics.to_log_dict(prefix="atoms/") == {
            "atoms/stability": 0.5,
            "atoms/retention": 0.75,
            "atoms/coherence": 0.25,
            "atoms/mean_atom_sparsity": 2.0,
        }


class TestEvalSaeAtomStability:
    def test_jumprelu_sae_metrics_agree_with_the_synthetic_eval(self) -> None:
        hidden_dim = 16
        num_features = 24

        feature_dict = FeatureDictionary(
            num_features=num_features,
            hidden_dim=hidden_dim,
            bias=False,
            initializer=None,
        )
        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.1,
        )

        sae = JumpReLUSAE(build_jumprelu_sae_cfg(d_in=hidden_dim, d_sae=num_features))
        init_sae_to_match_feature_dict(sae, feature_dict)
        with torch.no_grad():
            sae.threshold.zero_()

        num_samples = 200_000
        metrics = eval_sae_atom_stability(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=num_samples,
            batch_size=10_000,
        )
        eval_result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=num_samples,
            batch_size=10_000,
        )

        assert 0.0 <= metrics.stability <= 1.0
        assert metrics.retention >= metrics.stability
        assert 0.0 < metrics.coherence < 1.0
        # Both paths count mean L0 of the same SAE's latents over draws from the
        # same generator, so they must agree up to sampling error.
        assert metrics.mean_atom_sparsity == pytest.approx(eval_result.sae_l0, abs=0.05)

        naip = normalized_atom_inner_products(sae.W_dec)
        naip.fill_diagonal_(0.0)
        assert metrics.coherence == pytest.approx(naip.abs().max().item(), abs=1e-9)
