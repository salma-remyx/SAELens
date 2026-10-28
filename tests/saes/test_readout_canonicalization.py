import pytest
import torch

from sae_lens.saes.batchtopk_sae import BatchTopK, BatchTopKTrainingSAE
from sae_lens.saes.matryoshka_batchtopk_sae import MatryoshkaBatchTopKTrainingSAE
from sae_lens.saes.readout_canonicalization import compute_readout_canonicalization
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopK, TopKTrainingSAE
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_batchtopk_sae_training_cfg,
    build_matryoshka_batchtopk_sae_training_cfg,
    build_topk_sae_training_cfg,
    random_params,
)


def _decoder_with_spectrum(
    d_sae: int, d_in: int, spectrum: torch.Tensor
) -> torch.Tensor:
    """A decoder whose singular values are exactly `spectrum`."""
    left = torch.linalg.qr(torch.randn(d_sae, d_in, dtype=spectrum.dtype))[0]
    right = torch.linalg.qr(torch.randn(d_in, d_in, dtype=spectrum.dtype))[0]
    return left @ torch.diag(spectrum) @ right.T


def _topk_sae(
    readout_condition_number: float | None, dtype: str = "float64"
) -> TopKTrainingSAE:
    cfg = build_topk_sae_training_cfg(
        d_in=6,
        d_sae=18,
        k=4,
        dtype=dtype,
        readout_condition_number=readout_condition_number,
        rescale_acts_by_decoder_norm=False,
    )
    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    # a readout with a strongly skewed spectrum, so there are real gains to remove
    with torch.no_grad():
        sae.W_dec.data = _decoder_with_spectrum(
            cfg.d_sae, cfg.d_in, torch.logspace(0, 3, cfg.d_in, dtype=sae.dtype)
        )
    return sae


@pytest.mark.parametrize("condition_number", [1.0, 4.0, 100.0])
def test_readout_canonicalization_caps_condition_number_and_preserves_scale(
    condition_number: float,
):
    spectrum = torch.logspace(0, 3, 6)
    W_dec = _decoder_with_spectrum(20, 6, spectrum)

    canonicalization = compute_readout_canonicalization(W_dec, condition_number)
    canonicalized_spectrum = torch.linalg.svdvals(
        canonicalization.remove_from_decoder(W_dec)
    )

    assert canonicalized_spectrum[0] / canonicalized_spectrum[-1] == pytest.approx(
        condition_number, rel=1e-3
    )
    # the geometric mean (the readout's overall scale) is left alone
    assert canonicalized_spectrum.log().mean().exp() == pytest.approx(
        spectrum.log().mean().exp(), rel=1e-3
    )


def test_readout_canonicalization_matches_decoder_norm_rescaling_for_diagonal_readout():
    spectrum = torch.logspace(0, 2, 8)
    W_dec = torch.diag(spectrum)
    canonicalization = compute_readout_canonicalization(W_dec, 1.0)
    hidden_pre = torch.randn(32, 8)

    # for a readout whose gains already sit on individual features, equalizing the
    # spectrum is the existing per-feature decoder norm rescaling, up to the global
    # constant geo_mean
    geo_mean = spectrum.log().mean().exp()
    assert_close(
        canonicalization.apply(hidden_pre),
        hidden_pre * spectrum / geo_mean,
        rtol=1e-4,
    )


def test_readout_canonicalization_is_identity_for_large_condition_number():
    W_dec = _decoder_with_spectrum(24, 8, torch.logspace(0, 3, 8))
    canonicalization = compute_readout_canonicalization(W_dec, 1e9)

    hidden_pre = torch.randn(16, 24)
    assert torch.equal(canonicalization.apply(hidden_pre), hidden_pre)
    assert torch.equal(canonicalization.remove_from_decoder(W_dec), W_dec)


def test_TopKTrainingSAE_readout_canonicalization_changes_which_features_win():
    sae = _topk_sae(readout_condition_number=1.0)
    sae_in = torch.randn(128, sae.cfg.d_in, dtype=sae.dtype)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc

    canonicalized_acts = sae.encode(sae_in)
    plain_acts = TopK(sae.cfg.k)(hidden_pre)

    assert_not_close(canonicalized_acts, plain_acts)
    different_support = ((canonicalized_acts != 0) != (plain_acts != 0)).any(dim=-1)
    assert different_support.float().mean() > 0.75


def test_TopKTrainingSAE_readout_canonicalization_is_an_exact_wrapper():
    sae = _topk_sae(readout_condition_number=3.0)
    step_input = TrainStepInput(
        sae_in=torch.randn(32, sae.cfg.d_in, dtype=sae.dtype),
        # a mostly-dead SAE so the auxiliary loss path runs too
        dead_neuron_mask=torch.rand(sae.cfg.d_sae) > 0.3,
        coefficients={},
        n_training_steps=0,
        is_logging_step=False,
    )

    output = sae.training_forward_pass(step_input)

    sae.fold_W_dec_norm()

    assert sae.cfg.readout_condition_number is None
    folded_output = sae.training_forward_pass(step_input)
    assert_close(folded_output.feature_acts, output.feature_acts)
    assert_close(folded_output.sae_out, output.sae_out)
    assert_close(
        folded_output.losses["auxiliary_reconstruction_loss"],
        output.losses["auxiliary_reconstruction_loss"],
    )
    assert_close(folded_output.loss, output.loss)


def test_TopKTrainingSAE_folds_canonicalization_into_inference_state_dict():
    sae = _topk_sae(readout_condition_number=3.0)
    sae_in = torch.randn(24, sae.cfg.d_in, dtype=sae.dtype)
    feature_acts = sae.encode(sae_in)

    state_dict = sae.state_dict()
    sae.process_state_dict_for_saving_inference(state_dict)

    # the inference config has no readout canonicalization, so the folded weights
    # have to reproduce it on their own
    inference_sae = _topk_sae(readout_condition_number=None)
    inference_sae.load_state_dict(state_dict)
    assert_close(inference_sae.encode(sae_in), feature_acts)
    assert_close(
        inference_sae.decode(inference_sae.encode(sae_in)), sae.decode(feature_acts)
    )


def test_BatchTopKTrainingSAE_applies_readout_canonicalization():
    cfg = build_batchtopk_sae_training_cfg(
        d_in=6,
        d_sae=18,
        k=2,
        readout_condition_number=2.0,
        rescale_acts_by_decoder_norm=False,
    )
    sae = BatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(40, cfg.d_in)
    canonicalization = compute_readout_canonicalization(sae.W_dec, 2.0)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc

    assert_close(
        sae.encode(sae_in),
        BatchTopK(cfg.k)(canonicalization.apply(hidden_pre)),
    )


def test_readout_condition_number_takes_precedence_over_decoder_norm_rescaling():
    with pytest.warns(UserWarning, match="rescale_acts_by_decoder_norm"):
        cfg = build_topk_sae_training_cfg(
            d_in=6, d_sae=18, k=4, readout_condition_number=4.0
        )
    assert cfg.rescale_acts_by_decoder_norm is False

    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(32, cfg.d_in)
    canonicalization = compute_readout_canonicalization(sae.W_dec, 4.0)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc

    assert_close(sae.encode(sae_in), TopK(cfg.k)(canonicalization.apply(hidden_pre)))


def test_MatryoshkaBatchTopKTrainingSAE_rejects_readout_canonicalization():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=6,
        d_sae=18,
        matryoshka_widths=[6, 12, 18],
        readout_condition_number=2.0,
        rescale_acts_by_decoder_norm=False,
    )
    with pytest.raises(ValueError, match="matryoshka"):
        MatryoshkaBatchTopKTrainingSAE(cfg)
