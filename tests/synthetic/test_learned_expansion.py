import pytest
import torch

from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    train_toy_sae,
)
from sae_lens.synthetic.learned_expansion import (
    ExpansionArmResult,
    freeze_encoder,
    run_expansion_study,
)

HIDDEN_DIM = 32
NUM_FEATURES = 64


@pytest.fixture
def synthetic_setup() -> tuple[FeatureDictionary, ActivationGenerator]:
    return (
        FeatureDictionary(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM),
        ActivationGenerator(num_features=NUM_FEATURES, firing_probabilities=0.05),
    )


def build_topk_sae() -> TopKTrainingSAE:
    return TopKTrainingSAE(
        TopKTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=2 * HIDDEN_DIM, k=6)
    )


def test_frozen_encoder_stays_at_random_init_while_decoder_learns(
    synthetic_setup: tuple[FeatureDictionary, ActivationGenerator],
) -> None:
    feature_dict, activations_generator = synthetic_setup
    sae = freeze_encoder(build_topk_sae())

    w_enc_before = sae.W_enc.detach().clone()
    b_enc_before = sae.b_enc.detach().clone()
    w_dec_before = sae.W_dec.detach().clone()

    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        training_samples=4096,
        batch_size=1024,
    )

    assert torch.equal(sae.W_enc.detach(), w_enc_before)
    assert torch.equal(sae.b_enc.detach(), b_enc_before)
    assert not torch.equal(sae.W_dec.detach(), w_dec_before)
    assert sae.W_enc.requires_grad is False
    assert sae.b_enc.requires_grad is False


def test_unfrozen_encoder_updates_during_training(
    synthetic_setup: tuple[FeatureDictionary, ActivationGenerator],
) -> None:
    feature_dict, activations_generator = synthetic_setup
    sae = build_topk_sae()

    w_enc_before = sae.W_enc.detach().clone()
    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        training_samples=4096,
        batch_size=1024,
    )
    assert not torch.equal(sae.W_enc.detach(), w_enc_before)


def test_learned_expansion_beats_frozen_random_at_narrow_width(
    synthetic_setup: tuple[FeatureDictionary, ActivationGenerator],
) -> None:
    feature_dict, activations_generator = synthetic_setup

    result = run_expansion_study(
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        expansion_ratios=(2.0,),
        architectures=("topk",),
        training_samples=100_000,
        eval_samples=20_000,
    )

    learned = next(arm for arm in result.arms if arm.encoder == "learned")
    random = next(arm for arm in result.arms if arm.encoder == "random")
    assert learned.explained_variance > random.explained_variance + 0.1
    assert learned.encoder_drift > 0.0
    assert random.encoder_drift == 0.0
    assert result.learned_vs_random_gain() == {
        2.0: pytest.approx(learned.explained_variance - random.explained_variance)
    }


def test_run_expansion_study_covers_requested_grid(
    synthetic_setup: tuple[FeatureDictionary, ActivationGenerator],
) -> None:
    feature_dict, activations_generator = synthetic_setup

    result = run_expansion_study(
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        expansion_ratios=(2.0, 4.0),
        architectures=("topk", "standard"),
        training_samples=50_000,
        eval_samples=10_000,
    )

    assert result.d_in == HIDDEN_DIM
    assert len(result.arms) == 8
    assert {arm.architecture for arm in result.arms} == {"topk", "standard"}
    assert {arm.encoder for arm in result.arms} == {"learned", "random"}

    by_key = {(arm.architecture, arm.encoder, arm.d_sae): arm for arm in result.arms}
    for (architecture, encoder, d_sae), arm in by_key.items():
        assert arm.expansion_ratio == pytest.approx(d_sae / HIDDEN_DIM)
        assert arm.explained_variance <= 1.0
        if architecture == "topk":
            assert arm.k == int(round(0.1 * d_sae))
            assert arm.sae_l0 == pytest.approx(arm.k)
        else:
            assert arm.k is None
        # The topk selection keeps exactly k latents, so its L0 is fixed, while
        # the dense ReLU baseline of the same width is far less sparse.
        if architecture == "standard":
            assert arm.sae_l0 > by_key[("topk", encoder, d_sae)].sae_l0

    for architecture in ("topk", "standard"):
        gains = result.learned_vs_random_gain(architecture=architecture)
        assert set(gains) == {2.0, 4.0}
        for ratio, gain in gains.items():
            assert gain > 0.0


def test_expansion_arm_result_to_log_dict_is_flat() -> None:
    arm = ExpansionArmResult(
        architecture="topk",
        encoder="random",
        d_sae=64,
        expansion_ratio=2.0,
        k=6,
        encoder_drift=0.0,
        explained_variance=0.5,
        mcc=0.25,
        sae_l0=6.0,
        dead_latents=0,
    )
    log_dict = arm.to_log_dict(prefix="eas/")
    assert log_dict == {
        "eas/encoder": "random",
        "eas/d_sae": 64,
        "eas/expansion_ratio": 2.0,
        "eas/k": 6,
        "eas/encoder_drift": 0.0,
        "eas/explained_variance": 0.5,
        "eas/mcc": 0.25,
        "eas/sae_l0": 6.0,
        "eas/dead_latents": 0,
    }
