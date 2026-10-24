from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    zipfian_firing_probabilities,
)
from sae_lens.synthetic.learned_expansion import run_expansion_study

HIDDEN_DIM = 32
NUM_FEATURES = 64
EXPANSION_RATIOS = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


# Run with: poetry run pytest benchmark/test_learned_expansion_study.py -v -s
def test_learned_vs_random_expansion_across_widths():
    feature_dict = FeatureDictionary(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM)
    activations_generator = ActivationGenerator(
        num_features=NUM_FEATURES,
        firing_probabilities=zipfian_firing_probabilities(NUM_FEATURES, min_prob=0.05),
    )

    result = run_expansion_study(
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        expansion_ratios=EXPANSION_RATIOS,
        architectures=("topk", "standard"),
        training_samples=200_000,
        eval_samples=40_000,
    )

    print(
        f"\n{'ratio':>5} {'arch':>9} {'encoder':>8} {'d_sae':>5} "
        f"{'k':>4} {'L0':>7} {'EV':>7} {'MCC':>6} {'drift':>7}"
    )
    for arm in result.arms:
        print(
            f"{arm.expansion_ratio:>5.1f} {arm.architecture:>9} {arm.encoder:>8} "
            f"{arm.d_sae:>5} {str(arm.k):>4} {arm.sae_l0:>7.1f} "
            f"{arm.explained_variance:>7.3f} {arm.mcc:>6.3f} "
            f"{arm.encoder_drift:>7.3f}"
        )

    explained_variance_of = {
        (arm.architecture, arm.encoder, arm.expansion_ratio): arm.explained_variance
        for arm in result.arms
    }

    print("\nGain in explained variance from learning the expansion:")
    for architecture in ("topk", "standard"):
        gains = result.learned_vs_random_gain(architecture=architecture)
        print(f"  {architecture}:")
        for ratio in EXPANSION_RATIOS:
            gain = gains[ratio]
            learned_ev = explained_variance_of[(architecture, "learned", ratio)]
            print(
                f"    ratio {ratio:>4.1f}: {gain:+.3f} "
                f"(relative: {gain / learned_ev:+.3f})"
            )
        # Learning the expansion helps at every width, and its relative
        # advantage is largest where the expansion is narrowest.
        assert all(gain > 0.0 for gain in gains.values())
        relative = {
            ratio: gain / explained_variance_of[(architecture, "learned", ratio)]
            for ratio, gain in gains.items()
        }
        assert relative[EXPANSION_RATIOS[0]] > relative[EXPANSION_RATIOS[-1]]
