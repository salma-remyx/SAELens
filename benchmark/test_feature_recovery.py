from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    HierarchyNode,
    hierarchy_modifier,
    train_toy_sae,
)
from sae_lens.synthetic.feature_recovery import (
    SyntheticRecoveryEvalResult,
    eval_sae_recovery_on_synthetic_data,
)

NUM_PARENTS = 8
CHILDREN_PER_PARENT = 4
HIDDEN_DIM = 20
FIRING_PROBABILITY = 0.3
TRAINING_SAMPLES = 60_000
BATCH_SIZE = 512
EVAL_SAMPLES = 20_000


def create_nested_setup(
    nesting: bool,
) -> tuple[FeatureDictionary, ActivationGenerator, list[HierarchyNode] | None]:
    """
    Create a feature dictionary and activation generator with nested features.

    Parent features 0..NUM_PARENTS-1 each gate CHILDREN_PER_PARENT child
    features via a hierarchy modifier; with nesting disabled the same features
    fire independently, which is the flat control condition.

    Args:
        nesting: Whether children only fire when their parent fires

    Returns:
        Tuple of (feature_dict, activations_generator, hierarchy_roots)
    """
    num_features = NUM_PARENTS * (1 + CHILDREN_PER_PARENT)
    roots = None
    modify_activations = None
    if nesting:
        roots = [
            HierarchyNode(
                feature_index=parent,
                children=[
                    HierarchyNode(
                        feature_index=NUM_PARENTS
                        + parent * CHILDREN_PER_PARENT
                        + child
                    )
                    for child in range(CHILDREN_PER_PARENT)
                ],
            )
            for parent in range(NUM_PARENTS)
        ]
        modify_activations = hierarchy_modifier(roots)

    feature_dict = FeatureDictionary(
        num_features=num_features,
        hidden_dim=HIDDEN_DIM,
        bias=False,
        initializer=None,
    )
    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=FIRING_PROBABILITY,
        modify_activations=modify_activations,
    )
    return feature_dict, activations_gen, roots


def train_and_evaluate(k: float, nesting: bool) -> SyntheticRecoveryEvalResult:
    """Train a BatchTopK SAE on the setup and evaluate its recovery phase."""
    feature_dict, activations_gen, roots = create_nested_setup(nesting)

    cfg = BatchTopKTrainingSAEConfig(
        d_in=HIDDEN_DIM,
        d_sae=2 * feature_dict.num_features,
        k=k,
    )
    sae = BatchTopKTrainingSAE(cfg)

    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=TRAINING_SAMPLES,
        batch_size=BATCH_SIZE,
        lr=1e-3,
    )

    return eval_sae_recovery_on_synthetic_data(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        hierarchy_roots=roots,
        num_samples=EVAL_SAMPLES,
        batch_size=2048,
    )


# Run with: poetry run pytest benchmark/test_feature_recovery.py -v -s
def test_batchtopk_recovery_phase_across_nesting_and_sparsity():
    print(
        f"\nRecovery phase diagram: {NUM_PARENTS} parents x {CHILDREN_PER_PARENT} "
        f"children, hidden_dim={HIDDEN_DIM}, {TRAINING_SAMPLES:,} training samples"
    )
    header = (
        f"{'setup':<8} {'k':>5} {'phase':<10} {'ev':>7} {'med_cos':>8} "
        f"{'rec_frac':>9} {'merged':>7} {'edge_gap':>9} {'density':>8}"
    )
    print(header)

    for nesting in [True, False]:
        for k in [4.0, 12.0, 32.0]:
            result = train_and_evaluate(k, nesting)
            recovery = result.recovery
            print(
                f"{'nested' if nesting else 'flat':<8} "
                f"{k:>5.0f} "
                f"{result.phase:<10} "
                f"{result.aggregate.explained_variance:>7.3f} "
                f"{recovery.median_best_cosine:>8.3f} "
                f"{recovery.recovered_fraction:>9.3f} "
                f"{recovery.merged_edge_fraction:>7.3f} "
                f"{recovery.mean_edge_cosine_gap:>9.3f} "
                f"{result.density_ratio:>8.2f}"
            )

            assert 0.0 <= recovery.recovered_fraction <= 1.0
            assert 0.0 <= recovery.merged_edge_fraction <= 1.0
            assert result.phase in ("recovered", "merged", "diffuse")
            assert result.aggregate.explained_variance <= 1.0
            assert result.density_ratio > 0
            if nesting:
                # 8 parents x 4 children = 32 hierarchy edges must be reported
                assert recovery.num_hierarchy_edges == NUM_PARENTS * CHILDREN_PER_PARENT
            else:
                assert recovery.num_hierarchy_edges == 0
