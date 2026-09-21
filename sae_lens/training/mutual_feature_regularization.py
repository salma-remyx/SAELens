"""Mutual feature regularization for SAEs trained in parallel.

Features that are recovered independently by several SAEs are more likely
to correspond to features of the input than features found by a single
SAE, so training K SAEs in parallel and encouraging them to learn similar
features filters out spurious ones. This module implements that coupling
as an auxiliary penalty of one minus the mean-max cosine similarity
(MMCS) between the feature directions of parallel SAEs, added to each
SAE's reconstruction loss. Adapted from "Enhancing Neural Network
Interpretability with Feature-Aligned Sparse Autoencoders"
(arxiv:2411.01220), which validated the penalty on GPT-2 Small
activations.

Adaptations for SAELens:

- The paper ties the encoder and decoder, so coupling either matrix is
  equivalent there; SAELens architectures keep them separate, so the
  penalty couples the decoder rows (`W_dec`, one row per feature
  direction), where feature similarity is conventionally read.
- `MultiSAETrainer` gives each SAE its own optimizer and steps them one
  at a time, so peer weights are detached: each SAE is pulled toward its
  peers' features from the previous step rather than receiving gradients
  through another SAE's optimizer. Both directions of a pair still pull,
  matching the paper's joint penalty up to a one-step lag in the peer
  term.
- The pair term is the symmetric average of both MMCS directions. With
  two SAEs this is exactly the paper's penalty; with more it is the mean
  of the pair terms involving this SAE, whose gradients match the paper's
  C(K, 2)-normalized sum over all pairs (terms between other SAEs are
  constant w.r.t. this SAE's weights) up to a rescale of the coefficient.
- The coefficient goes through the standard `TrainCoefficientConfig`
  warm-up seam (the paper ramps the penalty in over ~100 steps).

The dead-feature reinitialization half of the paper's method is out of
scope: SAELens architectures already handle dead features (the TopK
auxiliary reconstruction loss, the resampling protocol).

Combine `MutualFeatureSAEMixin` with any training architecture and drive
a group of them with `MutualFeatureMultiSAETrainer`, a drop-in
`MultiSAETrainer`:

```python
class TopKMFRTrainingSAE(MutualFeatureSAEMixin, TopKTrainingSAE):
    pass


trainer = MutualFeatureMultiSAETrainer(
    cfg=cfg,
    saes={"a": TopKMFRTrainingSAE(cfg_a), "b": TopKMFRTrainingSAE(cfg_b)},
    hook_names={"a": hook, "b": hook},
    data_provider=data_provider,
    mfr_coefficient=TrainCoefficientConfig(value=3.0, warm_up_steps=100),
)
trainer.fit()
```

The penalty surfaces as the `mfr_loss` entry of each SAE's aux-loss dict,
so it appears in `TrainStepOutput.losses` and in wandb logging like any
other auxiliary loss. (The stock `MultiSAETrainingRunner` constructs a
plain `MultiSAETrainer` and never attaches peers, so MFR runs use
`MutualFeatureMultiSAETrainer` directly with their own data provider.)
"""

from typing import Any

import torch
from typing_extensions import override

from sae_lens.config import SAETrainerConfig
from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainStepInput,
    TrainingSAE,
)
from sae_lens.training.multi_sae_trainer import (
    MultiSAEEvaluatorProtocol,
    MultiSAETrainer,
)
from sae_lens.training.sae_trainer import SaveCheckpointFn
from sae_lens.training.types import MultiHookDataProvider


def mean_max_cosine_sim(
    features: torch.Tensor, other_features: torch.Tensor
) -> torch.Tensor:
    """
    Mean over the rows of `features` of the maximum cosine similarity to any
    row of `other_features`, where each row is a feature direction.

    Args:
        features: (n_features, d_in) feature directions being matched.
        other_features: (n_other_features, d_in) feature directions matched
            against. Directional: swapping the arguments is not a no-op.
    """
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    other_features = other_features / other_features.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return (features @ other_features.T).max(dim=-1).values.mean()


def mutual_feature_penalty(
    features: torch.Tensor,
    peer_features: list[torch.Tensor],
) -> torch.Tensor:
    """
    Mean over peers of one minus the symmetric MMCS between `features` and
    each peer's feature directions. Peer tensors are detached, so gradients
    only flow to `features`; a SAE with no peers has zero penalty.
    """
    if not peer_features:
        return features.new_zeros(())
    penalties = [
        1.0
        - 0.5
        * (
            mean_max_cosine_sim(features, peer.detach())
            + mean_max_cosine_sim(peer.detach(), features)
        )
        for peer in peer_features
    ]
    return torch.stack(penalties).mean()


class MutualFeatureSAEMixin(TrainingSAE[Any]):
    """
    Mixin adding the mutual feature regularization penalty to any training
    SAE architecture:

    ```python
    class TopKMFRTrainingSAE(MutualFeatureSAEMixin, TopKTrainingSAE):
        pass
    ```

    Peers and the penalty coefficient are attached by
    `MutualFeatureMultiSAETrainer`; until peers are attached the SAE trains
    exactly like the base architecture.
    """

    # Rebound (never mutated in place) per SAE by the coordinator.
    mfr_peers: list[TrainingSAE[Any]] = []
    mfr_coefficient: float | TrainCoefficientConfig = 0.0

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {**super().get_coefficients(), "mfr_coefficient": self.mfr_coefficient}

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_losses = super().calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        losses = (
            {"aux_loss": base_losses}
            if isinstance(base_losses, torch.Tensor)
            else dict(base_losses)
        )
        if self.mfr_peers:
            penalty = mutual_feature_penalty(
                self.W_dec, [peer.W_dec for peer in self.mfr_peers]
            )
            losses["mfr_loss"] = (
                step_input.coefficients.get("mfr_coefficient", 0.0) * penalty
            )
        return losses


class MutualFeatureMultiSAETrainer(MultiSAETrainer):
    """
    `MultiSAETrainer` whose SAEs are additionally coupled with mutual
    feature regularization: each SAE's training loss gains an `mfr_loss`
    term pulling its decoder feature directions toward the ones the other
    SAEs have learned.

    Args:
        mfr_coefficient: Weight on the penalty, either a plain float or a
            `TrainCoefficientConfig` for linear warm-up. All SAEs share it.
    """

    def __init__(
        self,
        cfg: SAETrainerConfig,
        saes: dict[str, MutualFeatureSAEMixin],
        hook_names: dict[str, str],
        data_provider: MultiHookDataProvider,
        mfr_coefficient: float | TrainCoefficientConfig = 1.0,
        evaluator: MultiSAEEvaluatorProtocol | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None:
        for name, sae in saes.items():
            if not isinstance(sae, MutualFeatureSAEMixin):
                raise TypeError(
                    f"SAEs trained with mutual feature regularization must combine "
                    f"MutualFeatureSAEMixin with a TrainingSAE architecture; "
                    f"{name!r} is a {type(sae).__name__}"
                )
        # Peers and the coefficient must be attached before super().__init__
        # builds the per-SAE trainers, which read get_coefficients() to set
        # up the coefficient schedulers.
        for sae in saes.values():
            sae.mfr_peers = [peer for peer in saes.values() if peer is not sae]
            sae.mfr_coefficient = mfr_coefficient
        super().__init__(
            cfg=cfg,
            saes=saes,
            hook_names=hook_names,
            data_provider=data_provider,
            evaluator=evaluator,
            save_checkpoint_fn=save_checkpoint_fn,
        )
