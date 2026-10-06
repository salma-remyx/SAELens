"""BatchTopK SAE training variant with a Hopkins-statistic topology regularizer.

Adapted from "Feature Space Topology Control via Hopkins Loss" (Vaaras & Airaksinen,
Proc. IEEE ICTAI 2025, https://arxiv.org/abs/2509.11154), which adds |H - H_T| as a
loss term to steer the feature space of a minibatch toward a target topology. The
paper applies the loss to classifier features and autoencoder bottlenecks; here the
controlled feature space is the SAE code space, i.e. the minibatch of feature
activation vectors, and the loss is added through the standard calculate_aux_loss
hook so it composes with the TopK auxiliary reconstruction loss and the
TrainCoefficientConfig warm-up plumbing. Reference implementation:
https://github.com/SPEECHCOG/hopkins_loss (MIT).

Importing this module registers the "hopkins_batchtopk" architecture, making it
usable from config-driven runners (e.g. LanguageModelSAETrainingRunner).
"""

from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.registry import register_sae_training_class
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainCoefficientConfig, TrainStepInput


def _chebyshev_knn_indices(
    x: torch.Tensor, points: torch.Tensor, k: int
) -> torch.Tensor:
    """Indices (num_points, k) of the k Chebyshev-nearest rows of x to each point."""
    # loop over the (few) query points instead of materializing the
    # (num_points, batch_size, num_features) pairwise difference tensor
    knn_indices = []
    for point in points:
        distances = (x - point).abs().amax(dim=1)
        knn_indices.append(distances.topk(k, largest=False).indices)
    return torch.stack(knn_indices)


def calculate_hopkins_statistic(
    feature_acts: torch.Tensor, sample_size: int, eps: float = 1e-12
) -> torch.Tensor:
    """Compute a differentiable Hopkins statistic over a batch of feature vectors.

    The statistic contrasts nearest-neighbour distances of random reference points
    with those of sampled data points (Banerjee & Davé 2004):

    1. Sample m < n points from x without replacement (x_tilde).
    2. Generate m points y uniformly at random in the per-dimension min/max
       bounding box of x.
    3. u_i is the Chebyshev distance from y_i to its nearest neighbour in x, and
       w_i is the distance from x_tilde_i to its nearest neighbour in x other
       than the point itself.
    4. H = sum(u_i) / (sum(u_i) + sum(w_i) + eps).

    H is close to 1 for clustered data, close to 0.5 for randomly (Poisson-like)
    spaced data, and close to 0 for regularly spaced data. Chebyshev distance is
    used because the paper found it was the only metric that preserved these
    properties across feature dimensionalities.

    The statistic is differentiable with respect to x: the nearest-neighbour search
    runs without gradients and only the selected nearest-neighbour distances are
    recomputed with gradients, which yields the same value and gradients as
    differentiating through a topk over the full pairwise distance matrix.

    Args:
        feature_acts: Tensor of shape (batch_size, num_features). Tensors with
            extra dimensions are flattened to (batch_size, num_features).
        sample_size: Number of sample points m used for the statistic.
        eps: Small value added to the denominator to avoid division by zero.
    """
    x = feature_acts
    if x.ndim > 2:
        x = x.flatten(start_dim=1)
    m = sample_size
    x_tilde = x[torch.randperm(x.shape[0])[:m]]
    mins = x.min(dim=0).values
    maxs = x.max(dim=0).values
    random_offsets = torch.rand(m, x.shape[1], device=x.device, dtype=x.dtype)
    y = (mins - maxs) * random_offsets + maxs

    with torch.no_grad():
        u_indices = _chebyshev_knn_indices(x, y, 1)[:, 0]
        w_indices = _chebyshev_knn_indices(x, x_tilde, 2)[:, 1]
    u_distances = (x[u_indices] - y).abs().amax(dim=1)
    w_distances = (x[w_indices] - x_tilde).abs().amax(dim=1)

    sum_u = u_distances.sum()
    sum_w = w_distances.sum()
    return sum_u / (sum_u + sum_w + eps)


@dataclass
class HopkinsBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a HopkinsBatchTopKTrainingSAE.

    This is a BatchTopK SAE with an additional Hopkins loss term that pushes the
    topology of the SAE code space toward a target value: e.g. a strongly
    clustered code space (hopkins_target=0.99), a random one (0.5, equivalent to
    disabling topology control in expectation), or a regular one where codes are
    evenly spread (0.01). The regularizer composes with the usual BatchTopK
    training losses, so reconstruction quality and dead-feature handling are
    unaffected unless the coefficient is made large.

    Args:
        hopkins_target (float): Target value H_T for the Hopkins statistic of the
            SAE code space. Values above ~0.7 correspond to a clustered topology,
            ~0.5 to a random topology, and below ~0.3 to a regular topology.
        hopkins_loss_coefficient (float): Coefficient for the Hopkins loss term.
            The default of 1/3 corresponds to the paper's loss weighting
            L = w * L_MSE + (1 - w) * L_Hopkins with w = 0.75.
        hopkins_warm_up_steps (int): Number of training steps over which to
            linearly warm up the Hopkins loss coefficient.
        hopkins_sample_fraction (float): Fraction of the batch used as the sample
            for the Hopkins statistic. The paper recommends 5%. Note that the
            exact nearest-neighbour search costs O(sample_size * batch_size * d_sae)
            per training step, so smaller values are cheaper.
        k (float): The number of features to keep active. Inherited from
            BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global topk
            threshold. Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    hopkins_target: float = 0.5
    hopkins_loss_coefficient: float = 1 / 3
    hopkins_warm_up_steps: int = 0
    hopkins_sample_fraction: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 < self.hopkins_target < 1.0:
            raise ValueError(
                f"cfg.hopkins_target must be in (0, 1), got {self.hopkins_target}."
            )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "hopkins_batchtopk"


class HopkinsBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    Global Batch TopK Training SAE with a Hopkins-statistic topology regularizer.

    Adds a Hopkins loss term |H - hopkins_target| on top of the standard BatchTopK
    training losses, where H is the differentiable Hopkins statistic of the
    minibatch of SAE code vectors. Minimizing it steers the code space toward the
    topology encoded by the target value: regular (< 0.3), random (~0.5), or
    clustered (> 0.7).
    """

    cfg: HopkinsBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aux_losses = super().calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        coefficient = step_input.coefficients["hopkins"]
        hopkins_loss = coefficient * self.calculate_hopkins_loss(feature_acts)
        return {**aux_losses, "hopkins_loss": hopkins_loss}

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "hopkins": TrainCoefficientConfig(
                value=self.cfg.hopkins_loss_coefficient,
                warm_up_steps=self.cfg.hopkins_warm_up_steps,
            ),
        }

    def calculate_hopkins_loss(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Distance between the batch's Hopkins statistic and the configured target."""
        sample_size = max(
            1, int(self.cfg.hopkins_sample_fraction * feature_acts.shape[0])
        )
        statistic = calculate_hopkins_statistic(feature_acts, sample_size)
        return (statistic - self.cfg.hopkins_target).abs()


register_sae_training_class(
    "hopkins_batchtopk",
    HopkinsBatchTopKTrainingSAE,
    HopkinsBatchTopKTrainingSAEConfig,
)
