"""Spectral prefix readout for Matryoshka BatchTopK SAEs.

Adapted from SpecMRL ("SpecMRL: Redesign Matryoshka Representation Learning
within Parametric Spectrum", https://arxiv.org/abs/2610.26824). SpecMRL
reorganizes nested-prefix (Matryoshka) training with a parameter-derived
spectral basis: prefixes are read out in a learned spectral space whose
directions are ordered by feature importance, rather than in raw coordinates.
The basis is maintained by streaming power iteration over the Gram matrix of a
learnable projection W_proj (U = QR(W_proj W_proj^T U_prev), with U_prev
detached), which adds no new loss terms: the standard nested losses themselves
teach the projection which directions matter.

Here the reorganized readout is the nested per-level reconstruction of
MatryoshkaBatchTopKTrainingSAE. Feature activations are rotated into the
spectral space and each level decodes its leading coefficients, which is
equivalent to reconstructing from the projection of the activations onto the
leading spectral subspace. The full-width reconstruction is invariant to the
rotation, so the outer SAE, the interpretability of individual decoder
directions, and the saved JumpReLU inference SAE are unchanged; only the
nested-prefix structure is reorganized.

To train with the standard runner, register the architecture first:

    from sae_lens.saes.spectral_matryoshka_sae import (
        register_spectral_matryoshka_sae,
    )

    register_spectral_matryoshka_sae()

after which the architecture name "spectral_matryoshka_batchtopk" can be used
anywhere a training architecture is selected.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from typing_extensions import override

from sae_lens.registry import register_sae_training_class
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import _sparse_matmul_nd

SPECTRAL_MATRYOSHKA_BATCHTOPK_ARCHITECTURE = "spectral_matryoshka_batchtopk"
SPECTRAL_BASIS_FILENAME = "spectral_basis.safetensors"


@dataclass
class SpectralMatryoshkaBatchTopKTrainingSAEConfig(
    MatryoshkaBatchTopKTrainingSAEConfig
):
    """
    Configuration class for training a SpectralMatryoshkaBatchTopKTrainingSAE.

    This extends Matryoshka BatchTopK training with a parameter-derived spectral
    basis: the nested level readouts are performed over prefixes of a learned
    spectral space ordered by feature importance, rather than over raw latent
    coordinates. No additional loss terms are introduced.

    Args:
        spectral_proj_init_std (float): Standard deviation of the normal
            initialization of the projection W_proj whose Gram matrix defines
            the spectral basis.
        spectral_proj_rank_multiplier (int): W_proj has shape
            (d_sae, spectral_proj_rank_multiplier * d_sae). Any value >= 1 keeps
            the Gram matrix full rank; SpecMRL over-parameterized 128x at
            embedding dimension 1024 with gains plateauing beyond that, at a
            linear memory and compute cost.
        matryoshka_widths (list[int]): The widths of the matryoshka levels.
            Inherited from MatryoshkaBatchTopKTrainingSAEConfig.
        k (float): The number of features to keep active. Inherited from
            BatchTopKTrainingSAEConfig.
        use_matryoshka_aux_loss (bool): Whether dead latents are encouraged to
            reconstruct the error of their own level. With spectral readout the
            level reconstructions are spectral while dead-neuron selection still
            uses raw coordinate slices, so this combination is heuristic.
            Inherited from MatryoshkaBatchTopKTrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    spectral_proj_init_std: float = 0.02
    spectral_proj_rank_multiplier: int = 4

    @override
    @classmethod
    def architecture(cls) -> str:
        return SPECTRAL_MATRYOSHKA_BATCHTOPK_ARCHITECTURE


class SpectralMatryoshkaBatchTopKTrainingSAE(MatryoshkaBatchTopKTrainingSAE):
    """
    Matryoshka BatchTopK training SAE with spectral prefix readout.

    Each training step runs one warm-started orthogonal iteration on the Gram
    matrix of the learnable projection W_proj and reads the matryoshka levels
    out of the resulting spectral space: a level of width w reconstructs from
    the projection of the feature activations onto the leading w spectral
    directions. Within a step the basis is differentiable through W_proj and
    the QR factor, so the standard nested losses train the projection; between
    steps the basis is cached detached. The full-width reconstruction is
    invariant to the rotation, so the outer SAE and the saved JumpReLU
    inference SAE are unchanged.
    """

    cfg: SpectralMatryoshkaBatchTopKTrainingSAEConfig  # type: ignore[assignment]
    W_proj: nn.Parameter
    spectral_basis: torch.Tensor

    def __init__(
        self,
        cfg: SpectralMatryoshkaBatchTopKTrainingSAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__(cfg, use_error_term)
        if cfg.spectral_proj_rank_multiplier < 1:
            raise ValueError("cfg.spectral_proj_rank_multiplier must be >= 1.")
        proj_rank = cfg.d_sae * cfg.spectral_proj_rank_multiplier
        self.W_proj = nn.Parameter(
            torch.randn(cfg.d_sae, proj_rank, dtype=self.dtype, device=self.device)
            * cfg.spectral_proj_init_std
        )
        # kept in float32 so the iteration and the saved basis stay precise
        # regardless of the SAE dtype
        self.register_buffer("spectral_basis", _initial_spectral_basis(self.W_proj))

    def _next_spectral_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one warm-started orthogonal iteration of the spectral basis
        (SpecMRL Eq. 7): U = QR(W_proj W_proj^T U_prev), with U_prev detached.
        Gradients flow through W_proj and the QR factor. Returns the new basis
        and its Frobenius-norm drift from the previous one.
        """
        # autocast is disabled and the Gram matrix is computed in float32,
        # since the iteration loses precision quickly in lower precision
        with torch.autocast(device_type=self.W_proj.device.type, enabled=False):
            W_proj = self.W_proj.float()
            U_prev = self.spectral_basis.detach().float()
            U_next, _ = torch.linalg.qr(W_proj @ W_proj.T @ U_prev)
        drift = (U_next.detach() - U_prev).norm()
        return U_next, drift

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        basis, drift = self._next_spectral_basis()
        # the refreshed basis is differentiable through W_proj and the QR factor
        # for every loss computed in this step
        self.spectral_basis = basis
        base_output = super().training_forward_pass(step_input)
        # cache the basis detached as the warm start for the next step, so no
        # gradients flow between steps
        self.spectral_basis = basis.detach()
        base_output.metrics["spectral_basis_drift"] = drift
        return base_output

    @override
    def _iterable_decode(
        self, feature_acts: torch.Tensor, include_outer_loss: bool = False
    ) -> Generator[tuple[int, torch.Tensor], None, None]:
        if self.cfg.rescale_acts_by_decoder_norm:
            # need to multiply by the inverse of the norm because division is
            # illegal with sparse tensors
            inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)
            feature_acts = feature_acts * inv_W_dec_norm
        spectral_basis = self.spectral_basis
        # rotate activations into the spectral space (SpecMRL Eq. 5): z = acts @ U
        if feature_acts.is_sparse:
            rotated_acts = _sparse_matmul_nd(
                feature_acts, spectral_basis.to(feature_acts.dtype)
            )
        else:
            rotated_acts = feature_acts @ spectral_basis.to(feature_acts.dtype)
        # decoder directions expressed in spectral coordinates: U^T W_dec
        rotated_W_dec = spectral_basis.T.to(self.W_dec.dtype) @ self.W_dec
        widths = self.cfg.matryoshka_widths
        prev_width = 0
        if not include_outer_loss:
            widths = widths[:-1]
        decoded = self.b_dec
        for width in widths:
            decoded = (
                rotated_acts[:, prev_width:width] @ rotated_W_dec[prev_width:width]
                + decoded
            )
            prev_width = width
            yield width, self.run_time_activation_norm_fn_out(decoded)

    @torch.no_grad()
    def freeze_spectral_basis(self) -> None:
        """
        Fix the spectral basis for inference with one final float32 orthogonal
        iteration from the cached basis, using the final W_proj. After this the
        basis no longer needs recomputation from W_proj.
        """
        basis, _ = self._next_spectral_basis()
        self.spectral_basis = basis

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        # the inference SAE is an ordinary JumpReLU SAE (the outer reconstruction
        # is invariant to the rotation); the spectral parameters only define the
        # nested prefix readout and ship alongside the weights instead
        state_dict.pop("W_proj", None)
        state_dict.pop("spectral_basis", None)

    @override
    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        self.freeze_spectral_basis()
        model_weights_path, cfg_path = super().save_inference_model(path)
        save_file(
            {"spectral_basis": self.spectral_basis.detach().cpu().contiguous()},
            str(Path(path) / SPECTRAL_BASIS_FILENAME),
        )
        return model_weights_path, cfg_path


def _initial_spectral_basis(W_proj: torch.Tensor) -> torch.Tensor:
    # the ordered eigenvectors of the initial Gram matrix warm-start the
    # iteration (SpecMRL Eq. 4); eigh returns ascending eigenvalues, so flip to
    # order the columns by descending importance
    W_proj = W_proj.detach().float()
    _, eigenvectors = torch.linalg.eigh(W_proj @ W_proj.T)
    return eigenvectors.flip(-1).contiguous()


def load_spectral_basis(path: str | Path) -> torch.Tensor:
    """
    Load the frozen spectral basis saved next to an inference SAE trained with
    spectral prefix readout.

    Column i is the i-th most important spectral direction: the compact readout
    at width w uses the leading w columns, i.e. activations @ U[:, :w] are the
    compact coefficients and U[:, :w].T @ W_dec are the matching decoder
    directions.

    Args:
        path: Directory holding the saved inference SAE.
    """
    return load_file(str(Path(path) / SPECTRAL_BASIS_FILENAME))["spectral_basis"]


def register_spectral_matryoshka_sae() -> None:
    """
    Register the spectral matryoshka BatchTopK training SAE under the
    architecture name "spectral_matryoshka_batchtopk" so it can be selected by
    architecture in SAETrainingRunner / LanguageModelSAETrainingRunner.

    Registration is explicit rather than at import time, so importing this
    module never mutates the global SAE registries. Raises if the architecture
    is already registered.
    """
    register_sae_training_class(
        SPECTRAL_MATRYOSHKA_BATCHTOPK_ARCHITECTURE,
        SpectralMatryoshkaBatchTopKTrainingSAE,
        SpectralMatryoshkaBatchTopKTrainingSAEConfig,
    )
