"""Readout canonicalization for TopK-style sparse autoencoders.

Adapted from "Readout Scaling Shapes Sparse Feature Recovery"
(https://arxiv.org/abs/2610.18462), which trains SAEs in
readout-canonicalized decision-space (RC-DS) coordinates: the singular-value
gains of the readout are removed from the space in which sparsity decisions are
made, while the information the readout retains is preserved exactly.

This is the spectrum-level generalization of
``TopKTrainingSAEConfig.rescale_acts_by_decoder_norm``. That option only divides
out per-feature column norms, which is a diagonal canonicalization and so cannot
see gains shared between features. Here the decoder's full singular value
spectrum is equalized down to a target condition number, which also removes
correlated gains.

The canonicalization is a reparameterization rather than extra information: it
is applied to pre-activations on the way into the activation function and undone
on the decoder on the way out, so the reconstruction for a given latent vector is
unchanged and only the sparsity decision (which features win, and at what
magnitude) moves into the canonicalized space.
"""

from dataclasses import dataclass

import torch

# Singular values below this fraction of the largest are treated as null
# directions of the readout and left uncanonicalized, so that equalizing the
# spectrum never amplifies noise directions into the decision space.
RANK_FLOOR = 1e-6


@dataclass
class ReadoutCanonicalization:
    """
    Change of latent coordinates that removes the readout's singular-value gains.

    Held in low-rank form so that nothing of size (d_sae, d_sae) is ever built.
    With ``U`` the retained left singular vectors of the decoder and ``ratio``
    the per-direction singular value ratio sigma / tau, the canonicalization is
    ``M = I + U diag(ratio - 1) U^T``, which is the identity wherever ratio is 1.

    Args:
        U (torch.Tensor): Retained left singular vectors of the decoder, shape
            (d_sae, r) with orthonormal columns.
        ratio (torch.Tensor): Singular value ratios sigma / tau, shape (r,).
    """

    U: torch.Tensor
    ratio: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map pre-activations into canonicalized decision space (x @ M).

        Accepts any shape whose final dimension is d_sae.
        """
        return _right_update(x, self.U, self.ratio - 1.0)

    def remove_from_decoder(self, W_dec: torch.Tensor) -> torch.Tensor:
        """
        Return M^-1 W_dec, the decoder as seen by canonicalized activations.

        Folding the inverse into the decoder (rather than applying it to the
        activations) keeps sparse activations sparse through the decoder
        multiplication.
        """
        return _left_update(W_dec, self.U, 1.0 / self.ratio - 1.0)


def compute_readout_canonicalization(
    W_dec: torch.Tensor, condition_number: float
) -> ReadoutCanonicalization:
    """
    Canonicalize a decoder so its retained singular values share a single scale.

    Each singular value sigma is mapped to tau = clamp(sigma, g / sqrt(kappa),
    g * sqrt(kappa)), where g is the geometric mean of the retained singular
    values. The geometric-mean anchor preserves the readout's overall scale while
    the clamp caps the condition number of the canonicalized readout at kappa, so
    that near-null directions are boosted less as kappa grows. A kappa of 1
    removes the singular-value gains entirely.

    Args:
        W_dec (torch.Tensor): Decoder weights of shape (d_sae, d_in).
        condition_number (float): Target condition number for the canonicalized
            readout spectrum. Must be at least 1.

    Returns:
        A canonicalization that is the identity when W_dec is numerically rank
        deficient, since there is then no well-conditioned spectrum to remove.
    """
    if condition_number < 1.0:
        raise ValueError(f"condition_number must be at least 1, got {condition_number}")
    # The SVD is numerically meaningless in low precision, so always compute it
    # in float32 and let the update helpers cast back to the target dtype.
    U, S, _ = torch.linalg.svd(W_dec.detach().float(), full_matrices=False)
    ratio = torch.ones_like(S)
    retained = S > S[0] * RANK_FLOOR
    if bool(retained.any()):
        geo_mean = S[retained].log().mean().exp()
        spread = condition_number**0.5
        tau = S.clamp(min=geo_mean / spread, max=geo_mean * spread)
        ratio = torch.where(retained, S / tau, ratio)
    return ReadoutCanonicalization(U=U, ratio=ratio)


@torch.no_grad()
def fold_readout_canonicalization(
    canonicalization: ReadoutCanonicalization,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
) -> None:
    """
    Absorb a canonicalization into SAE weights, in place.

    After folding, the encoder already produces canonicalized pre-activations and
    the decoder already maps canonicalized activations back to input space, so a
    saved inference SAE needs no transform at load time. Follows the convention
    of ``_fold_norm_topk``: rebinds ``.data`` on the tensors it is given.
    """
    b_enc.data = canonicalization.apply(b_enc.data)
    W_enc.data = canonicalization.apply(W_enc.data)
    W_dec.data = canonicalization.remove_from_decoder(W_dec.data)


def _right_update(x: torch.Tensor, U: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Return x + ((x @ U) * coef) @ U^T, i.e. x @ (I + U diag(coef) U^T)."""
    U = U.to(device=x.device, dtype=x.dtype)
    coef = coef.to(device=x.device, dtype=x.dtype)
    return x + ((x @ U) * coef) @ U.T


def _left_update(W: torch.Tensor, U: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Return W + U @ ((U^T @ W) * coef), i.e. (I + U diag(coef) U^T) @ W."""
    U = U.to(device=W.device, dtype=W.dtype)
    coef = coef.to(device=W.device, dtype=W.dtype)
    return W + U @ ((U.T @ W) * coef.unsqueeze(-1))
