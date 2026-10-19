"""Readout canonicalization for SAE training and evaluation.

A readout maps activations to task outputs via ``x @ readout.T`` (e.g. an LM
head mapping residual-stream activations to logits, or a label head mapping
them to class scores). Its singular values act as per-direction gains which
shape what an SAE learns from the data: directions the readout amplifies
dominate any loss computed downstream of the readout, even when the factors
generating the data are spread evenly across directions.

This module provides the pieces of readout-canonicalized decision-space
(RC-DS) training from "Readout Scaling Shapes Sparse Feature Recovery"
(arXiv:2610.18462), adapted to SAELens' activation normalization and
weight-folding pipeline:

- ``readout_canonicalizing_transform`` builds an invertible transform which
  removes the readout's singular-value gains while preserving retained
  decision information (directions the readout cannot see pass through
  unchanged).
- ``fold_activation_transform`` folds a linear activation transform into SAE
  weights exactly, generalizing ``SAE.fold_activation_norm_scaling_factor``
  (the scalar special case). A SAE trained on transformed activations
  ``z = x @ transform`` becomes a standard SAE consuming raw activations,
  reproducing the wrap-train-unfold pipeline to numerical precision.
- ``gain_free_reconstruction_error`` measures reconstruction error in the
  readout's decision space with singular-value gains removed, so that
  reconstruction quality is not judged by directions the readout amplifies
  or hides.
"""

import torch

from sae_lens.saes.sae import SAE


def readout_canonicalizing_transform(readout: torch.Tensor) -> torch.Tensor:
    """Build the readout-canonicalizing decision-space (RC-DS) transform.

    The returned matrix ``M`` makes ``readout @ M`` gain-free: each singular
    value of the readout corresponding to a direction it can see is rescaled
    to one, while directions outside the readout's row space (which carry no
    decision information) are left unchanged. ``M`` is the symmetric
    readout whitening ``V diag(1/s) V^T`` (i.e. ``(readout^T readout)^-1/2``
    on the readout's row space) with invisible directions given unit scale,
    and is always invertible, so ``z = x @ M`` is an information-preserving
    reparameterization of the activations.

    Args:
        readout: Matrix of shape (d_readout, d_in) mapping activation row
            vectors to outputs via ``x @ readout.T``.

    Returns:
        Invertible matrix of shape (d_in, d_in); ``z = x @ M`` gives the
        RC-DS coordinates of the activations. Returned in float64 for
        float64 inputs and float32 otherwise, so that the gain-free property
        is not quantized away for low-precision readouts.
    """
    compute_dtype = torch.float64 if readout.dtype == torch.float64 else torch.float32
    readout_f = readout.to(compute_dtype)
    d_in = readout_f.shape[-1]
    # Full SVD: readout = U diag(s) Vh with Vh square orthonormal, so the
    # columns of Vh.T form a complete basis of activation space whose first
    # s.numel() directions are the readout's decision directions with gains
    # given by s (padded with zeros for the invisible directions).
    _, s, vh = torch.linalg.svd(readout_f, full_matrices=True)
    inverse_gains = torch.ones(d_in, dtype=compute_dtype, device=readout_f.device)
    if s.max() > 0:
        observed = s > s.max() * d_in * torch.finfo(compute_dtype).eps
        inverse_gains[: s.numel()] = torch.where(observed, 1.0 / s, torch.ones_like(s))
    return (vh.T * inverse_gains) @ vh


@torch.no_grad()
def fold_activation_transform(sae: SAE, transform: torch.Tensor) -> None:
    """Fold a linear activation transform into SAE weights, in place.

    Intended after training on transformed activations ``z = x @ transform``:
    the folded SAE consumes raw activations ``x`` directly, via

    - ``W_enc <- transform @ W_enc``
    - ``W_dec <- W_dec @ transform^-1``
    - ``b_dec <- b_dec @ transform^-1``

    The fold is exact: for architectures whose encode / decode use the plain
    ``(x - b_dec) @ W_enc + b_enc`` and ``acts @ W_dec + b_dec`` contractions
    (standard, gated, JumpReLU, TopK with
    ``rescale_acts_by_decoder_norm=False``) and with
    ``normalize_activations="none"``, the folded SAE satisfies
    ``folded_sae(x) == sae(x @ transform) @ transform^-1`` to numerical
    precision — the wrapper changes training geometry without adding
    information. TopK training configs with
    ``rescale_acts_by_decoder_norm=True`` recompute decoder norms inside
    encode / decode, so folding is only function-preserving there for
    transforms which preserve decoder row norms (e.g. orthogonal ones).

    This generalizes ``SAE.fold_activation_norm_scaling_factor``, which is
    the scalar special case ``transform = scaling_factor * I``.

    Args:
        sae: SAE whose weights should be folded in place.
        transform: Invertible matrix of shape (d_in, d_in).
    """
    transform = transform.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    inverse = torch.linalg.inv(transform)
    sae.W_enc.data = transform @ sae.W_enc.data
    sae.W_dec.data = sae.W_dec.data @ inverse
    sae.b_dec.data = sae.b_dec.data @ inverse


def gain_free_reconstruction_error(
    sae_in: torch.Tensor,
    sae_out: torch.Tensor,
    readout: torch.Tensor,
) -> float:
    """Mean squared reconstruction error in gain-free decision coordinates.

    The error ``sae_out - sae_in`` is expressed in the readout's decision
    basis (its right singular vectors), squared, summed over the decision
    space and averaged over the batch — the same convention as the plain
    reconstruction loss ``((sae_out - sae_in) ** 2).sum(dim=-1).mean()``, but
    restricted to the subspace the readout retains and with its directional
    gains removed. Errors in directions the readout cannot see contribute
    nothing; errors of equal magnitude along high- and low-gain decision
    directions contribute equally.

    Args:
        sae_in: Input activations of shape (batch_size, d_in).
        sae_out: Reconstructed activations of shape (batch_size, d_in).
        readout: Matrix of shape (d_readout, d_in) mapping activations to
            outputs via ``x @ readout.T``.

    Returns:
        Mean gain-free squared reconstruction error.
    """
    compute_dtype = torch.float64 if sae_in.dtype == torch.float64 else torch.float32
    readout_f = readout.to(compute_dtype)
    _, s, vh = torch.linalg.svd(readout_f, full_matrices=False)
    if s.numel() and s.max() > 0:
        # keep only the directions the readout actually retains; the
        # remaining right singular vectors are an arbitrary basis of the
        # readout's null space and carry no decision information.
        retained = s > s.max() * readout_f.shape[-1] * torch.finfo(compute_dtype).eps
        vh = vh[retained]
    error = (sae_out - sae_in).to(compute_dtype)
    return (((error @ vh.T) ** 2).sum(dim=-1).mean()).item()
