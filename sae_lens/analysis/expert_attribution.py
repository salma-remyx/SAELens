"""Attribute SAE latent pre-activations to individual MoE experts.

Adapted from "SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts
Exhibit Human-Interpretable Fine-Grained Specialization"
(https://arxiv.org/abs/2611.08089).

An MoE layer writes a router-weighted sum of expert outputs,
``moe_output = sum_e g_e * expert_output_e``, and an SAE encoder is affine, so
the pre-activation of every latent decomposes exactly over the experts:
``contributions[..., e, j] = g_e * (expert_output_e @ W_enc[:, j])`` and
``sum_e contributions[..., e, :] + b_enc == moe_output @ W_enc + b_enc``.

Aggregated over a corpus, this gives every expert a dominant set of latents
whose activation it principally drives. Compared to counting how often each
expert is routed, these contribution shares are far more concentrated: most
latents have only one or two dominant experts, and the overlap between
dominant sets concentrates on a few expert pairs, revealing specialization
below coarse domains.

To use with a HookedSAETransformer: cache the expert outputs and router
weights of an MoE layer (e.g. via ``run_with_cache``), reconstruct
``moe_output`` from them, and pass the per-expert tensors here together with
the SAE. The expert-independent offsets (``b_enc``, and ``-b_dec @ W_enc``
when ``apply_b_dec_to_input`` is set) complete the exact decomposition but
cancel in the per-latent dominance shares.
"""

from dataclasses import dataclass

import torch

from sae_lens.saes.sae import SAE

_EPS = 1e-12


def expert_latent_contributions(
    expert_outputs: torch.Tensor,
    router_weights: torch.Tensor,
    W_enc: torch.Tensor,
) -> torch.Tensor:
    """
    Decompose SAE latent pre-activations into exact per-expert contributions.

    Args:
        expert_outputs: (..., num_experts, d_in) outputs of the individual MoE
            experts (before router weighting).
        router_weights: (..., num_experts) router weights applied to each
            expert.
        W_enc: (d_in, d_sae) encoder weight matrix of the SAE.

    Returns:
        (..., num_experts, d_sae) contribution of each expert to each latent
        pre-activation, with
        ``result.sum(dim=-2) + b_enc == moe_output @ W_enc + b_enc`` where
        ``moe_output = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)``.
    """
    if expert_outputs.shape[:-1] != router_weights.shape:
        raise ValueError(
            "expert_outputs shape (..., num_experts, d_in) of "
            f"{tuple(expert_outputs.shape)} is incompatible with "
            f"router_weights shape (..., num_experts) of {tuple(router_weights.shape)}"
        )
    return router_weights.unsqueeze(-1) * (expert_outputs @ W_enc)


def expert_latent_contributions_from_sae(
    sae: SAE,
    expert_outputs: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-expert latent contributions for a specific SAE, matching its
    ``encode_with_hidden_pre`` scaling.

    For TopK SAEs with ``rescale_acts_by_decoder_norm`` enabled the
    pre-activations are scaled by the decoder column norms, so the same
    per-latent scale is applied to every expert's contributions. The
    decomposition then satisfies ``result.sum(dim=-2) + b_enc * scale ==
    hidden_pre`` from ``sae.encode_with_hidden_pre(moe_output)`` whenever the
    SAE's input preprocessing is the identity (``apply_b_dec_to_input=False``
    and ``normalize_activations="none"``).

    Args:
        sae: SAE whose encoder defines the latent pre-activations.
        expert_outputs: (..., num_experts, d_in) outputs of the individual MoE
            experts.
        router_weights: (..., num_experts) router weights applied to each
            expert.

    Returns:
        (..., num_experts, d_sae) contribution of each expert to each latent
        pre-activation.
    """
    contributions = expert_latent_contributions(
        expert_outputs, router_weights, sae.W_enc
    )
    if getattr(sae.cfg, "rescale_acts_by_decoder_norm", False):
        contributions = contributions * sae.W_dec.norm(dim=-1)
    return contributions


def aggregate_expert_contributions(
    contributions: torch.Tensor,
    latent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Corpus-level positive contribution mass per (expert, latent).

    Negative contributions are clamped to zero: only positive pre-activation
    mass drives a latent's activation. Passing ``latent_mask`` (e.g.
    ``feature_acts > 0``) restricts the aggregation to positions where each
    latent is active, following the paper's focus on latents whose activation
    an expert principally drives.

    Args:
        contributions: (..., num_experts, d_sae) per-expert contributions from
            ``expert_latent_contributions``.
        latent_mask: optional (..., d_sae) boolean mask of active latents.

    Returns:
        (num_experts, d_sae) summed positive contribution mass.
    """
    mass = contributions.clamp_min(0.0)
    if latent_mask is not None:
        mass = mass * latent_mask.to(mass.dtype).unsqueeze(-2)
    num_experts = contributions.shape[-2]
    d_sae = contributions.shape[-1]
    return mass.reshape(-1, num_experts, d_sae).sum(dim=0)


def dominance_sets(
    aggregate: torch.Tensor,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    Dominance sets at threshold tau: expert e dominates latent j when its
    share of the aggregated contribution mass to j is at least tau.

    Args:
        aggregate: (num_experts, d_sae) aggregated contributions from
            ``aggregate_expert_contributions``.
        tau: minimum contribution share for dominance (the paper uses 0.2).

    Returns:
        (num_experts, d_sae) boolean dominance sets. Latents with no positive
        contribution mass have no dominators.
    """
    shares, active = _normalized_shares(aggregate)
    return active & (shares >= tau)


def dominant_set_overlap(dominant: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Jaccard overlap between the dominance sets of all expert pairs.

    Args:
        dominant: (num_experts, d_sae) boolean dominance sets.

    Returns:
        (num_experts, num_experts) overlap matrix where entry (e1, e2) is the
        size of the intersection of the two dominant sets divided by the size
        of their union. The diagonal is 1 for experts with non-empty dominant
        sets and 0 for experts with empty ones.
    """
    sets = dominant.to(torch.float32)
    intersections = sets @ sets.T
    sizes = sets.sum(dim=-1)
    unions = sizes.unsqueeze(1) + sizes.unsqueeze(0) - intersections
    return torch.where(
        unions > 0,
        intersections / unions.clamp_min(_EPS),
        torch.zeros_like(intersections),
    )


def effective_contributors(weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Participation ratio (inverse Herfindahl index) of non-negative weights
    normalized over dim: the effective number of entries sharing the mass.

    A value of 1.0 means all mass sits on a single expert (maximally
    concentrated); mass spread equally over n entries gives n.

    Args:
        weights: non-negative weights, e.g. (num_experts, d_sae) per-latent
            contribution mass or (num_experts,) corpus-level gating mass.
        dim: dimension to normalize and reduce over (the expert dimension).

    Returns:
        Tensor with ``dim`` removed.
    """
    shares = weights / weights.sum(dim=dim, keepdim=True).clamp_min(_EPS)
    return 1.0 / (shares * shares).sum(dim=dim)


def _normalized_shares(aggregate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    totals = aggregate.sum(dim=0)
    active = totals > 0
    shares = aggregate / totals.clamp_min(_EPS)
    return shares, active


@dataclass
class ExpertDominanceAnalysis:
    """
    Per-expert dominance analysis of an SAE trained on an MoE layer output.

    Attributes:
        aggregate: (num_experts, d_sae) positive contribution mass per expert
            and latent.
        shares: (num_experts, d_sae) per-latent contribution shares, summing
            to 1 over experts for latents with positive mass.
        dominant: (num_experts, d_sae) boolean dominance sets at threshold
            tau.
        dominators_per_latent: (d_sae,) number of experts dominating each
            latent.
        overlap: (num_experts, num_experts) Jaccard overlap between dominance
            sets of expert pairs.
        latent_concentration: (d_sae,) effective number of contributing
            experts per latent (0 for latents with no positive mass).
        gating_concentration: effective number of experts the router uses
            over the corpus.
    """

    aggregate: torch.Tensor
    shares: torch.Tensor
    dominant: torch.Tensor
    dominators_per_latent: torch.Tensor
    overlap: torch.Tensor
    latent_concentration: torch.Tensor
    gating_concentration: torch.Tensor


def analyze_expert_dominance(
    contributions: torch.Tensor,
    router_weights: torch.Tensor,
    tau: float = 0.2,
    latent_mask: torch.Tensor | None = None,
) -> ExpertDominanceAnalysis:
    """
    Run the full dominance analysis over per-expert latent contributions.

    This is the paper's post-hoc analysis: dominance sets at threshold tau,
    how many experts dominate each latent, pairwise overlap between dominant
    sets, and the concentration comparison showing that contributions are far
    more concentrated than gating (compare ``latent_concentration`` against
    ``gating_concentration``).

    Args:
        contributions: (..., num_experts, d_sae) per-expert contributions from
            ``expert_latent_contributions``.
        router_weights: (..., num_experts) router weights used to compute the
            contributions.
        tau: minimum per-latent contribution share for dominance.
        latent_mask: optional (..., d_sae) boolean mask of active latents,
            forwarded to ``aggregate_expert_contributions``.

    Returns:
        The full analysis result.
    """
    aggregate = aggregate_expert_contributions(contributions, latent_mask)
    shares, active = _normalized_shares(aggregate)
    dominant = active & (shares >= tau)
    gating_mass = router_weights.reshape(-1, router_weights.shape[-1]).mean(dim=0)
    return ExpertDominanceAnalysis(
        aggregate=aggregate,
        shares=shares,
        dominant=dominant,
        dominators_per_latent=dominant.sum(dim=0),
        overlap=dominant_set_overlap(dominant),
        latent_concentration=torch.where(
            active,
            effective_contributors(aggregate, dim=0),
            torch.zeros((), dtype=aggregate.dtype, device=aggregate.device),
        ),
        gating_concentration=effective_contributors(gating_mass, dim=0),
    )
