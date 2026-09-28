"""Set-level similarity and stability metrics over active SAE latent sets.

Treats each input's SAE code not as a dense vector but as the *set* of latent
indices that are active for it, and measures similarity between inputs as the
overlap of those sets (Jaccard). This is the set-level similarity measure
proposed in "Beyond a Bag of Features: Set-Level Instability in Sparse
Autoencoders" (arXiv:2608.11197), which studies how much active latent sets
change under small, controlled modifications of the input and finds evidence
that, outside idealised settings, SAE features do not compose via simple
bag-of-features semantics.

Adapted from the paper: the set-overlap similarity and the
modification-stability analysis are ported directly, while the paper's
natural-text and human-typicality benchmarks are left to downstream
evaluation.
"""

from typing import Any

import torch

from sae_lens.saes.sae import SAE


def active_latent_mask(
    sparse_codes: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Return a boolean mask of which latents are active in each SAE code.

    Args:
        sparse_codes: SAE feature activations of shape (..., d_sae), e.g. the
            output of sae.encode().
        threshold: Minimum activation value for a latent to count as active.

    Returns:
        Boolean tensor of shape (n, d_sae), where n is the product of all
        leading dimensions of sparse_codes.
    """
    return sparse_codes.reshape(-1, sparse_codes.shape[-1]) > threshold


def latent_set_jaccard(
    codes_a: torch.Tensor,
    codes_b: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Paired Jaccard overlap |A ∩ B| / |A ∪ B| between active latent sets.

    Rows are matched elementwise, so codes_a and codes_b must have the same
    shape. Two empty sets are treated as fully overlapping (1.0); an empty set
    against a non-empty one has zero overlap.

    Args:
        codes_a: SAE feature activations of shape (..., d_sae).
        codes_b: SAE feature activations of the same shape as codes_a.
        threshold: Minimum activation value for a latent to count as active.

    Returns:
        Float tensor of shape (n,) with the Jaccard overlap of each row pair.
    """
    mask_a = active_latent_mask(codes_a, threshold)
    mask_b = active_latent_mask(codes_b, threshold)
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            "codes_a and codes_b must have the same shape, got "
            f"{codes_a.shape} and {codes_b.shape}."
        )
    intersection = (mask_a & mask_b).sum(dim=-1).float()
    union = (mask_a | mask_b).sum(dim=-1).float()
    return torch.where(
        union > 0, intersection / union.clamp(min=1), torch.ones_like(union)
    )


def latent_set_jaccard_matrix(
    codes_a: torch.Tensor,
    codes_b: torch.Tensor | None = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    All-pairs Jaccard overlap between active latent sets.

    Useful for building set-level similarity neighbourhoods: unlike cosine
    similarity over dense representations, each comparison only involves the
    latents active for the two inputs.

    Args:
        codes_a: SAE feature activations of shape (..., d_sae).
        codes_b: Optional second tensor of shape (..., d_sae); when omitted,
            codes_a is compared against itself.
        threshold: Minimum activation value for a latent to count as active.

    Returns:
        Float tensor of shape (n_a, n_b) where entry (i, j) is the Jaccard
        overlap between row i of codes_a and row j of codes_b.
    """
    mask_a = active_latent_mask(codes_a, threshold).float()
    mask_b = (
        mask_a
        if codes_b is None
        else active_latent_mask(codes_b, threshold).float()
    )
    intersection = mask_a @ mask_b.T
    sizes_a = mask_a.sum(dim=-1)
    sizes_b = mask_b.sum(dim=-1)
    union = sizes_a[:, None] + sizes_b[None, :] - intersection
    return torch.where(
        union > 0, intersection / union.clamp(min=1), torch.ones_like(union)
    )


@torch.no_grad()
def latent_set_stability(
    sae: SAE[Any],
    base_activations: torch.Tensor,
    modified_activations: torch.Tensor,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Measure how much the SAE's active latent set changes under modifications.

    Both activation tensors are encoded with sae.encode() and matched
    elementwise, so each row of base_activations is compared with the same row
    of modified_activations. BatchTopK SAEs allocate their activation budget
    across the whole flattened batch, so keep the two tensors the same shape
    for a fair comparison.

    Args:
        sae: SAE whose encode() produces sparse feature activations.
        base_activations: Activations of shape (..., d_in).
        modified_activations: Activations of the same shape as
            base_activations.
        threshold: Minimum activation value for a latent to count as active.

    Returns:
        Dictionary of summary metrics:

        - n_examples: number of paired rows compared.
        - mean_jaccard_overlap: mean |A ∩ B| / |A ∪ B| over all pairs; 1.0
          means no active set changed at all.
        - identical_set_fraction: fraction of pairs whose active sets are
          exactly equal.
        - mean_retained_fraction: mean fraction of base active latents that
          stay active after the modification.
        - mean_active_set_size_base and mean_active_set_size_modified: mean
          number of active latents per row on each side.
        - per_example_jaccard: the per-pair overlaps the summaries aggregate.
    """
    if base_activations.shape != modified_activations.shape:
        raise ValueError(
            "base_activations and modified_activations must have the same "
            f"shape, got {base_activations.shape} and {modified_activations.shape}."
        )

    codes_base = sae.encode(base_activations)
    codes_modified = sae.encode(modified_activations)

    mask_base = active_latent_mask(codes_base, threshold)
    mask_modified = active_latent_mask(codes_modified, threshold)

    jaccard = latent_set_jaccard(codes_base, codes_modified, threshold)
    base_sizes = mask_base.sum(dim=-1).float()
    intersection = (mask_base & mask_modified).sum(dim=-1).float()
    retained = torch.where(
        base_sizes > 0,
        intersection / base_sizes.clamp(min=1),
        torch.ones_like(base_sizes),
    )

    return {
        "n_examples": int(jaccard.numel()),
        "mean_jaccard_overlap": float(jaccard.mean()),
        "identical_set_fraction": float((jaccard == 1.0).float().mean()),
        "mean_retained_fraction": float(retained.mean()),
        "mean_active_set_size_base": float(base_sizes.mean()),
        "mean_active_set_size_modified": float(
            mask_modified.sum(dim=-1).float().mean()
        ),
        "per_example_jaccard": jaccard.tolist(),
    }
