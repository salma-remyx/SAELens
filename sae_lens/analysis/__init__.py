from sae_lens.analysis.bilingual_alignment import (
    BilingualAlignment,
    LanguageSelectivity,
    Monosemanticity,
    bilingual_alignment,
    collect_token_attributions,
    compute_monosemanticity,
    feature_mask,
    language_selectivity,
    make_feature_injection_hook,
    peak_alignment_layer,
    transfer_decomposed_features,
)
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer

__all__ = [
    "BilingualAlignment",
    "HookedSAETransformer",
    "LanguageSelectivity",
    "Monosemanticity",
    "bilingual_alignment",
    "collect_token_attributions",
    "compute_monosemanticity",
    "feature_mask",
    "language_selectivity",
    "make_feature_injection_hook",
    "peak_alignment_layer",
    "transfer_decomposed_features",
]

try:
    from sae_lens.analysis.compat import has_transformer_bridge

    if has_transformer_bridge():
        from sae_lens.analysis.sae_transformer_bridge import (  # noqa: F401
            SAETransformerBridge,
        )

        __all__.append("SAETransformerBridge")
except ImportError:
    pass
