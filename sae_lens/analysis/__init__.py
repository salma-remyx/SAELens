from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.analysis.intervention_strength import (
    StrengthCalibration,
    calibrate_intervention_strength,
    output_kl,
    positionwise_norm_coefficients,
    run_with_feature_patch,
    scaled_patch,
    total_norm_coefficients,
)

__all__ = [
    "HookedSAETransformer",
    "StrengthCalibration",
    "calibrate_intervention_strength",
    "output_kl",
    "positionwise_norm_coefficients",
    "run_with_feature_patch",
    "scaled_patch",
    "total_norm_coefficients",
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
