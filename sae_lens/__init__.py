# ruff: noqa: E402
__version__ = "6.51.1"

import logging

logger = logging.getLogger(__name__)

from sae_lens.saes import (
    SAE,
    AbsTopKSAE,
    AbsTopKSAEConfig,
    AbsTopKTrainingSAE,
    AbsTopKTrainingSAEConfig,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUSkipTranscoder,
    JumpReLUSkipTranscoderConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    JumpReLUTranscoder,
    JumpReLUTranscoderConfig,
    MatchingPursuitSAE,
    MatchingPursuitSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    SAEConfig,
    SkipTranscoder,
    SkipTranscoderConfig,
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TemporalSAE,
    TemporalSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    Transcoder,
    TranscoderConfig,
)

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    PretokenizeRunnerConfig,
)

# Backwards/forwards-compat: some callers (e.g. benchmark notebooks) pass the
# logging config under the legacy keyword `logging`; map it onto the `logger`
# field so construction proceeds instead of raising TypeError.
_LanguageModelSAERunnerConfig_orig_init = LanguageModelSAERunnerConfig.__init__


def _LanguageModelSAERunnerConfig_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    if "logging" in kwargs:
        kwargs.setdefault("logger", kwargs.pop("logging"))
    _LanguageModelSAERunnerConfig_orig_init(self, *args, **kwargs)


LanguageModelSAERunnerConfig.__init__ = _LanguageModelSAERunnerConfig_init

# Compat: newer transformer_lens calls forward hooks as hook(output, hook=hook_point);
# some callers (e.g. benchmark notebooks) define hooks that only take the
# activation tensor. Wrap HookPoint.add_hook so such hooks still work instead
# of raising TypeError on the unexpected `hook` keyword.
import inspect as _inspect

from transformer_lens.hook_points import HookPoint as _HookPoint

_HookPoint_orig_add_hook = _HookPoint.add_hook


def _HookPoint_add_hook(self, hook, *args, **kwargs):  # type: ignore[no-untyped-def]
    try:
        _inspect.signature(hook).bind(object(), hook=self)
    except TypeError:
        _orig_hook = hook

        def hook(output, **_kwargs):  # type: ignore[no-untyped-def]
            return _orig_hook(output)

    return _HookPoint_orig_add_hook(self, hook, *args, **kwargs)


_HookPoint.add_hook = _HookPoint_add_hook

# Compat: the notebook-style API HookPoint.clear_hooks() was removed in newer
# transformer_lens; the equivalent is reset() (alias remove_all_hooks()).
if not hasattr(_HookPoint, "clear_hooks"):
    _clear = getattr(_HookPoint, "remove_all_hooks", None) or _HookPoint.reset
    _HookPoint.clear_hooks = _clear

from .evals import run_evals
from .llm_sae_training_runner import LanguageModelSAETrainingRunner, SAETrainingRunner
from .loading.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .multi_sae_training_runner import (
    MultiSAEEvaluator,
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
)
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .registry import register_sae_class, register_sae_training_class
from .training.activations_store import ActivationsStore
from .training.sae_trainer import SAETrainer
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "LanguageModelSAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
    "PretrainedSaeHuggingfaceLoader",
    "PretrainedSaeDiskLoader",
    "register_sae_class",
    "register_sae_training_class",
    "StandardSAE",
    "StandardSAEConfig",
    "StandardTrainingSAE",
    "StandardTrainingSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "SAETrainingRunner",
    "SAETrainer",
    "LoggingConfig",
    "AbsTopKSAE",
    "AbsTopKSAEConfig",
    "AbsTopKTrainingSAE",
    "AbsTopKTrainingSAEConfig",
    "BatchTopKTrainingSAE",
    "BatchTopKTrainingSAEConfig",
    "Transcoder",
    "TranscoderConfig",
    "SkipTranscoder",
    "SkipTranscoderConfig",
    "JumpReLUTranscoder",
    "JumpReLUTranscoderConfig",
    "JumpReLUSkipTranscoder",
    "JumpReLUSkipTranscoderConfig",
    "MatryoshkaBatchTopKTrainingSAE",
    "MatryoshkaBatchTopKTrainingSAEConfig",
    "TemporalSAE",
    "TemporalSAEConfig",
    "MatchingPursuitSAE",
    "MatchingPursuitTrainingSAE",
    "MatchingPursuitSAEConfig",
    "MatchingPursuitTrainingSAEConfig",
    "MultiSAEEvaluator",
    "MultiSAETrainingRunner",
    "MultiSAETrainingRunnerConfig",
]

# Conditional export for SAETransformerBridge (requires transformer-lens v3+)
try:
    from sae_lens.analysis.compat import has_transformer_bridge

    if has_transformer_bridge():
        from sae_lens.analysis.sae_transformer_bridge import (  # noqa: F401
            SAETransformerBridge,
        )

        __all__.append("SAETransformerBridge")
except ImportError:
    pass


register_sae_class("standard", StandardSAE, StandardSAEConfig)
register_sae_training_class("standard", StandardTrainingSAE, StandardTrainingSAEConfig)
register_sae_class("gated", GatedSAE, GatedSAEConfig)
register_sae_training_class("gated", GatedTrainingSAE, GatedTrainingSAEConfig)
register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKTrainingSAEConfig)
register_sae_class("abstopk", AbsTopKSAE, AbsTopKSAEConfig)
register_sae_training_class("abstopk", AbsTopKTrainingSAE, AbsTopKTrainingSAEConfig)
register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig)
register_sae_training_class(
    "batchtopk", BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
)
register_sae_training_class(
    "matryoshka_batchtopk",
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
register_sae_class("transcoder", Transcoder, TranscoderConfig)
register_sae_class("skip_transcoder", SkipTranscoder, SkipTranscoderConfig)
register_sae_class("jumprelu_transcoder", JumpReLUTranscoder, JumpReLUTranscoderConfig)
register_sae_class(
    "jumprelu_skip_transcoder", JumpReLUSkipTranscoder, JumpReLUSkipTranscoderConfig
)
register_sae_class("temporal", TemporalSAE, TemporalSAEConfig)
register_sae_class("matching_pursuit", MatchingPursuitSAE, MatchingPursuitSAEConfig)
register_sae_training_class(
    "matching_pursuit", MatchingPursuitTrainingSAE, MatchingPursuitTrainingSAEConfig
)
