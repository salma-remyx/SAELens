from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset, Features, Sequence, Value

from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sparse_activation_codec import (
    bitmask_column,
    bitmask_num_bytes,
    decode_sparse_rows,
    encode_sparse_rows,
    is_sparse_activation_cache,
    unpack_nonzero_mask,
    values_column,
)


def test_bitmask_packing_matches_hand_computed_bytes():
    acts = torch.tensor([0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 7.0])

    bitmask_rows, values_rows = encode_sparse_rows(acts.reshape(1, 1, 9))

    # nonzero positions 1, 3 land in byte 0 as 2^1 + 2^3, position 8 in byte 1 as 2^0
    assert bitmask_rows == [[10, 1]]
    assert values_rows == [[5.0, 3.0, 7.0]]
    decoded = decode_sparse_rows(bitmask_rows, values_rows, 1, 9, torch.float32)
    assert torch.equal(decoded, acts.reshape(1, 1, 9))


def test_all_zero_activations_round_trip_losslessly():
    acts = torch.zeros(3, 2, 5)

    bitmask_rows, values_rows = encode_sparse_rows(acts)

    assert bitmask_rows == [[0, 0]] * 3
    assert values_rows == [[], [], []]
    decoded = decode_sparse_rows(bitmask_rows, values_rows, 2, 5, torch.float32)
    assert torch.equal(decoded, acts)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_round_trip_is_lossless_for_random_sparse_activations(dtype: torch.dtype):
    acts = torch.randn(32, 4, 64, dtype=dtype)
    acts[acts.abs() < 1.0] = 0.0

    bitmask_rows, values_rows = encode_sparse_rows(acts)

    assert sum(len(values) for values in values_rows) > 0
    decoded = decode_sparse_rows(bitmask_rows, values_rows, 4, 64, dtype)
    assert torch.equal(decoded, acts)


def test_stored_size_is_bitmask_plus_nonzero_values_only():
    acts = torch.zeros(4, 8, 128)
    acts[:, :, ::8] = torch.randn(4, 8, 16)  # exactly 1 in 8 entries nonzero

    bitmask_rows, values_rows = encode_sparse_rows(acts)

    for bitmask_row in bitmask_rows:
        assert len(bitmask_row) == bitmask_num_bytes(8 * 128) == 128
    assert sum(len(values) for values in values_rows) == 4 * 8 * 16
    # 4 rows * (128 bitmask bytes + 128 values * 4 bytes) = 2560 stored bytes
    # vs 4 * 1024 * 4 = 16384 dense bytes
    stored_bytes = 4 * 128 + 4 * 8 * 16 * 4
    assert stored_bytes == 2560
    assert stored_bytes < acts.numel() * acts.element_size() / 4


def test_decode_raises_when_value_count_does_not_match_bitmask():
    acts = torch.randn(2, 3, 7)
    acts[0, 0, 0] = 0.0
    bitmask_rows, values_rows = encode_sparse_rows(acts)
    values_rows[0] = values_rows[0][1:]  # corrupt: drop one stored value

    with pytest.raises(ValueError, match="nonzero activations"):
        decode_sparse_rows(bitmask_rows, values_rows, 3, 7, torch.float32)


def test_unpack_rejects_wrong_bitmask_length():
    with pytest.raises(ValueError, match="expected"):
        unpack_nonzero_mask(torch.zeros(2, dtype=torch.uint8), 20)


def test_sparse_cache_column_detection():
    columns = [
        "blocks.0.hook_mlp_out_bitmask",
        "blocks.0.hook_mlp_out_values",
        "token_ids",
    ]

    assert is_sparse_activation_cache(columns, "blocks.0.hook_mlp_out")
    assert not is_sparse_activation_cache(columns, "blocks.1.hook_mlp_out")
    assert not is_sparse_activation_cache(
        ["blocks.0.hook_mlp_out_bitmask", "token_ids"], "blocks.0.hook_mlp_out"
    )


def _save_sparse_activation_cache(
    tmp_path: Path,
    hook_name: str,
    acts: torch.Tensor,
    token_ids: torch.Tensor,
) -> None:
    """Build an on-disk cache shaped exactly like a sparse CacheActivationsRunner shard."""
    context_size, d_in = acts.shape[1], acts.shape[2]
    bitmask_rows, values_rows = encode_sparse_rows(acts)
    shard = Dataset.from_dict(
        {
            bitmask_column(hook_name): bitmask_rows,
            values_column(hook_name): values_rows,
            "token_ids": token_ids.tolist(),
        },
        features=Features(
            {
                bitmask_column(hook_name): Sequence(
                    Value(dtype="uint8"),
                    length=bitmask_num_bytes(context_size * d_in),
                ),
                values_column(hook_name): Sequence(Value(dtype="float32")),
                "token_ids": Sequence(Value(dtype="int32"), length=context_size),
            }
        ),
    )
    shard.save_to_disk(str(tmp_path))


def _sparse_cache_sae_runner_config(
    tmp_path: Path, hook_name: str, context_size: int, d_in: int
) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
    return LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=d_in, d_sae=32),
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        model_name="test-model",
        hook_name=hook_name,
        dataset_path="",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=64,
        train_batch_size_tokens=8,
        n_batches_in_buffer=2,
        store_batch_size_prompts=8,
        device="cpu",
        seed=42,
        dtype="float32",
    )


def _mock_model(
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
) -> Any:
    class MockModel:
        def to_tokens(self, *args: tuple[Any, ...], **kwargs: Any) -> torch.Tensor:
            return torch.ones(cfg.context_size)

        @property
        def W_E(self) -> torch.Tensor:
            return torch.ones(16, 16)

        @property
        def cfg(self) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
            return cfg

    return MockModel()


def test_activations_store_decodes_sparse_cached_dataset_exactly(tmp_path: Path):
    hook_name = "blocks.0.hook_mlp_out"
    context_size, d_in, n_rows = 4, 16, 8
    keep = torch.rand(n_rows, context_size, d_in) < 0.25
    acts = torch.where(keep, torch.randn(n_rows, context_size, d_in), torch.zeros(1))
    token_ids = torch.randint(0, 1000, (n_rows, context_size), dtype=torch.int32)
    _save_sparse_activation_cache(tmp_path, hook_name, acts, token_ids)

    cfg = _sparse_cache_sae_runner_config(tmp_path, hook_name, context_size, d_in)
    store = ActivationsStore.from_config(
        _mock_model(cfg),
        cfg,
        override_dataset=Dataset.from_list([{"text": "hello world"}] * 16),
    )

    acts_buffer, token_ids_buffer = store.get_raw_llm_batch()

    assert torch.equal(acts_buffer, acts.reshape(n_rows * context_size, d_in))
    assert token_ids_buffer is not None
    assert torch.equal(token_ids_buffer, token_ids.reshape(n_rows * context_size))


def test_activations_store_rejects_sparse_cache_with_wrong_d_in(tmp_path: Path):
    hook_name = "blocks.0.hook_mlp_out"
    context_size, d_in, n_rows = 4, 16, 8
    acts = torch.randn(n_rows, context_size, d_in)
    token_ids = torch.zeros(n_rows, context_size, dtype=torch.int32)
    _save_sparse_activation_cache(tmp_path, hook_name, acts, token_ids)

    cfg = _sparse_cache_sae_runner_config(tmp_path, hook_name, context_size, d_in + 1)
    model = _mock_model(cfg)

    with pytest.raises(
        ValueError,
        match=r"bitmask length 8 does not match context_size \(4\) and d_in \(17\)",
    ):
        ActivationsStore.from_config(
            model,
            cfg,
            override_dataset=Dataset.from_list([{"text": "hello world"}] * 16),
        )


def test_cache_activations_runner_config_rejects_unknown_storage_format():
    with pytest.raises(ValueError, match="activation_storage_format must be one of"):
        CacheActivationsRunnerConfig(
            dataset_path="dummy",
            model_name="dummy",
            model_batch_size=1,
            hook_name="blocks.0.hook_mlp_out",
            d_in=4,
            training_tokens=16,
            context_size=4,
            activation_storage_format="quantized",
        )
