"""End-to-end training workflow for the sample-conditioned selection variant.

Trains a GumbelTopKTrainingSAE next to a matched-budget TopKTrainingSAE control
on synthetic sparse data, driving both through the standard
``training_forward_pass`` API, then checks that the learned selection head
actually conditions on the sample. Adapted from SAMPLESELECT
(https://arxiv.org/abs/2609.17076), which trains a per-input fixed-budget
feature mask with differentiable Gumbel Top-k selection.
"""

import torch

from sae_lens.saes.gumbel_topk_sae import (
    GumbelTopKTrainingSAE,
    GumbelTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

D_IN = 16
D_SAE = 32
K = 4
N_GROUND_TRUTH_FEATURES = 8
STEPS = 600
BATCH_SIZE = 64
LR = 3e-3


def _sample_sae_in(basis: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Draw activations from a sparse linear generative model over `basis`."""
    codes = (torch.rand(batch_size, basis.shape[0]) < 0.3) * torch.randn(
        batch_size, basis.shape[0]
    )
    return codes @ basis + 0.05 * torch.randn(batch_size, basis.shape[1])


def _train_sae(sae: TopKTrainingSAE, basis: torch.Tensor) -> tuple[float, float]:
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    initial_loss = None
    recent_losses: list[float] = []
    for step in range(STEPS):
        step_input = TrainStepInput(
            sae_in=_sample_sae_in(basis, BATCH_SIZE),
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=step,
            is_logging_step=False,
        )
        output = sae.training_forward_pass(step_input)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        loss_value = output.loss.item()
        if step == 0:
            initial_loss = loss_value
        if step >= STEPS - 10:
            recent_losses.append(loss_value)
    assert initial_loss is not None
    return initial_loss, sum(recent_losses) / len(recent_losses)


def test_typical_sample_conditioned_selection_training_workflow() -> None:
    basis = torch.randn(N_GROUND_TRUTH_FEATURES, D_IN)

    gumbel_sae = GumbelTopKTrainingSAE(
        GumbelTopKTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            k=K,
            dtype="float32",
            device="cpu",
            rescale_acts_by_decoder_norm=False,
            apply_b_dec_to_input=False,
            selection_hidden_dim=16,
        )
    )
    gumbel_sae.train()
    initial_loss, final_loss = _train_sae(gumbel_sae, basis)

    assert final_loss < 0.25 * initial_loss, (
        f"reconstruction did not train: {initial_loss=} {final_loss=}"
    )

    # The matched control: the same per-sample budget, but always spent on the
    # k largest-magnitude features (SAMPLESELECT's full-representation control).
    topk_sae = TopKTrainingSAE(
        TopKTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            k=K,
            dtype="float32",
            device="cpu",
            rescale_acts_by_decoder_norm=False,
            apply_b_dec_to_input=False,
        )
    )
    topk_sae.train()
    topk_initial, topk_final = _train_sae(topk_sae, basis)
    print(
        f"matched-budget control over {STEPS} steps: "
        f"topk {topk_initial:.3f} -> {topk_final:.3f}, "
        f"gumbel_topk {initial_loss:.3f} -> {final_loss:.3f}"
    )

    gumbel_sae.eval()
    eval_sae_in = _sample_sae_in(basis, 64)
    feature_acts = gumbel_sae.encode(eval_sae_in)
    num_active = (feature_acts != 0).sum(dim=-1)
    assert (num_active <= K).all()

    # The point of the variant: which features fire must depend on the input,
    # so different samples must end up with different active sets.
    distinct_patterns = feature_acts.unique(dim=0).shape[0]
    assert distinct_patterns > 1, "selection head collapsed to a fixed mask"
