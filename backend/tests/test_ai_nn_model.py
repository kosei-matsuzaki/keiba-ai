"""Tests for ai.nn.model (HorseEncoder, RaceModel)."""

from __future__ import annotations

import pytest
import torch

from ai.nn.model import HorseEncoder, RaceModel

BATCH = 2
MAX_HORSES = 8
HORSE_FEAT_DIM = 10
RACE_FEAT_DIM = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
N_HEADS = 4


@pytest.fixture()
def horse_features() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, MAX_HORSES, HORSE_FEAT_DIM)


@pytest.fixture()
def race_features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, RACE_FEAT_DIM)


@pytest.fixture()
def mask() -> torch.Tensor:
    """batch 0: all 8 valid, batch 1: first 6 valid."""
    m = torch.zeros(BATCH, MAX_HORSES, dtype=torch.bool)
    m[0, :] = True
    m[1, :6] = True
    return m


class TestHorseEncoder:
    def test_output_shape(self, horse_features, race_features):
        encoder = HorseEncoder(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM)
        out = encoder(horse_features, race_features)
        assert out.shape == (BATCH, MAX_HORSES, EMBED_DIM)

    def test_gradient_flows(self, horse_features, race_features):
        horse_features = horse_features.requires_grad_(True)
        encoder = HorseEncoder(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM)
        out = encoder(horse_features, race_features)
        out.sum().backward()
        assert horse_features.grad is not None

    def test_all_params_receive_grad(self, horse_features, race_features):
        encoder = HorseEncoder(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM)
        out = encoder(horse_features, race_features)
        out.sum().backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"param {name} has no gradient"


class TestRaceModel:
    def test_output_shape(self, horse_features, race_features, mask):
        model = RaceModel(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM, N_HEADS)
        scores = model(horse_features, race_features, mask)
        assert scores.shape == (BATCH, MAX_HORSES)

    def test_padded_positions_are_neg_inf(self, horse_features, race_features, mask):
        """Positions where mask=False should be -inf in scores."""
        model = RaceModel(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM, N_HEADS)
        scores = model(horse_features, race_features, mask)
        # batch 1: positions 6, 7 are padded
        assert torch.all(scores[1, 6:] == float("-inf"))

    def test_valid_positions_are_finite(self, horse_features, race_features, mask):
        model = RaceModel(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM, N_HEADS)
        scores = model(horse_features, race_features, mask)
        assert torch.all(torch.isfinite(scores[0]))
        assert torch.all(torch.isfinite(scores[1, :6]))

    def test_padded_scores_have_zero_gradient(self, horse_features, race_features, mask):
        """Backward should not propagate non-zero gradients through padded horse positions."""
        horse_features = horse_features.clone().requires_grad_(True)
        model = RaceModel(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM, N_HEADS)
        scores = model(horse_features, race_features, mask)

        # Only accumulate loss over valid (finite) scores to avoid NaN in backward
        valid_scores = scores[mask]
        valid_scores.sum().backward()

        # Gradient in padded positions (batch 1, horses 6-7) should be 0
        grad = horse_features.grad
        assert grad is not None
        assert torch.all(grad[1, 6:] == 0.0)

    def test_all_params_receive_grad(self, horse_features, race_features, mask):
        model = RaceModel(HORSE_FEAT_DIM, RACE_FEAT_DIM, EMBED_DIM, HIDDEN_DIM, N_HEADS)
        scores = model(horse_features, race_features, mask)
        valid_scores = scores[mask]
        valid_scores.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"param {name} has no gradient"
