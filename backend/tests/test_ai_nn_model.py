"""Tests for ai.nn.model (HorseEncoder, RaceModel, RaceTransformerModel)."""

from __future__ import annotations

import pytest
import torch

from ai.nn.model import HorseEncoder, HorseEncoderWithEmb, RaceModel, RaceTransformerModel

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


# ---------------------------------------------------------------------------
# Arch v2: HorseEncoderWithEmb / RaceTransformerModel
# ---------------------------------------------------------------------------


HORSE_CAT_POSITIONS = [0, 1]     # 2 categorical horse cols
HORSE_CAT_CARDINALITIES = [3, 7]
RACE_CAT_POSITIONS = [2]         # 1 categorical race col
RACE_CAT_CARDINALITIES = [4]
CAT_EMBED_DIM = 4


def _make_v2_features(
    batch: int = BATCH,
    n_horses: int = MAX_HORSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (horse, race) feature tensors with sensible category indices.

    Categorical positions hold values in {-1, 0, ..., cardinality-1}; the
    rest are arbitrary floats.
    """
    torch.manual_seed(7)
    horse = torch.randn(batch, n_horses, HORSE_FEAT_DIM)
    race = torch.randn(batch, RACE_FEAT_DIM)
    # Plant categorical indices
    horse[..., HORSE_CAT_POSITIONS[0]] = torch.randint(0, HORSE_CAT_CARDINALITIES[0], (batch, n_horses)).float()
    horse[..., HORSE_CAT_POSITIONS[1]] = torch.randint(-1, HORSE_CAT_CARDINALITIES[1], (batch, n_horses)).float()
    race[..., RACE_CAT_POSITIONS[0]] = torch.randint(0, RACE_CAT_CARDINALITIES[0], (batch,)).float()
    return horse, race


class TestHorseEncoderWithEmb:
    def test_output_shape(self):
        horse, race = _make_v2_features()
        encoder = HorseEncoderWithEmb(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            horse_cat_positions=HORSE_CAT_POSITIONS,
            horse_cat_cardinalities=HORSE_CAT_CARDINALITIES,
            race_cat_positions=RACE_CAT_POSITIONS,
            race_cat_cardinalities=RACE_CAT_CARDINALITIES,
            cat_embed_dim=CAT_EMBED_DIM,
        )
        out = encoder(horse, race)
        assert out.shape == (BATCH, MAX_HORSES, EMBED_DIM)

    def test_no_categoricals_still_works(self, horse_features, race_features):
        """Empty cat positions == fully continuous input — should still produce embeddings."""
        encoder = HorseEncoderWithEmb(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        out = encoder(horse_features, race_features)
        assert out.shape == (BATCH, MAX_HORSES, EMBED_DIM)

    def test_unknown_category_does_not_crash(self):
        """Value -1 (preprocessor's unknown sentinel) must map to index 0 cleanly."""
        horse, race = _make_v2_features()
        horse[..., HORSE_CAT_POSITIONS[0]] = -1.0  # all unknown
        race[..., RACE_CAT_POSITIONS[0]] = -1.0
        encoder = HorseEncoderWithEmb(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            horse_cat_positions=HORSE_CAT_POSITIONS,
            horse_cat_cardinalities=HORSE_CAT_CARDINALITIES,
            race_cat_positions=RACE_CAT_POSITIONS,
            race_cat_cardinalities=RACE_CAT_CARDINALITIES,
            cat_embed_dim=CAT_EMBED_DIM,
        )
        out = encoder(horse, race)
        assert torch.all(torch.isfinite(out))

    def test_grads_flow_through_embeddings(self):
        horse, race = _make_v2_features()
        encoder = HorseEncoderWithEmb(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            horse_cat_positions=HORSE_CAT_POSITIONS,
            horse_cat_cardinalities=HORSE_CAT_CARDINALITIES,
            race_cat_positions=RACE_CAT_POSITIONS,
            race_cat_cardinalities=RACE_CAT_CARDINALITIES,
            cat_embed_dim=CAT_EMBED_DIM,
        )
        out = encoder(horse, race)
        out.sum().backward()
        for emb in encoder.horse_cat_embeddings:
            assert emb.weight.grad is not None


class TestRaceTransformerModel:
    def test_output_shape(self, mask):
        horse, race = _make_v2_features()
        model = RaceTransformerModel(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_heads=N_HEADS,
            horse_cat_positions=HORSE_CAT_POSITIONS,
            horse_cat_cardinalities=HORSE_CAT_CARDINALITIES,
            race_cat_positions=RACE_CAT_POSITIONS,
            race_cat_cardinalities=RACE_CAT_CARDINALITIES,
            cat_embed_dim=CAT_EMBED_DIM,
            n_transformer_layers=2,
        )
        scores = model(horse, race, mask)
        assert scores.shape == (BATCH, MAX_HORSES)

    def test_padded_positions_are_neg_inf(self, mask):
        horse, race = _make_v2_features()
        model = RaceTransformerModel(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_heads=N_HEADS,
            horse_cat_positions=HORSE_CAT_POSITIONS,
            horse_cat_cardinalities=HORSE_CAT_CARDINALITIES,
            race_cat_positions=RACE_CAT_POSITIONS,
            race_cat_cardinalities=RACE_CAT_CARDINALITIES,
            n_transformer_layers=2,
        )
        scores = model(horse, race, mask)
        assert torch.all(scores[1, 6:] == float("-inf"))
        assert torch.all(torch.isfinite(scores[0]))
        assert torch.all(torch.isfinite(scores[1, :6]))

    def test_n_transformer_layers_increases_param_count(self, mask):
        kwargs = dict(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_heads=N_HEADS,
        )
        m1 = RaceTransformerModel(**kwargs, n_transformer_layers=1)
        m2 = RaceTransformerModel(**kwargs, n_transformer_layers=3)
        n1 = sum(p.numel() for p in m1.parameters())
        n2 = sum(p.numel() for p in m2.parameters())
        assert n2 > n1
