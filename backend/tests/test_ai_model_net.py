"""Tests for ai.model.net (HorseEncoderWithEmb, RaceTransformerModel)."""

from __future__ import annotations

import pytest
import torch

from ai.model.net import HorseEncoderWithEmb, RaceTransformerModel

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


HORSE_CAT_POSITIONS = [0, 1]     # 2 categorical horse cols
HORSE_CAT_CARDINALITIES = [3, 7]
RACE_CAT_POSITIONS = [2]         # 1 categorical race col
RACE_CAT_CARDINALITIES = [4]
CAT_EMBED_DIM = 4


def _make_features(
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
        horse, race = _make_features()
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
        horse, race = _make_features()
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
        horse, race = _make_features()
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
        horse, race = _make_features()
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
        horse, race = _make_features()
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


HISTORY_FEAT_DIM = 6
HISTORY_LEN = 4


def _history_inputs(lengths_pattern: list[list[int]] | None = None):
    """history_seq [B,N,L,Hf] + history_lengths [B,N]."""
    torch.manual_seed(11)
    seq = torch.randn(BATCH, MAX_HORSES, HISTORY_LEN, HISTORY_FEAT_DIM)
    if lengths_pattern is None:
        lengths = torch.full((BATCH, MAX_HORSES), HISTORY_LEN, dtype=torch.long)
    else:
        lengths = torch.tensor(lengths_pattern, dtype=torch.long)
    return seq, lengths


class TestRaceTransformerHistory:
    def _model(self, with_history: bool) -> RaceTransformerModel:
        # history_feat_dim>0 で履歴 GRU を構築、0 で履歴入力なし。
        return RaceTransformerModel(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_heads=N_HEADS,
            history_feat_dim=HISTORY_FEAT_DIM if with_history else 0,
            history_hidden=16,
        )

    def test_history_output_shape(self, horse_features, race_features, mask):
        model = self._model(with_history=True)
        seq, lengths = _history_inputs()
        scores = model(horse_features, race_features, mask, history_seq=seq, history_lengths=lengths)
        assert scores.shape == (BATCH, MAX_HORSES)
        # valid positions finite
        assert torch.isfinite(scores[mask]).all()

    def test_history_seq_omitted_uses_zero_embed(self, horse_features, race_features, mask):
        """履歴 GRU 付きでも history_seq を渡さなければ zero 埋めで動く (推論の graceful path)。"""
        model = self._model(with_history=True)
        assert model.history_encoder is not None
        scores = model(horse_features, race_features, mask)
        assert scores.shape == (BATCH, MAX_HORSES)
        assert torch.isfinite(scores[mask]).all()

    def test_no_history_has_no_gru(self, horse_features, race_features, mask):
        """history_feat_dim=0 のモデルは GRU を持たず、encoder の入力次元も増えない。"""
        model = self._model(with_history=False)
        assert model.history_encoder is None
        assert model.horse_encoder.history_embed_dim == 0
        scores = model(horse_features, race_features, mask)
        assert scores.shape == (BATCH, MAX_HORSES)
        assert torch.isfinite(scores[mask]).all()

    def test_zero_length_horse_does_not_crash(self, horse_features, race_features, mask):
        """過去走 0 件の馬 (length=0) を含んでも落ちず有限スコアを返す。"""
        model = self._model(with_history=True)
        seq, lengths = _history_inputs()
        lengths[0, 0] = 0  # 1 頭目を履歴なしに
        lengths[1, 5] = 0
        scores = model(horse_features, race_features, mask, history_seq=seq, history_lengths=lengths)
        assert torch.isfinite(scores[mask]).all()

    def test_history_grads_flow(self, horse_features, race_features, mask):
        model = self._model(with_history=True)
        seq, lengths = _history_inputs()
        scores = model(horse_features, race_features, mask, history_seq=seq, history_lengths=lengths)
        scores[mask].sum().backward()
        grads = [p.grad for p in model.history_encoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


ODDS_FEAT_DIM = 2


def _odds_features() -> torch.Tensor:
    torch.manual_seed(13)
    return torch.randn(BATCH, MAX_HORSES, ODDS_FEAT_DIM)


class TestRaceTransformerOddsHead:
    def _model(self, with_odds: bool) -> RaceTransformerModel:
        # odds_feat_dim>0 で head に odds を concat、0 で ability-only head。
        return RaceTransformerModel(
            horse_feat_dim=HORSE_FEAT_DIM,
            race_feat_dim=RACE_FEAT_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_heads=N_HEADS,
            odds_feat_dim=ODDS_FEAT_DIM if with_odds else 0,
        )

    def test_odds_head_output_shape(self, horse_features, race_features, mask):
        model = self._model(with_odds=True)
        odds = _odds_features()
        scores = model(horse_features, race_features, mask, odds_features=odds)
        assert scores.shape == (BATCH, MAX_HORSES)
        assert torch.isfinite(scores[mask]).all()

    def test_odds_head_missing_odds_fallback(self, horse_features, race_features, mask):
        """odds_features=None でも zeros fallback で finite を返す (欠損オッズ耐性)。"""
        model = self._model(with_odds=True)
        scores = model(horse_features, race_features, mask, odds_features=None)
        assert torch.isfinite(scores[mask]).all()

    def test_odds_head_dim_in_mlp(self):
        """head_mlp の入力次元 = embed + odds_feat_dim (odds を head で concat)。"""
        on = self._model(with_odds=True)
        assert on.odds_feat_dim == ODDS_FEAT_DIM
        assert on.head_mlp[0].in_features == EMBED_DIM + ODDS_FEAT_DIM

    def test_ability_only_head_dim(self, horse_features, race_features, mask):
        """odds_feat_dim=0 (exclude-odds) は head が ability のみを入力に取る。"""
        off = self._model(with_odds=False)
        assert off.odds_feat_dim == 0
        assert off.head_mlp[0].in_features == EMBED_DIM
        scores = off(horse_features, race_features, mask)
        assert torch.isfinite(scores[mask]).all()
        # 同構成同士は state_dict strict load 可
        off2 = self._model(with_odds=False)
        off2.load_state_dict(off.state_dict(), strict=True)

    def test_odds_head_grads_flow(self, horse_features, race_features, mask):
        model = self._model(with_odds=True)
        odds = _odds_features()
        scores = model(horse_features, race_features, mask, odds_features=odds)
        scores[mask].sum().backward()
        grads = [p.grad for p in model.head_mlp.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
