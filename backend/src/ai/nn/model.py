"""Neural network model: per-horse Encoder + race-level Set Transformer.

Two architectures live here:

* ``HorseEncoder`` + ``RaceModel`` (arch v1) — original single-MHA design,
  kept for backward-compatibility loading of legacy checkpoints.
* ``HorseEncoderWithEmb`` + ``RaceTransformerModel`` (arch v2) — adds
  ``nn.Embedding`` for categorical inputs and stacks
  ``nn.TransformerEncoderLayer`` blocks (GELU + pre-norm + FFN).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HorseEncoder(nn.Module):
    """MLP that encodes each horse independently using horse + race features.

    Args:
        horse_feat_dim: dimension of per-horse feature vector
        race_feat_dim: dimension of race-level feature vector (broadcast to each horse)
        embed_dim: output embedding dimension
        hidden_dim: hidden layer size
        dropout: dropout probability
    """

    def __init__(
        self,
        horse_feat_dim: int,
        race_feat_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = horse_feat_dim + race_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        horse_features: torch.Tensor,
        race_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode horses.

        Args:
            horse_features: [B, max_n_horses, horse_feat_dim]
            race_features:  [B, race_feat_dim]

        Returns:
            [B, max_n_horses, embed_dim]
        """
        # Broadcast race_features to each horse position: [B, 1, race_feat_dim]
        race_expanded = race_features.unsqueeze(1).expand(
            -1, horse_features.size(1), -1
        )
        combined = torch.cat([horse_features, race_expanded], dim=-1)
        return self.net(combined)


class RaceModel(nn.Module):
    """Set Transformer-style model: HorseEncoder + multi-head self-attention + score head.

    Produces a scalar score per horse.  Padded positions (mask=False) receive
    -inf so that downstream loss functions can safely ignore them.

    Args:
        horse_feat_dim: dimension of per-horse feature vector
        race_feat_dim: dimension of race-level feature vector
        embed_dim: embedding / attention dimension
        hidden_dim: hidden layer size in HorseEncoder and score head
        n_heads: number of attention heads
        dropout: dropout probability
    """

    def __init__(
        self,
        horse_feat_dim: int,
        race_feat_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horse_encoder = HorseEncoder(
            horse_feat_dim=horse_feat_dim,
            race_feat_dim=race_feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        horse_features: torch.Tensor,
        race_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-horse scores.

        Args:
            horse_features: [B, max_n_horses, horse_feat_dim]
            race_features:  [B, race_feat_dim]
            mask:           [B, max_n_horses] bool — True = valid horse, False = padded

        Returns:
            scores: [B, max_n_horses]  (padded positions = -inf)
        """
        encoded = self.horse_encoder(horse_features, race_features)

        # MultiheadAttention key_padding_mask: True means *ignore* that position
        key_padding_mask = ~mask  # [B, max_n_horses]

        attended, _ = self.attn(
            encoded, encoded, encoded, key_padding_mask=key_padding_mask
        )
        # Residual + LayerNorm
        residual = self.norm(encoded + attended)

        scores = self.head(residual).squeeze(-1)  # [B, max_n_horses]

        # Mask padded positions with -inf so losses can safely ignore them
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores


# ---------------------------------------------------------------------------
# Arch v2: categorical embeddings + stacked TransformerEncoder
# ---------------------------------------------------------------------------


class HorseEncoderWithEmb(nn.Module):
    """Per-horse MLP encoder with learned embeddings for categorical features.

    The input feature tensors retain the same shape as the v1 encoder
    (``horse_features: [B, N, horse_feat_dim]``, ``race_features: [B, race_feat_dim]``).
    Positions listed in ``horse_cat_positions`` / ``race_cat_positions`` are
    treated as categorical: the float value is cast to long, shifted by +1
    (so the preprocessor's -1 = unknown maps to index 0), and looked up in
    ``nn.Embedding``.  Remaining positions are passed through as continuous
    standardized values.
    """

    def __init__(
        self,
        horse_feat_dim: int,
        race_feat_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        horse_cat_positions: list[int] | None = None,
        horse_cat_cardinalities: list[int] | None = None,
        race_cat_positions: list[int] | None = None,
        race_cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        horse_cat_positions = list(horse_cat_positions or [])
        horse_cat_cardinalities = list(horse_cat_cardinalities or [])
        race_cat_positions = list(race_cat_positions or [])
        race_cat_cardinalities = list(race_cat_cardinalities or [])

        if len(horse_cat_positions) != len(horse_cat_cardinalities):
            raise ValueError("horse_cat_positions / cardinalities length mismatch")
        if len(race_cat_positions) != len(race_cat_cardinalities):
            raise ValueError("race_cat_positions / cardinalities length mismatch")

        self.horse_cat_positions = horse_cat_positions
        self.horse_cat_cardinalities = horse_cat_cardinalities
        self.race_cat_positions = race_cat_positions
        self.race_cat_cardinalities = race_cat_cardinalities
        self.cat_embed_dim = cat_embed_dim

        horse_cat_set = set(horse_cat_positions)
        race_cat_set = set(race_cat_positions)
        self.horse_num_positions = [
            i for i in range(horse_feat_dim) if i not in horse_cat_set
        ]
        self.race_num_positions = [
            i for i in range(race_feat_dim) if i not in race_cat_set
        ]

        # +1 slot reserved for "unknown" (preprocessor encodes unknown / NaN as -1)
        self.horse_cat_embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, cat_embed_dim) for c in horse_cat_cardinalities]
        )
        self.race_cat_embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, cat_embed_dim) for c in race_cat_cardinalities]
        )

        in_dim = (
            len(self.horse_num_positions)
            + len(horse_cat_cardinalities) * cat_embed_dim
            + len(self.race_num_positions)
            + len(race_cat_cardinalities) * cat_embed_dim
        )

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _lookup(
        self,
        features: torch.Tensor,
        positions: list[int],
        embeddings: nn.ModuleList,
        cardinalities: list[int],
    ) -> list[torch.Tensor]:
        """Run embedding lookups for the listed positions.

        ``features`` is either [B, N, F] (horse) or [B, F] (race).  The
        returned tensors share the leading dims of ``features``.
        """
        parts: list[torch.Tensor] = []
        for emb, pos, card in zip(embeddings, positions, cardinalities, strict=True):
            raw = features[..., pos] + 1.0  # shift -1 → 0
            idx = raw.long().clamp(min=0, max=card)
            parts.append(emb(idx))
        return parts

    def forward(
        self,
        horse_features: torch.Tensor,
        race_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode horses.

        Args:
            horse_features: [B, N, horse_feat_dim]
            race_features:  [B, race_feat_dim]

        Returns:
            [B, N, embed_dim]
        """
        b, n, _ = horse_features.shape

        if self.horse_num_positions:
            horse_num = horse_features[..., self.horse_num_positions]
        else:
            horse_num = horse_features.new_zeros(b, n, 0)

        if self.race_num_positions:
            race_num = race_features[..., self.race_num_positions]
        else:
            race_num = race_features.new_zeros(b, 0)

        horse_cat_parts = self._lookup(
            horse_features,
            self.horse_cat_positions,
            self.horse_cat_embeddings,
            self.horse_cat_cardinalities,
        )
        race_cat_parts = self._lookup(
            race_features,
            self.race_cat_positions,
            self.race_cat_embeddings,
            self.race_cat_cardinalities,
        )

        horse_concat = torch.cat([horse_num, *horse_cat_parts], dim=-1)
        race_concat = torch.cat([race_num, *race_cat_parts], dim=-1)

        race_expanded = race_concat.unsqueeze(1).expand(-1, n, -1)
        combined = torch.cat([horse_concat, race_expanded], dim=-1)
        return self.net(combined)


class RaceTransformerModel(nn.Module):
    """Set Transformer with categorical embeddings + stacked encoder blocks.

    Compared to :class:`RaceModel` (v1):

    * Categorical inputs feed ``nn.Embedding`` tables (see
      :class:`HorseEncoderWithEmb`).
    * Cross-horse interaction is modelled by a multi-layer
      ``nn.TransformerEncoder`` with GELU + pre-norm + FFN, instead of a
      single ``nn.MultiheadAttention`` layer.
    """

    def __init__(
        self,
        horse_feat_dim: int,
        race_feat_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        horse_cat_positions: list[int] | None = None,
        horse_cat_cardinalities: list[int] | None = None,
        race_cat_positions: list[int] | None = None,
        race_cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 4,
        n_transformer_layers: int = 2,
        ff_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.horse_encoder = HorseEncoderWithEmb(
            horse_feat_dim=horse_feat_dim,
            race_feat_dim=race_feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horse_cat_positions=horse_cat_positions,
            horse_cat_cardinalities=horse_cat_cardinalities,
            race_cat_positions=race_cat_positions,
            race_cat_cardinalities=race_cat_cardinalities,
            cat_embed_dim=cat_embed_dim,
        )

        ff = ff_dim if ff_dim is not None else hidden_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        horse_features: torch.Tensor,
        race_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-horse scores.

        Args:
            horse_features: [B, N, horse_feat_dim]
            race_features:  [B, race_feat_dim]
            mask:           [B, N] bool — True = valid horse, False = padded

        Returns:
            scores: [B, N] (padded positions = -inf)
        """
        encoded = self.horse_encoder(horse_features, race_features)
        key_padding_mask = ~mask
        attended = self.transformer(encoded, src_key_padding_mask=key_padding_mask)
        scores = self.head(attended).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores
