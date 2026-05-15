"""Neural network model: per-horse Encoder + race-level Set Transformer.

HorseEncoder: MLP that maps (horse_features, race_features) -> embed
RaceModel: HorseEncoder + multi-head self-attention head -> scores per horse
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
