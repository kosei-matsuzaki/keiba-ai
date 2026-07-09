"""Neural network model: per-horse Encoder + race-level Set Transformer.

The production architecture is :class:`HorseEncoderWithEmb` + :class:`RaceTransformerModel`:

* The per-horse encoder assesses **ability** — aggregate + per-race history
  features fed through ``nn.Embedding`` (categoricals) + an MLP.  Odds are
  deliberately *not* part of the ability encoder.
* Cross-horse interaction is modelled by a stacked ``nn.TransformerEncoder``
  (GELU + pre-norm + FFN).
* The scoring head combines the ability representation with standardised
  **odds** (market value) — ``head_norm(ability) ⊕ odds → head_mlp``.

``history_feat_dim`` / ``odds_feat_dim`` are *dimensions*: 0 means that input
is absent (e.g. ``odds_feat_dim=0`` under ``KEIBA_EXCLUDE_ODDS_FEATURES``),
otherwise the corresponding sub-module is built.  Production training always
supplies both.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        history_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.history_embed_dim = history_embed_dim
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
            + history_embed_dim  # 履歴エンコーダ出力 (0 = 履歴なし → 現行と同一)
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
        history_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode horses.

        Args:
            horse_features: [B, N, horse_feat_dim]
            race_features:  [B, race_feat_dim]
            history_embed:  optional [B, N, history_embed_dim] — 履歴エンコーダ出力。
                            None なら現行 (集約のみ) と完全に同一。

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

        horse_parts = [horse_num, *horse_cat_parts]
        if history_embed is not None:
            horse_parts.append(history_embed)
        horse_concat = torch.cat(horse_parts, dim=-1)
        race_concat = torch.cat([race_num, *race_cat_parts], dim=-1)

        race_expanded = race_concat.unsqueeze(1).expand(-1, n, -1)
        combined = torch.cat([horse_concat, race_expanded], dim=-1)
        return self.net(combined)


class RaceTransformerModel(nn.Module):
    """Set Transformer with categorical embeddings + stacked encoder blocks.

    * Categorical inputs feed ``nn.Embedding`` tables (see
      :class:`HorseEncoderWithEmb`).
    * The ability encoder optionally consumes a per-race history sequence
      (``history_feat_dim > 0``) via a GRU.
    * Cross-horse interaction is modelled by a multi-layer
      ``nn.TransformerEncoder`` with GELU + pre-norm + FFN.
    * The scoring head combines the ability representation with standardised
      odds (``odds_feat_dim > 0``) — ability→value separation.
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
        history_feat_dim: int = 0,
        history_hidden: int = 64,
        odds_feat_dim: int = 0,
    ) -> None:
        super().__init__()

        # 履歴エンコーダ (per-past-race 系列 → 馬ごとの 1 ベクトル)。
        # history_feat_dim=0 のとき履歴入力なし (GRU を作らない)。本番は常に >0。
        self.history_hidden = history_hidden
        if history_feat_dim > 0:
            self.history_encoder = nn.GRU(
                input_size=history_feat_dim,
                hidden_size=history_hidden,
                batch_first=True,
            )
            history_embed_dim = history_hidden
        else:
            self.history_encoder = None
            history_embed_dim = 0

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
            history_embed_dim=history_embed_dim,
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
        # norm_first=True のため nested tensor 高速パスは使えない。
        # enable_nested_tensor を明示的に False にして無害な UserWarning を抑止。
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False,
        )

        # スコアリング head: ability(transformer 出力) を LayerNorm したものに
        # 標準化済み odds(市場価値) を concat して MLP に通す (ability→value 分離)。
        # odds_feat_dim=0 (exclude-odds) のとき odds concat なし = ability-only。
        self.odds_feat_dim = max(odds_feat_dim, 0)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(embed_dim + self.odds_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_history(
        self, history_seq: torch.Tensor, history_lengths: torch.Tensor
    ) -> torch.Tensor:
        """[B, N, L, Hf] 系列 → [B, N, history_hidden]。過去走 0 件の馬は zero。"""
        b, n, ll, hf = history_seq.shape
        flat = history_seq.reshape(b * n, ll, hf)
        lengths = history_lengths.reshape(b * n)
        clamped = lengths.clamp(min=1)  # pack は length>=1 を要求
        packed = nn.utils.rnn.pack_padded_sequence(
            flat, clamped.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hn = self.history_encoder(packed)  # [1, B*N, hidden]
        emb = hn.squeeze(0)  # [B*N, hidden]
        # 過去走 0 件 (length==0) の馬は履歴寄与なし → zero ベクトルに
        emb = emb * (lengths > 0).unsqueeze(-1).to(emb.dtype)
        return emb.reshape(b, n, -1)

    def forward(
        self,
        horse_features: torch.Tensor,
        race_features: torch.Tensor,
        mask: torch.Tensor,
        history_seq: torch.Tensor | None = None,
        history_lengths: torch.Tensor | None = None,
        odds_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-horse scores.

        Args:
            horse_features: [B, N, horse_feat_dim]
            race_features:  [B, race_feat_dim]
            mask:           [B, N] bool — True = valid horse, False = padded
            history_seq:    optional [B, N, L, history_feat_dim] — 過去走系列
            history_lengths: optional [B, N] — 各馬の実履歴長 (0 可)
            odds_features:  optional [B, N, odds_feat_dim] — 標準化済み odds。
                            head で ability とは分離して使う。None なら zero 埋め。

        Returns:
            scores: [B, N] (padded positions = -inf)
        """
        history_embed = None
        if self.history_encoder is not None:
            if history_seq is not None and history_lengths is not None:
                history_embed = self._encode_history(history_seq, history_lengths)
            else:
                # 履歴系列が渡されない場合は zero ベクトル (encoder の入力次元を保つ)。
                b, n, _ = horse_features.shape
                history_embed = horse_features.new_zeros(b, n, self.history_hidden)
        encoded = self.horse_encoder(horse_features, race_features, history_embed=history_embed)
        key_padding_mask = ~mask
        attended = self.transformer(encoded, src_key_padding_mask=key_padding_mask)

        normed = self.head_norm(attended)  # [B, N, embed]
        if self.odds_feat_dim > 0:
            if odds_features is None:
                # 欠損オッズ: ability-only で評価 (zeros = 標準化平均)
                b, n, _ = normed.shape
                odds_features = normed.new_zeros(b, n, self.odds_feat_dim)
            normed = torch.cat([normed, odds_features], dim=-1)  # [B, N, embed+odds]
        scores = self.head_mlp(normed).squeeze(-1)

        scores = scores.masked_fill(~mask, float("-inf"))
        return scores
