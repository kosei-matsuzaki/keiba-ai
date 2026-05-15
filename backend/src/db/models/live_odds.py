"""LiveOdds ORM model — 当日リアルタイム連系オッズ。

fetch_live_odds ジョブが race.netkeiba.com から取得したオッズを保存する。
payouts テーブルはレース確定後の払戻記録であり、live_odds は当日オッズの記録。
両者は別テーブルで役割が異なる。

race_id → races: CASCADE  (レース削除時に live_odds も連動削除)
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class LiveOdds(Base):
    __tablename__ = "live_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    # 単勝 | 複勝 | 枠連 | 馬連 | ワイド | 馬単 | 三連複 | 三連単
    bet_type: Mapped[str] = mapped_column(String, nullable=False)
    # payouts.combo と同じ形式 ("3", "3-7" 昇順, "3→5" 順序つき等)
    combo: Mapped[str] = mapped_column(String, nullable=False)
    # min/max あり券種は最小オッズ、それ以外は確定オッズ
    odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 複勝・ワイドの最大オッズ（それ以外の券種は NULL）
    odds_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fetched_at: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index("ix_live_odds_race_id", "race_id"),
        Index("ix_live_odds_race_id_bet_type", "race_id", "bet_type"),
        UniqueConstraint("race_id", "bet_type", "combo", name="uq_live_odds_race_id_bet_type_combo"),
    )
