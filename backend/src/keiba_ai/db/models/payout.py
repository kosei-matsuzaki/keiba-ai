"""Payout ORM model —払戻詳細（bet_type ごと）。

race_id → races: CASCADE  (レース削除時に払戻記録も連動削除)
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from keiba_ai.db.base import Base


class Payout(Base):
    __tablename__ = "payouts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    bet_type: Mapped[str] = mapped_column(String, nullable=False)  # '単勝' | '複勝' | '馬連' etc.
    combo: Mapped[str] = mapped_column(String, nullable=False)     # e.g. "3" or "3-7"
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    popularity: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_payouts_race_id_bet_type", "race_id", "bet_type"),
    )
