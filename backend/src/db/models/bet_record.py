"""BetRecord ORM model — ユーザが実際に購入した（または推奨どおり購入したと仮定した）ベット記録。

race_id → races: RESTRICT  (レース削除前に bet_records を先に消す必要あり)
recommendation_id は将来の recommendations テーブル追加用の整数列であり、
現時点では外部キー制約なし。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.race import Race


class BetRecord(Base):
    __tablename__ = "bet_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)  # ISO 8601
    race_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("races.race_id", ondelete="RESTRICT"),
        nullable=False,
    )
    bet_type: Mapped[str] = mapped_column(String, nullable=False)  # validation is done by Pydantic
    combo: Mapped[str] = mapped_column(String, nullable=False)
    stake: Mapped[int] = mapped_column(Integer, nullable=False)             # 円
    source: Mapped[str] = mapped_column(String, nullable=False)             # 'recommendation' | 'manual'
    recommendation_id: Mapped[int | None] = mapped_column(Integer)          # 将来の FK 用; 現状は制約なし
    settled_at: Mapped[str | None] = mapped_column(String)                  # ISO 8601; NULL = 未確定
    payout: Mapped[int | None] = mapped_column(Integer)                     # 払戻金 (円); 外れ = 0
    profit: Mapped[int | None] = mapped_column(Integer)                     # payout - stake
    notes: Mapped[str | None] = mapped_column(String)

    __table_args__ = (
        Index("ix_bet_records_race_id", "race_id"),
        Index("ix_bet_records_created_at", "created_at"),
        Index("ix_bet_records_settled_at", "settled_at"),
    )

    # SQLAlchemy unit-of-work が parent-first insert を解決できるよう scalar
    # relationship を 1 本だけ張る (back_populates 等は意図的に省略)。
    race: Mapped[Race] = relationship("Race", foreign_keys=[race_id])
