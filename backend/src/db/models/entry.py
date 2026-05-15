"""Entry ORM model — one row per horse per race.

FK CASCADE policy (明示):
  - race_id → races: CASCADE  (レース削除時に出走記録を連動削除)
  - horse_id → horses: RESTRICT  (馬の履歴を保持。entries を先に消さないと馬を削除できない)
  - jockey_id → jockeys: SET NULL  (騎手引退でもエントリは残す)
  - trainer_id → trainers: SET NULL  (調教師も同様)
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Entry(Base):
    __tablename__ = "entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    horse_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("horses.horse_id", ondelete="RESTRICT"),
        nullable=False,
    )
    post_position: Mapped[int | None] = mapped_column(Integer)
    jockey_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("jockeys.jockey_id", ondelete="SET NULL"),
    )
    trainer_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("trainers.trainer_id", ondelete="SET NULL"),
    )
    weight_carried: Mapped[float | None] = mapped_column(Float)
    age: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String)
    horse_weight: Mapped[int | None] = mapped_column(Integer)
    horse_weight_diff: Mapped[int | None] = mapped_column(Integer)
    odds_win: Mapped[float | None] = mapped_column(Float)
    popularity: Mapped[int | None] = mapped_column(Integer)
    finish_position: Mapped[int | None] = mapped_column(Integer)
    finish_time: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[str | None] = mapped_column(String)
    agari_3f: Mapped[float | None] = mapped_column(Float, nullable=True)
    passing: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_entries_race_id_horse_id", "race_id", "horse_id"),
        Index("ix_entries_horse_id_finish_position", "horse_id", "finish_position"),
        UniqueConstraint("race_id", "horse_id", name="uq_entries_race_id_horse_id"),
    )
