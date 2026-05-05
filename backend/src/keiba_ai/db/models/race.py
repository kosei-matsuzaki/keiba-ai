"""Race ORM model — one row per race event."""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from keiba_ai.db.base import Base


class Race(Base):
    __tablename__ = "races"

    # race_id is TEXT (structured identifier like "202406010101"); not an integer.
    race_id: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, nullable=False)
    course: Mapped[str] = mapped_column(String, nullable=False)
    surface: Mapped[str] = mapped_column(String, nullable=False)
    distance: Mapped[int] = mapped_column(Integer, nullable=False)
    weather: Mapped[str | None] = mapped_column(String)
    track_condition: Mapped[str | None] = mapped_column(String)
    race_class: Mapped[str | None] = mapped_column(String)
    n_runners: Mapped[int | None] = mapped_column(Integer)
    payout_win: Mapped[int | None] = mapped_column(Integer)
    payout_place: Mapped[str | None] = mapped_column(String)  # JSON string
    name: Mapped[str | None] = mapped_column(String, nullable=True)
