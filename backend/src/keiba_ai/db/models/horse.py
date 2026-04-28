"""Horse master ORM model."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from keiba_ai.db.base import Base


class Horse(Base):
    __tablename__ = "horses"

    # horse_id is TEXT; may contain leading zeros in netkeiba IDs.
    horse_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    sex: Mapped[str | None] = mapped_column(String)
    birth_date: Mapped[str | None] = mapped_column(String)
    sire: Mapped[str | None] = mapped_column(String)  # 父馬名
    dam: Mapped[str | None] = mapped_column(String)   # 母馬名
