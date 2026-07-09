"""Horse master ORM model."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Horse(Base):
    __tablename__ = "horses"

    # horse_id is TEXT; may contain leading zeros in netkeiba IDs.
    horse_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    sex: Mapped[str | None] = mapped_column(String)
    birth_date: Mapped[str | None] = mapped_column(String)
    # sire/dam は系統特徴量 (compute_pedigree_features) で
    # `WHERE sire == ?` / `WHERE dam == ?` 検索されるため index 必須。
    # 無いと entries 全件 (~50万行) スキャンになり 1 レース推論で十数秒かかる。
    sire: Mapped[str | None] = mapped_column(String, index=True)  # 父馬名
    dam: Mapped[str | None] = mapped_column(String, index=True)   # 母馬名
