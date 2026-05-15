"""Trainer master ORM model."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Trainer(Base):
    __tablename__ = "trainers"

    trainer_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
