"""ModelRun ORM model — 学習履歴 (PyTorch NN)。

is_active フラグは spec.md で定義されているが、推論エンドポイントは M5 以降のため
カラムのみ定義し、M4/M5 で活用する。
"""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)    # ISO 8601
    model_path: Mapped[str] = mapped_column(String, nullable=False)    # data/models/<ts>/
    params_json: Mapped[str | None] = mapped_column(String)            # model params JSON
    train_range: Mapped[str | None] = mapped_column(String)            # e.g. "2022-01-01/2024-01-01"
    valid_range: Mapped[str | None] = mapped_column(String)
    metrics_json: Mapped[str | None] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(String)
    is_active: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )  # 0/1 flag — model と migration の default を揃える
    model_type: Mapped[str] = mapped_column(
        String, nullable=False, default="nn", server_default="nn"
    )  # 常に "nn"（NN 専用化済み。列は履歴互換のため残置）
