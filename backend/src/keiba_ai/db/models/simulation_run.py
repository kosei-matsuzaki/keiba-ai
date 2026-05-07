"""SimulationRun ORM model — シミュレーション実行履歴。

各 Ledger 「シミュレーション」 タブの実行ごとに 1 行 insert される。
新規 insert 時は古いものから順に削除し、合計 50 件以下に保つ
(persistence helper 側で実装)。
"""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from keiba_ai.db.base import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)    # ISO 8601 UTC

    # Input parameters
    budget: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)      # conservative|balanced|aggressive
    window_start: Mapped[str | None] = mapped_column(String)           # YYYY-MM-DD
    window_end: Mapped[str | None] = mapped_column(String)
    model_path: Mapped[str] = mapped_column(String, nullable=False)

    # Top-level result fields (FK 不要 / 検索用)
    n_races: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_settled_races: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    final_bankroll: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    peak_bankroll: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Heavy result payload (json text)
    summary_json: Mapped[str] = mapped_column(String, nullable=False)
    by_bet_type_json: Mapped[str] = mapped_column(String, nullable=False)
    by_race_class_json: Mapped[str] = mapped_column(String, nullable=False)
    by_course_json: Mapped[str] = mapped_column(String, nullable=False)
    bankroll_timeseries_json: Mapped[str] = mapped_column(String, nullable=False)
