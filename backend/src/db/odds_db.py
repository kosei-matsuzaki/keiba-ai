"""Separate SQLite store for scraped confirmed combination odds.

Why a second database file (``odds.db``) instead of new tables in keiba.db:

  - The odds dataset is large (≈260k rows over 2015→ at one row per
    race × bet_type, ~1GB compressed) and write-heavy during the multi-day
    backfill. Keeping that workload off keiba.db means a corruption there
    (the project has hit this before) costs nothing irreplaceable — odds are
    simply re-scraped from netkeiba.
  - It uses its **own declarative Base** so ``create_all`` only ever touches
    ``race_odds`` and never the main schema. Alembic is intentionally *not*
    wired up here; the table is created on demand. A future Postgres migration
    can fold this back into the main schema deliberately.

Storage layout: one row per ``(race_id, bet_type)``, the full per-combo odds
dict gzip-compressed into a BLOB. Existence of a row == that bet_type was
already fetched for that race (used for resume). Read path decompresses one
race's rows into the ``{bet_type: {combo: [...]}}`` shape that
``compute_race_odds_with_sources`` consumes.

Journal mode is **TRUNCATE, not WAL**: keiba.db's past corruption is the
classic WAL-on-DrvFS (/mnt/c) failure mode when touched from both Windows and
WSL. A rollback journal sidesteps the WAL shared-memory file entirely, which
matters most for a file that two OSes may open.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Engine,
    Integer,
    LargeBinary,
    String,
    create_engine,
    event,
    select,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from core.paths import odds_db_path


class OddsBase(DeclarativeBase):
    """Dedicated Base so create_all only manages odds.db tables."""


class RaceOdds(OddsBase):
    """Confirmed odds for one (race, bet_type), all combos in one gzip BLOB."""

    __tablename__ = "race_odds"

    race_id: Mapped[str] = mapped_column(String, primary_key=True)
    bet_type: Mapped[str] = mapped_column(String, primary_key=True)
    # netkeiba data.official_datetime — lets us tell a confirmed snapshot from
    # an intermediate one. None when the feed omitted it.
    official_datetime: Mapped[str | None] = mapped_column(String, nullable=True)
    n_combos: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    fetched_at: Mapped[str] = mapped_column(String, nullable=False)
    # 1 = 確定オッズ (status="result")、0 = 発走前ライブ (status="middle")。
    # 確定バックフィル (jobs.ingest_odds) はライブ行 (0) を resume skip せず再取得し
    # 確定値で上書きする。既存 DB の行はすべて確定 backfill 由来なので default 1。
    is_confirmed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )


def make_odds_engine(path: Path | None = None) -> Engine:
    """Engine for odds.db with a rollback journal (no WAL) for DrvFS safety."""
    target = path or odds_db_path()
    engine = create_engine(
        f"sqlite:///{target}",
        echo=False,
        future=True,
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # noqa: ANN001
        dbapi_conn.execute("PRAGMA journal_mode=TRUNCATE")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        dbapi_conn.execute("PRAGMA busy_timeout=30000")

    return engine


def init_odds_db(engine: Engine) -> None:
    """Create the race_odds table if it does not exist.

    Alembic は使わない (db/odds_db.py の方針) ため、後付けの ``is_confirmed`` 列は
    ここで冪等に ADD COLUMN する。既存行は確定 backfill 由来なので DEFAULT 1。
    """
    OddsBase.metadata.create_all(engine)

    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(race_odds)")}
        if "is_confirmed" not in cols:
            # 並行 init で別プロセスが先に追加した場合（duplicate column）は無視。
            with suppress(OperationalError):
                conn.exec_driver_sql(
                    "ALTER TABLE race_odds ADD COLUMN is_confirmed INTEGER NOT NULL DEFAULT 1"
                )


@contextmanager
def odds_session_scope(engine: Engine) -> Iterator[Session]:
    """Single-transaction Session for odds.db (commit on success, rollback on error)."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── (de)serialisation ──────────────────────────────────────────────────────

def compress_odds(combos: dict[str, list[float | int]]) -> bytes:
    """gzip(JSON) of a per-combo odds dict. Compact separators, no ASCII escaping."""
    raw = json.dumps(combos, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=6)


def decompress_odds(blob: bytes) -> dict[str, list[float | int]]:
    return json.loads(gzip.decompress(blob).decode("utf-8"))


# ── write / read helpers ────────────────────────────────────────────────────

def upsert_race_odds(
    session: Session,
    race_id: str,
    bet_type: str,
    official_datetime: str | None,
    combos: dict[str, list[float | int]],
    is_confirmed: bool = True,
) -> None:
    """Insert or replace the odds blob for one (race_id, bet_type).

    is_confirmed=False は発走前ライブ snapshot を表し、確定バックフィルが後で
    上書きできるようにする（resume skip させない）。
    """
    blob = compress_odds(combos)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    row = session.get(RaceOdds, (race_id, bet_type))
    if row is None:
        session.add(
            RaceOdds(
                race_id=race_id,
                bet_type=bet_type,
                official_datetime=official_datetime,
                n_combos=len(combos),
                data=blob,
                fetched_at=now,
                is_confirmed=1 if is_confirmed else 0,
            )
        )
    else:
        row.official_datetime = official_datetime
        row.n_combos = len(combos)
        row.data = blob
        row.fetched_at = now
        row.is_confirmed = 1 if is_confirmed else 0


def fetched_bet_types(
    session: Session, race_id: str, *, confirmed_only: bool = False
) -> set[str]:
    """bet_types already stored for a race (resume: skip these).

    confirmed_only=True なら確定行 (is_confirmed=1) のみ数える。確定バックフィルは
    これを使い、発走前ライブ行 (0) を resume skip せず確定値で上書きする。
    """
    stmt = select(RaceOdds.bet_type).where(RaceOdds.race_id == race_id)
    if confirmed_only:
        stmt = stmt.where(RaceOdds.is_confirmed == 1)
    rows = session.execute(stmt).all()
    return {r[0] for r in rows}


def load_race_odds(
    session: Session, race_id: str
) -> dict[str, dict[str, list[float | int]]]:
    """All stored bet_types for a race, decompressed into nested dicts."""
    rows = session.execute(
        select(RaceOdds.bet_type, RaceOdds.data).where(RaceOdds.race_id == race_id)
    ).all()
    return {bet_type: decompress_odds(blob) for bet_type, blob in rows}
