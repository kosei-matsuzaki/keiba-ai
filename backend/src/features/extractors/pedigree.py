"""Pedigree features: sire/dam progeny win rates.

All aggregations are strictly before before_date to prevent target leakage.
Sire/dam strings are matched exactly as stored in the horses table (including
long foreign-horse names like "フォーウィールドライブFour Wheel Drive(米)").

Two implementations (mirror of horse_history.py / jockey.py / trainer.py):
  compute_pedigree_features                — per-call SQL (1-2 queries)
  build_pedigree_cache +
    compute_pedigree_features_from_cache   — preload everything once, then
                                             pure pandas slicing
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race


def _progeny_win_rate(
    session: Session,
    field: str,
    name: str,
    before_date_str: str,
) -> float | None:
    """Compute win rate for progeny of a single sire or dam.

    Args:
        field: "sire" or "dam" (attribute name on Horse model)
        name: the sire/dam name to filter on
        before_date_str: ISO date string; only races strictly before this date count

    Returns:
        float win rate [0.0, 1.0], or None if no qualifying starts exist
    """
    horse_attr = getattr(Horse, field)

    base = (
        select(func.count())
        .select_from(Entry)
        .join(Horse, Horse.horse_id == Entry.horse_id)
        .join(Race, Race.race_id == Entry.race_id)
        .where(horse_attr == name)
        .where(Race.date < before_date_str)
    )
    n_total = session.execute(base).scalar() or 0

    if n_total == 0:
        return None

    wins_stmt = base.where(Entry.finish_position == 1)
    n_wins = session.execute(wins_stmt).scalar() or 0

    return n_wins / n_total


def compute_pedigree_features(
    session: Session,
    sire: str | None,
    dam: str | None,
    before_date: object,  # date | str accepted
) -> dict[str, float | None]:
    """Compute sire/dam progeny win rates using only races before before_date.

    Args:
        session: SQLAlchemy session
        sire: sire name as stored in horses.sire (None → skip)
        dam: dam name as stored in horses.dam (None → skip)
        before_date: date object or ISO date string

    Returns:
        {
            'sire_progeny_win_rate': float | None,
            'dam_progeny_win_rate': float | None,
        }
    """
    before_str = before_date.isoformat() if hasattr(before_date, "isoformat") else str(before_date)

    return {
        "sire_progeny_win_rate": (
            _progeny_win_rate(session, "sire", sire, before_str) if sire else None
        ),
        "dam_progeny_win_rate": (
            _progeny_win_rate(session, "dam", dam, before_str) if dam else None
        ),
    }


# ---------------------------------------------------------------------------
# Bulk-preload variant
# ---------------------------------------------------------------------------


@dataclass
class PedigreeCache:
    """Pre-loaded pedigree data for fast feature lookup.

    Two indexed tables:
      by_sire: sire(str) -> DataFrame of (date, finish_position) for all
               progeny race-rows whose sire matches.
      by_dam:  same, keyed by dam name.

    Plus horse_to_sire_dam: horse_id -> (sire, dam). This eliminates the
    per-entry session.get(Horse, ...) call in builder.py.
    """

    by_sire: dict[str, pd.DataFrame]
    by_dam: dict[str, pd.DataFrame]
    horse_to_sire_dam: dict[str, tuple[str | None, str | None]]


def build_pedigree_cache(session: Session) -> PedigreeCache:
    """Load all (horse, sire, dam) + their race history in 2 SQL passes.

    Pass 1: horses table → horse_id → (sire, dam).
    Pass 2: entries × horses × races → DataFrame with sire / dam / date /
            finish_position columns. Grouped by sire and dam separately.

    NULL sire/dam rows are skipped from grouping (cannot lookup anyway).
    """
    # Pass 1
    horse_rows = session.execute(
        select(Horse.horse_id, Horse.sire, Horse.dam)
    ).all()
    horse_to_sire_dam: dict[str, tuple[str | None, str | None]] = {
        r.horse_id: (r.sire, r.dam) for r in horse_rows
    }

    # Pass 2
    query = (
        select(
            Horse.sire,
            Horse.dam,
            Race.date,
            Entry.finish_position,
        )
        .select_from(Entry)
        .join(Horse, Horse.horse_id == Entry.horse_id)
        .join(Race, Race.race_id == Entry.race_id)
    )
    rows = session.execute(query).all()
    df = pd.DataFrame(
        rows,
        columns=["sire", "dam", "date", "finish_position"],
    )

    by_sire: dict[str, pd.DataFrame] = {}
    if not df.empty:
        sire_rows = df[df["sire"].notna()]
        by_sire = {s: g for s, g in sire_rows.groupby("sire", sort=False)}

    by_dam: dict[str, pd.DataFrame] = {}
    if not df.empty:
        dam_rows = df[df["dam"].notna()]
        by_dam = {d: g for d, g in dam_rows.groupby("dam", sort=False)}

    return PedigreeCache(
        by_sire=by_sire,
        by_dam=by_dam,
        horse_to_sire_dam=horse_to_sire_dam,
    )


def _rate_from_cache(
    by_parent: dict[str, pd.DataFrame],
    name: str | None,
    before_str: str,
) -> float | None:
    """Compute progeny win rate from pre-grouped frame, semantics mirroring _progeny_win_rate."""
    if not name:
        return None
    df = by_parent.get(name)
    if df is None or df.empty:
        return None
    rows = df[df["date"] < before_str]
    n_total = len(rows)
    if n_total == 0:
        return None
    n_wins = int((rows["finish_position"] == 1).sum())
    return n_wins / n_total


def compute_pedigree_features_from_cache(
    cache: PedigreeCache,
    sire: str | None,
    dam: str | None,
    before_date: object,
) -> dict[str, float | None]:
    """Cached counterpart of compute_pedigree_features; bit-for-bit identical output."""
    before_str = (
        before_date.isoformat() if hasattr(before_date, "isoformat") else str(before_date)
    )
    return {
        "sire_progeny_win_rate": _rate_from_cache(cache.by_sire, sire, before_str),
        "dam_progeny_win_rate": _rate_from_cache(cache.by_dam, dam, before_str),
    }
