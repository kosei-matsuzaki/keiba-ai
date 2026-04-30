"""Pedigree features: sire/dam progeny win rates.

All aggregations are strictly before before_date to prevent target leakage.
Sire/dam strings are matched exactly as stored in the horses table (including
long foreign-horse names like "フォーウィールドライブFour Wheel Drive(米)").
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race


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
