"""Baseline odds estimation from historical payout data.

These functions compute average odds from the payouts table to serve as
a proxy for current-day odds until live netkeiba combination odds scraping
is implemented (separate Issue).

The hardcoded fallback constants are placeholders only — real odds deviate
significantly from these race-by-race, so EV figures computed with fallbacks
are rough estimates at best.
"""

from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.live_odds import LiveOdds
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race

# ---------------------------------------------------------------------------
# Hardcoded fallback odds (amount in yen / 100 = odds multiplier).
# Used when the payouts table has no rows for a given bet_type.
# Replace these with live odds once scraping is available.
# ---------------------------------------------------------------------------

_FALLBACK_AMOUNTS: dict[str, int] = {
    "単勝": 1000,
    "複勝": 200,
    "枠連": 3000,
    "馬連": 5000,
    "ワイド": 1500,
    "馬単": 10000,
    "三連複": 10000,
    "三連単": 50000,
}

_ALL_BET_TYPES = list(_FALLBACK_AMOUNTS.keys())


def _amounts_to_odds(amounts: dict[str, float]) -> dict[str, float]:
    """Convert payout amounts (yen per 100-yen bet) to odds multiplier."""
    return {bet_type: amt / 100.0 for bet_type, amt in amounts.items()}


def _fallback_odds() -> dict[str, float]:
    return _amounts_to_odds({k: float(v) for k, v in _FALLBACK_AMOUNTS.items()})


def compute_baseline_odds(session: Session) -> dict[str, float]:
    """Compute average odds per bet_type from the payouts table.

    Aggregates over all races in the database. Returns a dict mapping
    bet_type (e.g. '単勝', '馬連') to average odds (amount / 100).

    If payouts table is empty or a bet_type has no rows, falls back to
    hardcoded placeholder values defined in _FALLBACK_AMOUNTS.

    NOTE: These are historical average payouts, not current-race odds.
    They will be replaced by live pre-race odds once the scraper for
    combination odds is implemented.
    """
    rows = session.execute(
        select(Payout.bet_type, func.avg(Payout.amount).label("avg_amount"))
        .group_by(Payout.bet_type)
    ).all()

    db_odds: dict[str, float] = {}
    for row in rows:
        db_odds[row.bet_type] = row.avg_amount / 100.0

    result = _fallback_odds()
    result.update(db_odds)
    return result


def compute_race_odds(
    session: Session,
    race_id: str,
) -> dict[str, dict[str, float]]:
    """live_odds テーブルから特定レースのオッズを返す。

    live_odds テーブルに指定レースのデータが存在する場合のみ値を返す。
    データが無い場合は空 dict を返す（baseline へのフォールバックは呼び出し側の責務）。

    Args:
        session: SQLAlchemy Session.
        race_id: 対象レースの race_id。

    Returns:
        {bet_type: {combo: odds}} 形式の 2 段ネスト dict。
        例: {'馬連': {'3-7': 25.4, '3-9': 18.2}, ...}
        複勝/ワイドは最小オッズ (odds) を使用し、odds_max は含めない（EV 計算用の単一値）。
        オッズ未確定 (odds=None) の combo は結果から除外する。
    """
    rows = session.execute(
        select(LiveOdds.bet_type, LiveOdds.combo, LiveOdds.odds)
        .where(LiveOdds.race_id == race_id)
        .where(LiveOdds.odds.is_not(None))
    ).all()

    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.bet_type not in result:
            result[row.bet_type] = {}
        result[row.bet_type][row.combo] = row.odds

    return result


def compute_past_race_odds(
    session: Session,
    race_id: str,
) -> dict[str, dict[str, float]]:
    """過去レースの確定オッズを返す（取れた combo のみ）。

    - 単勝: entries.odds_win から全馬の確定オッズ（締切時オッズ）
    - 複勝: races.payout_place JSON の 1〜3 着馬のみ payout/100 = 確定オッズ
    - 連系（馬連/ワイド/馬単/三連複/三連単）: payouts テーブルの amount/100 のみ（的中 combo のみ）

    取得不能な combo（複勝の 4 着以下、連系の外れ）は dict に含めない。
    呼び出し側は「ない combo は est_odds=None」で扱うこと。

    Args:
        session: SQLAlchemy Session.
        race_id: 対象レースの race_id。

    Returns:
        {bet_type: {combo: odds}} 形式の 2 段ネスト dict。
        compute_race_odds と同じ構造。
    """
    result: dict[str, dict[str, float]] = {}

    # ── 単勝: entries.odds_win から全馬 ─────────────────────────────────────
    entry_rows = session.execute(
        select(Entry.post_position, Entry.odds_win)
        .where(Entry.race_id == race_id)
        .where(Entry.odds_win.is_not(None))
        .where(Entry.post_position.is_not(None))
    ).all()

    if entry_rows:
        result["単勝"] = {
            str(row.post_position): row.odds_win
            for row in entry_rows
        }

    # ── 複勝: races.payout_place JSON の 1〜3 着馬 ─────────────────────────
    race_row = session.execute(
        select(Race.payout_place)
        .where(Race.race_id == race_id)
    ).first()

    if race_row is not None and race_row.payout_place is not None:
        try:
            # payout_place は {"1": 110, "2": 160, "3": 150} 形式の JSON 文字列
            payout_place_map: dict[str, int] = json.loads(race_row.payout_place)
        except (json.JSONDecodeError, TypeError):
            payout_place_map = {}

        if payout_place_map:
            # finish_position ごとの post_position を引く
            # payout_place_map のキーは finish_position（着順）
            finish_positions = [int(k) for k in payout_place_map if k.isdigit()]
            if finish_positions:
                place_entry_rows = session.execute(
                    select(Entry.post_position, Entry.finish_position)
                    .where(Entry.race_id == race_id)
                    .where(Entry.finish_position.in_(finish_positions))
                    .where(Entry.post_position.is_not(None))
                ).all()

                finish_to_post = {
                    row.finish_position: row.post_position
                    for row in place_entry_rows
                }

                fuku_odds: dict[str, float] = {}
                for finish_pos_str, amount in payout_place_map.items():
                    if not finish_pos_str.isdigit():
                        continue
                    finish_pos = int(finish_pos_str)
                    post_pos = finish_to_post.get(finish_pos)
                    if post_pos is not None and amount is not None:
                        fuku_odds[str(post_pos)] = amount / 100.0

                if fuku_odds:
                    result["複勝"] = fuku_odds

    # ── 連系: payouts テーブルから的中 combo のみ ─────────────────────────
    payout_rows = session.execute(
        select(Payout.bet_type, Payout.combo, Payout.amount)
        .where(Payout.race_id == race_id)
        .where(Payout.amount.is_not(None))
    ).all()

    for row in payout_rows:
        if row.bet_type in ("単勝", "複勝"):
            # 単勝・複勝は上記で別途処理済み（payouts の単勝/複勝も入れると重複するが
            # payouts テーブルの方が着順 1 位の確定値なので上書きでも OK）
            continue
        if row.bet_type not in result:
            result[row.bet_type] = {}
        result[row.bet_type][row.combo] = row.amount / 100.0

    return result


def compute_baseline_odds_by_class(
    session: Session,
    race_class: str | None = None,
    surface: str | None = None,
    distance: int | None = None,
    min_samples: int = 30,
) -> dict[str, float]:
    """Compute average odds filtered by race conditions.

    Filters payouts by race_class / surface / distance (any combination).
    Falls back to the overall average (compute_baseline_odds) if the
    filtered result has fewer than min_samples rows for a given bet_type,
    which avoids unstable estimates from tiny sub-groups.

    Args:
        session: SQLAlchemy Session.
        race_class: Filter by races.race_class if provided.
        surface: Filter by races.surface if provided.
        distance: Filter by races.distance if provided.
        min_samples: Minimum rows required to use filtered estimate.
            If fewer rows exist for a bet_type, falls back to overall average.
    """
    base_odds = compute_baseline_odds(session)

    # Build filtered query joining races
    q = (
        select(
            Payout.bet_type,
            func.avg(Payout.amount).label("avg_amount"),
            func.count(Payout.id).label("n"),
        )
        .join(Race, Race.race_id == Payout.race_id)
        .group_by(Payout.bet_type)
    )
    if race_class is not None:
        q = q.where(Race.race_class == race_class)
    if surface is not None:
        q = q.where(Race.surface == surface)
    if distance is not None:
        q = q.where(Race.distance == distance)

    rows = session.execute(q).all()

    result = dict(base_odds)  # start with overall averages as fallback
    for row in rows:
        if row.n >= min_samples:
            result[row.bet_type] = row.avg_amount / 100.0

    return result
