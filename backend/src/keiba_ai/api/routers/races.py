"""Race endpoints: upcoming list, recent list, by_date, and race detail."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_or_404, get_session
from keiba_ai.api.schemas import EntrySummary, RaceDetail, RaceSummary, UpcomingRacesResponse
from keiba_ai.core.dates import this_weekend_dates
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race

router = APIRouter()


def _race_summary(race: Race) -> RaceSummary:
    return RaceSummary(
        race_id=race.race_id,
        date=race.date,
        course=race.course,
        surface=race.surface,
        distance=race.distance,
        race_class=race.race_class,
        n_runners=race.n_runners,
        name=race.name,
    )


def _build_entry_summaries(entries: list[Entry], session: Session) -> list[EntrySummary]:
    """Build EntrySummary list with horse_name populated via a single bulk load."""
    horse_ids = {e.horse_id for e in entries}
    horses: dict[str, str | None] = {}
    if horse_ids:
        horse_rows = session.scalars(
            select(Horse).where(Horse.horse_id.in_(horse_ids))
        ).all()
        horses = {h.horse_id: h.name for h in horse_rows}

    return [
        EntrySummary(
            horse_id=entry.horse_id,
            horse_name=horses.get(entry.horse_id),
            post_position=entry.post_position,
            jockey_id=entry.jockey_id,
            trainer_id=entry.trainer_id,
            age=entry.age,
            sex=entry.sex,
            odds_win=entry.odds_win,
            popularity=entry.popularity,
            finish_position=entry.finish_position,
        )
        for entry in entries
    ]


@router.get("/races/this_weekend", response_model=UpcomingRacesResponse)
def get_this_weekend_races(
    session: Annotated[Session, Depends(get_session)],
) -> UpcomingRacesResponse:
    """今週末 (土・日) の JRA レース一覧を返す。

    DB に保存済みのレース（shutuba ingest 済み）を JST の今週土・日に絞って返す。
    未 ingest の場合は空リストを返す（404 ではない）。
    """
    sat, sun = this_weekend_dates()
    stmt = (
        select(Race)
        .where(Race.date.in_([sat.isoformat(), sun.isoformat()]))
        .order_by(Race.date, Race.race_id)
    )
    races = session.scalars(stmt).all()
    return UpcomingRacesResponse(races=[_race_summary(r) for r in races])


@router.get("/races/upcoming", response_model=UpcomingRacesResponse)
def get_upcoming_races(
    session: Annotated[Session, Depends(get_session)],
    days: int = 7,
) -> UpcomingRacesResponse:
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=days)).isoformat()
    stmt = (
        select(Race)
        .where(Race.date >= today, Race.date <= until)
        .order_by(Race.date)
    )
    races = session.scalars(stmt).all()
    return UpcomingRacesResponse(races=[_race_summary(r) for r in races])


@router.get("/races/recent", response_model=UpcomingRacesResponse)
def get_recent_races(
    session: Annotated[Session, Depends(get_session)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    from_: Annotated[
        str | None,
        Query(alias="from", description="Start date YYYY-MM-DD (overrides days when both from and to are given)"),
    ] = None,
    to: Annotated[
        str | None,
        Query(description="End date YYYY-MM-DD (overrides days when both from and to are given)"),
    ] = None,
) -> UpcomingRacesResponse:
    """Return past races, ordered by date desc.

    - If both `from` and `to` are provided, the result is filtered to
      `from <= date <= to` (inclusive).
    - Otherwise, falls back to `days` mode: `today - days <= date < today`.
    """
    if from_ and to:
        try:
            d_from = date.fromisoformat(from_)
            d_to = date.fromisoformat(to)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid date format (expected YYYY-MM-DD): {exc}",
            ) from exc
        if d_from > d_to:
            raise HTTPException(
                status_code=422,
                detail="`from` must be on or before `to`.",
            )
        if (d_to - d_from).days > 365:
            raise HTTPException(
                status_code=422,
                detail="Date range must not exceed 365 days.",
            )
        stmt = (
            select(Race)
            .where(Race.date >= d_from.isoformat(), Race.date <= d_to.isoformat())
            .order_by(Race.date.desc())
            .limit(limit)
        )
    else:
        today = date.today().isoformat()
        since = (date.today() - timedelta(days=days)).isoformat()
        stmt = (
            select(Race)
            .where(Race.date < today, Race.date >= since)
            .order_by(Race.date.desc())
            .limit(limit)
        )
    races = session.scalars(stmt).all()
    return UpcomingRacesResponse(races=[_race_summary(r) for r in races])


@router.get("/races/by_date", response_model=UpcomingRacesResponse)
def get_races_by_date(
    session: Annotated[Session, Depends(get_session)],
    date_: Annotated[
        str,
        Query(alias="date", description="Target date YYYY-MM-DD"),
    ],
) -> UpcomingRacesResponse:
    """Return all races on a single date, ordered by race_id ascending.

    Returns an empty list (not 404) when no races exist for the given date.
    """
    try:
        target = date.fromisoformat(date_)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format (expected YYYY-MM-DD): {exc}",
        ) from exc

    stmt = (
        select(Race)
        .where(Race.date == target.isoformat())
        .order_by(Race.race_id)
    )
    races = session.scalars(stmt).all()
    return UpcomingRacesResponse(races=[_race_summary(r) for r in races])


@router.get("/races/{race_id}", response_model=RaceDetail)
def get_race_detail(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> RaceDetail:
    race = get_or_404(session, Race, race_id, label="Race")

    entries_stmt = select(Entry).where(Entry.race_id == race_id).order_by(Entry.post_position)
    entries = list(session.scalars(entries_stmt).all())

    return RaceDetail(
        race_id=race.race_id,
        date=race.date,
        course=race.course,
        surface=race.surface,
        distance=race.distance,
        race_class=race.race_class,
        n_runners=race.n_runners,
        name=race.name,
        weather=race.weather,
        track_condition=race.track_condition,
        entries=_build_entry_summaries(entries, session),
        payout_win=race.payout_win,
        payout_place=race.payout_place,
    )
