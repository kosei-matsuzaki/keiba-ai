"""Race endpoints: upcoming list and race detail."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_session
from keiba_ai.api.schemas import EntrySummary, RaceDetail, RaceSummary, UpcomingRacesResponse
from keiba_ai.db.models.entry import Entry
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
    )


def _entry_summary(entry: Entry) -> EntrySummary:
    return EntrySummary(
        horse_id=entry.horse_id,
        post_position=entry.post_position,
        jockey_id=entry.jockey_id,
        trainer_id=entry.trainer_id,
        age=entry.age,
        sex=entry.sex,
        odds_win=entry.odds_win,
        popularity=entry.popularity,
        finish_position=entry.finish_position,
    )


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
    races = list(session.scalars(stmt).all())
    return UpcomingRacesResponse(races=[_race_summary(r) for r in races])


@router.get("/races/{race_id}", response_model=RaceDetail)
def get_race_detail(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> RaceDetail:
    race = session.get(Race, race_id)
    if race is None:
        raise HTTPException(status_code=404, detail=f"Race {race_id!r} not found")

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
        weather=race.weather,
        track_condition=race.track_condition,
        entries=[_entry_summary(e) for e in entries],
        payout_win=race.payout_win,
        payout_place=race.payout_place,
    )
