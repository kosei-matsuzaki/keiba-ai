"""Race endpoints: upcoming list, recent list, by_date, and race detail."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.deps import get_or_404, get_session
from api.schemas import (
    CalendarDay,
    CalendarResponse,
    DataCoverage,
    EntrySummary,
    PayoutEntry,
    RaceDetail,
    RaceSummary,
    UpcomingRacesResponse,
)
from core.bet_types import normalize_combo
from core.dates import this_weekend_dates
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.payout import Payout
from db.models.race import Race

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
    """Build EntrySummary list with horse_name / jockey_name populated via bulk loads."""
    horse_ids = {e.horse_id for e in entries}
    horses: dict[str, str | None] = {}
    if horse_ids:
        horse_rows = session.scalars(
            select(Horse).where(Horse.horse_id.in_(horse_ids))
        ).all()
        horses = {h.horse_id: h.name for h in horse_rows}

    jockey_ids = {e.jockey_id for e in entries if e.jockey_id is not None}
    jockeys: dict[str, str | None] = {}
    if jockey_ids:
        jockey_rows = session.scalars(
            select(Jockey).where(Jockey.jockey_id.in_(jockey_ids))
        ).all()
        jockeys = {j.jockey_id: j.name for j in jockey_rows}

    return [
        EntrySummary(
            horse_id=entry.horse_id,
            horse_name=horses.get(entry.horse_id),
            post_position=entry.post_position,
            jockey_id=entry.jockey_id,
            jockey_name=jockeys.get(entry.jockey_id) if entry.jockey_id else None,
            trainer_id=entry.trainer_id,
            age=entry.age,
            sex=entry.sex,
            horse_weight=entry.horse_weight,
            horse_weight_diff=entry.horse_weight_diff,
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


# 重賞ほど前に来るように並べる。カレンダーに 1 つだけ名前を出すときの優先度。
_CLASS_PRIORITY = ["G1", "G2", "G3", "OP", "L", "3勝", "2勝", "1勝", "新馬", "未勝利"]


def _class_rank(race_class: str | None) -> int:
    """race_class の格付け順位 (小さいほど格上)。未知/None は最下位。"""
    if not race_class:
        return len(_CLASS_PRIORITY)
    for i, key in enumerate(_CLASS_PRIORITY):
        if key in race_class:
            return i
    return len(_CLASS_PRIORITY)


@router.get("/races/calendar", response_model=CalendarResponse)
def get_races_calendar(
    session: Annotated[Session, Depends(get_session)],
    from_: Annotated[str, Query(alias="from", description="開始日 YYYY-MM-DD")],
    to: Annotated[str, Query(description="終了日 YYYY-MM-DD (含む)")],
) -> CalendarResponse:
    """期間内の日ごとの取込状況を返す (カレンダー表示用)。

    1 レースも無い日は返さない。呼び出し側は「返って来なかった日 = 未取得」
    として扱えばよい。
    """
    try:
        start = date.fromisoformat(from_)
        end = date.fromisoformat(to)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format (expected YYYY-MM-DD): {exc}",
        ) from exc
    if end < start:
        raise HTTPException(status_code=422, detail="'to' must not be earlier than 'from'")

    races = session.scalars(
        select(Race)
        .where(Race.date >= start.isoformat(), Race.date <= end.isoformat())
        .order_by(Race.date, Race.race_id)
    ).all()
    if not races:
        return CalendarResponse(days=[])

    # 着順が入っている race_id (= 結果まで取り込めているレース)
    finished_ids = set(
        session.scalars(
            select(Entry.race_id)
            .join(Race, Race.race_id == Entry.race_id)
            .where(
                Race.date >= start.isoformat(),
                Race.date <= end.isoformat(),
                Entry.finish_position.is_not(None),
            )
            .distinct()
        ).all()
    )

    by_date: dict[str, list[Race]] = {}
    for r in races:
        by_date.setdefault(r.date, []).append(r)

    days: list[CalendarDay] = []
    for day, day_races in sorted(by_date.items()):
        # 開催場は出現順を保ったまま重複を除く
        courses: list[str] = []
        for r in day_races:
            if r.course not in courses:
                courses.append(r.course)

        highlight = min(day_races, key=lambda r: (_class_rank(r.race_class), r.race_id))
        # 平場しか無い日は名前を出さない (「未勝利」と出しても情報にならない)
        has_feature = _class_rank(highlight.race_class) < _CLASS_PRIORITY.index("3勝")

        days.append(
            CalendarDay(
                date=day,
                race_count=len(day_races),
                result_count=sum(1 for r in day_races if r.race_id in finished_ids),
                courses=courses,
                highlight_race_id=highlight.race_id if has_feature else None,
                highlight_name=(highlight.name if has_feature else None),
                highlight_class=(highlight.race_class if has_feature else None),
            )
        )

    return CalendarResponse(days=days)


@router.get("/races/coverage", response_model=DataCoverage)
def get_data_coverage(
    session: Annotated[Session, Depends(get_session)],
) -> DataCoverage:
    """取込済みデータ全体の状況を返す。

    「いつからいつまで、何レース入っていて、そのうち何レースが結果まで
    取れているか」を 1 レスポンスで返す。
    """
    first_date = session.scalar(select(func.min(Race.date)))
    last_date = session.scalar(select(func.max(Race.date)))
    race_count = session.scalar(select(func.count()).select_from(Race)) or 0
    entry_count = session.scalar(select(func.count()).select_from(Entry)) or 0
    result_count = (
        session.scalar(
            select(func.count(func.distinct(Entry.race_id))).where(
                Entry.finish_position.is_not(None)
            )
        )
        or 0
    )

    # 直近 90 日のうち、レースを取り込めている日数
    span_days = 90
    since = (date.today() - timedelta(days=span_days)).isoformat()
    recent_days_with_data = (
        session.scalar(
            select(func.count(func.distinct(Race.date))).where(Race.date >= since)
        )
        or 0
    )

    return DataCoverage(
        first_date=first_date,
        last_date=last_date,
        race_count=race_count,
        result_count=result_count,
        entry_count=entry_count,
        recent_days_with_data=recent_days_with_data,
        recent_days_span=span_days,
    )


@router.get("/races/{race_id}", response_model=RaceDetail)
def get_race_detail(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> RaceDetail:
    race = get_or_404(session, Race, race_id, label="Race")

    entries_stmt = select(Entry).where(Entry.race_id == race_id).order_by(Entry.post_position)
    entries = list(session.scalars(entries_stmt).all())

    # 全券種の確定払戻。答え合わせで推奨買目を突き合わせるのに使う。
    payouts = list(session.scalars(select(Payout).where(Payout.race_id == race_id)).all())

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
        payouts=[
            PayoutEntry(
                bet_type=p.bet_type,
                # 画面は買い目 (``1-10``) と突き合わせるので表記を揃えて返す
                combo=normalize_combo(p.combo),
                amount=p.amount,
                popularity=p.popularity,
            )
            for p in payouts
        ],
    )
