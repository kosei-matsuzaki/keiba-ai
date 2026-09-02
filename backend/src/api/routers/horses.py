"""GET /api/horses/{horse_id}/history — その日より前の過去走。

出走馬一覧から「この馬は前走どうだったか」をその場で見るための API。
AI の予想の根拠 (履歴 GRU が食べているもの) と同じ範囲を人も見られるようにする。

**必ず `before` より厳密に過去だけを返す。** 特徴量側と同じ制約で、当日以降の
情報が混ざると「予想の根拠」として読めなくなる (`features/builder.py` と同じ考え方)。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import get_session
from db.models.entry import Entry
from db.models.jockey import Jockey
from db.models.race import Race

router = APIRouter()


class HorsePastRun(BaseModel):
    """過去走 1 走分。出走馬一覧の行を開いたときに出す。"""

    race_id: str
    date: str
    course: str
    race_name: str | None
    race_class: str | None
    surface: str
    distance: int
    track_condition: str | None
    n_runners: int | None
    post_position: int | None
    finish_position: int | None
    #: 単勝オッズと人気 (当時)
    odds_win: float | None
    popularity: int | None
    jockey_name: str | None
    weight_carried: float | None
    horse_weight: int | None
    finish_time: float | None
    agari_3f: float | None
    passing: str | None
    margin: str | None


class HorseHistoryResponse(BaseModel):
    horse_id: str
    #: この日より**厳密に前**の走りだけを返す。
    before: str | None
    runs: list[HorsePastRun]


@router.get("/horses/{horse_id}/history", response_model=HorseHistoryResponse)
def get_horse_history(
    horse_id: str,
    session: Annotated[Session, Depends(get_session)],
    before: Annotated[
        str | None,
        Query(description="この日より前の走りだけ返す (YYYY-MM-DD)。未指定なら全部"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=30, description="最大何走返すか")] = 6,
) -> HorseHistoryResponse:
    """馬の過去走を新しい順に返す。

    レース詳細の出走馬一覧から、その**レース当日より前**の成績を引くのに使う。
    """
    stmt = (
        select(Entry, Race, Jockey.name)
        .join(Race, Entry.race_id == Race.race_id)
        .join(Jockey, Entry.jockey_id == Jockey.jockey_id, isouter=True)
        .where(Entry.horse_id == horse_id)
    )
    if before is not None:
        # **厳密に過去**。同日のレースも入れない (当日の結果は根拠にできない)
        stmt = stmt.where(Race.date < before)
    stmt = stmt.order_by(Race.date.desc()).limit(limit)

    runs = [
        HorsePastRun(
            race_id=race.race_id,
            date=race.date,
            course=race.course,
            race_name=race.name,
            race_class=race.race_class,
            surface=race.surface,
            distance=race.distance,
            track_condition=race.track_condition,
            n_runners=race.n_runners,
            post_position=entry.post_position,
            finish_position=entry.finish_position,
            odds_win=entry.odds_win,
            popularity=entry.popularity,
            jockey_name=jockey_name,
            weight_carried=entry.weight_carried,
            horse_weight=entry.horse_weight,
            finish_time=entry.finish_time,
            agari_3f=entry.agari_3f,
            passing=entry.passing,
            margin=entry.margin,
        )
        for entry, race, jockey_name in session.execute(stmt).all()
    ]
    return HorseHistoryResponse(horse_id=horse_id, before=before, runs=runs)
