"""Bet record endpoints: CRUD for /api/bets."""

from __future__ import annotations

import datetime
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_session
from keiba_ai.api.schemas import BetRecordIn, BetRecordList, BetRecordOut, BetRecordUpdate
from keiba_ai.db.models.bet_record import BetRecord
from keiba_ai.db.models.race import Race
from keiba_ai.services.bet_settlement import settle_bet

router = APIRouter()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _to_out(record: BetRecord) -> BetRecordOut:
    return BetRecordOut(
        id=record.id,
        created_at=record.created_at,
        race_id=record.race_id,
        bet_type=record.bet_type,
        combo=record.combo,
        stake=record.stake,
        source=record.source,
        recommendation_id=record.recommendation_id,
        settled_at=record.settled_at,
        payout=record.payout,
        profit=record.profit,
        notes=record.notes,
    )


@router.post("/bets", response_model=BetRecordOut, status_code=201)
def create_bet(
    body: BetRecordIn,
    session: Annotated[Session, Depends(get_session)],
) -> BetRecordOut:
    """bet_record を登録し、即時突合せを試みる。

    races テーブルに該当 race_id が存在しない場合は 404 を返す。
    payouts またはfallback フィールドが揃っていれば登録と同時に確定する。
    """
    race = session.get(Race, body.race_id)
    if race is None:
        raise HTTPException(status_code=404, detail=f"Race {body.race_id!r} not found")

    record = BetRecord(
        created_at=_now_iso(),
        race_id=body.race_id,
        bet_type=body.bet_type,
        combo=body.combo,
        stake=body.stake,
        source=body.source,
        recommendation_id=body.recommendation_id,
        notes=body.notes,
    )
    session.add(record)
    session.flush()  # id を確定させてから突合せ

    # 即時突合せ試行 (結果データが既にあれば確定される)
    settle_bet(session, record)
    session.commit()
    session.refresh(record)
    return _to_out(record)


@router.get("/bets", response_model=BetRecordList)
def list_bets(
    session: Annotated[Session, Depends(get_session)],
    race_id: str | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    source: str | None = Query(default=None),
    settled: bool | None = Query(default=None),
) -> BetRecordList:
    """bet_records 一覧取得。各種フィルタに対応。"""
    stmt = select(BetRecord)

    if race_id is not None:
        stmt = stmt.where(BetRecord.race_id == race_id)
    if from_ is not None:
        stmt = stmt.where(BetRecord.created_at >= from_)
    if to is not None:
        stmt = stmt.where(BetRecord.created_at <= to)
    if source is not None:
        stmt = stmt.where(BetRecord.source == source)
    if settled is True:
        stmt = stmt.where(BetRecord.settled_at.is_not(None))
    elif settled is False:
        stmt = stmt.where(BetRecord.settled_at.is_(None))

    stmt = stmt.order_by(BetRecord.created_at.desc())
    records = session.scalars(stmt).all()
    return BetRecordList(total=len(records), items=[_to_out(r) for r in records])


@router.get("/bets/{bet_id}", response_model=BetRecordOut)
def get_bet(
    bet_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> BetRecordOut:
    record = session.get(BetRecord, bet_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"BetRecord {bet_id} not found")
    return _to_out(record)


@router.put("/bets/{bet_id}", response_model=BetRecordOut)
def update_bet(
    bet_id: int,
    body: BetRecordUpdate,
    session: Annotated[Session, Depends(get_session)],
) -> BetRecordOut:
    """notes のみ更新可。settled な bet の更新は 409 を返す。"""
    record = session.get(BetRecord, bet_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"BetRecord {bet_id} not found")
    if record.settled_at is not None:
        raise HTTPException(
            status_code=409,
            detail=f"BetRecord {bet_id} is already settled and cannot be modified",
        )
    record.notes = body.notes
    session.commit()
    session.refresh(record)
    return _to_out(record)


@router.delete("/bets/{bet_id}", status_code=204)
def delete_bet(
    bet_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """bet_record を削除する。settled な bet は 409 を返す。"""
    record = session.get(BetRecord, bet_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"BetRecord {bet_id} not found")
    if record.settled_at is not None:
        raise HTTPException(
            status_code=409,
            detail=f"BetRecord {bet_id} is already settled and cannot be deleted",
        )
    session.delete(record)
    session.commit()
