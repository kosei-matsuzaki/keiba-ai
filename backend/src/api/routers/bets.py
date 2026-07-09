"""Bet record endpoints: CRUD for /api/bets and aggregation endpoints.

期間フィルタ（from/to）は settled_at（確定日）ベースで動作する。
pending（未確定）bet は settled_at が NULL のため期間フィルタ後の集計から除外される。
"""

from __future__ import annotations

import csv
import datetime
import io
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.deps import get_or_404, get_session
from api.schemas import (
    BetBreakdown,
    BetBreakdownRow,
    BetBulkDeleteIn,
    BetRecordBulkIn,
    BetRecordIn,
    BetRecordList,
    BetRecordOut,
    BetRecordUpdate,
    BetSummary,
    BetTimeseries,
    BetTimeseriesPoint,
)
from db.models.bet_record import BetRecord
from db.models.race import Race
from services import bet_analytics
from services.bet_settlement import settle_bet

router = APIRouter()

_MAX_CSV_DAYS = 366
_MAX_CSV_ROWS = 50_000


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


# ── GET /api/bets/summary ─────────────────────────────────────────────────────

@router.get("/bets/summary", response_model=BetSummary)
def get_bet_summary(
    session: Annotated[Session, Depends(get_session)],
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    bet_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> BetSummary:
    """累計損益サマリ。settled_at（確定日）期間・券種・ソースでフィルタ可能。

    - from/to は settled_at ベース。未確定（pending）bet は期間フィルタ後に除外される。
    - range_from / range_to は settled 期間を示す。
    - pending_bets は指定期間内に created されたが未確定の bet 数を返す（summary 整合のため）。
    """
    stmt = bet_analytics.apply_bet_filters(select(BetRecord), from_, to, bet_type, source)
    records = session.scalars(stmt).all()

    return BetSummary(
        **bet_analytics.summarize(records),
        range_from=from_,
        range_to=to,
    )


# ── GET /api/bets/timeseries ──────────────────────────────────────────────────

@router.get("/bets/timeseries", response_model=BetTimeseries)
def get_bet_timeseries(
    session: Annotated[Session, Depends(get_session)],
    bucket: Literal["day", "week", "month"] = Query(default="day"),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    bet_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> BetTimeseries:
    """時系列損益データ。bucket で集約粒度を指定。

    - settled_at（確定日）ベースで集計。settled_at IS NULL の bet は除外。
    - from/to 指定時は期間内の全 bucket を 0 で初期化し、空日を含む連続データを返す。
    - cumulative_profit は前 bucket の累計を持ち越す。
    """
    stmt = bet_analytics.apply_bet_filters(select(BetRecord), from_, to, bet_type, source)
    # settled_at でソート（NULL は除外済みなので安全）
    stmt = stmt.where(BetRecord.settled_at.is_not(None))
    stmt = stmt.order_by(BetRecord.settled_at.asc())
    records = session.scalars(stmt).all()

    points = bet_analytics.timeseries_points(records, bucket, from_, to)
    return BetTimeseries(
        bucket=bucket,
        points=[BetTimeseriesPoint(**p) for p in points],
    )


# ── GET /api/bets/breakdown ───────────────────────────────────────────────────

@router.get("/bets/breakdown", response_model=BetBreakdown)
def get_bet_breakdown(
    session: Annotated[Session, Depends(get_session)],
    group_by: Literal["bet_type", "race_class", "month", "source"] = Query(default="bet_type"),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    bet_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> BetBreakdown:
    """グルーピング別の損益ブレイクダウン。settled_at（確定日）ベースでフィルタ。"""
    stmt = bet_analytics.apply_bet_filters(select(BetRecord), from_, to, bet_type, source)
    records = session.scalars(stmt).all()

    # Bulk load race_class to avoid N+1 queries in race_class grouping
    race_class_map: dict[str, str] = {}
    if group_by == "race_class":
        race_ids = {r.race_id for r in records}
        race_class_map = {
            r.race_id: r.race_class
            for r in session.scalars(select(Race).where(Race.race_id.in_(race_ids))).all()
            if r.race_class
        }

    rows = bet_analytics.breakdown_rows(records, group_by, race_class_map)
    return BetBreakdown(
        group_by=group_by,
        rows=[BetBreakdownRow(**row) for row in rows],
    )


# ── GET /api/bets/export.csv ──────────────────────────────────────────────────

_CSV_COLUMNS = ["id", "created_at", "race_id", "bet_type", "combo", "stake", "source",
                "settled_at", "payout", "profit", "notes"]


@router.get("/bets/export.csv")
def export_bets_csv(
    session: Annotated[Session, Depends(get_session)],
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    bet_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
    limit: int = Query(default=10_000, ge=1, le=_MAX_CSV_ROWS),
) -> StreamingResponse:
    """現在フィルタ条件の bet_records を BOM 付き UTF-8 CSV で返す。

    - 期間は settled_at（確定日）でフィルタ。
    - 期間範囲は最大 366 日。超過時は 400 を返す。
    - from/to 両方未指定の場合は直近 1 年分を返す。
    - limit パラメータ（デフォルト 10000、最大 50000）で件数を制限。
    """
    # Resolve default date range (last 1 year) when both from/to are absent
    effective_from = from_
    effective_to = to
    if effective_from is None and effective_to is None:
        today = datetime.date.today()
        effective_from = (today - datetime.timedelta(days=365)).isoformat()
        effective_to = today.isoformat()

    # DoS guard: period must be ≤ 366 days when both bounds are specified
    if effective_from is not None and effective_to is not None:
        try:
            d_from = datetime.date.fromisoformat(effective_from[:10])
            d_to = datetime.date.fromisoformat(effective_to[:10])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}") from exc
        if (d_to - d_from).days > _MAX_CSV_DAYS:
            raise HTTPException(
                status_code=400,
                detail="Period exceeds 1 year. Please narrow the date range.",
            )

    stmt = bet_analytics.apply_bet_filters(select(BetRecord), effective_from, effective_to, bet_type, source)
    stmt = stmt.order_by(BetRecord.settled_at.asc()).limit(limit)
    records = session.scalars(stmt).all()

    buf = io.BytesIO()
    # Write BOM for Excel UTF-8 compatibility
    buf.write(b"\xef\xbb\xbf")
    text_wrapper = io.TextIOWrapper(buf, encoding="utf-8", newline="")
    writer = csv.DictWriter(text_wrapper, fieldnames=_CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for r in records:
        writer.writerow({
            "id": r.id,
            "created_at": r.created_at,
            "race_id": r.race_id,
            "bet_type": r.bet_type,
            "combo": r.combo,
            "stake": r.stake,
            "source": r.source,
            "settled_at": r.settled_at or "",
            "payout": r.payout if r.payout is not None else "",
            "profit": r.profit if r.profit is not None else "",
            "notes": r.notes or "",
        })
    text_wrapper.flush()
    text_wrapper.detach()

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=bet_records.csv"},
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
    get_or_404(session, Race, body.race_id, label="Race")

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


@router.post("/bets/bulk", response_model=BetRecordList, status_code=201)
def create_bets_bulk(
    body: BetRecordBulkIn,
    session: Annotated[Session, Depends(get_session)],
) -> BetRecordList:
    """買い方（流し / ボックス / フォーメーション）を展開した複数点をまとめて登録する。

    各点 (combo, stake) を独立した bet_record として保存し、それぞれ即時突合せを試みる。
    1 トランザクションで処理するので、途中失敗時は全件ロールバックされる。
    races テーブルに該当 race_id が無ければ 404。
    """
    get_or_404(session, Race, body.race_id, label="Race")

    now = _now_iso()
    records = [
        BetRecord(
            created_at=now,
            race_id=body.race_id,
            bet_type=body.bet_type,
            combo=item.combo,
            stake=item.stake,
            source=body.source,
            notes=body.notes,
        )
        for item in body.combos
    ]
    session.add_all(records)
    session.flush()  # id を確定させてから突合せ

    for record in records:
        settle_bet(session, record)
    session.commit()

    for record in records:
        session.refresh(record)
    return BetRecordList(total=len(records), items=[_to_out(r) for r in records])


@router.post("/bets/bulk_delete")
def bulk_delete_bets(
    body: BetBulkDeleteIn,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, int]:
    """指定した id 群の bet_record をまとめて削除する（買い方単位の削除に使う）。

    個人台帳なので確定済みも削除可。存在しない id は無視する。
    """
    res = session.execute(delete(BetRecord).where(BetRecord.id.in_(body.ids)))
    session.commit()
    return {"deleted": int(res.rowcount or 0)}


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
    record = get_or_404(session, BetRecord, bet_id, label="BetRecord")
    return _to_out(record)


@router.put("/bets/{bet_id}", response_model=BetRecordOut)
def update_bet(
    bet_id: int,
    body: BetRecordUpdate,
    session: Annotated[Session, Depends(get_session)],
) -> BetRecordOut:
    """notes のみ更新可。settled な bet の更新は 409 を返す。"""
    record = get_or_404(session, BetRecord, bet_id, label="BetRecord")
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
    """bet_record を削除する。

    個人の購入台帳なので、誤登録の訂正のため確定済み (settled) の記録も削除できる。
    手動登録は過去レースだと登録と同時に自動確定するため、確定済みを消せないと
    訂正手段が無くなる。
    """
    record = get_or_404(session, BetRecord, bet_id, label="BetRecord")
    session.delete(record)
    session.commit()
