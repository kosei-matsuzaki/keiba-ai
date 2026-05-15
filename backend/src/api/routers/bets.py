"""Bet record endpoints: CRUD for /api/bets and aggregation endpoints.

期間フィルタ（from/to）は settled_at（確定日）ベースで動作する。
pending（未確定）bet は settled_at が NULL のため期間フィルタ後の集計から除外される。
"""

from __future__ import annotations

import csv
import datetime
import io
from typing import Annotated, Literal

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import get_or_404, get_session
from api.schemas import (
    BetBreakdown,
    BetBreakdownRow,
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


def _apply_common_filters(
    stmt: sa.Select,
    from_: str | None,
    to: str | None,
    bet_type: str | None,
    source: str | None,
) -> sa.Select:
    """Reusable WHERE clause for aggregation endpoints.

    期間フィルタは settled_at（確定日）ベースで動作する。
    settled_at IS NULL の bet（未確定）は from/to 指定時に除外される。
    """
    if from_ is not None:
        stmt = stmt.where(BetRecord.settled_at >= from_)
    if to is not None:
        # Inclusive upper bound: treat 'to' as end-of-day by appending 'T23:59:59'
        stmt = stmt.where(BetRecord.settled_at <= f"{to}T23:59:59")
    if bet_type is not None:
        stmt = stmt.where(BetRecord.bet_type == bet_type)
    if source is not None:
        stmt = stmt.where(BetRecord.source == source)
    return stmt


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
    stmt = _apply_common_filters(select(BetRecord), from_, to, bet_type, source)
    records = session.scalars(stmt).all()

    total_bets = len(records)
    settled = [r for r in records if r.settled_at is not None]
    settled_bets = len(settled)
    pending_bets = total_bets - settled_bets

    total_invested = sum(r.stake for r in records)
    # payout / profit are None for pending bets — treat as 0 in aggregation
    total_payout = sum(r.payout or 0 for r in records)
    total_profit = sum(r.profit or 0 for r in records)
    payback_rate = total_payout / total_invested if total_invested > 0 else 0.0
    hit_count = sum(1 for r in settled if (r.payout or 0) > 0)
    hit_rate = hit_count / settled_bets if settled_bets > 0 else 0.0

    return BetSummary(
        total_bets=total_bets,
        settled_bets=settled_bets,
        pending_bets=pending_bets,
        total_invested=total_invested,
        total_payout=total_payout,
        total_profit=total_profit,
        payback_rate=payback_rate,
        hit_rate=hit_rate,
        range_from=from_,
        range_to=to,
    )


# ── GET /api/bets/timeseries ──────────────────────────────────────────────────

_BucketType = Literal["day", "week", "month"]


def _format_bucket_key(d: datetime.date, bucket: _BucketType) -> str:
    """date から bucket key 文字列を生成する。"""
    if bucket == "day":
        return d.isoformat()
    if bucket == "week":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{d.year}-{d.month:02d}"


def _bucket_date(settled_at: str, bucket: _BucketType) -> str:
    """ISO 8601 settled_at 文字列から bucket key を生成する。"""
    return _format_bucket_key(datetime.date.fromisoformat(settled_at[:10]), bucket)


def _iter_buckets(start: datetime.date, end: datetime.date, bucket: _BucketType) -> list[str]:
    """[start, end] の範囲をカバーする bucket key を昇順に列挙する。"""
    keys: list[str] = []
    if bucket == "day":
        cur = start
        step = datetime.timedelta(days=1)
        while cur <= end:
            keys.append(_format_bucket_key(cur, bucket))
            cur += step
        return keys

    if bucket == "week":
        # ISO 週の月曜にアラインしてから 1 週ずつ進める
        cur = start - datetime.timedelta(days=start.weekday())
        step = datetime.timedelta(days=7)
        while cur <= end:
            keys.append(_format_bucket_key(cur, bucket))
            cur += step
        return keys

    # month: 年月を 1 ずつインクリメント
    year, month = start.year, start.month
    end_ym = (end.year, end.month)
    while (year, month) <= end_ym:
        keys.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return keys


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
    stmt = _apply_common_filters(select(BetRecord), from_, to, bet_type, source)
    # settled_at でソート（NULL は除外済みなので安全）
    stmt = stmt.where(BetRecord.settled_at.is_not(None))
    stmt = stmt.order_by(BetRecord.settled_at.asc())
    records = session.scalars(stmt).all()

    # Group by bucket key (settled_at ベース)
    data_buckets: dict[str, dict] = {}
    for r in records:
        key = _bucket_date(r.settled_at, bucket)  # type: ignore[arg-type]
        if key not in data_buckets:
            data_buckets[key] = {"invested": 0, "payout": 0, "profit": 0, "bets": 0}
        data_buckets[key]["invested"] += r.stake
        data_buckets[key]["payout"] += r.payout or 0
        data_buckets[key]["profit"] += r.profit or 0
        data_buckets[key]["bets"] += 1

    # Determine date range for bucket enumeration
    if data_buckets or from_ is not None or to is not None:
        if from_ is not None:
            range_start = datetime.date.fromisoformat(from_)
        elif records:
            range_start = datetime.date.fromisoformat(records[0].settled_at[:10])  # type: ignore[index]
        else:
            range_start = None

        if to is not None:
            range_end = datetime.date.fromisoformat(to)
        elif records:
            range_end = datetime.date.fromisoformat(records[-1].settled_at[:10])  # type: ignore[index]
        else:
            range_end = None
    else:
        range_start = range_end = None

    # Build ordered bucket keys with 0-filled gaps
    if range_start is not None and range_end is not None:
        all_keys = _iter_buckets(range_start, range_end, bucket)
    else:
        all_keys = sorted(data_buckets.keys())

    # Build points with window-accumulated cumulative_profit
    cumulative = 0
    points: list[BetTimeseriesPoint] = []
    for key in all_keys:
        b = data_buckets.get(key, {"invested": 0, "payout": 0, "profit": 0, "bets": 0})
        cumulative += b["profit"]
        points.append(BetTimeseriesPoint(
            date=key,
            invested=b["invested"],
            payout=b["payout"],
            profit=b["profit"],
            cumulative_profit=cumulative,
            bets=b["bets"],
        ))

    return BetTimeseries(bucket=bucket, points=points)


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
    stmt = _apply_common_filters(select(BetRecord), from_, to, bet_type, source)
    records = session.scalars(stmt).all()

    # Bulk load Race records to avoid N+1 queries in race_class grouping
    if group_by == "race_class":
        race_ids = {r.race_id for r in records}
        races_map: dict[str, Race] = {
            r.race_id: r
            for r in session.scalars(select(Race).where(Race.race_id.in_(race_ids))).all()
        }

    def _group_key(r: BetRecord) -> str:
        if group_by == "bet_type":
            return r.bet_type
        if group_by == "source":
            return r.source
        if group_by == "month":
            # settled_at ベースで月集計。NULL は _apply_common_filters で除外されるが念のため
            date_src = r.settled_at or r.created_at
            return date_src[:7]  # "YYYY-MM"
        # race_class: use bulk-loaded map
        race = races_map.get(r.race_id)
        return race.race_class if (race and race.race_class) else "不明"

    groups: dict[str, dict] = {}
    for r in records:
        key = _group_key(r)
        if key not in groups:
            groups[key] = {"bets": 0, "invested": 0, "payout": 0, "profit": 0, "hits": 0, "settled": 0}
        groups[key]["bets"] += 1
        groups[key]["invested"] += r.stake
        groups[key]["payout"] += r.payout or 0
        groups[key]["profit"] += r.profit or 0
        if r.settled_at is not None:
            groups[key]["settled"] += 1
            if (r.payout or 0) > 0:
                groups[key]["hits"] += 1

    rows = []
    for key, g in groups.items():
        payback_rate = g["payout"] / g["invested"] if g["invested"] > 0 else 0.0
        hit_rate = g["hits"] / g["settled"] if g["settled"] > 0 else 0.0
        rows.append(BetBreakdownRow(
            group_key=key,
            bets=g["bets"],
            invested=g["invested"],
            payout=g["payout"],
            profit=g["profit"],
            payback_rate=payback_rate,
            hit_rate=hit_rate,
        ))

    return BetBreakdown(group_by=group_by, rows=rows)


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

    stmt = _apply_common_filters(select(BetRecord), effective_from, effective_to, bet_type, source)
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
    """bet_record を削除する。settled な bet は 409 を返す。"""
    record = get_or_404(session, BetRecord, bet_id, label="BetRecord")
    if record.settled_at is not None:
        raise HTTPException(
            status_code=409,
            detail=f"BetRecord {bet_id} is already settled and cannot be deleted",
        )
    session.delete(record)
    session.commit()
