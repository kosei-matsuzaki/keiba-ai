"""Bet record aggregation logic (pure, DB/HTTP 非依存).

api/routers/bets.py から集計アルゴリズムを切り出したもの。
- 入力は読み出し済みの BetRecord シーケンス（SQL は呼び出し側 router が担当）。
- 戻り値は素の dict / list[dict]。api.schemas には依存しない（依存方向 api → services を保つ）。
- 期間フィルタは settled_at（確定日）ベースという呼び出し側の規約をそのまま受ける。
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import sqlalchemy as sa

from db.models.bet_record import BetRecord

BucketType = Literal["day", "week", "month"]


def apply_bet_filters(
    stmt: sa.Select,
    from_: str | None,
    to: str | None,
    bet_type: str | None,
    source: str | None,
) -> sa.Select:
    """集計エンドポイント共通の WHERE 句。

    期間フィルタは settled_at（確定日）ベース。settled_at IS NULL の bet（未確定）は
    from/to 指定時に除外される。
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


def summarize(records: Sequence[BetRecord]) -> dict:
    """累計損益サマリを集計する。

    payout / profit は未確定 bet では None なので 0 として扱う。
    """
    total_bets = len(records)
    settled = [r for r in records if r.settled_at is not None]
    settled_bets = len(settled)

    total_invested = sum(r.stake for r in records)
    total_payout = sum(r.payout or 0 for r in records)
    total_profit = sum(r.profit or 0 for r in records)
    hit_count = sum(1 for r in settled if (r.payout or 0) > 0)

    return {
        "total_bets": total_bets,
        "settled_bets": settled_bets,
        "pending_bets": total_bets - settled_bets,
        "total_invested": total_invested,
        "total_payout": total_payout,
        "total_profit": total_profit,
        "payback_rate": total_payout / total_invested if total_invested > 0 else 0.0,
        "hit_rate": hit_count / settled_bets if settled_bets > 0 else 0.0,
    }


def _format_bucket_key(d: datetime.date, bucket: BucketType) -> str:
    """date から bucket key 文字列を生成する。"""
    if bucket == "day":
        return d.isoformat()
    if bucket == "week":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{d.year}-{d.month:02d}"


def _bucket_date(settled_at: str, bucket: BucketType) -> str:
    """ISO 8601 settled_at 文字列から bucket key を生成する。"""
    return _format_bucket_key(datetime.date.fromisoformat(settled_at[:10]), bucket)


def _iter_buckets(start: datetime.date, end: datetime.date, bucket: BucketType) -> list[str]:
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


def timeseries_points(
    records: Sequence[BetRecord],
    bucket: BucketType,
    from_: str | None,
    to: str | None,
) -> list[dict]:
    """時系列損益データの points を構築する。

    - records は settled_at 昇順に並んでいること（未確定 bet は呼び出し側で除外済み）。
    - from/to 指定時は期間内の全 bucket を 0 で初期化し、空 bucket を含む連続データを返す。
    - cumulative_profit は前 bucket の累計を持ち越す。
    """
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
    points: list[dict] = []
    for key in all_keys:
        b = data_buckets.get(key, {"invested": 0, "payout": 0, "profit": 0, "bets": 0})
        cumulative += b["profit"]
        points.append({
            "date": key,
            "invested": b["invested"],
            "payout": b["payout"],
            "profit": b["profit"],
            "cumulative_profit": cumulative,
            "bets": b["bets"],
        })
    return points


GroupByType = Literal["bet_type", "race_class", "month", "source"]


def breakdown_rows(
    records: Iterable[BetRecord],
    group_by: GroupByType,
    race_class_map: Mapping[str, str] | None = None,
) -> list[dict]:
    """グルーピング別の損益ブレイクダウンを集計する。

    race_class グルーピング時は race_id → race_class の辞書を race_class_map で受ける
    （N+1 回避のため SQL 一括ロードは呼び出し側 router が担当）。
    """
    race_class_map = race_class_map or {}

    def _group_key(r: BetRecord) -> str:
        if group_by == "bet_type":
            return r.bet_type
        if group_by == "source":
            return r.source
        if group_by == "month":
            # settled_at ベースで月集計。NULL は apply_bet_filters で除外されるが念のため
            date_src = r.settled_at or r.created_at
            return date_src[:7]  # "YYYY-MM"
        # race_class: use bulk-loaded map
        return race_class_map.get(r.race_id) or "不明"

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

    rows: list[dict] = []
    for key, g in groups.items():
        rows.append({
            "group_key": key,
            "bets": g["bets"],
            "invested": g["invested"],
            "payout": g["payout"],
            "profit": g["profit"],
            "payback_rate": g["payout"] / g["invested"] if g["invested"] > 0 else 0.0,
            "hit_rate": g["hits"] / g["settled"] if g["settled"] > 0 else 0.0,
        })
    return rows
