"""Bet settlement service — payouts テーブルまたは races の fallback フィールドを用いて
bet_records を確定（settled_at / payout / profit を更新）する。

優先順位:
  1. payouts テーブル (bet_type + combo の厳密マッチ) — 全馬券種対応
  2. races.payout_win  (単勝のみ fallback、payouts が未 ingest の場合のみ)
  3. races.payout_place JSON (複勝のみ fallback、payouts が未 ingest の場合のみ)

payouts にヒットした場合は payout_win / payout_place は参照しない。
payouts テーブルに該当 race のデータが存在するが combo が一致しない場合は外れ（payout=0）。
payouts テーブルに該当 race のデータが存在しない場合のみ fallback を使用する。
fallback は combo を entries で検証してから払戻を返す。
"""

from __future__ import annotations

import datetime
import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.bet_record import BetRecord
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _calc_payout(amount_per_100: int, stake: int) -> int:
    """100 円基準の払戻金 amount_per_100 と stake 円から払戻金を計算する。

    JRA の払戻表示は 100 円あたりの金額なので、stake / 100 を掛ける。
    stake が 100 の倍数でなくても比例計算する。
    """
    return int(amount_per_100 * stake / 100)


def _resolve_payout_from_payouts_table(
    session: Session, race_id: str, bet_type: str, combo: str
) -> int | None:
    """payouts テーブルから payout amount (100 円基準) を取得する。ヒットしなければ None。"""
    row = session.execute(
        select(Payout).where(
            Payout.race_id == race_id,
            Payout.bet_type == bet_type,
            Payout.combo == combo,
        ).limit(1)
    ).scalar_one_or_none()
    return row.amount if row is not None else None


def _resolve_payout_win_fallback(session: Session, race_id: str, combo: str) -> int | None:
    """単勝の fallback. combo (post_position 文字列) が実際の 1 着馬と一致したら payout_win を返す。

    payouts テーブルが未 ingest のレガシーデータ向け。
    entries から 1 着馬の post_position を取得して combo と照合する。
    一致すれば当たり (payout_win)、不一致なら外れ (0)。
    """
    race = session.get(Race, race_id)
    if race is None or race.payout_win is None:
        return None
    winner = session.execute(
        select(Entry.post_position).where(
            Entry.race_id == race_id,
            Entry.finish_position == 1,
        ).limit(1)
    ).scalar_one_or_none()
    if winner is None:
        return None  # 結果未確定または entries 未着
    if str(winner) == combo:
        return race.payout_win  # ベットが 1 着馬と一致 → 当たり
    return 0  # 1 着馬と一致せず → 外れ


def _resolve_payout_place_fallback(session: Session, race_id: str, combo: str) -> int | None:
    """複勝の fallback.

    combo は post_position（馬番）。races.payout_place は
    {"1": 110, "2": 160, "3": 150} 形式の JSON 文字列（キー = finish_position 1/2/3、値 = その着順の馬の払戻）。

    手順:
      1. combo (post_position) から entries の finish_position を引く
      2. finish_position が 1/2/3 なら payout_place JSON でその fp の値を返す
      3. 4 着以下 / NULL なら 0（外れ）
      4. entries が無い / payout_place が無いなら None（確定不能）
    """
    race = session.get(Race, race_id)
    if race is None or race.payout_place is None:
        return None
    try:
        post_pos = int(combo)
    except (ValueError, TypeError):
        return None
    fp = session.execute(
        select(Entry.finish_position).where(
            Entry.race_id == race_id,
            Entry.post_position == post_pos,
        ).limit(1)
    ).scalar_one_or_none()
    if fp is None:
        return None  # entries 未着または馬番なし
    if fp not in (1, 2, 3):
        return 0  # 4 着以下 → 外れ
    try:
        place_map: dict[str, int] = json.loads(race.payout_place)
    except (json.JSONDecodeError, TypeError):
        return None
    return place_map.get(str(fp))  # str fp で引く（2 着なら "2"）


def settle_bet(session: Session, bet_record: BetRecord) -> None:
    """単一の BetRecord を確定する。

    既に settled_at が設定済みのレコードは再確定しない。
    payout / profit / settled_at を更新する（コミットは呼び出し側の責務）。
    """
    if bet_record.settled_at is not None:
        return

    amount_per_100: int | None = _resolve_payout_from_payouts_table(
        session, bet_record.race_id, bet_record.bet_type, bet_record.combo
    )

    # payouts テーブルにヒットしなかった場合の処理
    if amount_per_100 is None:
        # 該当 race に payouts のデータが存在するか確認する
        has_payouts = session.execute(
            select(Payout.id).where(Payout.race_id == bet_record.race_id).limit(1)
        ).first() is not None

        if has_payouts:
            # payouts は ingest 済みだが該当 combo が無い → 外れ確定
            amount_per_100 = 0
        else:
            # payouts 未 ingest（レガシーデータ）→ races.payout_* fallback（combo 検証つき）
            if bet_record.bet_type == "単勝":
                amount_per_100 = _resolve_payout_win_fallback(
                    session, bet_record.race_id, bet_record.combo
                )
            elif bet_record.bet_type == "複勝":
                amount_per_100 = _resolve_payout_place_fallback(
                    session, bet_record.race_id, bet_record.combo
                )
            # 単勝・複勝以外で payouts が空なら確定不能（None のまま）

    if amount_per_100 is None:
        # 払戻データ未着（結果未確定 or データ欠損）— 確定しない
        return

    payout = _calc_payout(amount_per_100, bet_record.stake) if amount_per_100 > 0 else 0
    bet_record.payout = payout
    bet_record.profit = payout - bet_record.stake
    bet_record.settled_at = _now_iso()


def settle_bets_for_race(session: Session, race_id: str) -> int:
    """指定 race_id の未確定 bet を一括確定する。確定した件数を返す。

    `session.commit()` は呼ばない（呼び出し側の責務）。ingest 経路では
    結果取込みのトランザクション内で他の upsert と一緒に commit され、
    API 経路では各エンドポイントが個別に commit する。
    """
    pending = session.scalars(
        select(BetRecord).where(
            BetRecord.race_id == race_id,
            BetRecord.settled_at.is_(None),
        )
    ).all()

    count = 0
    for bet in pending:
        before = bet.settled_at
        settle_bet(session, bet)
        if bet.settled_at is not None and before is None:
            count += 1

    return count


def settle_all_pending(session: Session) -> int:
    """全未確定 bet を一括確定する。

    payouts / payout_win / payout_place のデータが存在する場合のみ確定する。
    確定できなかった bet（データ未着）はスキップされる。
    確定した件数を返す。
    """
    pending = session.scalars(
        select(BetRecord).where(BetRecord.settled_at.is_(None))
    ).all()

    count = 0
    for bet in pending:
        before = bet.settled_at
        settle_bet(session, bet)
        if bet.settled_at is not None and before is None:
            count += 1

    return count
