"""Unit tests for bet_settlement service."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from db.models.bet_record import BetRecord
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.payout import Payout
from db.models.race import Race
from services.bet_settlement import (
    settle_all_pending,
    settle_bet,
    settle_bets_for_race,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _add_race(
    session: Session,
    race_id: str = "202406010101",
    payout_win: int | None = None,
    payout_place: dict | None = None,
) -> Race:
    race = Race(
        race_id=race_id,
        date="2024-06-01",
        course="東京",
        surface="芝",
        distance=2400,
        payout_win=payout_win,
        payout_place=json.dumps(payout_place) if payout_place is not None else None,
    )
    session.add(race)
    session.flush()
    return race


def _add_payout(
    session: Session,
    race_id: str,
    bet_type: str,
    combo: str,
    amount: int,
) -> Payout:
    p = Payout(race_id=race_id, bet_type=bet_type, combo=combo, amount=amount, popularity=1)
    session.add(p)
    session.flush()
    return p


def _add_bet(
    session: Session,
    race_id: str = "202406010101",
    bet_type: str = "単勝",
    combo: str = "5",
    stake: int = 1000,
    settled_at: str | None = None,
) -> BetRecord:
    bet = BetRecord(
        created_at="2024-06-01T10:00:00+00:00",
        race_id=race_id,
        bet_type=bet_type,
        combo=combo,
        stake=stake,
        source="manual",
        settled_at=settled_at,
        payout=None,
        profit=None,
    )
    session.add(bet)
    session.flush()
    return bet


def _add_horse(session: Session, horse_id: str) -> Horse:
    horse = Horse(horse_id=horse_id, name=f"horse_{horse_id}")
    session.add(horse)
    session.flush()
    return horse


def _add_entry(
    session: Session,
    race_id: str = "202406010101",
    horse_id: str = "h001",
    post_position: int | None = None,
    finish_position: int | None = None,
) -> Entry:
    entry = Entry(
        race_id=race_id,
        horse_id=horse_id,
        post_position=post_position,
        finish_position=finish_position,
    )
    session.add(entry)
    session.flush()
    return entry


# ── settle_bet: payouts テーブル経由 ─────────────────────────────────────────

class TestSettleBetViaPayoutsTable:
    def test_tansho_win(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=1000)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 2800   # 280 * 1000 / 100
        assert bet.profit == 1800

    def test_tansho_lose(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        bet = _add_bet(db_session, bet_type="単勝", combo="3", stake=500)  # combo 不一致→外れ

        settle_bet(db_session, bet)

        # payouts は ingest 済み（combo="5" が存在）。combo="3" はヒットしない → 外れ確定（payout=0）
        assert bet.settled_at is not None
        assert bet.payout == 0
        assert bet.profit == -500

    def test_fukusho_win(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "複勝", "3", 150)
        bet = _add_bet(db_session, bet_type="複勝", combo="3", stake=500)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 750    # 150 * 500 / 100

    def test_wakuren(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "枠連", "3-7", 1200)
        bet = _add_bet(db_session, bet_type="枠連", combo="3-7", stake=200)

        settle_bet(db_session, bet)

        assert bet.payout == 2400
        assert bet.profit == 2200

    def test_umaren(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "馬連", "3-7", 2500)
        bet = _add_bet(db_session, bet_type="馬連", combo="3-7", stake=100)

        settle_bet(db_session, bet)

        assert bet.payout == 2500
        assert bet.profit == 2400

    def test_wide(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "ワイド", "3-7", 600)
        bet = _add_bet(db_session, bet_type="ワイド", combo="3-7", stake=300)

        settle_bet(db_session, bet)

        assert bet.payout == 1800

    def test_umatan(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "馬単", "7-3", 8000)
        bet = _add_bet(db_session, bet_type="馬単", combo="7-3", stake=100)

        settle_bet(db_session, bet)

        assert bet.payout == 8000

    def test_sanrenpuku(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "三連複", "3-7-12", 15000)
        bet = _add_bet(db_session, bet_type="三連複", combo="3-7-12", stake=100)

        settle_bet(db_session, bet)

        assert bet.payout == 15000

    def test_sanrentan(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "三連単", "7-3-12", 120000)
        bet = _add_bet(db_session, bet_type="三連単", combo="7-3-12", stake=100)

        settle_bet(db_session, bet)

        assert bet.payout == 120000

    def test_all_bet_types_lose(self, db_session):
        """payouts に一致なし → payouts が ingest 済みなので全て外れ確定（payout=0）。"""
        _add_race(db_session)
        bet_types_combos = [
            ("単勝", "1"),
            ("複勝", "1"),
            ("枠連", "1-2"),
            ("馬連", "1-2"),
            ("ワイド", "1-2"),
            ("馬単", "1-2"),
            ("三連複", "1-2-3"),
            ("三連単", "1-2-3"),
        ]
        # payouts は「5」系のみ登録しておく（= payouts は ingest 済み）
        _add_payout(db_session, "202406010101", "単勝", "5", 280)

        for bt, combo in bet_types_combos:
            bet = _add_bet(db_session, bet_type=bt, combo=combo, stake=100)
            settle_bet(db_session, bet)
            # payouts が ingest 済みで該当 combo が無い → 外れ確定（payout=0、settled_at が設定される）
            assert bet.settled_at is not None, f"{bt}/{combo} should be settled as loss"
            assert bet.payout == 0, f"{bt}/{combo} should have payout=0"
            assert bet.profit == -100, f"{bt}/{combo} should have profit=-stake"


# ── 既に settled なレコードは再確定されない ───────────────────────────────────

class TestAlreadySettled:
    def test_already_settled_is_not_overwritten(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        bet = _add_bet(
            db_session,
            settled_at="2024-06-01T12:00:00+00:00",
        )
        bet.payout = 9999  # 既存の値をダミー設定
        bet.profit = 8999

        settle_bet(db_session, bet)

        # settled_at が変わらず、payout も書き換えられない
        assert bet.settled_at == "2024-06-01T12:00:00+00:00"
        assert bet.payout == 9999


# ── payout_win / payout_place fallback ───────────────────────────────────────

class TestFallback:
    def test_tansho_fallback_payout_win(self, db_session):
        """payouts テーブルが空でも、combo が 1 着馬と一致すれば races.payout_win で単勝が確定できる。"""
        _add_race(db_session, payout_win=430)
        _add_horse(db_session, "h003")
        _add_entry(db_session, post_position=3, finish_position=1, horse_id="h003")
        bet = _add_bet(db_session, bet_type="単勝", combo="3", stake=200)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 860    # 430 * 200 / 100

    def test_fukusho_fallback_payout_place(self, db_session):
        """payouts テーブルが空でも、races.payout_place JSON（キー=finish_position）で複勝が確定できる。

        payout_place のキーは finish_position (1/2/3)、値はその着順の馬の払戻。
        combo は post_position 馬番。entries から finish_position を引いてから JSON を参照する。
        """
        # payout_place キー = finish_position。1着=110円、2着=180円、3着=150円
        _add_race(db_session, payout_place={"1": 110, "2": 180, "3": 150})
        # 馬番7（post_position=7）が 2 着（finish_position=2）
        _add_horse(db_session, "h007")
        _add_entry(db_session, post_position=7, finish_position=2, horse_id="h007")
        bet = _add_bet(db_session, bet_type="複勝", combo="7", stake=500)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 900    # 180 * 500 / 100

    def test_fukusho_fallback_combo_not_in_entries(self, db_session):
        """combo の post_position に対応する entries が無い場合は確定不能（settled_at=None）。"""
        _add_race(db_session, payout_place={"1": 150, "2": 180})
        # post_position=12 の entry が存在しない → finish_position を解決できない
        bet = _add_bet(db_session, bet_type="複勝", combo="12", stake=100)

        settle_bet(db_session, bet)

        assert bet.settled_at is None

    def test_payouts_table_takes_priority_over_payout_win(self, db_session):
        """payouts テーブルにヒットした場合は payout_win を無視する。"""
        _add_race(db_session, payout_win=999)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=100)

        settle_bet(db_session, bet)

        # payouts.amount=280 を優先、payout_win=999 は無視
        assert bet.payout == 280

    def test_tansho_fallback_no_payout_win(self, db_session):
        """payout_win が NULL かつ payouts 空の場合は未確定。"""
        _add_race(db_session, payout_win=None)
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=100)

        settle_bet(db_session, bet)

        assert bet.settled_at is None


# ── settle_bets_for_race ──────────────────────────────────────────────────────

class TestSettleBetsForRace:
    def test_settles_all_pending_for_race(self, db_session):
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        _add_payout(db_session, "202406010101", "複勝", "3", 150)

        _add_bet(db_session, bet_type="単勝", combo="5")
        _add_bet(db_session, bet_type="複勝", combo="3")

        count = settle_bets_for_race(db_session, "202406010101")
        assert count == 2

    def test_skips_already_settled(self, db_session):
        _add_race(db_session)
        # payouts は登録しない（未 ingest）。payout_win / payout_place も None。
        # → どちらの bet も fallback で確定不能（None）になる。

        _add_bet(db_session, settled_at="2024-06-01T12:00:00+00:00")  # 既に確定
        _add_bet(db_session, bet_type="複勝", combo="3")               # 未確定（payouts/fallback なし）

        count = settle_bets_for_race(db_session, "202406010101")
        assert count == 0  # settled 済み bet はスキップ、複勝は確定不能

    def test_returns_zero_for_unknown_race(self, db_session):
        _add_race(db_session)
        count = settle_bets_for_race(db_session, "999999999999")
        assert count == 0


# ── settle_all_pending ────────────────────────────────────────────────────────

class TestSettleAllPending:
    def test_settles_multiple_races(self, db_session):
        for race_num, win_combo in (("202406010101", "5"), ("202406010201", "3")):
            _add_race(db_session, race_id=race_num)
            _add_payout(db_session, race_num, "単勝", win_combo, 280)
            _add_bet(db_session, race_id=race_num, bet_type="単勝", combo=win_combo)

        count = settle_all_pending(db_session)
        assert count == 2

    def test_empty_pending(self, db_session):
        count = settle_all_pending(db_session)
        assert count == 0

    def test_stake_fractional(self, db_session):
        """50 円の stake でも比例計算が正しく動作する。"""
        _add_race(db_session)
        _add_payout(db_session, "202406010101", "単勝", "5", 280)
        bet = _add_bet(db_session, stake=50)

        settle_all_pending(db_session)

        assert bet.payout == 140    # 280 * 50 / 100
        assert bet.profit == 90


# ── 新規: payouts 存在判定 + 検証つき fallback ─────────────────────────────────

class TestPayoutsExistenceAndFallbackValidation:
    def test_payouts_ingest_miss_combo_settles_as_loss(self, db_session):
        """payouts が ingest 済みで該当 combo が無い → 外れ確定（payout=0）。"""
        _add_race(db_session)
        # race_id に payouts が存在する（= ingest 済み）が combo="5" のみ
        _add_payout(db_session, "202406010101", "単勝", "3", 180)
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=300)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 0
        assert bet.profit == -300

    def test_tansho_fallback_winner_mismatch_is_loss(self, db_session):
        """単勝 fallback: 1 着馬と一致しない combo → payout=0（外れ）。"""
        _add_race(db_session, payout_win=180)
        # 1 着馬は post_position=3
        _add_horse(db_session, "h003")
        _add_entry(db_session, post_position=3, finish_position=1, horse_id="h003")
        # ユーザは馬番5にベット
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=500)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 0
        assert bet.profit == -500

    def test_tansho_fallback_winner_match_pays_out(self, db_session):
        """単勝 fallback: 1 着馬と一致する combo → payout_win 値が返る。"""
        _add_race(db_session, payout_win=250)
        _add_horse(db_session, "h008")
        _add_entry(db_session, post_position=8, finish_position=1, horse_id="h008")
        bet = _add_bet(db_session, bet_type="単勝", combo="8", stake=400)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 1000   # 250 * 400 / 100

    def test_fukusho_fallback_fourth_place_is_loss(self, db_session):
        """複勝 fallback: 4 着馬のベット → 外れ（payout=0）。"""
        _add_race(db_session, payout_place={"1": 110, "2": 140, "3": 160})
        _add_horse(db_session, "h006")
        _add_entry(db_session, post_position=6, finish_position=4, horse_id="h006")
        bet = _add_bet(db_session, bet_type="複勝", combo="6", stake=200)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 0
        assert bet.profit == -200

    def test_fukusho_fallback_second_place_pays_out(self, db_session):
        """複勝 fallback: 2 着馬のベット → payout_place JSON の "2" の値が返る。"""
        _add_race(db_session, payout_place={"1": 110, "2": 160, "3": 130})
        _add_horse(db_session, "h010")
        _add_entry(db_session, post_position=10, finish_position=2, horse_id="h010")
        bet = _add_bet(db_session, bet_type="複勝", combo="10", stake=500)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 800    # 160 * 500 / 100

    def test_fukusho_fallback_payout_place_keys_are_str_finish_positions(self, db_session):
        """payout_place JSON のキーは str 型の finish_position (1/2/3) であることを確認する。"""
        # JSON キーは文字列 "1", "2", "3"（整数キーは JSON 仕様上ありえないが、str で引けることを明示）
        _add_race(db_session, payout_place={"1": 120, "2": 150, "3": 180})
        _add_horse(db_session, "h001")
        _add_entry(db_session, post_position=1, finish_position=3, horse_id="h001")
        bet = _add_bet(db_session, bet_type="複勝", combo="1", stake=100)

        settle_bet(db_session, bet)

        assert bet.settled_at is not None
        assert bet.payout == 180    # 180 * 100 / 100（3着の払戻）

    def test_no_payouts_no_payout_win_remains_unsettled(self, db_session):
        """payouts も payout_win も無い（結果未着またはデータ欠損）→ 確定されない。"""
        _add_race(db_session, payout_win=None, payout_place=None)
        bet = _add_bet(db_session, bet_type="単勝", combo="5", stake=100)

        settle_bet(db_session, bet)

        assert bet.settled_at is None
        assert bet.payout is None
        assert bet.profit is None
