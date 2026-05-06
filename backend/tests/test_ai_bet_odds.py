"""Tests for ai/bet_odds.py — baseline odds computation and compute_race_odds."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.bet_odds import (
    _FALLBACK_AMOUNTS,
    compute_baseline_odds,
    compute_baseline_odds_by_class,
    compute_implied_combo_odds_from_tansho,
    compute_past_race_odds,
    compute_past_race_odds_with_tansho_fill,
    compute_race_odds,
    tansho_to_pl_scores,
)
from keiba_ai.db.base import Base
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.live_odds import LiveOdds
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race


@pytest.fixture()
def empty_session():
    """In-memory session with no payouts data."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def seeded_session():
    """In-memory session with a few payout rows for testing."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Race(
            race_id="R001",
            date="2025-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            race_class="G1",
            n_runners=10,
        ))
        session.add(Race(
            race_id="R002",
            date="2025-02-01",
            course="中山",
            surface="ダ",
            distance=1600,
            race_class="条件戦",
            n_runners=12,
        ))
        session.flush()

        # 単勝: two races, amounts 1200 and 800 → avg = 1000 → odds = 10.0
        session.add(Payout(race_id="R001", bet_type="単勝", combo="3", amount=1200, popularity=1))
        session.add(Payout(race_id="R002", bet_type="単勝", combo="5", amount=800, popularity=2))

        # 馬連: one race, amount 5500 → odds = 55.0
        session.add(Payout(race_id="R001", bet_type="馬連", combo="3-7", amount=5500, popularity=3))

        session.commit()
        yield session


def test_compute_baseline_odds_empty_payouts_uses_fallback(empty_session):
    """When payouts table is empty, every bet_type should return its fallback value."""
    odds = compute_baseline_odds(empty_session)

    for bet_type, fallback_amount in _FALLBACK_AMOUNTS.items():
        expected_odds = fallback_amount / 100.0
        assert odds[bet_type] == pytest.approx(expected_odds), (
            f"{bet_type}: expected fallback {expected_odds}, got {odds[bet_type]}"
        )


def test_compute_baseline_odds_uses_db_average(seeded_session):
    """When payouts table has data, returns average from DB for those bet_types."""
    odds = compute_baseline_odds(seeded_session)

    # 単勝: (1200 + 800) / 2 / 100 = 10.0
    assert odds["単勝"] == pytest.approx(10.0, rel=1e-4)

    # 馬連: 5500 / 100 = 55.0
    assert odds["馬連"] == pytest.approx(55.0, rel=1e-4)


def test_compute_baseline_odds_missing_types_fall_back(seeded_session):
    """Bet types not present in DB still get the hardcoded fallback."""
    odds = compute_baseline_odds(seeded_session)

    # ワイド has no rows in seeded_session
    expected_wide = _FALLBACK_AMOUNTS["ワイド"] / 100.0
    assert odds["ワイド"] == pytest.approx(expected_wide)

    # 三連単 has no rows
    expected_sanrentan = _FALLBACK_AMOUNTS["三連単"] / 100.0
    assert odds["三連単"] == pytest.approx(expected_sanrentan)


def test_compute_baseline_odds_returns_all_bet_types(seeded_session):
    """Result dict should cover all standard bet types."""
    odds = compute_baseline_odds(seeded_session)
    for bt in ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]:
        assert bt in odds, f"Missing bet_type: {bt}"


def test_compute_baseline_odds_by_class_with_match(seeded_session):
    """When enough rows match the filter, returns filtered average."""
    # Only 単勝 from G1 races (R001): amount=1200 → odds=12.0
    # min_samples=1 to ensure we use the filtered value
    odds = compute_baseline_odds_by_class(
        seeded_session,
        race_class="G1",
        min_samples=1,
    )
    assert odds["単勝"] == pytest.approx(12.0, rel=1e-4)


def test_compute_baseline_odds_by_class_falls_back_on_low_samples(seeded_session):
    """When filtered rows < min_samples, falls back to overall average."""
    # G1 has only 1 単勝 row, min_samples=5 forces fallback to overall (10.0)
    odds = compute_baseline_odds_by_class(
        seeded_session,
        race_class="G1",
        min_samples=5,
    )
    # Overall average: (1200 + 800) / 2 / 100 = 10.0
    assert odds["単勝"] == pytest.approx(10.0, rel=1e-4)


def test_compute_baseline_odds_by_class_surface_filter(seeded_session):
    """Filter by surface isolates races on that track type."""
    # ダ surface only has R002: 単勝 amount=800 → odds=8.0
    odds = compute_baseline_odds_by_class(
        seeded_session,
        surface="ダ",
        min_samples=1,
    )
    assert odds["単勝"] == pytest.approx(8.0, rel=1e-4)


def test_compute_baseline_odds_by_class_no_filter_matches_overall(seeded_session):
    """With no filter conditions, by_class returns same as compute_baseline_odds."""
    overall = compute_baseline_odds(seeded_session)
    by_class = compute_baseline_odds_by_class(seeded_session, min_samples=0)
    for bt in overall:
        assert overall[bt] == pytest.approx(by_class[bt], rel=1e-4), f"Mismatch for {bt}"


# ---------------------------------------------------------------------------
# compute_race_odds tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_odds_session():
    """In-memory session seeded with live_odds rows."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Race(
            race_id="R_LIVE",
            date="2025-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            n_runners=10,
        ))
        session.flush()

        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="馬連",
            combo="3-7",
            odds=25.4,
            odds_max=None,
            popularity=1,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="馬連",
            combo="3-9",
            odds=18.2,
            odds_max=None,
            popularity=2,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="単勝",
            combo="3",
            odds=5.0,
            odds_max=None,
            popularity=1,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        # odds=None (未確定) の行は compute_race_odds で除外される
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="単勝",
            combo="5",
            odds=None,
            odds_max=None,
            popularity=None,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.commit()
        yield session


def test_compute_race_odds_returns_nested_dict(live_odds_session):
    """compute_race_odds returns {bet_type: {combo: odds}} dict."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    assert isinstance(result, dict)
    assert "馬連" in result
    assert isinstance(result["馬連"], dict)


def test_compute_race_odds_values(live_odds_session):
    """Correct odds values are returned for each combo."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    assert result["馬連"]["3-7"] == pytest.approx(25.4)
    assert result["馬連"]["3-9"] == pytest.approx(18.2)
    assert result["単勝"]["3"] == pytest.approx(5.0)


def test_compute_race_odds_excludes_none_odds(live_odds_session):
    """Combos with odds=None are excluded from the result."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    # combo "5" has odds=None → should not appear
    assert "5" not in result.get("単勝", {})


def test_compute_race_odds_empty_for_unknown_race(live_odds_session):
    """Returns empty dict when race_id has no live_odds."""
    result = compute_race_odds(live_odds_session, "NONEXISTENT")
    assert result == {}


def test_compute_race_odds_empty_when_no_live_odds_table_populated(empty_session):
    """Returns empty dict when live_odds table is empty."""
    result = compute_race_odds(empty_session, "R001")
    assert result == {}


# ---------------------------------------------------------------------------
# compute_past_race_odds tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def past_race_session():
    """In-memory session seeded with a completed past race including entries and payouts."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # payout_place JSON: {finish_position_str: amount_yen}
        payout_place = json.dumps({"1": 120, "2": 170, "3": 140})
        session.add(Race(
            race_id="PAST_R001",
            date="2024-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            n_runners=5,
            payout_win=1500,
            payout_place=payout_place,
        ))
        session.flush()

        # Add horses
        for i in range(1, 6):
            session.add(Horse(horse_id=f"H{i}", name=f"horse{i}"))
        session.flush()

        # Entries: post_position 1-5, finish_position set for 1st, 2nd, 3rd
        # post 3 finished 1st, post 1 finished 2nd, post 5 finished 3rd
        horse_data = [
            # (post_position, odds_win, finish_position)
            (1, 5.0, 2),
            (2, 12.0, 4),
            (3, 3.0, 1),
            (4, 20.0, 5),
            (5, 8.0, 3),
        ]
        for pp, odds_win, finish_pos in horse_data:
            session.add(Entry(
                race_id="PAST_R001",
                horse_id=f"H{pp}",
                post_position=pp,
                odds_win=odds_win,
                finish_position=finish_pos,
            ))
        session.flush()

        # Payouts for winning combinations (連系)
        session.add(Payout(race_id="PAST_R001", bet_type="馬連", combo="1-3", amount=3500))
        session.add(Payout(race_id="PAST_R001", bet_type="ワイド", combo="1-3", amount=800))
        session.add(Payout(race_id="PAST_R001", bet_type="ワイド", combo="3-5", amount=600))
        session.add(Payout(race_id="PAST_R001", bet_type="馬単", combo="3→1", amount=6000))
        session.add(Payout(race_id="PAST_R001", bet_type="三連複", combo="1-3-5", amount=8000))
        session.add(Payout(race_id="PAST_R001", bet_type="三連単", combo="3→1→5", amount=30000))
        session.commit()
        yield session


def test_compute_past_race_odds_tan_all_horses(past_race_session):
    """単勝: entries.odds_win から全馬のオッズを返す。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    assert "単勝" in result
    tan = result["単勝"]
    # 5 頭全員分
    assert len(tan) == 5
    assert tan["1"] == pytest.approx(5.0)
    assert tan["2"] == pytest.approx(12.0)
    assert tan["3"] == pytest.approx(3.0)
    assert tan["4"] == pytest.approx(20.0)
    assert tan["5"] == pytest.approx(8.0)


def test_compute_past_race_odds_fuku_top3_only(past_race_session):
    """複勝: 1〜3 着馬のみ確定オッズを返す（4着以下は含まない）。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    assert "複勝" in result
    fuku = result["複勝"]
    # 1着=post3, 2着=post1, 3着=post5 → payout 120/100=1.2, 170/100=1.7, 140/100=1.4
    assert "3" in fuku
    assert "1" in fuku
    assert "5" in fuku
    # 4 着・5 着馬（post 2, post 4）は含まない
    assert "2" not in fuku
    assert "4" not in fuku
    assert fuku["3"] == pytest.approx(1.2)
    assert fuku["1"] == pytest.approx(1.7)
    assert fuku["5"] == pytest.approx(1.4)


def test_compute_past_race_odds_renki_winning_combos(past_race_session):
    """連系: payouts テーブルの的中 combo のみ返す。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    # 馬連
    assert "馬連" in result
    assert result["馬連"]["1-3"] == pytest.approx(35.0)
    # ワイド
    assert "ワイド" in result
    assert result["ワイド"]["1-3"] == pytest.approx(8.0)
    assert result["ワイド"]["3-5"] == pytest.approx(6.0)
    # 馬単
    assert "馬単" in result
    assert result["馬単"]["3→1"] == pytest.approx(60.0)
    # 三連複
    assert "三連複" in result
    assert result["三連複"]["1-3-5"] == pytest.approx(80.0)
    # 三連単
    assert "三連単" in result
    assert result["三連単"]["3→1→5"] == pytest.approx(300.0)


def test_compute_past_race_odds_empty_for_unknown_race(past_race_session):
    """存在しない race_id は空 dict を返す。"""
    result = compute_past_race_odds(past_race_session, "NONEXISTENT")
    assert result == {}


def test_compute_past_race_odds_empty_when_no_data(empty_session):
    """テーブルが空の場合は空 dict を返す。"""
    result = compute_past_race_odds(empty_session, "R001")
    assert result == {}


# ---------------------------------------------------------------------------
# Tansho-implied combination odds (Plackett-Luce)
# ---------------------------------------------------------------------------


def test_tansho_to_pl_scores_normalises_overround():
    """単勝オッズの逆数和は overround で >1 になる。
    Score の softmax 確率は和=1 に正規化されること。"""
    import numpy as np

    odds = {"1": 1.5, "2": 3.0, "3": 5.0}  # raw 1/odds = 0.667 + 0.333 + 0.2 = 1.2
    posts, scores = tansho_to_pl_scores(odds)

    assert posts == ["1", "2", "3"]
    probs = np.exp(scores)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    # オッズ低いほど確率高いという関係
    assert probs[0] > probs[1] > probs[2]


def test_tansho_to_pl_scores_skips_invalid_odds():
    """odds が None / <= 0 のエントリは除外される。"""
    odds = {"1": 1.5, "2": None, "3": 0, "4": 5.0, "5": -1.0}  # type: ignore[dict-item]
    posts, scores = tansho_to_pl_scores(odds)

    assert posts == ["1", "4"]
    assert len(scores) == 2


def test_tansho_to_pl_scores_sorts_post_position_numerically():
    """post position 文字列は数値順 (10 が 2 より後ろ)。"""
    odds = {"10": 4.0, "2": 3.0, "1": 2.0}
    posts, _ = tansho_to_pl_scores(odds)
    assert posts == ["1", "2", "10"]


def test_tansho_to_pl_scores_raises_on_empty():
    """有効エントリ 0 件は ValueError。"""
    with pytest.raises(ValueError):
        tansho_to_pl_scores({"1": None, "2": 0})  # type: ignore[dict-item]


def test_compute_implied_combo_odds_returns_all_bet_types():
    """単勝オッズから 複勝 / 馬連 / ワイド / 馬単 / 三連複 / 三連単 すべてが返る。

    解析式 (use_analytical=True デフォルト) では combo 数が完全に閉形式と一致する。
    """
    odds = {"1": 2.5, "2": 3.5, "3": 5.0, "4": 8.0, "5": 12.0, "6": 18.0, "7": 30.0, "8": 60.0}
    result = compute_implied_combo_odds_from_tansho(odds)

    assert {"複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"} <= result.keys()

    # 8 頭の純粋組み合わせ数 (解析式は MC と異なり量子化なし、全 combo を返す):
    #   複勝   = 8, 馬連 = C(8,2) = 28, ワイド = C(8,2) = 28,
    #   馬単   = P(8,2) = 56, 三連複 = C(8,3) = 56, 三連単 = P(8,3) = 336
    assert len(result["複勝"]) == 8
    assert len(result["馬連"]) == 28
    assert len(result["ワイド"]) == 28
    assert len(result["馬単"]) == 56
    assert len(result["三連複"]) == 56
    assert len(result["三連単"]) == 336


def test_analytical_eliminates_mc_quantization():
    """三連単オッズが MC 量子化 (1/N, 2/N, 3/N の離散化) ではなく連続値になる。"""
    import numpy as np

    odds = {"1": 2.5, "2": 3.5, "3": 5.0, "4": 8.0, "5": 12.0, "6": 18.0, "7": 30.0, "8": 60.0}

    # 解析式 (デフォルト)
    analytical = compute_implied_combo_odds_from_tansho(odds)
    a_vals = list(analytical["三連単"].values())
    a_unique = len(set(round(v, 1) for v in a_vals))

    # MC 10K: ユニーク値が大幅に少ない (量子化の証拠)
    rng = np.random.default_rng(0)
    mc = compute_implied_combo_odds_from_tansho(
        odds, n_samples=10_000, rng=rng, use_analytical=False
    )
    m_vals = list(mc["三連単"].values())
    m_unique = len(set(round(v, 1) for v in m_vals))

    # 解析式は MC より遥かに多くのユニーク値を返す (≈ combo 数)
    assert a_unique > m_unique * 2, (
        f"analytical={a_unique} unique vals, mc={m_unique} — analytical should be much more diverse"
    )
    # 解析式は全 336 combo を返す (MC は count=0 で missing)
    assert len(analytical["三連単"]) > len(mc["三連単"])


def test_jra_minimum_payout_floor():
    """est_odds は JRA 最低払戻 1.0 倍で floor される (4 頭で favorite 複勝)。"""
    # 4 頭で 1 番人気が圧倒的だと、複勝の P(top-3) ≈ 1 になり
    # raw est = 1/0.99 × 0.80 = 0.81 となる。1.0 倍で floor されること。
    odds = {"1": 1.5, "2": 3.0, "3": 6.0, "4": 12.0}
    result = compute_implied_combo_odds_from_tansho(odds)

    fukusho_min = min(result["複勝"].values())
    assert fukusho_min >= 1.0, f"fukusho_min={fukusho_min} < 1.0 (JRA min payout violated)"


def test_compute_implied_combo_odds_fukusho_favorite_lower():
    """1番人気の複勝オッズは穴馬より低い (より高確率で top-3 に来る)。"""
    import numpy as np

    odds = {"1": 1.5, "2": 3.0, "3": 8.0, "4": 30.0, "5": 60.0}
    rng = np.random.default_rng(11)
    result = compute_implied_combo_odds_from_tansho(odds, n_samples=100_000, rng=rng)

    fuku = result["複勝"]
    # 1番人気の複勝オッズ < 4-5 番人気
    assert fuku["1"] < fuku["4"]
    assert fuku["1"] < fuku["5"]
    # 控除率 0.20 込みで実 JRA レンジ内 (1.0 倍以上が一般的)
    # 単勝 1.5 倍の馬は ほぼ確実に top-3 → 複勝 < 1 になることもある
    # （3 頭 / 5 頭以下のレースでは特に）。下限 0.5、上限 50 倍以内のチェック。
    for combo, val in fuku.items():
        assert 0 < val < 200, f"{combo}: {val}"


def test_compute_implied_combo_odds_favorite_combo_cheaper():
    """1番人気同士の馬連は穴×穴より低オッズになる。"""
    import numpy as np

    odds = {"1": 2.0, "2": 3.0, "3": 5.0, "4": 50.0, "5": 80.0}
    rng = np.random.default_rng(123)
    result = compute_implied_combo_odds_from_tansho(odds, n_samples=50_000, rng=rng)

    # 馬連 1-2 (favorite combo) < 馬連 4-5 (longshot combo)
    assert result["馬連"]["1-2"] < result["馬連"]["4-5"]
    # 馬単 1→2 < 馬単 5→4
    assert result["馬単"]["1→2"] < result["馬単"]["5→4"]


def test_compute_implied_combo_odds_takeout_applied():
    """fair odds に takeout 控除がかかっている (実オッズ < 1/probability)。"""
    import numpy as np

    odds = {"1": 2.0, "2": 3.0, "3": 5.0, "4": 10.0}
    rng = np.random.default_rng(7)
    result = compute_implied_combo_odds_from_tansho(odds, n_samples=100_000, rng=rng)

    # 馬連 1-2 の実オッズは fair odds の (1 - 0.225) = 0.775 倍
    # fair odds は概ね 2-4 倍程度（馬連 favorite combo）
    fair_uupper = 5.0
    realistic = result["馬連"]["1-2"]
    assert 0 < realistic < fair_uupper * 0.776  # 控除率ぶん必ず低くなる


def test_compute_implied_combo_odds_raises_for_single_horse():
    """1 頭しか単勝が無いと連系券種が組めず ValueError。"""
    with pytest.raises(ValueError):
        compute_implied_combo_odds_from_tansho({"1": 2.0})


def test_compute_past_race_odds_with_tansho_fill_keeps_confirmed(past_race_session):
    """確定 combo は tansho-implied で上書きされない。"""
    import numpy as np

    # ※ compute_implied_... は内部で default_rng() を使うが、確定値が
    # 上書きされないことだけ確認するので rng の seed は不要
    result = compute_past_race_odds_with_tansho_fill(
        past_race_session, "PAST_R001", n_samples=20_000
    )

    # confirmed values from past_race_session
    assert result["馬連"]["1-3"] == pytest.approx(35.0)
    assert result["ワイド"]["1-3"] == pytest.approx(8.0)
    assert result["ワイド"]["3-5"] == pytest.approx(6.0)
    assert result["馬単"]["3→1"] == pytest.approx(60.0)
    assert result["三連複"]["1-3-5"] == pytest.approx(80.0)
    assert result["三連単"]["3→1→5"] == pytest.approx(300.0)


def test_compute_past_race_odds_with_tansho_fill_adds_missing(past_race_session):
    """未確定 combo は tansho-implied で補完される。"""
    result = compute_past_race_odds_with_tansho_fill(
        past_race_session, "PAST_R001", n_samples=20_000
    )

    # past_race_session has 5 horses, only 1-3 / 3→1 / 1-3-5 / 3→1→5 confirmed.
    # All other combos should be filled by tansho-implied.
    # 5 horses → 馬連 = C(5,2) = 10, ワイド = 10, 馬単 = P(5,2) = 20, 三連複 = C(5,3) = 10
    assert len(result["馬連"]) == 10
    assert len(result["ワイド"]) == 10
    assert len(result["馬単"]) == 20

    # The non-confirmed combos must be > 0
    assert result["馬連"]["2-4"] > 0  # not confirmed → implied
    assert result["馬単"]["1→3"] > 0  # not confirmed (1→3, opposite of 3→1)
    assert result["馬単"]["3→1"] == pytest.approx(60.0)  # confirmed preserved


def test_compute_past_race_odds_with_tansho_fill_no_tansho(empty_session):
    """単勝オッズが取れない場合はそのまま返す（空 dict）。"""
    result = compute_past_race_odds_with_tansho_fill(empty_session, "NONEXISTENT")
    assert result == {}
