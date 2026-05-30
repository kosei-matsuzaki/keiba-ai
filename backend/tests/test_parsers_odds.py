"""Tests for scraper.parsers.odds.parse_odds_payload."""

from __future__ import annotations

from scraper.parsers.odds import parse_odds_payload


def _payload(odds: dict, *, status: str = "result", dt: str = "2025-12-14 11:23:52") -> dict:
    return {"status": status, "data": {"official_datetime": dt, "odds": odds}}


def test_tansho_fukusho_bundled() -> None:
    # type=1 returns groups "1" (単勝) + "2" (複勝 range).
    payload = _payload(
        {
            "1": {"01": ["8.6", "0.0", "4"], "07": ["1.9", "0.0", "1"]},
            "2": {"01": ["2.1", "3.5", "6"], "07": ["1.1", "1.3", "1"]},
        }
    )
    dt, odds = parse_odds_payload(payload)
    assert dt == "2025-12-14 11:23:52"
    assert odds["単勝"]["1"] == [8.6, 0.0, 4]
    assert odds["単勝"]["7"] == [1.9, 0.0, 1]
    # 複勝 keeps the min/max range.
    assert odds["複勝"]["1"] == [2.1, 3.5, 6]
    assert odds["複勝"]["7"] == [1.1, 1.3, 1]


def test_umaren_unordered_sorted() -> None:
    # 馬連 (type=4): unordered -> ascending "-" combo.
    payload = _payload({"4": {"0102": ["22.7", "0.0", "8"], "0201": ["22.7", "0.0", "8"]}})
    _, odds = parse_odds_payload(payload)
    # both map to the same canonical "1-2" key.
    assert odds["馬連"] == {"1-2": [22.7, 0.0, 8]}


def test_umatan_ordered_preserved() -> None:
    # 馬単 (type=6): ordered -> "→" preserves finish order.
    payload = _payload({"6": {"0102": ["49.6", "0.0", "19"], "0201": ["55.0", "0.0", "22"]}})
    _, odds = parse_odds_payload(payload)
    assert odds["馬単"]["1→2"] == [49.6, 0.0, 19]
    assert odds["馬単"]["2→1"] == [55.0, 0.0, 22]


def test_wide_is_range() -> None:
    payload = _payload({"5": {"0102": ["7.2", "8.6", "8"]}})
    _, odds = parse_odds_payload(payload)
    assert odds["ワイド"]["1-2"] == [7.2, 8.6, 8]


def test_sanrentan_comma_and_order() -> None:
    # 三連単 (type=8): ordered triple, big number with thousands comma.
    payload = _payload({"8": {"010203": ["2,378.3", "0.0", "439"]}})
    _, odds = parse_odds_payload(payload)
    assert odds["三連単"]["1→2→3"] == [2378.3, 0.0, 439]


def test_sanrenpuku_sorted() -> None:
    payload = _payload({"7": {"030102": ["697.1", "0.0", "115"]}})
    _, odds = parse_odds_payload(payload)
    assert odds["三連複"] == {"1-2-3": [697.1, 0.0, 115]}


def test_non_result_status_returns_empty() -> None:
    dt, odds = parse_odds_payload(_payload({"4": {"0102": ["1.0", "0.0", "1"]}}, status="error"))
    assert dt is None
    assert odds == {}


def test_placeholder_odds_skipped() -> None:
    # Unconfirmed cells ("---.-") must not become combos.
    payload = _payload({"4": {"0102": ["---.-", "0.0", "0"], "0103": ["5.0", "0.0", "2"]}})
    _, odds = parse_odds_payload(payload)
    assert odds["馬連"] == {"1-3": [5.0, 0.0, 2]}


def test_invalid_combo_keys_skipped() -> None:
    # odd-length key ("012") and a zero 馬番 ("0002") are both rejected.
    payload = _payload(
        {"4": {"012": ["3.0", "0.0", "1"], "0002": ["9.0", "0.0", "3"], "0102": ["4.0", "0.0", "2"]}}
    )
    _, odds = parse_odds_payload(payload)
    assert odds["馬連"] == {"1-2": [4.0, 0.0, 2]}
