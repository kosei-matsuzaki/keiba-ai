"""Parser for netkeiba's ``api_get_jra_odds.html`` JSON odds feed.

The endpoint returns *confirmed* (post-close) odds for **every** combination of a
race — not just the winning combo that the ``payouts`` table stores. One request
fetches one ``type``; the response groups results by bet-type code:

    type=1 -> groups "1" (単勝) + "2" (複勝)        # bundled
    type=3 -> "3" 枠連                              (these single-type calls
    type=4 -> "4" 馬連                               return only their own group)
    type=5 -> "5" ワイド
    type=6 -> "6" 馬単
    type=7 -> "7" 三連複
    type=8 -> "8" 三連単

Raw value per combo is ``[v1, v2, popularity]`` (all strings, big numbers carry a
thousands comma e.g. ``"4,002.6"``):
  - single-odds types (単勝/枠連/馬連/馬単/三連複/三連単): v1=odds, v2="0.0"
  - range types (複勝/ワイド): v1=min odds, v2=max odds

Combo keys are concatenated zero-padded 馬番 (枠番 for 枠連): ``"0102"`` / ``"010203"``.

:func:`parse_odds_payload` normalises everything into the **same combo string
format the rest of the app uses** (see ``compute_past_race_odds`` /
``predict_race_with_combinations``), so the ingested data drops straight into
``compute_race_odds_with_sources`` without further translation:

  - 単勝 / 複勝 / 枠連 単穴: ``"3"``
  - 馬連 / ワイド / 三連複: ``"3-7"`` / ``"3-5-7"``  (ascending, "-" 区切り)
  - 馬単 / 三連単:          ``"3→7"`` / ``"3→5→7"`` (着順そのまま, "→" 区切り)
  - 枠連:                   ``"1-2"``                (枠番昇順)
"""

from __future__ import annotations

# odds group code (= netkeiba ``type`` for single-type calls) -> 券種名
_GROUP_TO_BET_TYPE: dict[str, str] = {
    "1": "単勝",
    "2": "複勝",
    "3": "枠連",
    "4": "馬連",
    "5": "ワイド",
    "6": "馬単",
    "7": "三連複",
    "8": "三連単",
}

# 着順を保持する（ordered）券種。それ以外は昇順ソートして "-" で繋ぐ。
_ORDERED_BET_TYPES = frozenset({"馬単", "三連単"})


def _to_float(raw: str) -> float | None:
    """"4,002.6" / "7.2" -> float。"---.-" 等の未確定/非数値は None。"""
    if raw is None:
        return None
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _split_numbers(key: str) -> list[int] | None:
    """"0102" -> [1, 2], "010203" -> [1, 2, 3]。2 桁ずつに割る。

    桁数が奇数 / 非数値 / "00" を含む（無効馬番）場合は None。
    """
    if not key or len(key) % 2 != 0:
        return None
    try:
        nums = [int(key[i : i + 2]) for i in range(0, len(key), 2)]
    except ValueError:
        return None
    if any(n <= 0 for n in nums):
        return None
    return nums


def _format_combo(numbers: list[int], bet_type: str) -> str:
    """正規化された combo 文字列を返す（券種ごとに区切り/順序が異なる）。"""
    if len(numbers) == 1:
        return str(numbers[0])
    if bet_type in _ORDERED_BET_TYPES:
        return "→".join(str(n) for n in numbers)
    return "-".join(str(n) for n in sorted(numbers))


def parse_odds_payload(
    payload: dict,
) -> tuple[str | None, dict[str, dict[str, list[float | int]]]]:
    """netkeiba odds JSON (1 リクエスト分) を正規化する。

    Args:
        payload: ``json.loads`` 済みのレスポンス dict。

    Returns:
        ``(official_datetime, odds)``:
          - official_datetime: ``data.official_datetime`` 文字列（無ければ None）。
            確定スナップショットか判定するのに使う。
          - odds: ``{bet_type: {combo: [v1, v2, popularity]}}``。
            v1/v2 は float（range 券種は min/max、それ以外は v2=0.0）、
            popularity は int。status が "result" でない / odds が空なら ``{}``。
    """
    if not isinstance(payload, dict) or payload.get("status") != "result":
        return None, {}

    data = payload.get("data") or {}
    official_datetime = data.get("official_datetime")
    raw_odds = data.get("odds") or {}

    result: dict[str, dict[str, list[float | int]]] = {}
    for group_code, combos in raw_odds.items():
        bet_type = _GROUP_TO_BET_TYPE.get(str(group_code))
        if bet_type is None or not isinstance(combos, dict):
            continue

        parsed: dict[str, list[float | int]] = {}
        for raw_key, raw_val in combos.items():
            numbers = _split_numbers(str(raw_key))
            if numbers is None:
                continue
            if not isinstance(raw_val, (list, tuple)) or len(raw_val) < 1:
                continue

            v1 = _to_float(str(raw_val[0]))
            if v1 is None:  # 未確定/欠損 odds は捨てる（呼び出し側で combo 不在扱い）
                continue
            v2 = _to_float(str(raw_val[1])) if len(raw_val) > 1 else 0.0
            try:
                pop = int(str(raw_val[2]).replace(",", "")) if len(raw_val) > 2 else 0
            except ValueError:
                pop = 0

            parsed[_format_combo(numbers, bet_type)] = [v1, v2 or 0.0, pop]

        if parsed:
            result[bet_type] = parsed

    return official_datetime, result
