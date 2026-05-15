"""Date utilities for keiba-ai.

JST (Asia/Tokyo) を基準に「今週末」などの日付を計算する。
"""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")


def today_jst() -> date:
    """Return today's date in JST."""
    from datetime import datetime

    return datetime.now(_JST).date()


def this_weekend_dates(today: date | None = None) -> tuple[date, date]:
    """今日 (default: 当日 JST) を含む or 直近の Sat と Sun を返す。

    今日が:
    - 月〜金 → 今週末の Sat と Sun
    - 土 → 当日 (Sat) と 翌日 (Sun)
    - 日 → 前日 (Sat) と 当日 (Sun)

    JST タイムゾーンで日付計算。
    """
    if today is None:
        today = today_jst()

    weekday = today.weekday()  # 0=Mon … 5=Sat, 6=Sun

    if weekday == 5:  # Saturday
        sat = today
        sun = today + timedelta(days=1)
    elif weekday == 6:  # Sunday
        sat = today - timedelta(days=1)
        sun = today
    else:  # Mon(0) … Fri(4)
        days_until_sat = 5 - weekday
        sat = today + timedelta(days=days_until_sat)
        sun = sat + timedelta(days=1)

    return sat, sun
