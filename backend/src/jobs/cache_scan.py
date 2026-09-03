"""data/raw の HTML キャッシュを走査して、retro-fill の対象レースを選ぶ。

refill_payouts と refill_race_meta が同じ走査を別々に持っていて、**同じバグを
2 つ抱えていた**ので 1 本にした。

バグは「race_id を日付として parse する」こと。netkeiba の race_id は
年(4) + 競馬場(2) + 回(2) + 日(2) + R(2) の構造化文字列で、`race_id[4:6]` は
月ではなく **競馬場コード** である (CLAUDE.md)。競馬場は 01〜10、回次は 1〜6 なので
`datetime.date()` は例外を出さず、**静かに違う日付になる**。結果として --start /
--end を付けるとほとんどのレースが範囲外に落ちていた。

期間は必ず `races.date` で判定する。ここが唯一の走査経路なので、同じ間違いが
もう一度別の場所に生えることはない。
"""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.race import Race

# キャッシュのファイル名は <race_id>.html (12 桁数字)。
RACE_ID_RE = re.compile(r"^(\d{12})\.html$")

# SQLite の bind 変数上限 (古いビルドで 999) に触らない大きさ。
_ID_CHUNK = 500


def collect_cache_files(raw: Path) -> list[tuple[str, Path]]:
    """data/raw/<yyyy>/<mm>/<race_id>.html を列挙して (race_id, path) を返す。

    日付では絞らない。絞るのは `select_cached_races` の仕事で、判定には
    races.date を使う (モジュール冒頭の注意を参照)。
    """
    result: list[tuple[str, Path]] = []
    if not raw.exists():
        return result

    for yyyy_dir in sorted(raw.iterdir()):
        if not yyyy_dir.is_dir() or not yyyy_dir.name.isdigit():
            continue
        for mm_dir in sorted(yyyy_dir.iterdir()):
            if not mm_dir.is_dir() or not mm_dir.name.isdigit():
                continue
            for html_file in sorted(mm_dir.iterdir()):
                m = RACE_ID_RE.match(html_file.name)
                if m:
                    result.append((m.group(1), html_file))
    return result


def race_dates(session: Session, race_ids: list[str]) -> dict[str, str]:
    """race_id -> races.date。races に行が無い race_id はキーを持たない。

    1 件ずつ SELECT すると数万ファイルで効かないので、まとめて引く。
    """
    out: dict[str, str] = {}
    for i in range(0, len(race_ids), _ID_CHUNK):
        chunk = race_ids[i : i + _ID_CHUNK]
        rows = session.execute(
            select(Race.race_id, Race.date).where(Race.race_id.in_(chunk))
        ).all()
        for race_id, date in rows:
            if date is not None:
                out[race_id] = date
    return out


def select_cached_races(
    session: Session,
    raw: Path,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    limit: int | None = None,
) -> tuple[list[tuple[str, Path]], int]:
    """処理対象の (race_id, path) と、races 行が無くて落とした件数を返す。

    順序は「races にある → 期間で絞る → limit」。limit を先に掛けると、範囲外の
    ファイルで枠を使い切って 1 件も処理されないことがある。
    """
    files = collect_cache_files(raw)
    dates = race_dates(session, [race_id for race_id, _ in files])

    selected: list[tuple[str, Path]] = []
    skipped_no_race = 0
    lo = start.isoformat() if start else None
    hi = end.isoformat() if end else None
    for race_id, path in files:
        date = dates.get(race_id)
        if date is None:
            skipped_no_race += 1
            continue
        if (lo is not None and date < lo) or (hi is not None and date > hi):
            continue
        selected.append((race_id, path))

    if limit is not None:
        selected = selected[:limit]
    return selected, skipped_no_race


def add_range_args(parser: argparse.ArgumentParser) -> None:
    """--start / --end / --limit を足す。2 つの refill ジョブで同じもの。

    以前は片方のヘルプに "Filters by race_id date prefix" と書いてあり、しかも
    実装もそう動いていた (それがこのモジュール冒頭のバグ)。文言も 1 箇所にする。
    """
    parser.add_argument(
        "--start", default=None, metavar="YYYY-MM-DD",
        help="Start date (inclusive). Filters by races.date.",
    )
    parser.add_argument(
        "--end", default=None, metavar="YYYY-MM-DD",
        help="End date (inclusive). Filters by races.date.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Maximum number of races to process, applied after the date window (debug use).",
    )
