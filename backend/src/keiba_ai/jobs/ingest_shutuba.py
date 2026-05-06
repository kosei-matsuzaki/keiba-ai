"""Shutuba (出馬表) ingest job — registers upcoming races before they run.

Usage:
    uv run keiba-ingest-shutuba --date 2025-05-05
    uv run keiba-ingest-shutuba --date 2025-05-05 --limit 3
    uv run keiba-ingest-shutuba --race-ids 202506050911,202506050912
    uv run keiba-ingest-shutuba --date 2025-05-05 --race-ids 202506050911
    python -m keiba_ai.jobs.ingest_shutuba --date 2025-05-05

フロー (--date のみ):
  1. race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD から当日 race_id 一覧取得
  2. 各 race_id について shutuba page を fetch してパース
  3. races / entries テーブルに upsert

フロー (--race-ids 指定時):
  calendar fetch を skip して与えられた race_id 群について直接 shutuba HTML を取得して ingest する。
  --date も指定できるが calendar fetch には使わず、DB に保存する date 値として使う。
  両方指定された場合は --race-ids 優先で calendar fetch は行わない。

  ⚠ calendar 取得の現状:
    race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD は AJAX で race_id を取得する
    仕様のため、静的 HTML には race_id が含まれない。
    サーバ側 API (/api/api_get_jra_digest2.html) が空レスポンスを返す場合があり、
    calendar 経由の自動取得が不安定。--race-ids で直接指定することを推奨する。

upsert ポリシー:
  - races: race row が存在しない場合のみ INSERT。
    既存 row (finish_position 確定済み) があっても race メタ情報 (surface/distance) を
    shutuba で上書きする必要はほぼないため、INSERT OR IGNORE 相当を採用。
    ただし n_runners は最新値で更新する（除外馬が出た場合などに対応）。
  - entries: finish_position が NULL の entry は最新の odds_win/popularity で上書き。
    finish_position が既に入っている entry は skip（結果確定済み）。
  - 新規 entry は INSERT。
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os

import httpx
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from keiba_ai.core.config import load_settings
from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.base import Base
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.jockey import Jockey
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.models.trainer import Trainer
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.scraper import cache as cache_module
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.parsers.race_card_calendar import parse_race_ids_from_card_calendar
from keiba_ai.scraper.parsers.shutuba import ParsedShutuba, ShutubaEntry, parse_shutuba
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache
from keiba_ai.scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_CARD_CALENDAR_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"
_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def _already_scraped(session: Session, url: str) -> bool:
    row = session.execute(
        select(ScrapeLog).where(ScrapeLog.url == url, ScrapeLog.status == "ok").limit(1)
    ).first()
    return row is not None


def _record_scrape_log(
    session: Session,
    url: str,
    status: str,
    content_hash: str | None = None,
) -> None:
    fetched_at = datetime.datetime.now(datetime.UTC).isoformat()
    session.add(ScrapeLog(url=url, fetched_at=fetched_at, status=status, content_hash=content_hash))


def _upsert_race_from_shutuba(session: Session, result: ParsedShutuba) -> None:
    """Upsert race row from shutuba data.

    既存の race row がある場合は n_runners のみ更新する（除外馬対応）。
    payout_win / payout_place / track_condition / race_class は
    shutuba では取得できないため、既存値を保持する。
    race row が存在しない場合は新規 INSERT する。
    """
    stmt = sqlite_insert(Race).values(
        race_id=result.race_id,
        # date は初回登録値を保持する設計（on_conflict 側の set_ に含めない）。
        # CLI 経由では --date で必ず埋まるが、万一 "" になっても shutuba ingest で
        # 上書きされないため、偶発的な空文字 row が生じても既存値を破壊しない。
        date=result.date or "",
        course=result.course or "",
        surface=result.surface or "",
        distance=result.distance or 0,
        weather=result.weather,
        track_condition=None,
        race_class=result.race_class,
        name=result.name,
        n_runners=result.n_runners,
        payout_win=None,
        payout_place=None,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["race_id"],
        set_={
            # shutuba で確定できるフィールドのみ更新
            "n_runners": stmt.excluded.n_runners,
            # name / race_class は COALESCE で既存値を保護
            "name": sa.func.coalesce(stmt.excluded.name, Race.name),
            "race_class": sa.func.coalesce(stmt.excluded.race_class, Race.race_class),
            # 既存の確定済みデータ (payout / track_condition) は保持
            # date / course / surface / distance は初回登録値を尊重し上書きしない
        },
    )
    session.execute(stmt)


def _upsert_masters_from_shutuba(session: Session, result: ParsedShutuba) -> None:
    """Upsert horses, jockeys, trainers from shutuba entries.

    出馬表段階では horse_detail / pedigree の fetch は行わない（最小限 ingest）。
    既存の name / sire / dam は COALESCE で保持する。
    """
    horses_seen: set[str] = set()
    jockeys_seen: set[str] = set()
    trainers_seen: set[str] = set()

    for e in result.entries:
        if e.horse_id and e.horse_id not in horses_seen:
            horses_seen.add(e.horse_id)
            stmt = sqlite_insert(Horse).values(horse_id=e.horse_id, name=e.horse_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=["horse_id"],
                set_={"name": sa.func.coalesce(stmt.excluded.name, Horse.name)},
            )
            session.execute(stmt)

        if e.jockey_id and e.jockey_id not in jockeys_seen:
            stmt = sqlite_insert(Jockey).values(jockey_id=e.jockey_id, name=e.jockey_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=["jockey_id"],
                set_={"name": sa.func.coalesce(stmt.excluded.name, Jockey.name)},
            )
            session.execute(stmt)
            jockeys_seen.add(e.jockey_id)

        if e.trainer_id and e.trainer_id not in trainers_seen:
            stmt = sqlite_insert(Trainer).values(trainer_id=e.trainer_id, name=e.trainer_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=["trainer_id"],
                set_={"name": sa.func.coalesce(stmt.excluded.name, Trainer.name)},
            )
            session.execute(stmt)
            trainers_seen.add(e.trainer_id)


def _upsert_entry_from_shutuba(session: Session, e: ShutubaEntry) -> None:
    """Upsert one entry row.

    shutuba ingest は finish_position が NULL の entry にのみ書き込む。
    - finish_position が NULL の entry → 全 shutuba カラムを最新値で上書き
    - finish_position が確定済みの entry → どの shutuba カラムも触らない
    - 新規 entry → INSERT

    結果データ (finish_position / finish_time / margin / agari_3f / passing) は
    shutuba では取得できないため set_ に含めない。
    """
    stmt = sqlite_insert(Entry).values(
        race_id=e.race_id,
        horse_id=e.horse_id,
        post_position=e.post_position,
        jockey_id=e.jockey_id,
        trainer_id=e.trainer_id,
        weight_carried=e.weight_carried,
        age=e.age,
        sex=e.sex,
        horse_weight=e.horse_weight,
        horse_weight_diff=e.horse_weight_diff,
        odds_win=e.odds_win,
        popularity=e.popularity,
        finish_position=None,
        finish_time=None,
        margin=None,
        agari_3f=None,
        passing=None,
    )
    # finish_position が NULL (レース前) の entry だけ shutuba 由来カラムを更新する。
    # finish_position が確定済みの場合は全カラムを既存値で据え置く（結果データ保護）。
    # SQLAlchemy sqlite_insert の on_conflict_do_update は WHERE をサポートしないため
    # sa.case() で既存 finish_position の NULL チェックを行う。
    def _shutuba_case(new_val, existing_col):
        """finish_position が NULL のときのみ新値を採用するヘルパー。"""
        return sa.case(
            (Entry.finish_position.is_(None), new_val),
            else_=existing_col,
        )

    stmt = stmt.on_conflict_do_update(
        index_elements=["race_id", "horse_id"],
        set_={
            "post_position": _shutuba_case(stmt.excluded.post_position, Entry.post_position),
            "odds_win": _shutuba_case(stmt.excluded.odds_win, Entry.odds_win),
            "popularity": _shutuba_case(stmt.excluded.popularity, Entry.popularity),
            # 馬体重・騎手・斤量も shutuba 最新値で更新（出走取消・乗り替わり対応）
            "horse_weight": _shutuba_case(stmt.excluded.horse_weight, Entry.horse_weight),
            "horse_weight_diff": _shutuba_case(stmt.excluded.horse_weight_diff, Entry.horse_weight_diff),
            "jockey_id": _shutuba_case(stmt.excluded.jockey_id, Entry.jockey_id),
            "weight_carried": _shutuba_case(stmt.excluded.weight_carried, Entry.weight_carried),
        },
    )
    session.execute(stmt)


async def _ingest_race_ids(
    race_ids: list[str],
    date_str: str | None,
    client: NetkeibaClient,
    session: Session,
    limit: int | None = None,
) -> dict[str, int]:
    """race_id リストを元に shutuba ingest を実行する。

    --race-ids と --date 両方の ingest フローから呼ばれる共通ロジック。
    date_str は HTML から日付が取得できない場合の fallback のみに使う。
    """
    counters = {"fetched": 0, "skipped": 0, "errors": 0}

    if limit is not None:
        race_ids = race_ids[:limit]
        logger.info("Limiting to %d races (--limit)", limit)

    for race_id in race_ids:
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during shutuba race loop")

        shutuba_url = _SHUTUBA_URL.format(race_id=race_id)

        # shutuba はオッズが変動するので、常に最新を取得する（キャッシュ短め or 無効）
        # scrape_log の "ok" チェックは skip しない（再実行でオッズを更新するため）
        try:
            html = await client.fetch(shutuba_url, cache_max_age_hours=1)
            parsed = parse_shutuba(html, race_id)
            # HTML から抽出した date を最優先する。
            # HTML に date が無い場合のみ date_str を fallback として使う。
            # どちらも無い場合はこのレースをスキップしてエラー扱いにする
            # （空文字の date row が無限に溜まるのを防ぐ）。
            if not parsed.date:
                if date_str:
                    parsed.date = date_str  # HTML 解析失敗時の fallback
                else:
                    logger.warning(
                        "Shutuba HTML lacks date for race %s; skipping", race_id
                    )
                    _record_scrape_log(session, shutuba_url, "error")
                    session.commit()
                    counters["errors"] += 1
                    continue

            _upsert_race_from_shutuba(session, parsed)
            _upsert_masters_from_shutuba(session, parsed)

            for e in parsed.entries:
                _upsert_entry_from_shutuba(session, e)

            _record_scrape_log(session, shutuba_url, "ok", cache_module.content_hash(html))
            session.commit()

            counters["fetched"] += 1
            logger.info("Ingested shutuba race %s (%d entries)", race_id, len(parsed.entries))

        except ScraperStopped:
            raise
        except Exception as exc:
            logger.error("Error ingesting shutuba race %s: %s", race_id, exc)
            session.rollback()
            try:
                _record_scrape_log(session, shutuba_url, "error")
                session.commit()
            except Exception:
                session.rollback()
            counters["errors"] += 1

    return counters


async def run_ingest_shutuba(
    date_str: str | None,
    client: NetkeibaClient,
    session: Session,
    limit: int | None = None,
    race_ids: list[str] | None = None,
) -> dict[str, int]:
    """Core shutuba ingest logic; returns summary counters.

    Args:
        date_str: Race date (YYYY-MM-DD) used as fallback when the HTML lacks a date.
            None は許容される — HTML から日付が取れる場合は使われない。
            calendar fetch には必須（race_ids 未指定時）。
        client: NetkeibaClient instance.
        session: SQLAlchemy Session.
        limit: Max number of races to fetch (debug use).
        race_ids: If provided, skip calendar fetch and ingest only these race IDs.
            --race-ids CLI フラグと対応する。calendar 取得が壊れている場合の回避策として使う。
    """
    if race_ids is not None:
        # --race-ids 指定時: calendar fetch を skip して直接 ingest
        logger.info(
            "Ingesting %d race(s) from --race-ids (calendar fetch skipped)", len(race_ids)
        )
        return await _ingest_race_ids(race_ids, date_str, client, session, limit=limit)

    # calendar 経由で race_id 一覧を取得（date_str が必須）
    if not date_str:
        raise ValueError("date_str is required for calendar-based shutuba ingest")
    card_calendar_url = _CARD_CALENDAR_URL.format(date=date_str.replace("-", ""))
    logger.info("Fetching race-card calendar: %s", card_calendar_url)
    # 当日オッズは変動するため、shutuba ページは短いキャッシュ TTL (1 時間) を使う
    calendar_html = await client.fetch(card_calendar_url, cache_max_age_hours=1)
    include_nar = os.getenv("KEIBA_INCLUDE_NAR", "0") == "1"
    fetched_race_ids = parse_race_ids_from_card_calendar(calendar_html, include_nar=include_nar)

    logger.info("Found %d race IDs for shutuba ingest on %s", len(fetched_race_ids), date_str)

    return await _ingest_race_ids(fetched_race_ids, date_str, client, session, limit=limit)


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)

    rate_limiter = AsyncRateLimiter(load_settings())
    robots_cache = RobotsCache(load_settings().user_agent)

    # --race-ids が指定されている場合は --date を省略可能にするため、
    # date のデフォルトを今日の日付にする。
    date_str: str = args.date or datetime.date.today().isoformat()

    race_ids: list[str] | None = None
    if args.race_ids:
        race_ids = [rid.strip() for rid in args.race_ids.split(",") if rid.strip()]
        if not race_ids:
            logger.error("--race-ids is empty after splitting; aborting")
            return 1

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, load_settings())
        with session_scope(engine) as session:
            try:
                counters = await run_ingest_shutuba(
                    date_str, client, session, limit=args.limit, race_ids=race_ids
                )
            except ScraperStopped:
                logger.warning("Scraper stopped by stop flag")
                return 1

    logger.info(
        "Shutuba ingest complete — fetched=%d skipped=%d errors=%d",
        counters["fetched"], counters["skipped"], counters["errors"],
    )
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest shutuba (出馬表) pages for upcoming races"
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Race date to ingest shutuba for (e.g. 2025-05-05). "
            "Used for calendar fetch (--date のみ指定時) and as DB date value. "
            "--race-ids と併用した場合は calendar fetch を skip し、date は DB 保存値として使う。"
        ),
    )
    parser.add_argument(
        "--race-ids",
        default=None,
        metavar="ID1,ID2,...",
        help=(
            "Comma-separated list of race IDs to ingest directly, skipping calendar fetch. "
            "例: --race-ids 202506050911,202506050912 "
            "race.netkeiba.com の race_list は AJAX 取得のため calendar 経由の自動取得が "
            "不安定な場合にこのオプションで直接指定する。"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of races to fetch (debug use)",
    )
    return parser.parse_args()


def cli_main() -> int:
    args = _parse_args()
    if args.date is None and args.race_ids is None:
        print("Error: either --date or --race-ids must be specified")
        return 1
    return asyncio.run(main(args))


if __name__ == "__main__":
    raise SystemExit(cli_main())
