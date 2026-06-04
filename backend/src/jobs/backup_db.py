"""CLI: snapshot SQLite DBs into data/backups/ with generation rotation.

Uses SQLite's **online backup API** (``sqlite3.Connection.backup``) which takes
a consistent, WAL-aware snapshot **while the DB is in use** — safe to run during
an active ingest. Plain ``cp`` of a live DB can capture a torn WAL state, so it
is intentionally avoided.

Targets keiba.db (irreplaceable race/entry data) and odds.db (re-scrapeable but
large). Each DB keeps the newest ``--keep`` generations; older snapshots are
pruned. Backups land in ``data/backups/<name>-<YYYYMMDD-HHMMSS>.db``.

Usage:
  uv run keiba-backup                      # keiba + odds, keep 7
  uv run keiba-backup --db keiba           # keiba only
  uv run keiba-backup --keep 14            # keep 14 generations

WSL 注意: `/mnt/c` 上の DB を WSL から開くと WAL の mmap 問題で失敗することが
ある (drvfs)。バックアップは Windows 側 (`uv run keiba-backup`) から実行する
のが確実。ext4 上の DB なら WSL でも問題ない。
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from core.logging import configure_logging, get_logger
from core.paths import data_dir, db_path, odds_db_path

log = get_logger(__name__)

DEFAULT_KEEP = 7

# DB に付随するサイドカーファイル (WAL モード / rollback journal)。
_DB_SIDECARS = ("-wal", "-shm", "-journal")


def backups_dir() -> Path:
    d = data_dir() / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rotate(prefix: str, keep: int, *, _dir: Path | None = None) -> list[Path]:
    """Keep the newest ``keep`` ``<prefix>-*.db`` files; unlink the rest.

    Timestamped names sort chronologically, so lexical sort == time order.
    Returns the list of removed paths.
    """
    bdir = _dir or backups_dir()
    files = sorted(bdir.glob(f"{prefix}-*.db"), key=lambda p: p.name)
    if keep <= 0:
        return []
    removed = files[:-keep] if len(files) > keep else []
    for p in removed:
        p.unlink()
        log.info("rotated out old backup: %s", p.name)
    return removed


def _online_backup(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` via SQLite's online backup API (read-only src)."""
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(dest)
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def backup_one(src: Path, prefix: str, keep: int = DEFAULT_KEEP) -> Path | None:
    """Online-backup ``src`` to data/backups/<prefix>-<ts>.db, then rotate.

    Returns the created backup path, or None when src does not exist.
    """
    if not src.exists():
        log.warning("source DB not found, skipping: %s", src)
        return None

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = backups_dir() / f"{prefix}-{ts}.db"

    try:
        _online_backup(src, dest)
    except sqlite3.DatabaseError as exc:
        # WSL/drvfs: a live WAL on /mnt/c can't be mmapped (`-shm`), so the
        # direct open fails with "file is not a database". Fall back to copying
        # the DB + sidecar files to a local temp dir (ext4) and backing up from
        # there. Assumes the source isn't being heavily written concurrently
        # (true for keiba.db; odds.db uses a rollback journal and opens directly).
        dest.unlink(missing_ok=True)
        log.warning(
            "direct backup of %s failed (%s); retrying via local temp copy "
            "(WSL/drvfs workaround)", src.name, exc,
        )
        with tempfile.TemporaryDirectory(prefix="keiba-backup-") as td:
            for suffix in ("", *_DB_SIDECARS):
                f = src.with_name(src.name + suffix)
                if f.exists():
                    shutil.copy2(f, Path(td) / f.name)
            _online_backup(Path(td) / src.name, dest)

    size_mb = dest.stat().st_size / 1e6
    log.info("backed up %s -> backups/%s (%.1f MB)", src.name, dest.name, size_mb)
    _rotate(prefix, keep)
    return dest


def run_backup(targets: list[str], keep: int = DEFAULT_KEEP) -> list[Path]:
    """Back up the requested targets ('keiba' / 'odds'). Returns created paths."""
    created: list[Path] = []
    if "keiba" in targets:
        p = backup_one(db_path(), "keiba", keep)
        if p is not None:
            created.append(p)
    if "odds" in targets:
        p = backup_one(odds_db_path(), "odds", keep)
        if p is not None:
            created.append(p)
    return created


def cli_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        choices=["keiba", "odds", "both"],
        default="both",
        help="どの DB をバックアップするか (default: both)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        help=f"DB ごとに保持する世代数 (default: {DEFAULT_KEEP})",
    )
    args = parser.parse_args()
    configure_logging(log_name="backup_db")

    targets = ["keiba", "odds"] if args.db == "both" else [args.db]
    created = run_backup(targets, keep=args.keep)
    if not created:
        log.warning("no backups were created")
    else:
        log.info("done: %d backup(s) in %s", len(created), backups_dir())


if __name__ == "__main__":
    cli_main()
