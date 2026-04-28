"""Inline DDL for M2.  TODO(M3): replace with Alembic migrations.

Only the tables required by the ingest job are created here.  The full schema
(horses, jockeys, trainers, payouts, model_runs) will be added in M3 when
Alembic is introduced.
"""

from __future__ import annotations

import sqlite3


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    course          TEXT NOT NULL,
    surface         TEXT NOT NULL,
    distance        INTEGER NOT NULL,
    weather         TEXT,
    track_condition TEXT,
    race_class      TEXT,
    n_runners       INTEGER,
    payout_win      INTEGER,
    payout_place    TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL REFERENCES races(race_id),
    horse_id            TEXT NOT NULL,
    post_position       INTEGER,
    jockey_id           TEXT,
    trainer_id          TEXT,
    weight_carried      REAL,
    age                 INTEGER,
    sex                 TEXT,
    horse_weight        INTEGER,
    horse_weight_diff   INTEGER,
    odds_win            REAL,
    popularity          INTEGER,
    finish_position     INTEGER,
    finish_time         REAL,
    margin              TEXT
);

CREATE TABLE IF NOT EXISTS scrape_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    etag         TEXT,
    content_hash TEXT
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Execute CREATE TABLE IF NOT EXISTS for all M2 tables."""
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
