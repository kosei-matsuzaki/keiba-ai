"""Synthetic race data generation for CI/testing.

Generates deterministic data that exercises the full training pipeline
without requiring real netkeiba data.  Each horse appears in multiple
races to enable leakage-prevention tests.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session

import db.models  # noqa: F401 — populate Base.metadata
from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.race import Race
from db.models.trainer import Trainer

COURSES = ["東京", "中山", "京都"]
SURFACES = ["芝", "ダ"]
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2400]
WEATHER_OPTIONS = ["晴", "曇", "雨"]
TRACK_CONDITIONS = ["良", "稍重", "重"]
RACE_CLASSES = ["G1", "G3", "条件戦", "オープン"]
SEXES = ["牡", "牝", "セ"]


def make_synthetic_db(
    engine: Engine,
    n_races: int = 30,
    n_horses_per_race: int = 10,
    days_back: int = 180,
    seed: int = 42,
) -> None:
    """Populate engine with synthetic races, horses, jockeys, trainers, and entries.

    Horses are drawn from a fixed pool smaller than n_races * n_horses_per_race
    so each horse appears in multiple races — required for leakage tests.
    Tables are created if they don't exist.
    """
    Base.metadata.create_all(engine)
    rng = random.Random(seed)

    n_horse_pool = max(n_horses_per_race * 3, 30)
    n_jockey_pool = 10
    n_trainer_pool = 8

    horse_ids = [f"H{i:04d}" for i in range(n_horse_pool)]
    jockey_ids = [f"J{i:03d}" for i in range(n_jockey_pool)]
    trainer_ids = [f"T{i:03d}" for i in range(n_trainer_pool)]

    today = date.today()
    start_date = today - timedelta(days=days_back)

    with Session(engine) as session:
        # Masters
        for hid in horse_ids:
            if not session.get(Horse, hid):
                session.add(Horse(horse_id=hid, name=None))
        for jid in jockey_ids:
            if not session.get(Jockey, jid):
                session.add(Jockey(jockey_id=jid, name=None))
        for tid in trainer_ids:
            if not session.get(Trainer, tid):
                session.add(Trainer(trainer_id=tid, name=None))
        session.flush()

        # Races spread evenly over days_back
        step = max(days_back // n_races, 1)
        for i in range(n_races):
            race_date = start_date + timedelta(days=i * step)
            race_id = f"SYN{race_date.strftime('%Y%m%d')}{i:02d}"
            course = rng.choice(COURSES)
            surface = rng.choice(SURFACES)
            distance = rng.choice(DISTANCES)
            n_runners = n_horses_per_race

            race = Race(
                race_id=race_id,
                date=race_date.isoformat(),
                course=course,
                surface=surface,
                distance=distance,
                weather=rng.choice(WEATHER_OPTIONS),
                track_condition=rng.choice(TRACK_CONDITIONS),
                race_class=rng.choice(RACE_CLASSES),
                n_runners=n_runners,
                payout_win=rng.randint(100, 5000),
                payout_place=None,
            )
            session.add(race)
            session.flush()

            # Pick horses for this race (sampling without replacement from pool)
            race_horses = rng.sample(horse_ids, n_horses_per_race)
            # Assign random finish positions (1..n without ties)
            positions = list(range(1, n_horses_per_race + 1))
            rng.shuffle(positions)

            for pos_idx, horse_id in enumerate(race_horses):
                finish_pos = positions[pos_idx]
                jockey_id = rng.choice(jockey_ids)
                trainer_id = rng.choice(trainer_ids)
                base_weight = rng.randint(430, 540)
                odds_win = round(rng.uniform(1.1, 50.0), 1)

                entry = Entry(
                    race_id=race_id,
                    horse_id=horse_id,
                    post_position=pos_idx + 1,
                    jockey_id=jockey_id,
                    trainer_id=trainer_id,
                    weight_carried=54.0 + rng.uniform(-2, 2),
                    age=rng.randint(2, 8),
                    sex=rng.choice(SEXES),
                    horse_weight=base_weight,
                    horse_weight_diff=rng.randint(-10, 10),
                    odds_win=odds_win,
                    popularity=pos_idx + 1,
                    finish_position=finish_pos,
                    finish_time=None,
                    margin=None,
                )
                session.add(entry)

        session.commit()


def train_synthetic_nn(db_file, *, train_end=None, valid_months=2, test_months=1, max_epochs=5):
    """Train a tiny NN model on a synthetic DB and return its model_dir.

    Shared test helper for integration tests that need a real model bundle
    (evaluate / diagnosis). Uses few epochs + cpu and skips temperature /
    combo calibrator fitting for speed. Callers must set KEIBA_DATA_DIR so the
    model is saved under a temp dir.
    """
    from pathlib import Path

    from ai.training.train_nn import train_nn

    result = train_nn(
        db=db_file,
        train_end=train_end,
        valid_months=valid_months,
        test_months=test_months,
        max_epochs=max_epochs,
        device="cpu",
        fit_temperature=False,
    )
    return Path(result["model_dir"])
