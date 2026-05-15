"""Tests for ai/registry.py — model generation management and active switching."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from ai.gbm.train import train
from ai.registry import get_active, list_models, load_model, set_active
from db.base import Base
from db.models.model_run import ModelRun
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def two_models(tmp_path):
    """Train two separate models and return their directories."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=5)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

    r1 = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    # Small sleep to ensure distinct timestamps
    time.sleep(1.1)
    r2 = train(db=db_file, train_end=None, valid_months=2, test_months=1)

    return engine, db_file, Path(r1["model_dir"]), Path(r2["model_dir"])


def test_list_models_returns_two(two_models):
    _, _, dir1, dir2 = two_models
    models = list_models()
    paths = [m.path for m in models]
    assert dir1 in paths
    assert dir2 in paths
    assert len([m for m in models if m.path in (dir1, dir2)]) == 2


def test_load_model_returns_booster(two_models):
    import lightgbm as lgb
    _, _, dir1, _ = two_models
    model = load_model(dir1)
    assert isinstance(model, lgb.Booster)


def test_set_and_get_active(two_models):
    engine, db_file, dir1, dir2 = two_models

    with Session(engine) as session:
        # Register model_runs manually (train already inserts them)
        runs = session.query(ModelRun).all()
        # Assign model paths to distinguish
        assert len(runs) >= 2
        runs[0].model_path = str(dir1)
        runs[1].model_path = str(dir2)
        session.commit()

    with Session(engine) as session:
        set_active(dir1, session)
        session.commit()

    with Session(engine) as session:
        active = get_active(session)

    assert active == dir1


def test_set_active_deactivates_others(two_models):
    engine, db_file, dir1, dir2 = two_models

    with Session(engine) as session:
        runs = session.query(ModelRun).all()
        runs[0].model_path = str(dir1)
        runs[1].model_path = str(dir2)
        session.commit()

    # Activate dir2 first
    with Session(engine) as session:
        set_active(dir2, session)
        session.commit()

    # Then switch to dir1
    with Session(engine) as session:
        set_active(dir1, session)
        session.commit()

    with Session(engine) as session:
        active = get_active(session)
        inactive_run = session.query(ModelRun).filter(
            ModelRun.model_path == str(dir2)
        ).first()

    assert active == dir1
    assert inactive_run.is_active == 0


def test_get_active_none_when_no_active(tmp_path):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        active = get_active(session)
    assert active is None
    engine.dispose()
