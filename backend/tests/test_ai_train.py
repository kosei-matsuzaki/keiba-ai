"""Integration test: train pipeline on synthetic data.

Verifies:
- model.txt and meta.json are created
- model_runs table gets a new row
- Training completes without errors on small synthetic data
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.train import train
from keiba_ai.db.models.model_run import ModelRun
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def syn_engine(tmp_path):
    """In-file SQLite DB in tmp_path (LightGBM needs real files for some ops)."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    # Need n_races large enough to get non-empty train/valid/test splits
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=42)
    yield engine, db_file
    engine.dispose()


def test_train_creates_model_files(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine

    # Override data_dir so models go into tmp_path
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(
        db=db_file,
        train_end=None,
        valid_months=2,  # small to ensure all splits have data
        test_months=1,
    )

    model_dir = Path(result["model_dir"])
    assert (model_dir / "model.txt").exists(), "model.txt not found"
    assert (model_dir / "meta.json").exists(), "meta.json not found"

    meta = json.loads((model_dir / "meta.json").read_text())
    assert "params" in meta
    assert "metrics" in meta
    assert "feature_columns" in meta


def test_train_inserts_model_run(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with Session(engine) as session:
        count_before = session.scalar(select(func.count()).select_from(ModelRun))

    train(db=db_file, train_end=None, valid_months=2, test_months=1)

    with Session(engine) as session:
        count_after = session.scalar(select(func.count()).select_from(ModelRun))

    assert count_after == count_before + 1


def test_train_metrics_keys_present(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)

    assert "model_dir" in result
    # Metrics keys may be nan for small data but must be present
    for key in ("valid_ndcg1", "valid_ndcg3", "test_ndcg1", "test_ndcg3"):
        assert key in result
