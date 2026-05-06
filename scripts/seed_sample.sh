#!/usr/bin/env bash
# Seed the local SQLite database with synthetic race/horse/entry data for
# development and smoke-testing without requiring real netkeiba data.
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

BACKEND_DIR="$SCRIPT_DIR/../backend"

cd "$BACKEND_DIR"

echo "[seed_sample] Seeding synthetic data (20 races, 10 horses each)..."

uv run python - << 'PYEOF'
import sys
import os

# Ensure the backend src is importable when called from the script dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.core.paths import db_path

# Run alembic upgrade to make sure the schema is up to date.
import subprocess
subprocess.run(["uv", "run", "alembic", "upgrade", "head"], check=True)

engine = make_engine(db_path())

# Import the synthetic data helper from the test suite if available,
# otherwise fall back to a lightweight inline implementation.
try:
    sys.path.insert(0, "tests")
    from synthetic import make_synthetic_db  # type: ignore[import]
    make_synthetic_db(engine, n_races=20, n_horses_per_race=10, days_back=180, seed=42)
    print(f"Seeded 20 races via tests/synthetic.py at {db_path()}")
except ImportError:
    # Inline fallback: insert minimal rows so the API returns something.
    from datetime import date, timedelta
    from sqlalchemy import text

    print("tests/synthetic.py not found — using inline fallback seeder")
    today = date.today()
    with engine.begin() as conn:
        for i in range(20):
            race_date = (today - timedelta(days=i * 7)).isoformat()
            race_id = f"2026{i+1:04d}01"
            conn.execute(text(
                "INSERT OR IGNORE INTO races (race_id, date, course, surface, distance, race_class, n_runners) "
                "VALUES (:id, :date, :course, :surface, :dist, :cls, :n)"
            ), {"id": race_id, "date": race_date, "course": "東京", "surface": "芝",
                "dist": 1600, "cls": "G1", "n": 10})
            for h in range(10):
                horse_id = f"H{i:04d}{h:04d}"
                conn.execute(text(
                    "INSERT OR IGNORE INTO horses (horse_id, name) VALUES (:id, :name)"
                ), {"id": horse_id, "name": f"Horse{i}-{h}"})
                conn.execute(text(
                    "INSERT OR IGNORE INTO entries "
                    "(race_id, horse_id, post_position, odds_win, popularity, finish_position) "
                    "VALUES (:rid, :hid, :pos, :odds, :pop, :fin)"
                ), {"rid": race_id, "hid": horse_id, "pos": h + 1,
                    "odds": round(2.0 + h * 1.5, 1), "pop": h + 1, "fin": h + 1})

    print(f"Inline fallback: seeded 20 races at {db_path()}")
PYEOF

echo "[seed_sample] Done."
