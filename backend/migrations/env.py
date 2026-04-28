"""Alembic environment configuration.

- Resolves the DB URL from keiba_ai.core.paths.db_path() so that the
  same path logic used at runtime is reused here.
- render_as_batch=True is required for SQLite ALTER TABLE limitations.
- compare_type=True ensures column type changes are detected by autogen.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure the src directory is on sys.path so keiba_ai can be imported
_backend_dir = Path(__file__).resolve().parent.parent
_src_dir = _backend_dir / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import Base and all models so metadata is fully populated
import keiba_ai.db.models  # noqa: E402, F401  (side-effect import to register all mappers)
from keiba_ai.db.base import Base  # noqa: E402

target_metadata = Base.metadata

alembic_config = context.config

# Override sqlalchemy.url with the runtime-resolved path only when using the
# placeholder value from alembic.ini.  If the caller (e.g. test suite) has
# already set a concrete URL via cfg.set_main_option(), we leave it untouched.
_current_url = alembic_config.get_main_option("sqlalchemy.url", "")
if not _current_url or "placeholder" in _current_url:
    try:
        from keiba_ai.core.paths import db_path

        alembic_config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path()}")
    except Exception:
        # Fallback: keep whatever is in alembic.ini
        pass

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the DB (--sql mode)."""
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
