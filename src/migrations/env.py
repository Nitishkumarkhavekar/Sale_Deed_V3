"""Alembic environment for SQLAlchemy 2.0 + psycopg v3 on PostgreSQL.

Three edits to the generated scaffold, each required:

1. `target_metadata` points at `Base.metadata`. The template leaves it `None`,
   which makes `--autogenerate` compare the database against nothing and emit an
   empty migration.
2. The URL comes from `SALEDEED_DB_URL`, normalised onto the psycopg v3 driver.
   The template ships the placeholder `driver://user:pass@localhost/dbname`; the
   alternative to reading the environment is putting a real password into a
   committed file.
3. Offline mode is wired up, so `alembic upgrade head --sql` emits migration SQL
   **without a database connection**. That is what allows the schema to be
   generated and reviewed before a PostgreSQL server exists.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project importable so `core.db` resolves when Alembic runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db.engine import dsn_from_env  # noqa: E402
from core.db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#: Autogenerate compares the live database against this.
target_metadata = Base.metadata


def get_url() -> str:
    """DSN from the environment, forced onto the psycopg v3 driver.

    An explicit `-x url=...` or a real value in alembic.ini still wins; the
    template's `driver://` placeholder is ignored.
    """
    override = config.get_main_option("sqlalchemy.url", "") or ""
    if override and not override.startswith("driver://"):
        return override
    return dsn_from_env(dict(os.environ))


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting.

    The path that works today: the schema can be generated, reviewed and handed
    over before any server is reachable.
    """
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply. Requires a running PostgreSQL server."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool,
        # Bound the handshake so a missing server fails fast instead of hanging.
        connect_args={"connect_timeout": 5},
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # One transaction per migration, so a failure leaves no half-applied DDL.
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
