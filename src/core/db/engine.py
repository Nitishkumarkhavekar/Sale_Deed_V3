"""Engine and session management for SQLAlchemy 2.0 + psycopg v3.

Two modes:

**Connected** - `build_engine()` returns a pooled engine talking to a live
PostgreSQL server through psycopg v3.

**Offline** - `render_ddl()` compiles the full schema to PostgreSQL DDL without
opening a socket. This is what makes the schema reviewable and testable before a
server exists: SQLAlchemy renders the exact `CREATE TABLE` statements the real
database will receive.

DSN scheme is always `postgresql+psycopg://` (psycopg v3). `postgresql://` alone
resolves to psycopg2, which is not installed here, and the resulting ImportError
is confusing enough to be worth preventing outright.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, create_mock_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

#: PostgreSQL is the application database (ADR-012). SQLite is retained only as a
#: verification target for `tools/db_setup.py --sqlite`, never for the app.
DEFAULT_DSN = "postgresql+psycopg://saledeed:saledeed@localhost:5432/saledeed"


class DatabaseUnavailableError(RuntimeError):
    """Raised when no PostgreSQL server can be reached.

    Carries an explanation rather than a driver traceback, because the usual cause
    is simply that no server is running - psycopg is a client and cannot store
    data on its own.
    """


def normalise_dsn(dsn: str) -> str:
    """Force the psycopg v3 driver into the URL.

    `postgresql://` and `postgresql+psycopg2://` both select psycopg2. Only
    psycopg v3 is installed, so both are rewritten.
    """
    if dsn.startswith("postgresql+psycopg://"):
        return dsn
    return re.sub(r"^postgres(?:ql)?(?:\+psycopg2)?://", "postgresql+psycopg://", dsn)


def dsn_from_env(env: dict[str, str] | None = None) -> str:
    source = env if env is not None else dict(os.environ)
    return normalise_dsn(source.get("SALEDEED_DB_URL", DEFAULT_DSN))


def build_engine(
    dsn: str | None = None,
    *,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    statement_timeout_ms: int = 30_000,
    connect_timeout_s: int = 5,
) -> Engine:
    """Create a pooled engine.

    `pool_pre_ping` is on deliberately: this application runs for hours and the
    AI server's stage workers hold connections across long GPU waits, during
    which an idle connection can be dropped by the server or a firewall. Without
    pre-ping the next query fails rather than transparently reconnecting.
    """
    url = normalise_dsn(dsn or dsn_from_env())
    kwargs: dict[str, Any] = {"echo": echo, "future": True}

    if url.startswith("postgresql"):
        # Bound the TCP handshake. Without this, connecting to a host where
        # nothing is listening blocks for the OS default (tens of seconds on
        # Windows), turning a health probe into an apparent hang.
        connect_args: dict[str, Any] = {"connect_timeout": connect_timeout_s}

        if statement_timeout_ms:
            # Passed as a libpq startup option rather than executed as `SET`.
            # `SET statement_timeout` is transaction-scoped, and the pool issues
            # ROLLBACK when a connection is returned (reset_on_return defaults to
            # 'rollback'), which silently reverted it. The measured effect: the
            # first query on a fresh connection had a 30 s limit and every query
            # after it had none - so the runaway query this is meant to bound
            # could still stall a batch indefinitely. A startup option is applied
            # by the server at connection time and survives rollback.
            connect_args["options"] = f"-c statement_timeout={int(statement_timeout_ms)}"

        kwargs.update(
            connect_args=connect_args,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    elif url.startswith("sqlite"):
        # Verification target only (tools/db_setup.py --sqlite), not the app.
        # SQLite rejects the PostgreSQL pool arguments outright, and in-memory
        # needs StaticPool: the default pool hands every session its own empty
        # database, so writes in one transaction are invisible to the next.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool

    return create_engine(url, **kwargs)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory with autoflush off.

    Autoflush is disabled because stage workers mutate documents while iterating
    query results; implicit flushes mid-iteration produce ordering surprises that
    are hard to trace. Flushes are explicit in the repositories.
    """
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                        future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception.

    The unit of work for **one document**. The specification requires continuous
    commit - each document is durable the moment it finishes, never deferred to
    batch end - so callers must scope this per document, not per batch.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(engine: Engine) -> tuple[bool, str]:
    """Probe the server. Returns (ok, detail); never raises."""
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar_one()
        return True, str(version)
    except Exception as exc:  # noqa: BLE001 - a probe must not propagate
        return False, (
            f"{type(exc).__name__}: {exc}\n"
            "psycopg is a client driver and cannot store data by itself - a "
            "PostgreSQL server must be running and reachable at the configured DSN."
        )


# ---------------------------------------------------------------------------
# Offline DDL
# ---------------------------------------------------------------------------


def render_ddl(dialect_name: str = "postgresql") -> str:
    """Compile the whole schema to DDL without connecting to anything.

    Uses SQLAlchemy's mock engine, so the statements produced are exactly what a
    live server would receive. This is how the schema is reviewed and verified
    before any database exists.
    """
    statements: list[str] = []

    def collect(sql: Any, *_args: Any, **_kwargs: Any) -> None:
        rendered = str(sql.compile(dialect=engine.dialect)).strip()
        if rendered:
            statements.append(rendered + ";")

    engine = create_mock_engine(f"{dialect_name}+psycopg://", collect)
    Base.metadata.create_all(engine, checkfirst=False)  # type: ignore[arg-type]
    return "\n\n".join(statements)


def schema_summary() -> list[tuple[str, int, int]]:
    """(table, column count, index count) for every mapped table."""
    return [
        (table.name, len(table.columns), len(table.indexes))
        for table in Base.metadata.sorted_tables
    ]


def create_all(engine: Engine) -> None:
    """Create tables directly. Prefer Alembic for anything but a scratch database."""
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    if "--ddl" in sys.argv:
        print(render_ddl())
        raise SystemExit(0)

    print(f"DSN: {dsn_from_env()}")
    print()
    print(f"{'table':<22}{'cols':>6}{'idx':>6}")
    print("-" * 34)
    total_cols = 0
    for name, cols, idx in schema_summary():
        print(f"{name:<22}{cols:>6}{idx:>6}")
        total_cols += cols
    print("-" * 34)
    print(f"{len(schema_summary())} tables, {total_cols} columns")
    print()
    print("Offline DDL renders without a server: python -m core.db.engine --ddl")
