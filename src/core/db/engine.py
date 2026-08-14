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

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, create_mock_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

#: Defaults for a machine that says nothing about its database. Each is
#: overridable on its own, because a target machine may differ from this one in
#: exactly one respect - a non-default port, a shared server, a renamed
#: database - and forcing a full DSN to change a port is how installers end up
#: with credentials pasted into scripts.
DB_DEFAULTS = {
    "SALEDEED_DB_HOST": "localhost",
    "SALEDEED_DB_PORT": "5432",
    "SALEDEED_DB_NAME": "saledeed",
    "SALEDEED_DB_USER": "saledeed",
    "SALEDEED_DB_PASSWORD": "saledeed",
}


def build_dsn(env: dict[str, str] | None = None) -> str:
    """Assemble a DSN from its parts.

    `SALEDEED_DB_URL` remains the single override that wins outright - a
    complete URL is unambiguous and is what the installer writes. These parts
    exist for the case where only one of them differs, and for the installer to
    fill in per machine rather than shipping a development machine's values.

    The password is quoted: a generated password may contain characters that
    would otherwise terminate the URL early and produce a baffling parse error
    rather than an authentication failure.
    """
    from urllib.parse import quote

    source = env if env is not None else dict(os.environ)
    part = lambda key: source.get(key) or DB_DEFAULTS[key]  # noqa: E731
    user = quote(part("SALEDEED_DB_USER"), safe="")
    password = quote(part("SALEDEED_DB_PASSWORD"), safe="")
    return (f"postgresql+psycopg://{user}:{password}@"
            f"{part('SALEDEED_DB_HOST')}:{part('SALEDEED_DB_PORT')}/"
            f"{part('SALEDEED_DB_NAME')}")


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


def _load_env_file() -> None:
    """Make `.env` authoritative for every entry point, not just the launcher.

    `launcher/config.py` reads `.env`, so the desktop application saw the
    configured credential - but pytest, `alembic`, and every tool script bypass
    the launcher entirely and fell through to `DEFAULT_DSN`. The configured
    password was therefore ignored by most of the ways this code is actually
    run, which is also why a hard-coded fallback could survive unnoticed.

    Values already in the environment win: an operator who exported
    `SALEDEED_DB_URL` for one command means it.
    """
    try:
        from core import paths

        path = paths.ROOT / ".env"
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:  # noqa: BLE001 - configuration must never break startup
        return


def dsn_from_env(env: dict[str, str] | None = None) -> str:
    """The configured DSN.

    Falls back to `DEFAULT_DSN` only when nothing is configured. That fallback
    carries a guessable password and exists so a developer machine works out of
    the box; a deployed system must set `SALEDEED_DB_URL`, which the installer
    writes to `.env`.
    """
    if env is None:
        _load_env_file()
    source = env if env is not None else dict(os.environ)
    configured = source.get("SALEDEED_DB_URL")
    if configured:
        return normalise_dsn(configured)

    # No complete URL. Assemble one from the parts, any of which may have been
    # set individually; only warn when *nothing* was configured, because a
    # machine that set a host and port has been configured deliberately.
    if not any(source.get(key) for key in DB_DEFAULTS):
        _log_default_dsn_use()
    return normalise_dsn(build_dsn(source))


def _log_default_dsn_use() -> None:
    """Say, once, that the built-in credential is in use.

    Not an exception: raising here would break every machine that works today,
    including this project's own test suite. A warning names the problem at the
    moment it matters and points at the fix, without taking a working system
    down to make a point.
    """
    global _WARNED_DEFAULT_DSN
    if _WARNED_DEFAULT_DSN:
        return
    _WARNED_DEFAULT_DSN = True
    logging.getLogger("saledeed.db").warning(
        "SALEDEED_DB_URL is not set; connecting with the built-in development "
        "credential. Write a generated password to .env - "
        "`System Setup.bat` does this automatically on a new machine.")


_WARNED_DEFAULT_DSN = False


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
