"""Database layer - SQLAlchemy 2.0 + psycopg (v3) + Alembic on PostgreSQL.

Stack is mandated; see docs/DOCUMENTATION.md. The DSN scheme is
`postgresql+psycopg://` - psycopg v3, not psycopg2. Mixing the two is a common
and confusing failure, so the driver is named explicitly everywhere.

    models        declarative table definitions
    engine        engine and session factory, plus offline DDL rendering
    repositories  repository pattern and unit of work

The schema is written to the PostgreSQL dialect. It can be authored, compiled and
reviewed without a running server: `engine.render_ddl()` emits the exact CREATE
statements offline. Storing rows, naturally, needs a live server.
"""

from .models import (
    Base,
    Batch,
    BatchState,
    Document,
    Extraction,
    LogEntry,
    OcrPage,
    Person,
    PersonRelation,
    Property,
    Setting,
    StageState,
    User,
    ValidationResult,
)

__all__ = [
    "Base",
    "Batch",
    "BatchState",
    "Document",
    "Extraction",
    "LogEntry",
    "OcrPage",
    "Person",
    "PersonRelation",
    "Property",
    "Setting",
    "StageState",
    "User",
    "ValidationResult",
]
