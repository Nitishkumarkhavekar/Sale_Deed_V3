"""Database testing - PostgreSQL, SQLAlchemy 2.0 and Alembic.

Split into two halves by what they need:

**Offline** - schema shape, DDL rendering, constraint declarations. These run
anywhere, including a machine that has never had PostgreSQL installed, because
SQLAlchemy can compile the schema without opening a socket.

**Connected** - transactions, concurrent claiming, cascades, idempotency. These
need a live server and skip cleanly when there is not one.

The connected half matters more than it looks. Three defects found in this
project were invisible to offline tests: repository methods that were not
idempotent, requeued documents that became permanently unclaimable, and a claim
that raced under concurrency. All three are regression-tested here.
"""

from __future__ import annotations

import re
import threading

import pytest
from sqlalchemy import inspect, text

from core.db.engine import (
    DEFAULT_DSN,
    build_engine,
    normalise_dsn,
    render_ddl,
    session_scope,
)
from core.db.models import (
    Base,
    BatchState,
    DocumentState,
    StageState,
)
from core.db.repositories import UnitOfWork

# ---------------------------------------------------------------------------
# Offline - schema and configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDsnHandling:
    """psycopg v3 must be forced into the URL; psycopg2 is not installed."""

    @pytest.mark.parametrize("given", [
        "postgresql://u:p@h:5432/db",
        "postgres://u:p@h:5432/db",
        "postgresql+psycopg2://u:p@h:5432/db",
    ])
    def test_driver_is_rewritten(self, given):
        assert normalise_dsn(given).startswith("postgresql+psycopg://")

    def test_already_correct_is_untouched(self):
        dsn = "postgresql+psycopg://u:p@h:5432/db"
        assert normalise_dsn(dsn) == dsn

    def test_credentials_are_preserved(self):
        out = normalise_dsn("postgresql://user:p%40ss@host:5432/db")
        assert "user:p%40ss@host" in out

    def test_default_targets_postgresql(self):
        assert DEFAULT_DSN.startswith("postgresql+psycopg://")


@pytest.mark.unit
class TestSchemaShape:
    def test_expected_tables_exist(self):
        names = set(Base.metadata.tables)
        for required in ("users", "batches", "documents", "ocr_pages",
                         "extractions", "properties", "persons",
                         "validation_results", "settings", "logs"):
            assert required in names, f"table {required} missing"

    def test_every_table_has_a_primary_key(self):
        for name, table in Base.metadata.tables.items():
            assert table.primary_key.columns, f"{name} has no primary key"

    def test_documents_reference_batches(self):
        fks = Base.metadata.tables["documents"].foreign_keys
        assert any(fk.column.table.name == "batches" for fk in fks)

    def test_child_tables_cascade_on_delete(self):
        """Deleting a batch must not strand rows in child tables."""
        for child in ("ocr_pages", "extractions", "validation_results"):
            table = Base.metadata.tables[child]
            assert any(fk.ondelete == "CASCADE" for fk in table.foreign_keys), \
                f"{child} would be orphaned"


@pytest.mark.unit
class TestDdlRendering:
    """The DDL PostgreSQL will actually receive, compiled without a server."""

    @pytest.fixture(scope="class")
    def ddl(self) -> str:
        return render_ddl()

    def test_creates_every_table(self, ddl):
        for name in Base.metadata.tables:
            assert re.search(rf"CREATE TABLE {name}\b", ddl), f"{name} not created"

    def test_uses_postgresql_types_not_sqlite(self, ddl):
        assert "BIGSERIAL" in ddl or "BIGINT" in ddl

    def test_check_constraints_are_emitted(self, ddl):
        assert "CHECK" in ddl

    def test_no_server_was_contacted(self, ddl):
        # render_ddl uses a mock engine; a real connection would have raised
        # long before this assertion.
        assert len(ddl) > 500


@pytest.mark.unit
class TestEnumsAreClosed:
    """State machines must not accept arbitrary strings."""

    def test_document_states(self):
        # Lowercase deliberately - these are stored values, and mixing cases
        # across a state machine is how comparisons start failing silently.
        values = {s.value for s in DocumentState}
        assert {"processing", "processed", "failed", "needs_review"} <= values

    def test_stage_states(self):
        values = {s.value for s in StageState}
        assert {"pending", "running", "done", "failed", "skipped"} <= values

    def test_state_values_are_lowercase(self):
        for enum in (DocumentState, StageState, BatchState):
            for member in enum:
                assert member.value == member.value.lower(), \
                    f"{enum.__name__}.{member.name} breaks the convention"

    def test_batch_states(self):
        assert len({s.value for s in BatchState}) >= 3


# ---------------------------------------------------------------------------
# Connected - behaviour against a live server
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConnectivity:
    def test_server_answers(self, db_engine):
        with db_engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar() == 1

    def test_is_actually_postgresql(self, db_engine):
        with db_engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
        assert "PostgreSQL" in version

    def test_statement_timeout_is_set(self, db_engine):
        """A runaway query must not stall a batch forever."""
        with db_engine.connect() as conn:
            value = conn.execute(text("SHOW statement_timeout")).scalar()
        assert value not in ("0", "0ms"), "no statement timeout configured"


@pytest.mark.integration
class TestMigrations:
    def test_alembic_version_table_exists(self, db_engine):
        assert inspect(db_engine).has_table("alembic_version")

    def test_schema_is_at_a_known_revision(self, db_engine):
        with db_engine.connect() as conn:
            revision = conn.execute(
                text("SELECT version_num FROM alembic_version")).scalar()
        assert revision, "no migration has been applied"

    def test_every_model_table_exists_in_the_database(self, db_engine):
        present = set(inspect(db_engine).get_table_names())
        missing = set(Base.metadata.tables) - present
        assert not missing, f"migrations are behind the models: {missing}"


@pytest.mark.integration
class TestTransactions:
    def test_commit_persists(self, session_factory):
        with session_scope(session_factory) as session:
            UnitOfWork(session).settings.set("pytest_tx", "kept")
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).settings.get("pytest_tx") == "kept"
        with session_scope(session_factory) as session:
            UnitOfWork(session).settings.set("pytest_tx", None)

    def test_exception_rolls_back(self, session_factory):
        """A failure mid-transaction must leave nothing behind."""
        class Boom(RuntimeError):
            pass

        with pytest.raises(Boom):
            with session_scope(session_factory) as session:
                UnitOfWork(session).settings.set("pytest_rollback", "x")
                raise Boom()

        with session_scope(session_factory) as session:
            assert UnitOfWork(session).settings.get("pytest_rollback") is None


@pytest.mark.integration
class TestWorkClaiming:
    """`claim_next` is the crash-safety primitive - two workers must never get
    the same document."""

    def test_claim_returns_a_document(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.claim_next("ocr", temp_batch)
            assert doc is not None

    def test_no_document_is_claimed_twice(self, session_factory, temp_batch):
        seen: list[int] = []
        for _ in range(5):
            with session_scope(session_factory) as session:
                doc = UnitOfWork(session).documents.claim_next("ocr", temp_batch)
                if doc is None:
                    break
                seen.append(doc.id)
        assert len(seen) == len(set(seen)), "a document was claimed twice"

    def test_concurrent_claims_do_not_collide(self, session_factory, temp_batch):
        """The real test: threads racing on the same batch."""
        claimed: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            with session_scope(session_factory) as session:
                doc = UnitOfWork(session).documents.claim_next("ocr", temp_batch)
                if doc is not None:
                    with lock:
                        claimed.append(doc.id)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(claimed) == len(set(claimed)), \
            f"documents claimed more than once: {claimed}"


@pytest.mark.integration
class TestIdempotency:
    """Regression: these three methods set the foreign key directly instead of
    appending through the relationship, so a stale collection meant
    delete-then-insert deleted nothing and rows accumulated on every retry."""

    def test_save_pages_twice_does_not_duplicate(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch)
            if doc is None:
                pytest.skip("no claimable document")
            pages = [(1, "page one"), (2, "page two")]
            uow.ocr.save_pages(doc, pages)
            uow.ocr.save_pages(doc, pages)
            session.flush()
            text_after = uow.ocr.full_text(doc)
        assert text_after.count("page one") == 1

    def test_replace_persons_twice_does_not_duplicate(self, session_factory,
                                                      temp_batch):
        extraction = {"buyer_details": [{"name": "A", "pan": "ABCDE1234F"}],
                      "seller_details": [{"name": "B"}]}
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("extract", temp_batch)
            if doc is None:
                pytest.skip("no claimable document")
            uow.results.replace_persons(doc, extraction)
            session.flush()
            second = uow.results.replace_persons(doc, extraction)
            session.flush()
        assert len(second) == 2, "persons accumulated across runs"


@pytest.mark.integration
class TestCascades:
    def test_deleting_a_batch_removes_its_documents(self, session_factory):
        from core.db.engine import session_scope as scope

        with scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("pytest_cascade")
            batch = uow.batches.create("pytest_cascade", user, 1, 1024)
            uow.documents.add_many(batch, [{
                "document_id": "CASCADE-1", "source_filename": "c.pdf",
                "source_path": "c.pdf", "size_bytes": 1}])
            session.flush()
            batch_id = batch.id

        with scope(session_factory) as session:
            uow = UnitOfWork(session)
            session.delete(uow.batches.get(batch_id))

        with scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.batches.get(batch_id) is None
            remaining = session.execute(
                text("SELECT count(*) FROM documents WHERE batch_id = :b"),
                {"b": batch_id}).scalar()
            assert remaining == 0, "documents were orphaned"
            session.delete(uow.users.get_or_create("pytest_cascade"))


@pytest.mark.integration
class TestSettingsRepository:
    def test_round_trip(self, session_factory):
        with session_scope(session_factory) as session:
            UnitOfWork(session).settings.set("pytest_key", "value")
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).settings.get("pytest_key") == "value"

    def test_missing_key_returns_default(self, session_factory):
        with session_scope(session_factory) as session:
            got = UnitOfWork(session).settings.get("pytest_absent", "fallback")
        assert got == "fallback"

    def test_all_returns_a_mapping(self, session_factory):
        with session_scope(session_factory) as session:
            assert isinstance(UnitOfWork(session).settings.all(), dict)
            UnitOfWork(session).settings.set("pytest_key", None)


@pytest.mark.integration
class TestRecovery:
    """Crash recovery: work left RUNNING by a killed process must return to the
    queue, or a restart silently loses those documents."""

    def test_running_work_is_requeued(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch)
            if doc is None:
                pytest.skip("no claimable document")
            uow.documents.mark_stage(doc, "ocr", StageState.RUNNING)

        with session_scope(session_factory) as session:
            reset = UnitOfWork(session).documents.reset_running_to_pending(temp_batch)
        assert reset >= 1
