"""Repository pattern and unit of work.

Two requirements shape everything here.

**Continuous commit.** A document is durable the moment its stage finishes -
never deferred to batch end. So the unit of work is scoped to one document, and
every repository method is written to be called inside such a scope.

**Crash-safe resume with no duplicate processing.** Stage workers claim documents
with `SELECT ... FOR UPDATE SKIP LOCKED`. Two workers can pull work concurrently
without ever taking the same row, and a worker that dies mid-document releases
its lock on disconnect, so the document becomes claimable again rather than
stranded. This is a genuine reason PostgreSQL suits this workload: SQLite has no
equivalent and would need application-level locking.

Business rules enforced here rather than by database constraints, so failures can
explain themselves: the four-batch queue cap, per-stage retry limits, and the
file-count and size ceilings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .models import (
    Batch,
    BatchState,
    Document,
    DocumentState,
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

#: Queue and upload ceilings from the specification.
MAX_QUEUED_BATCHES = 4
MAX_FILES_PER_BATCH = 1000
MAX_BATCH_BYTES = 25 * 1024**3

STAGE_COLUMNS = {
    "ocr": Document.ocr_state,
    "extract": Document.extract_state,
    "translate": Document.translate_state,
    "validate": Document.validate_state,
}
STAGE_ATTEMPTS = {"ocr": Document.ocr_attempts, "extract": Document.extract_attempts}
#: Stage order. A stage may only claim a document whose predecessors are done.
STAGE_ORDER = ("ocr", "extract", "translate", "validate")


class RepositoryError(RuntimeError):
    """A business-rule violation, phrased for an operator."""


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageProgress:
    stage: str
    done: int
    failed: int
    total: int

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total if self.total else 0.0


@dataclass(frozen=True)
class BatchProgress:
    """Everything the dashboard needs for one batch, in a single query."""

    batch_id: int
    name: str
    state: BatchState
    total: int
    completed: int
    failed: int
    needs_review: int
    stages: dict[str, StageProgress]

    @property
    def percent(self) -> float:
        return 100.0 * (self.completed + self.failed) / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Unit of work
# ---------------------------------------------------------------------------


class UnitOfWork:
    """Groups repositories over one session.

    Not a context manager itself - the session lifecycle belongs to
    `engine.session_scope()`, so commit and rollback stay in one place.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.batches = BatchRepository(session)
        self.documents = DocumentRepository(session)
        self.ocr = OcrRepository(session)
        self.extractions = ExtractionRepository(session)
        self.results = ResultRepository(session)
        self.settings = SettingsRepository(session)
        self.maintenance = MaintenanceRepository(session)

    def flush(self) -> None:
        self.session.flush()


class _Base:
    def __init__(self, session: Session) -> None:
        self.session = session


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class UserRepository(_Base):
    def get_or_create(self, username: str) -> User:
        username = (username or "").strip()
        if not username:
            raise RepositoryError("username is required")
        existing = self.session.scalar(select(User).where(User.username == username))
        if existing:
            return existing
        user = User(username=username)
        self.session.add(user)
        try:
            self.session.flush()
        except IntegrityError:
            # Lost a race with another process; the row now exists.
            self.session.rollback()
            return self.session.scalar(
                select(User).where(User.username == username))  # type: ignore[return-value]
        return user


class BatchRepository(_Base):
    def queued_count(self) -> int:
        return self.session.scalar(
            select(func.count()).select_from(Batch)
            .where(Batch.state == BatchState.QUEUED)) or 0

    def create(self, name: str, user: User | None, file_count: int,
               total_bytes: int) -> Batch:
        """Create a queued batch, enforcing the upload ceilings."""
        if not (name or "").strip():
            raise RepositoryError("batch name is required")
        if file_count <= 0:
            raise RepositoryError("a batch must contain at least one file")
        if file_count > MAX_FILES_PER_BATCH:
            raise RepositoryError(
                f"{file_count} files exceeds the {MAX_FILES_PER_BATCH}-file limit")
        if total_bytes > MAX_BATCH_BYTES:
            raise RepositoryError(
                f"{total_bytes / 1024**3:.1f} GB exceeds the "
                f"{MAX_BATCH_BYTES / 1024**3:.0f} GB limit")

        queued = self.queued_count()
        if queued >= MAX_QUEUED_BATCHES:
            raise RepositoryError(
                f"{queued} batches already queued; the maximum is "
                f"{MAX_QUEUED_BATCHES}. Wait for one to finish before adding another.")

        next_position = (self.session.scalar(
            select(func.max(Batch.queue_position))
            .where(Batch.state.in_((BatchState.QUEUED, BatchState.RUNNING)))) or 0) + 1

        batch = Batch(name=name.strip(), user=user, file_count=file_count,
                      total_bytes=total_bytes, state=BatchState.QUEUED,
                      queue_position=next_position)
        self.session.add(batch)
        self.session.flush()
        return batch

    def get(self, batch_id: int) -> Batch | None:
        return self.session.get(Batch, batch_id)

    def next_queued(self) -> Batch | None:
        """Head of the queue, locked so two schedulers cannot start the same batch."""
        return self.session.scalar(
            select(Batch).where(Batch.state == BatchState.QUEUED)
            .order_by(Batch.queue_position, Batch.id)
            .limit(1).with_for_update(skip_locked=True))

    def active(self) -> Batch | None:
        return self.session.scalar(
            select(Batch).where(Batch.state == BatchState.RUNNING)
            .order_by(Batch.started_at).limit(1))

    def list_paginated(self, page: int = 1, per_page: int = 5,
                       states: Sequence[BatchState] | None = None) -> tuple[list[Batch], int]:
        """Page of batches plus the total count. Dashboard paginates by 5."""
        # `user` is eager-loaded: every caller renders `batch.user.username`, and
        # a lazy load there is one extra SELECT per row - ten queued batches on
        # the dashboard meant ten round trips for ten short strings.
        stmt: Select[Any] = select(Batch).options(selectinload(Batch.user))
        count_stmt = select(func.count()).select_from(Batch)
        if states:
            stmt = stmt.where(Batch.state.in_(states))
            count_stmt = count_stmt.where(Batch.state.in_(states))
        total = self.session.scalar(count_stmt) or 0
        page = max(1, page)
        rows = list(self.session.scalars(
            stmt.order_by(Batch.created_at.desc())
            .offset((page - 1) * per_page).limit(per_page)))
        return rows, total

    def set_state(self, batch: Batch, state: BatchState) -> None:
        batch.state = state
        if state is BatchState.RUNNING and batch.started_at is None:
            batch.started_at = _now()
        if state in (BatchState.COMPLETED, BatchState.FAILED):
            batch.finished_at = _now()
        self.session.flush()

    def progress(self, batch_id: int) -> BatchProgress | None:
        """Per-stage counts for one batch, in a single round trip."""
        return self.progress_many([batch_id]).get(batch_id)

    def progress_many(self, batch_ids: Sequence[int]) -> dict[int, BatchProgress]:
        """Per-stage counts for several batches, in a single round trip.

        This used to be one query *per stage per batch*. The docstring said "one
        grouped query" and it issued seven - a total, four stage group-bys and an
        overall - and the dashboard then called it once per completed batch in a
        loop. A dashboard page showing five finished batches cost about forty
        round trips, and it grew linearly with the page size.

        Conditional aggregation collapses all of it: every count the caller needs
        is a `SUM(CASE ...)` over the same scan, grouped by batch. `SUM(CASE)`
        rather than `COUNT(*) FILTER (...)` because the filter clause is
        PostgreSQL-and-recent-SQLite only, and `claim_next` documents that this
        layer runs on both.
        """
        ids = [int(b) for b in batch_ids]
        if not ids:
            return {}

        def tally(column: Any, value: Any) -> Any:
            return func.coalesce(func.sum(case((column == value, 1), else_=0)), 0)

        columns = [Document.batch_id, func.count().label("total")]
        # Order is fixed and read back positionally below.
        for stage, column in STAGE_COLUMNS.items():
            columns.append(tally(column, StageState.DONE).label(f"{stage}_done"))
            columns.append(tally(column, StageState.FAILED).label(f"{stage}_failed"))
        columns.append(tally(Document.overall_state, DocumentState.PROCESSED)
                       .label("processed"))
        columns.append(tally(Document.overall_state, DocumentState.FAILED)
                       .label("overall_failed"))
        columns.append(tally(Document.overall_state, DocumentState.NEEDS_REVIEW)
                       .label("needs_review"))

        rows = self.session.execute(
            select(*columns).where(Document.batch_id.in_(ids))
            .group_by(Document.batch_id)).mappings().all()
        counts = {row["batch_id"]: row for row in rows}

        result: dict[int, BatchProgress] = {}
        for batch_id in ids:
            batch = self.get(batch_id)
            if batch is None:
                continue
            # A batch with no documents yet produces no group; it is not an
            # error, and it must still report a progress of zero rather than
            # vanishing from the dashboard.
            row = counts.get(batch_id)
            total = int(row["total"]) if row else 0
            result[batch_id] = BatchProgress(
                batch_id=batch_id, name=batch.name, state=batch.state, total=total,
                completed=int(row["processed"]) if row else 0,
                failed=int(row["overall_failed"]) if row else 0,
                needs_review=int(row["needs_review"]) if row else 0,
                stages={
                    stage: StageProgress(
                        stage=stage,
                        done=int(row[f"{stage}_done"]) if row else 0,
                        failed=int(row[f"{stage}_failed"]) if row else 0,
                        total=total)
                    for stage in STAGE_COLUMNS
                })
        return result

    def is_finished(self, batch_id: int) -> bool:
        remaining = self.session.scalar(
            select(func.count()).select_from(Document)
            .where(Document.batch_id == batch_id,
                   Document.overall_state == DocumentState.PROCESSING)) or 0
        return remaining == 0


class DocumentRepository(_Base):
    def add_many(self, batch: Batch, entries: Iterable[dict[str, Any]]) -> list[Document]:
        """Register a batch's documents. Duplicate document_ids are skipped.

        Idempotent by design: re-registering after a crash mid-upload must not
        create a second row for the same document.
        """
        created: list[Document] = []
        existing = set(self.session.scalars(
            select(Document.document_id).where(Document.batch_id == batch.id)))
        for entry in entries:
            doc_id = str(entry.get("document_id") or "").strip()
            if not doc_id or doc_id in existing:
                continue
            doc = Document(
                batch_id=batch.id,
                document_id=doc_id,
                source_filename=str(entry.get("source_filename") or ""),
                source_path=entry.get("source_path"),
                page_count=int(entry.get("page_count") or 0),
                size_bytes=int(entry.get("size_bytes") or 0))
            self.session.add(doc)
            created.append(doc)
            existing.add(doc_id)
        self.session.flush()
        return created

    def get(self, document_pk: int) -> Document | None:
        return self.session.get(Document, document_pk)

    def claim_next(self, stage: str, batch_id: int | None = None,
                   max_attempts: int = 1) -> Document | None:
        """Atomically claim one document for a stage, or return None.

        Claiming is a **compare-and-swap**, which is portable and correct on both
        backends: find a candidate, then conditionally UPDATE it only while it is
        still PENDING. If another worker got there first the UPDATE matches zero
        rows and this returns None; the caller simply asks again.

        Why not rely on `SELECT ... FOR UPDATE SKIP LOCKED` alone: SQLite has no
        row-level locking and silently ignores `FOR UPDATE`, so on SQLite that
        approach would let two workers claim the same document. The CAS closes
        that. On PostgreSQL `SKIP LOCKED` is still applied to the candidate
        select, so concurrent workers pick *different* candidates and the CAS
        almost never has to retry - best of both.

        Also enforces stage ordering (a document is claimable only once its
        predecessor stages are done) and the retry cap.
        """
        if stage not in STAGE_COLUMNS:
            raise RepositoryError(f"unknown stage {stage!r}")
        column = STAGE_COLUMNS[stage]
        attempts = STAGE_ATTEMPTS.get(stage)

        conditions = [
            column == StageState.PENDING,
            Document.overall_state == DocumentState.PROCESSING,
        ]
        if batch_id is not None:
            conditions.append(Document.batch_id == batch_id)
        for earlier in STAGE_ORDER[:STAGE_ORDER.index(stage)]:
            conditions.append(STAGE_COLUMNS[earlier] == StageState.DONE)
        if attempts is not None:
            conditions.append(attempts <= max_attempts)

        candidate = select(Document.id).where(*conditions).order_by(Document.id).limit(1)
        # Row-level skip is a PostgreSQL capability; requesting it on SQLite would
        # be silently dropped, so only ask where it is honoured.
        if self.session.bind is not None and self.session.bind.dialect.name == "postgresql":
            candidate = candidate.with_for_update(skip_locked=True)

        document_pk = self.session.scalar(candidate)
        if document_pk is None:
            return None

        values: dict[str, Any] = {f"{stage}_state": StageState.RUNNING}
        if attempts is not None:
            values[f"{stage}_attempts"] = attempts + 1

        # The guard on `column == PENDING` is the compare half of the swap.
        result = self.session.execute(
            update(Document)
            .where(Document.id == document_pk, column == StageState.PENDING)
            .values(**values))
        if (result.rowcount or 0) != 1:
            return None  # lost the race; caller retries

        self.session.flush()
        doc = self.session.get(Document, document_pk)
        if doc is not None:
            # The UPDATE bypassed the identity map, so refresh any stale copy.
            self.session.refresh(doc)
        return doc

    def mark_stage(self, doc: Document, stage: str, state: StageState,
                   *, reason: str | None = None,
                   processing_status: str | None = None) -> None:
        if stage not in STAGE_COLUMNS:
            raise RepositoryError(f"unknown stage {stage!r}")
        setattr(doc, f"{stage}_state", state)
        if processing_status:
            doc.processing_status = processing_status
        if state is StageState.FAILED:
            doc.overall_state = DocumentState.FAILED
            doc.failure_reason = reason
        self.session.flush()

    def mark_overall(self, doc: Document, state: DocumentState,
                     reason: str | None = None) -> None:
        doc.overall_state = state
        if reason:
            doc.failure_reason = reason
        self.session.flush()

    def reset_running_to_pending(self, batch_id: int | None = None) -> int:
        """Crash recovery, run at startup.

        A document left RUNNING means the process died mid-stage. Returning it to
        PENDING makes it claimable again. Safe because stages are idempotent: the
        work is redone, not double-counted.
        """
        total = 0
        for stage in STAGE_ORDER:
            column = STAGE_COLUMNS[stage]
            stmt = update(Document).where(column == StageState.RUNNING)
            if batch_id is not None:
                stmt = stmt.where(Document.batch_id == batch_id)
            result = self.session.execute(
                stmt.values(**{f"{stage}_state": StageState.PENDING}))
            total += result.rowcount or 0
        self.session.flush()
        return total

    def list_for_batch(self, batch_id: int, page: int = 1, per_page: int = 10,
                       states: Sequence[DocumentState] | None = None,
                       search: str | None = None) -> tuple[list[Document], int]:
        """Paginated document list. Batch popup shows 10 per page.

        The three child collections are eager-loaded because every caller reads
        them - the Data View renders parties and consideration per row, and the
        export reads all of them for every document. Lazily they cost three
        SELECTs per row: a ten-document page issued thirty round trips to render
        one table, and a thousand-document export issued three thousand.

        `selectinload` rather than a join: these are collections, and joining
        would multiply the document rows by the number of persons before the
        ORM de-duplicated them, which breaks LIMIT/OFFSET on the outer query.
        """
        stmt: Select[Any] = select(Document).where(
            Document.batch_id == batch_id).options(
                selectinload(Document.persons),
                selectinload(Document.property_),
                selectinload(Document.validations))
        count_stmt = select(func.count()).select_from(Document).where(
            Document.batch_id == batch_id)
        if states:
            stmt = stmt.where(Document.overall_state.in_(states))
            count_stmt = count_stmt.where(Document.overall_state.in_(states))
        if search:
            like = f"%{search.strip()}%"
            stmt = stmt.where(Document.document_id.ilike(like))
            count_stmt = count_stmt.where(Document.document_id.ilike(like))
        total = self.session.scalar(count_stmt) or 0
        rows = list(self.session.scalars(
            stmt.order_by(Document.id)
            .offset((max(1, page) - 1) * per_page).limit(per_page)))
        return rows, total

    def failed_for_batch(self, batch_id: int) -> list[Document]:
        return list(self.session.scalars(
            select(Document).where(Document.batch_id == batch_id,
                                   Document.overall_state == DocumentState.FAILED)
            .order_by(Document.id)))

    def reprocess_failed(self, batch_id: int) -> int:
        """Requeue failed documents from the start, clearing their attempt counts."""
        docs = self.failed_for_batch(batch_id)
        for doc in docs:
            doc.overall_state = DocumentState.PROCESSING
            doc.failure_reason = None
            doc.ocr_attempts = 0
            doc.extract_attempts = 0
            for stage in STAGE_ORDER:
                setattr(doc, f"{stage}_state", StageState.PENDING)
        self.session.flush()
        return len(docs)

    # -- documents whose OCR failed ----------------------------------------
    #
    # Kept separate from `failed_for_batch`, which asks a different question.
    # A document can be FAILED for a reason that has nothing to do with OCR -
    # extraction gave up, validation rejected it - and rerunning OCR on those
    # would redo minutes of GPU work that was never the problem. The stage
    # column is the honest test: `ocr_state is FAILED` means OCR is what failed.

    def failed_ocr(self, batch_id: int | None = None, *,
                   page: int = 1, per_page: int = 25) -> tuple[list[Document], int]:
        """Documents whose OCR stage failed, newest batch first.

        Paginated: a 1000-file batch can fail 200 documents on a bad scanner
        run, and rendering all of them into the page would freeze the window.
        """
        conditions = [Document.ocr_state == StageState.FAILED]
        if batch_id is not None:
            conditions.append(Document.batch_id == batch_id)
        total = self.session.scalar(
            select(func.count()).select_from(Document).where(*conditions)) or 0
        rows = list(self.session.scalars(
            select(Document).where(*conditions)
            .order_by(Document.batch_id.desc(), Document.id)
            .offset((max(1, page) - 1) * per_page).limit(per_page)))
        return rows, total

    def record_validation(self, doc: Document, result: Any) -> None:
        """Store a `pdf_validation.ValidationResult` against the document.

        Overwrites any previous result: revalidation exists precisely so a
        repaired file can supersede the verdict on the broken one, and keeping
        the old row would leave the interface showing a corruption that has
        been fixed.
        """
        doc.validation_status = result.status
        doc.validation_error_code = result.error_code
        doc.validation_error_message = result.error_message
        doc.corrupted_pages = (",".join(str(p) for p in result.corrupted_pages)
                               or None)
        doc.validated_at = result.validated_at
        doc.validator_version = result.validator_version
        doc.is_retryable = result.retryable
        if result.page_count and not doc.page_count:
            doc.page_count = result.page_count
        self.session.flush()

    def corrupted(self, batch_id: int | None = None, *, page: int = 1,
                  per_page: int = 25) -> tuple[list[Document], int]:
        """Documents whose PDF the validator found fault with.

        Filtered on the validation verdict rather than on the stage that failed:
        the question this list answers is "which files are broken", and a file
        can be broken whichever stage happened to notice.
        """
        from core.pdf_validation import CORRUPT_STATUSES

        conditions: list[Any] = [
            Document.validation_status.in_(sorted(CORRUPT_STATUSES))]
        if batch_id is not None:
            conditions.append(Document.batch_id == batch_id)
        total = self.session.scalar(
            select(func.count()).select_from(Document).where(*conditions)) or 0
        rows = list(self.session.scalars(
            select(Document).where(*conditions)
            .order_by(Document.batch_id.desc(), Document.id)
            .offset((max(1, page) - 1) * per_page).limit(per_page)))
        return rows, total

    def pending_ocr_count(self, batch_id: int) -> int:
        """Documents in this batch still waiting for the OCR stage.

        Used by the runner to decide whether to keep draining OCR or to carry a
        finished document straight through the rest of its pipeline.
        """
        return self.session.scalar(
            select(func.count()).select_from(Document)
            .where(Document.batch_id == batch_id,
                   Document.ocr_state == StageState.PENDING,
                   Document.overall_state == DocumentState.PROCESSING)) or 0

    def failed_ocr_count(self, batch_id: int | None = None) -> int:
        conditions = [Document.ocr_state == StageState.FAILED]
        if batch_id is not None:
            conditions.append(Document.batch_id == batch_id)
        return self.session.scalar(
            select(func.count()).select_from(Document).where(*conditions)) or 0

    def requeue_ocr(self, document_pks: Sequence[int]) -> list[Document]:
        """Make the named documents claimable by the OCR stage again.

        Every later stage is reset with it. OCR output is the input to
        extraction, translation and validation, so a document that reruns OCR
        and keeps a DONE extraction would export results derived from text that
        no longer exists.

        `ocr_attempts` is cleared because the retry cap has already been spent -
        leaving it would make `claim_next` skip the document and the rerun would
        silently do nothing, which is the worst possible outcome for a button
        the operator pressed on purpose.

        Documents that did not fail OCR are ignored rather than reset: this is
        reachable from the UI with a stale document id, and quietly restarting a
        healthy document would destroy finished work.
        """
        if not document_pks:
            return []
        docs = list(self.session.scalars(
            select(Document).where(Document.id.in_(list(document_pks)),
                                   Document.ocr_state == StageState.FAILED)))
        for doc in docs:
            doc.overall_state = DocumentState.PROCESSING
            doc.failure_reason = None
            doc.processing_status = None
            doc.ocr_attempts = 0
            doc.extract_attempts = 0
            for stage in STAGE_ORDER:
                setattr(doc, f"{stage}_state", StageState.PENDING)
        self.session.flush()
        return docs


class OcrRepository(_Base):
    def save_pages(self, doc: Document, pages: Iterable[tuple[int, str]]) -> int:
        """Replace a document's OCR pages. Idempotent across retries.

        Children are appended to `doc.ocr_pages` rather than constructed with an
        explicit document_id. Setting the FK directly leaves the already-loaded
        collection stale, so a second call deletes nothing and the insert then
        violates the (document_id, page_number) unique constraint - which would
        break retry safety, the very property this method exists to provide.
        """
        doc.ocr_pages.clear()
        self.session.flush()

        count = 0
        for number, text in pages:
            doc.ocr_pages.append(
                OcrPage(page_number=number, text=text, char_count=len(text)))
            count += 1
        doc.page_count = max(doc.page_count, count)
        self.session.flush()
        return count

    def full_text(self, doc: Document) -> str:
        """Reassemble cleaned OCR with canonical page markers."""
        pages = self.session.scalars(
            select(OcrPage).where(OcrPage.document_id == doc.id)
            .order_by(OcrPage.page_number))
        return "\n".join(f"===== PAGE {p.page_number} =====\n{p.text}" for p in pages)

    def purge_expired(self, ttl_days: int = 30) -> int:
        """Delete OCR text past its TTL. Never backed up; safe to drop."""
        cutoff = _now() - timedelta(days=ttl_days)
        result = self.session.execute(
            OcrPage.__table__.delete().where(OcrPage.created_at < cutoff))
        self.session.flush()
        return result.rowcount or 0


class ExtractionRepository(_Base):
    def record(self, doc: Document, *, attempt: int, raw_output: str | None,
               parsed_ok: bool, pan_coverage: float | None,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               truncated: bool = False, model_name: str | None = None,
               quantisation: str | None = None, prompt_name: str | None = None,
               duration_s: float | None = None) -> Extraction:
        """Record one attempt. Re-recording the same attempt overwrites it."""
        existing = self.session.scalar(
            select(Extraction).where(Extraction.document_id == doc.id,
                                     Extraction.attempt == attempt))
        target = existing or Extraction(document_id=doc.id, attempt=attempt)
        target.raw_output = raw_output
        target.parsed_ok = parsed_ok
        target.pan_coverage = pan_coverage
        target.prompt_tokens = prompt_tokens
        target.completion_tokens = completion_tokens
        target.truncated = truncated
        target.model_name = model_name
        target.quantisation = quantisation
        target.prompt_name = prompt_name
        target.duration_s = duration_s
        if existing is None:
            self.session.add(target)
        self.session.flush()
        return target

    def latest(self, doc: Document) -> Extraction | None:
        return self.session.scalar(
            select(Extraction).where(Extraction.document_id == doc.id)
            .order_by(Extraction.attempt.desc()).limit(1))


class ResultRepository(_Base):
    """Persists the structured result: property, persons, validation flags."""

    def save_property(self, doc: Document, data: dict[str, Any],
                      stamp_value: int | None = None) -> Property:
        prop = doc.property_ or Property(document_id=doc.id)
        prop.schedule_c_address = data.get("schedule_c_property_address")
        prop.state = data.get("state")
        prop.sale_consideration = _as_int(data.get("sale_consideration"))
        prop.registration_fee = _as_int(data.get("registration_fee"))
        prop.stamp_value = stamp_value
        paid = data.get("paid_in_cash")
        prop.paid_in_cash = None if paid not in ("yes", "no") else paid == "yes"
        if doc.property_ is None:
            self.session.add(prop)
        self.session.flush()
        return prop

    def save_document_meta(self, prop: Property, data: dict[str, Any]) -> None:
        raw = data.get("transaction_date")
        prop.transaction_date = _as_date(raw)
        prop.registration_office = data.get("registration_office")
        self.session.flush()

    def replace_persons(self, doc: Document, extraction: dict[str, Any]) -> list[Person]:
        """Replace all persons for a document. Idempotent across retries."""
        # Clear through the relationship so the in-session collection stays in
        # step; delete-orphan cascade issues the DELETEs.
        doc.persons.clear()
        self.session.flush()

        created: list[Person] = []
        for key, relation in (("buyer_details", PersonRelation.BUYER),
                              ("seller_details", PersonRelation.SELLER)):
            for ordinal, entry in enumerate(extraction.get(key) or [], start=1):
                if not isinstance(entry, dict):
                    continue
                person = Person(
                    relation=relation, ordinal=ordinal,
                    name=entry.get("name"),
                    name_translated=entry.get("name_translated"),
                    gender=entry.get("gender"),
                    father_name=entry.get("father_name"),
                    father_name_translated=entry.get("father_name_translated"),
                    aadhaar_number=_digits_or_none(entry.get("aadhaar_number"), 12),
                    pan_card_number=_pan_or_none(entry.get("pan_card_number")),
                    address=entry.get("address"),
                    address_translated=entry.get("address_translated"),
                    state=entry.get("state"))
                doc.persons.append(person)
                created.append(person)
        self.session.flush()
        return created

    def record_flags(self, doc: Document, entries: Iterable[dict[str, Any]]) -> int:
        """Replace a document's validation findings."""
        doc.validations.clear()
        self.session.flush()

        count = 0
        for entry in entries:
            doc.validations.append(ValidationResult(
                person_id=entry.get("person_id"),
                flag_code=str(entry.get("flag_code") or "")[:12],
                field=entry.get("field"),
                detail=entry.get("detail"),
                confidence=entry.get("confidence")))
            count += 1
        self.session.flush()
        return count


class SettingsRepository(_Base):
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.session.get(Setting, key)
        return row.value if row and row.value is not None else default

    def set(self, key: str, value: str | None) -> None:
        row = self.session.get(Setting, key)
        if row is None:
            self.session.add(Setting(key=key, value=value))
        else:
            row.value = value
        self.session.flush()

    def all(self) -> dict[str, str | None]:
        return {row.key: row.value for row in self.session.scalars(select(Setting))}


class MaintenanceRepository(_Base):
    """Backup rotation, cache expiry and growth checks."""

    def documents_before(self, year: int) -> int:
        cutoff = datetime(year, 1, 1, tzinfo=UTC)
        return self.session.scalar(
            select(func.count()).select_from(Document)
            .where(Document.created_at < cutoff)) or 0

    def purge_year(self, year: int) -> int:
        """Delete batches created before `year`. Archive to /backup first.

        Cascades to documents, OCR pages, extractions, persons and validations.
        """
        cutoff = datetime(year, 1, 1, tzinfo=UTC)
        batches = list(self.session.scalars(
            select(Batch).where(Batch.created_at < cutoff)))
        for batch in batches:
            self.session.delete(batch)
        self.session.flush()
        return len(batches)

    def purge_logs(self, retention_days: int = 30) -> int:
        cutoff = _now() - timedelta(days=retention_days)
        result = self.session.execute(
            LogEntry.__table__.delete().where(LogEntry.created_at < cutoff))
        self.session.flush()
        return result.rowcount or 0

    def table_counts(self) -> dict[str, int]:
        """Row counts for the health view, so growth is visible before it hurts."""
        counts: dict[str, int] = {}
        for model in (Batch, Document, OcrPage, Extraction, Person,
                      ValidationResult, LogEntry):
            counts[model.__tablename__] = self.session.scalar(
                select(func.count()).select_from(model)) or 0
        return counts


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _as_date(value: Any) -> Any:
    """Accept ISO `YYYY-MM-DD` only; anything else becomes NULL.

    The model is instructed to emit ISO. Guessing at other formats risks
    transposing day and month, which would be a silent data error.
    """
    text = str(value or "").strip()
    if len(text) != 10:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _digits_or_none(value: Any, length: int) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits if len(digits) == length else None


def _pan_or_none(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if len(text) != 10:
        return None
    ok = (text[:5].isalpha() and text[5:9].isdigit() and text[9].isalpha())
    return text if ok else None
