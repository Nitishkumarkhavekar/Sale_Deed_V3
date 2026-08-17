"""Batch runner - the orchestration layer.

Drives documents through the stages, persists every step, and honours the queue
rules. Everything database-coupled lives here so `stages.py` stays testable
without a server.

**Per-document pipelining, not stage-by-stage.** A worker takes one document all
the way through OCR -> extract -> validate -> translate before picking up the
next. Two reasons:

  * A document becomes durable and complete as early as possible, which is what
    "continuous commit" is for. Stage-by-stage would leave every document in the
    batch half-finished until the last stage started.
  * It matches the hardware. CPU work (PDF rendering, OCR, validation) overlaps
    across workers while the GPU stage serialises behind the governor's lease -
    so the 4 GB card never has two models resident, and the CPU is not idle
    waiting for it.

Each stage commits in its own transaction. A crash loses at most the stage in
flight, and `reset_running_to_pending()` at startup makes that stage claimable
again. Stages are idempotent (delete-then-insert), so redoing work never
double-counts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from .. import paths
from ..db.engine import session_scope
from ..db.models import Batch, BatchState, Document, DocumentState, StageState
from ..db.repositories import UnitOfWork
from ..transaction_id import extract as extract_transaction_id

#: Cleaned copies live here. Beside the exports rather than beside the
#: originals: the source directory is the user's, and writing into it
#: would put derived files among their records.
CLEANED_DIR = paths.CLEANED_DIR

log = logging.getLogger("saledeed.pipeline.runner")
from .stages import (
    ExtractStage,
    OcrStage,
    StageName,
    StageOutcome,
    TranslateStage,
    ValidateStage,
    find_surya,
)


#: Returned by a stage wrapper when the document was put back to PENDING for a
#: retry. Distinct from None (permanent failure) because a requeued document must
#: NOT be finalised: `claim_next` only considers documents whose overall_state is
#: still PROCESSING, so marking one FAILED or NEEDS_REVIEW after requeueing it
#: strands it forever - requeued but permanently unclaimable.
REQUEUED = object()


class RunnerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class BatchMode(str, Enum):
    MANUAL = "manual"
    #: Start the next queued batch automatically after a cooldown.
    AUTO = "auto"


@dataclass
class RunnerStats:
    documents_processed: int = 0
    documents_failed: int = 0
    documents_review: int = 0
    batches_completed: int = 0
    started_at: float | None = None
    last_document_at: float | None = None
    #: Rolling per-document durations, for the dashboard's ETA and speed figures.
    recent_durations: list[float] = field(default_factory=list)

    def record(self, duration: float, outcome: DocumentState) -> None:
        self.last_document_at = time.monotonic()
        self.recent_durations.append(duration)
        if len(self.recent_durations) > 50:
            del self.recent_durations[0]
        if outcome is DocumentState.PROCESSED:
            self.documents_processed += 1
        elif outcome is DocumentState.FAILED:
            self.documents_failed += 1
        elif outcome is DocumentState.NEEDS_REVIEW:
            self.documents_review += 1

    @property
    def seconds_per_document(self) -> float:
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)

    def eta_seconds(self, remaining: int) -> float:
        rate = self.seconds_per_document
        return remaining * rate if rate else 0.0


@dataclass
class Stages:
    """The stage objects, injected so tests can substitute doubles."""

    ocr: OcrStage
    extract: ExtractStage
    validate: ValidateStage
    translate: TranslateStage


class BatchRunner:
    """Owns the processing loop for one machine.

    Thread-safe. `start`, `pause` and `stop` may be called from the UI thread; the
    work happens on worker threads and never blocks the caller.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        stages: Stages,
        *,
        governor: Any | None = None,
        mode: BatchMode = BatchMode.MANUAL,
        auto_cooldown_s: float = 60.0,
        max_workers: int = 2,
        ocr_retry_limit: int = 1,
        extract_retry_limit: int = 1,
        on_document: Callable[[Document, DocumentState], None] | None = None,
        on_batch_complete: Callable[[int], None] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.stages = stages
        self.governor = governor
        self.mode = mode
        self.auto_cooldown_s = auto_cooldown_s
        self.max_workers = max(1, max_workers)
        self.ocr_retry_limit = ocr_retry_limit
        self.extract_retry_limit = extract_retry_limit
        self.on_document = on_document
        self.on_batch_complete = on_batch_complete

        self.stats = RunnerStats()
        self._state = RunnerState.IDLE
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._threads: list[threading.Thread] = []
        self._current_batch_id: int | None = None
        self._detail = "idle"

    # -- state ------------------------------------------------------------

    @property
    def state(self) -> RunnerState:
        with self._lock:
            return self._state

    def _set_state(self, state: RunnerState, detail: str = "") -> None:
        with self._lock:
            self._state = state
            if detail:
                self._detail = detail

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "detail": self._detail,
                "mode": self.mode.value,
                "current_batch_id": self._current_batch_id,
                "workers": len([t for t in self._threads if t.is_alive()]),
                "processed": self.stats.documents_processed,
                "failed": self.stats.documents_failed,
                "review": self.stats.documents_review,
                "batches_completed": self.stats.batches_completed,
                "seconds_per_document": round(self.stats.seconds_per_document, 2),
            }

    # -- control ----------------------------------------------------------

    def recover(self) -> int:
        """Reset documents stranded mid-stage by a crash. Call once at startup.

        Anything left RUNNING means the process died holding it. Returning those
        to PENDING makes them claimable again; safe because every stage is
        idempotent.

        Batches caught mid-stop are settled here too, and the order matters:
        documents are released *first*, so `settle_stopping` then sees zero in
        flight and completes the stop. Reversed, a batch killed during a stop
        would stay in `STOPPING` - a state with no exit and no Run button -
        until someone edited the database by hand.
        """
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            released = uow.documents.reset_running_to_pending()
            settled = uow.batches.settle_stopping()
        if settled:
            log.info("settled %d batch(es) left mid-stop by a restart",
                     len(settled), extra={"batches": settled})
        self._fail_stranded()
        return released

    def _fail_stranded(self, batch_id: int | None = None) -> int:
        """Give a terminal state to documents nothing can ever claim.

        The backstop for the condition described on `reset_running_to_pending`:
        a stage at PENDING with the retry cap already exceeded is unclaimable,
        but `overall_state` is still PROCESSING, so `is_finished` is false for
        ever and the batch holds `RUNNING` - and every queued batch waits behind
        it indefinitely. Marking these FAILED is what they already are in fact;
        it just says so, so the queue can move and the operator can see them on
        the Failed OCR page and retry them deliberately.
        """
        limits = {"ocr": self.ocr_retry_limit, "extract": self.extract_retry_limit}
        failed = 0
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            for doc_pk, stage in uow.documents.stranded(limits, batch_id):
                doc = uow.documents.get(doc_pk)
                if doc is None:
                    continue
                reason = (f"{stage} gave up after "
                          f"{getattr(doc, f'{stage}_attempts', 0)} attempts")
                uow.documents.mark_stage(doc, stage, StageState.FAILED,
                                         reason=reason,
                                         processing_status="OCR_F"
                                         if stage == "ocr" else None)
                self._append_failure(uow, doc, stage, reason, None)
                failed += 1
        if failed:
            log.warning("%d document(s) had exhausted their retries while still "
                        "marked in-progress; failed them so the batch can finish",
                        failed, extra={"documents": failed, "batch": batch_id})
        return failed

    def start(self) -> None:
        """Begin processing. Resumes from PAUSED without losing progress."""
        with self._lock:
            if self._state is RunnerState.RUNNING:
                return
            self._stop.clear()
            self._pause.clear()
            self._state = RunnerState.RUNNING
            self._detail = "running"
            if self.stats.started_at is None:
                self.stats.started_at = time.monotonic()
            alive = [t for t in self._threads if t.is_alive()]
            self._threads = alive
            needed = self.max_workers - len(alive)
            log.info("batch runner started: %d worker(s), mode=%s",
                     self.max_workers, self.mode.value,
                     extra={"workers": self.max_workers, "mode": self.mode.value,
                            "ocr": self.stages.ocr.engine,
                            "translate": self.stages.translate.engine})

        for i in range(max(0, needed)):
            thread = threading.Thread(target=self._loop, name=f"pipeline-{i}",
                                      daemon=True)
            thread.start()
            with self._lock:
                self._threads.append(thread)
        self._wake.set()

    def pause(self) -> None:
        """Stop taking new documents; let in-flight ones finish.

        Never kills work mid-document - that would lose a document's progress.
        Because every stage commits, resuming continues exactly where it stopped.
        """
        self._pause.set()
        self._set_state(RunnerState.PAUSING, "pausing - finishing in-flight documents")

    def stop(self, timeout: float = 120.0) -> None:
        """Halt and join workers. In-flight documents are allowed to finish."""
        log.info("batch runner stopping: %d processed, %d review, %d failed",
                 self.stats.documents_processed, self.stats.documents_review,
                 self.stats.documents_failed,
                 extra={"processed": self.stats.documents_processed,
                        "review": self.stats.documents_review,
                        "failed": self.stats.documents_failed})
        self._set_state(RunnerState.STOPPING, "stopping")
        self._stop.set()
        self._wake.set()
        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(timeout=max(0.1, deadline - time.monotonic()))
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
        self._set_state(RunnerState.STOPPED, "stopped")

    # -- main loop --------------------------------------------------------

    def _idle(self, seconds: float) -> None:
        """Wait, and mean it.

        `_wake` is set by `start` and `stop` to interrupt an idle worker
        promptly. Nothing ever cleared it, so once `start` had run every
        `_wake.wait(timeout=...)` in this loop returned immediately and all
        three idle paths became busy-waits: an application with nothing to do
        re-ran the claim query - a locking `SELECT ... FOR UPDATE` per worker -
        as fast as the database would answer, for as long as it was open.

        Clearing the flag after the wait restores the intended behaviour. Order
        matters: cleared *before* waiting, a `set` racing in between would be
        swallowed and the worker would sleep through a start it was meant to
        react to.
        """
        self._wake.wait(timeout=seconds)
        self._wake.clear()

    def _loop(self) -> None:
        idle_sleep = 1.0
        while not self._stop.is_set():
            if self._pause.is_set():
                self._set_state(RunnerState.PAUSED, "paused")
                self._idle(1.0)
                continue

            if self.governor is not None:
                plan = self.governor.plan()
                if not plan.admit_new_work:
                    # Resource pressure. Wait rather than pile on.
                    self._set_state(RunnerState.RUNNING,
                                    f"waiting - system pressure {plan.pressure.label}")
                    time.sleep(5.0)
                    continue

            batch_id = self._ensure_active_batch()
            if batch_id is None:
                self._set_state(RunnerState.RUNNING, "no batch queued")
                self._idle(idle_sleep)
                continue

            if not self._process_one(batch_id):
                # Nothing claimable in this batch: either finished, or every
                # remaining document is held by another worker - or none of
                # them can ever be claimed again. The last case is the one that
                # used to wedge the queue permanently, so it is checked here as
                # well as at startup: this is the only moment the runner can
                # observe "a RUNNING batch with no claimable work".
                if self._fail_stranded(batch_id):
                    continue  # something changed; re-evaluate immediately
                self._finalise_if_complete(batch_id)
                self._idle(idle_sleep)

    def _ensure_active_batch(self) -> int | None:
        """Return the running batch, promoting a queued one if needed.

        Also the point where a stop takes effect. A batch asked to stop leaves
        `RUNNING`, so it stops being returned here and no further document is
        claimed from it; the worker still inside one finishes and releases it,
        and the next pass through settles the batch to `STOPPED`. Nothing is
        interrupted and nothing is lost.

        Because the stop is recorded on the batch row rather than on the runner,
        stopping one batch leaves every other batch - and the runner itself -
        untouched: the queue simply advances to the next one.
        """
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            for stopped_id in uow.batches.settle_stopping():
                log.info("batch stopped; in-flight work finished cleanly",
                         extra={"batch": stopped_id})
                if self._current_batch_id == stopped_id:
                    self._current_batch_id = None
                    self._detail = f"stopped batch {stopped_id}"

            active = uow.batches.active()
            if active is not None:
                self._current_batch_id = active.id
                return active.id

            if self.mode is BatchMode.AUTO and self.stats.last_document_at:
                elapsed = time.monotonic() - self.stats.last_document_at
                if elapsed < self.auto_cooldown_s:
                    self._detail = (f"auto cooldown "
                                    f"{self.auto_cooldown_s - elapsed:.0f}s remaining")
                    return None

            nxt = uow.batches.next_queued()
            if nxt is None:
                return None
            uow.batches.set_state(nxt, BatchState.RUNNING)
            self._current_batch_id = nxt.id
            self._detail = f"started batch {nxt.name}"
            return nxt.id

    # -- one document -----------------------------------------------------

    def _process_one(self, batch_id: int) -> bool:
        """Claim and fully process one document. False when nothing was claimed."""
        started = time.monotonic()
        log.debug("looking for work", extra={"batch": batch_id})

        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", batch_id,
                                           max_attempts=self.ocr_retry_limit + 1)
            if doc is None:
                # Nothing needs OCR; a later stage may still have work.
                doc = self._claim_downstream(uow, batch_id)
                if doc is None:
                    return False
            doc_pk = doc.id
            # The prepared copy when there is one, the original otherwise.
            # Everything downstream - OCR, the viewer, re-processing - reads
            # this one path, so there is a single notion of "the document".
            pdf_path = doc.cleaned_path or doc.source_path or doc.source_filename
            needs_ocr = doc.ocr_state is StageState.RUNNING

        ocr_text: str | None = None

        if needs_ocr:
            # Prepared once, before OCR reads it. Removing a diagonal stamp
            # measurably helps Surya, which recognises a *rendered image* of the
            # page - an overlay across the text degrades it exactly as it would
            # a human reader.
            pdf_path = self._prepare_document(doc_pk, pdf_path)
            result = self._do_ocr(doc_pk, pdf_path)
            if result is REQUEUED:
                # Back in the queue; leave overall_state alone so it stays claimable.
                return True
            if result is None:
                self._finish(doc_pk, DocumentState.FAILED, started)
                return True
            ocr_text = result  # type: ignore[assignment]

            # Stop here and let the loop claim the next document needing OCR.
            #
            # Carrying straight on to extraction makes the batch alternate
            # between the two GPU models - Surya out, the language model in, back
            # out again - once per document. Each swap costs about 5 s measured,
            # so a run paid ~10 s per document to move weights around.
            #
            # Returning here drains OCR first; `_claim_downstream` then picks
            # these documents up for extraction, which is exactly the case it
            # was written for ("OCR ran in an earlier run"). The number of swaps
            # per batch drops from two-per-document to two.
            #
            # Nothing is lost if the process dies in between: `ocr_state` is
            # committed DONE with the text, and the document stays claimable by
            # the later stages. That is the same guarantee as before - it is why
            # resume works at all.
            if self._more_ocr_pending(batch_id):
                return True
        else:
            with session_scope(self.session_factory) as session:
                uow = UnitOfWork(session)
                doc = uow.documents.get(doc_pk)
                ocr_text = uow.ocr.full_text(doc) if doc else None
            if not ocr_text:
                self._finish(doc_pk, DocumentState.FAILED, started,
                             "no OCR text available")
                return True

        extracted = self._do_extract(doc_pk, ocr_text)
        if extracted is REQUEUED:
            return True
        if extracted is None:
            # _do_extract already recorded NEEDS_REVIEW with the reason; only the
            # timing stats are outstanding.
            self._finish(doc_pk, DocumentState.NEEDS_REVIEW, started)
            return True
        parsed = extracted  # type: ignore[assignment]

        disposition = self._do_validate(doc_pk, parsed, ocr_text)
        self._do_translate(doc_pk, parsed)

        final = (DocumentState.PROCESSED if disposition == "accept"
                 else DocumentState.NEEDS_REVIEW)
        self._finish(doc_pk, final, started)
        return True

    @staticmethod
    def _append_failure(uow: UnitOfWork, doc: Document, stage: str,
                        detail: str, verdict: Any = None) -> None:
        """Append the diagnosis to the document's history.

        Best-effort by design: a failure to *record* a failure must never turn
        one bad document into a dead batch. The stage columns are already
        written by the caller, so losing an event costs history, not state.
        """
        try:
            from core.failure_codes import classify_text

            code, message, retryable = classify_text(
                detail, getattr(verdict, "status", None))
            uow.documents.record_failure(
                doc, stage=stage, code=code, message=message,
                technical=detail or "", retryable=retryable)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not record failure event: %s", exc)

    @staticmethod
    def _validate_failed_pdf(pdf_path: str, doc: Document, detail: str) -> Any:
        """Ask what the file actually is, after a stage has failed on it.

        Returns a `ValidationResult`, or None if the validator itself could not
        run - in which case the original stage failure stands unchanged, which
        is the honest outcome: nothing new was learned.

        The source file is examined, not the prepared copy. A cleaned copy is
        this application's own output; if it is damaged, the operator needs to
        know whether their original is sound before they go looking for a file
        to replace.
        """
        from core.pdf_validation import Status, ValidationResult, validate_pdf

        try:
            original = doc.source_path or doc.source_filename
            result = validate_pdf(original or pdf_path)
        except Exception as exc:  # noqa: BLE001 - never break the failure path
            log.warning("[PDF_VALIDATOR] validator itself failed on %s: %s",
                        doc.source_filename, exc)
            return None

        if result.is_valid:
            # The file is fine, so the failure was somewhere else. Saying so is
            # the single most useful thing this feature does: it stops an
            # operator hunting for a corrupt file that does not exist.
            return ValidationResult(
                is_valid=True, status=Status.PROCESSING_ERROR,
                page_count=result.page_count,
                error_code="PROCESSING_ERROR",
                error_message=("The PDF itself is readable; processing failed "
                               f"for another reason ({(detail or '')[:120]})."))
        return result

    def _more_ocr_pending(self, batch_id: int) -> bool:
        """Is there another document waiting for OCR in this batch?

        Asked so the last document of a batch is not left sitting with its OCR
        done and nothing to trigger the rest of its pipeline: when this is the
        only one left, processing continues inline exactly as it always did.
        """
        with session_scope(self.session_factory) as session:
            return UnitOfWork(session).documents.pending_ocr_count(batch_id) > 0

    def _claim_downstream(self, uow: UnitOfWork, batch_id: int) -> Document | None:
        """Pick up a document whose OCR is done but later stages are pending.

        Happens after a crash between stages, or when OCR ran in an earlier run.
        """
        for stage in ("extract", "validate", "translate"):
            limit = self.extract_retry_limit + 1 if stage == "extract" else 1
            doc = uow.documents.claim_next(stage, batch_id, max_attempts=limit)
            if doc is not None:
                return doc
        return None

    def _lease(self, stage: object, name: str) -> Any:
        """GPU lease, but only for stages that actually touch the GPU.

        On a card too small for model co-residency the lease is what stops Surya,
        the LLM and IndicTrans occupying VRAM simultaneously. Taking it for a
        CPU-only backend would serialise CPU work behind GPU work and lose
        throughput for nothing.
        """
        if self.governor is None or not getattr(stage, "uses_gpu", False):
            return _NullLease()
        return self.governor.gpu_lease(name)

    # -- stage wrappers, each its own transaction -------------------------

    def _prepare_document(self, doc_pk: int, pdf_path: str) -> str:
        """Remove separable overlays and record the cleaned copy.

        Returns the path to use. Any failure returns the original: a deed that
        cannot be cleaned must still be processed, and preparation is an
        improvement to the input, never a precondition for it.
        """
        from ..pdf_prepare import prepare

        try:
            with session_scope(self.session_factory) as session:
                doc = UnitOfWork(session).documents.get(doc_pk)
                if doc is None:
                    return pdf_path
                if doc.cleaned_path and Path(doc.cleaned_path).is_file():
                    # Already prepared - a retry must not redo the work, and
                    # re-running removal on an already-cleaned file would search
                    # for a watermark that is no longer there.
                    return doc.cleaned_path

            result = prepare(pdf_path, CLEANED_DIR)
            if not result.ok:
                return pdf_path

            with session_scope(self.session_factory) as session:
                doc = UnitOfWork(session).documents.get(doc_pk)
                if doc is not None:
                    doc.cleaned_path = str(result.output)
            return str(result.output)
        except Exception as exc:  # noqa: BLE001 - never lose a document to this
            log.error("document preparation failed", extra={
                "document": doc_pk, "error": f"{type(exc).__name__}: {exc}"})
            return pdf_path

    def _log_stage(self, stage: str, doc_pk: int, outcome: Any) -> None:
        """One line per stage per document, at the point the stage returns.

        Emitted here rather than inside each stage because this is the only
        place that knows both the outcome and the document it belongs to -
        logging in both would print every step twice, which is the usual way a
        log stops being read.
        """
        seconds = outcome.data.get("duration_s") or outcome.duration_s
        if outcome.ok:
            log.info("%s ok in %.2fs", stage, seconds or 0.0,
                     extra={"document": doc_pk, "stage": stage,
                            **{k: v for k, v in outcome.data.items()
                               if k in ("pages", "chars", "translated", "engine",
                                        "device", "disposition", "confidence")}})
        else:
            log.warning("%s failed: %s", stage, outcome.detail,
                        extra={"document": doc_pk, "stage": stage,
                               "retryable": outcome.retryable})

    def _do_ocr(self, doc_pk: int, pdf_path: str) -> str | object | None:
        # Surya is a GPU model and must not be resident alongside the LLM.
        with self._lease(self.stages.ocr, "ocr"):
            outcome = self.stages.ocr.run(pdf_path)
            self._log_stage("ocr", doc_pk, outcome)
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return None
            if not outcome.ok:
                retry = outcome.retryable and doc.ocr_attempts <= self.ocr_retry_limit

                # The file is examined only now, when something has already gone
                # wrong. OCR reads the PDF, so its failure is the moment at
                # which "is the file itself broken?" becomes worth answering -
                # and answering it costs a few KB of reads against a stage that
                # has just spent minutes of GPU time.
                #
                # Never allowed to raise: this runs inside the failure path, and
                # an exception here would turn one bad document into a dead
                # batch, which is the opposite of the point.
                verdict = self._validate_failed_pdf(pdf_path, doc, outcome.detail)
                if verdict is not None and verdict.is_corrupt:
                    # A file that cannot be opened will not open on the next
                    # attempt either. Stop retrying it and say why.
                    retry = False

                uow.documents.mark_stage(
                    doc, "ocr",
                    StageState.PENDING if retry else StageState.FAILED,
                    reason=(verdict.error_message
                            if verdict is not None and verdict.is_corrupt
                            else outcome.detail),
                    processing_status=None if retry else "OCR_F")
                if verdict is not None:
                    uow.documents.record_validation(doc, verdict)
                self._append_failure(uow, doc, "ocr", outcome.detail, verdict)
                # REQUEUED leaves overall_state PROCESSING so the document stays
                # claimable; None means give up on it.
                return REQUEUED if retry else None

            uow.ocr.save_pages(doc, outcome.data.get("page_texts") or [])
            doc.page_count = outcome.data.get("pages") or doc.page_count

            # The registration number, read off the deed rather than taken from
            # the filename. `document_id` seeded as the file stem at upload,
            # which put "275.pdf" in a column meant to hold "BGP-1-00275-2025-26"
            # - the identifier the registry and the receiving system use.
            #
            # Done here, immediately after OCR, because it needs only the text
            # and every later stage benefits from the document being correctly
            # identified in the logs.
            identity = extract_transaction_id(
                outcome.data.get("text") or "",
                source=doc.source_filename,
                ocr_used=self.stages.ocr.engine != "textlayer")
            # Written to its own column, never over `document_id`. That field is
            # the internal handle, seeded from the file name so the row can exist
            # before anything is read; overwriting it made the two impossible to
            # tell apart, and a deed whose number could not be read exported its
            # file name as the Transaction Identity (R-043).
            doc.transaction_identity = identity.value if identity.found else None
            doc.transaction_identity_confidence = (
                identity.confidence if identity.found else None)

            uow.documents.mark_stage(doc, "ocr", StageState.DONE,
                                      processing_status="OCR_P")
            text = outcome.data.get("text")

        # Outside the session: this touches a file, not the database, and it must
        # not hold a connection open for the length of a save.
        self._make_searchable(pdf_path, outcome.data.get("lines") or [])
        return text

    def _make_searchable(self, pdf_path: str, lines: list) -> None:
        """Give scanned pages an invisible text layer, so the deed is selectable.

        Only pages that carry no text of their own are touched: a page with a
        real text layer already selects correctly, and writing a second copy over
        it would make every drag return the text twice.

        Best-effort throughout. The cleaned PDF is what the operator reads and
        what "Copy Text" copies from, but OCR text is already saved in the
        database by this point - so a failure here costs searchability, never
        data, and must not fail the document.
        """
        if not lines:
            return
        try:
            import pymupdf

            from ..pdf_prepare import add_text_layer, pages_without_text

            target = Path(pdf_path)
            empty = set(pages_without_text(target))
            if not empty:
                return

            # Surya reports boxes as fractions of the rendered image; the page is
            # measured in points. Scaling by the page rectangle keeps the two in
            # register whatever DPI the runner chose.
            placed: dict[int, list[tuple[Any, str]]] = {}
            with pymupdf.open(target) as doc:
                for number in sorted(empty):
                    if number > len(lines):
                        continue
                    rect = doc[number - 1].rect
                    placed[number] = [
                        ((x0 * rect.width, y0 * rect.height,
                          x1 * rect.width, y1 * rect.height), text)
                        for x0, y0, x1, y1, text in lines[number - 1]]

            written = add_text_layer(target, placed)
            if written:
                log.info("made %d scanned page(s) searchable", written,
                         extra={"file": target.name, "pages": written})
        except Exception as exc:  # noqa: BLE001 - searchability is a bonus
            log.warning("could not add a text layer", extra={
                "file": pdf_path, "error": f"{type(exc).__name__}: {exc}"})

    def _do_extract(self, doc_pk: int, ocr_text: str) -> dict[str, Any] | object | None:
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return None
            if doc.extract_state is not StageState.RUNNING:
                claimed = uow.documents.claim_next(
                    "extract", doc.batch_id,
                    max_attempts=self.extract_retry_limit + 1)
                if claimed is None or claimed.id != doc_pk:
                    # A failed claim is not this document's own failure, and it
                    # used to be treated as one: returning None here sent
                    # `_process_one` on to mark the document NEEDS_REVIEW while
                    # `extract_state` was still PENDING. `claim_next` admits
                    # only PROCESSING documents, so that combination is
                    # permanently unclaimable - the document sat with OCR done
                    # and extraction never attempted, and nothing would ever
                    # pick it up again. One real document was found in exactly
                    # that state.
                    if (doc.extract_attempts or 0) > self.extract_retry_limit:
                        uow.documents.mark_stage(
                            doc, "extract", StageState.FAILED,
                            reason="extraction retries exhausted")
                        return None
                    # Another worker holds it, or it is not yet due. Leave the
                    # states alone so it stays claimable.
                    return REQUEUED
            attempt = doc.extract_attempts or 1
            document_number = doc.document_id

        # The GPU stage runs under the governor's lease: on a card too small for
        # co-residency this is what stops OCR, LLM and translation models
        # colliding in VRAM.
        with self.governor.gpu_lease("extract") if self.governor else _NullLease():
            outcome = self.stages.extract.run(ocr_text, document_number, attempt)
            self._log_stage("extract", doc_pk, outcome)

        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return None
            uow.extractions.record(
                doc, attempt=attempt,
                raw_output=outcome.data.get("raw_output"),
                parsed_ok=bool(outcome.data.get("parsed")),
                pan_coverage=outcome.data.get("pan_coverage"),
                prompt_tokens=outcome.data.get("prompt_tokens") or 0,
                completion_tokens=outcome.data.get("completion_tokens") or 0,
                truncated=bool(outcome.data.get("truncated")),
                model_name=getattr(self.stages.extract, "model_name", None),
                quantisation=getattr(self.stages.extract, "quantisation", None),
                duration_s=outcome.data.get("duration_s"))

            if not outcome.ok:
                retry = (outcome.retryable
                         and doc.extract_attempts <= self.extract_retry_limit)
                if retry:
                    uow.documents.mark_stage(doc, "extract", StageState.PENDING,
                                             reason=outcome.detail)
                    return REQUEUED

                # DONE only when the model actually answered. The rule was
                # "not a stage failure: the model answered, the answer is not
                # trustworthy" - which is right, but it was applied even when
                # there was no answer at all. A deed rejected by the server for
                # exceeding the context produced no output on any attempt and
                # was still recorded DONE, so a stage that ran three times and
                # returned nothing looked like a success and no failed-document
                # report ever mentioned it.
                answered = bool(str(outcome.data.get("raw_output") or "").strip())
                if not answered:
                    uow.documents.mark_stage(doc, "extract", StageState.FAILED,
                                             reason=outcome.detail)
                    self._append_failure(uow, doc, "extract", outcome.detail)
                    return None

                uow.documents.mark_stage(doc, "extract", StageState.DONE,
                                         reason=outcome.detail)
                self._append_failure(uow, doc, "extract", outcome.detail)
                # The model answered and the answer is not trustworthy. A human
                # decides.
                uow.documents.mark_overall(doc, DocumentState.NEEDS_REVIEW,
                                            outcome.detail)
                return None

            uow.documents.mark_stage(doc, "extract", StageState.DONE)
            return outcome.data.get("parsed")

    def _do_validate(self, doc_pk: int, parsed: dict[str, Any],
                     ocr_text: str) -> str:
        outcome = self.stages.validate.run(parsed, ocr_text)
        self._log_stage("validate", doc_pk, outcome)
        report = outcome.data.get("report")

        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return "review"

            # `report.stamp_value` is the derived figure - the registration fee,
            # halved for transactions before the cutoff (docs/DECISIONS ADR-010).
            # Validation computed it and then nobody carried it here, so
            # `Property.stamp_value` stayed NULL and the CSV column was blank on
            # every document ever exported. Worse, `save_property` assigns the
            # parameter unconditionally, so a re-run also erased any value that
            # had been set by hand. See R-040.
            prop = uow.results.save_property(
                doc, parsed.get("property_details") or {},
                stamp_value=getattr(report, "stamp_value", None))
            uow.results.save_document_meta(prop, parsed.get("document_details") or {})
            persons = uow.results.replace_persons(doc, parsed)

            ids = {(p.relation.value, p.ordinal): p.id for p in persons}
            if report is not None:
                uow.results.record_flags(
                    doc, self.stages.validate.flag_rows(report, ids))
            uow.documents.mark_stage(doc, "validate", StageState.DONE)

        return str(outcome.data.get("disposition") or "review")

    def _do_translate(self, doc_pk: int, parsed: dict[str, Any]) -> None:
        with self._lease(self.stages.translate, "translate"):
            outcome = self.stages.translate.run(parsed)
            self._log_stage("translate", doc_pk, outcome)

        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return
            if outcome.ok:
                uow.documents.mark_stage(doc, "translate", StageState.DONE)
            else:
                # Translation being unavailable must not fail a document whose
                # extraction is sound: the data is usable, just untranslated.
                uow.documents.mark_stage(doc, "translate", StageState.SKIPPED,
                                          reason=outcome.detail)

    # -- completion -------------------------------------------------------

    def _finish(self, doc_pk: int, state: DocumentState, started: float,
                reason: str | None = None) -> None:
        duration = time.monotonic() - started
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(doc_pk)
            if doc is None:
                return
            uow.documents.mark_overall(doc, state, reason)
            self.stats.record(duration, state)
            if self.on_document is not None:
                try:
                    self.on_document(doc, state)
                except Exception:  # noqa: BLE001 - a UI callback must not stop the batch
                    pass

    def _finalise_if_complete(self, batch_id: int) -> None:
        with session_scope(self.session_factory) as session:
            uow = UnitOfWork(session)
            if not uow.batches.is_finished(batch_id):
                return
            batch = uow.batches.get(batch_id)
            if batch is None or batch.state is not BatchState.RUNNING:
                return
            uow.batches.set_state(batch, BatchState.COMPLETED)
            self.stats.batches_completed += 1
            self._current_batch_id = None
            self._detail = f"completed batch {batch.name}"

        if self.on_batch_complete is not None:
            try:
                self.on_batch_complete(batch_id)
            except Exception:  # noqa: BLE001
                pass


class _NullLease:
    """No-op lease for when no governor is supplied (tests)."""

    def __enter__(self) -> _NullLease:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def build_stages(
    *,
    ai_base_url: str = "http://127.0.0.1:8077",
    prompt_file: str | Path = paths.PROMPT_FILE,
    ocr_engine: str = "auto",
    translator_engine: str = "auto",
    retry_supported: bool = False,
    **kwargs: Any,
) -> Stages:
    """Assemble the standard stage set from configuration.

    `ocr_engine="auto"` resolves to Surya when it is installed and to the
    embedded text layer when it is not, so a machine without Surya still
    processes the digitally generated deeds that make up most of a batch instead
    of failing every document.
    """
    prompt = ""
    path = Path(prompt_file)
    if path.is_file():
        prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        # Loudly. Without its prompt the fine-tuned model is just an
        # instruction-tuned model looking at an unlabelled wall of OCR, and it
        # responds the way one does: a prose summary of the deed. Every
        # extraction then fails with "no parseable JSON" and every CSV column
        # comes out empty - with nothing anywhere saying why. See R-040.
        log.error("extraction prompt is missing or empty at %s - the model will "
                  "not produce JSON and every document will fail", path,
                  extra={"prompt_file": str(path), "exists": path.is_file()})

    ocr_kwargs = dict(kwargs.get("ocr", {}))
    if ocr_engine == "auto":
        interpreter, script = find_surya()
        if interpreter:
            ocr_engine = "surya"
            ocr_kwargs.setdefault("surya_python", interpreter)
            ocr_kwargs.setdefault("surya_script", script)
        else:
            ocr_engine = "textlayer"
    elif ocr_engine == "surya" and "surya_python" not in ocr_kwargs:
        interpreter, script = find_surya()
        if interpreter:
            ocr_kwargs.setdefault("surya_python", interpreter)
            ocr_kwargs.setdefault("surya_script", script)

    # Translation is configured by `core.translation.build_config()`, which
    # TranslateStage calls for itself. The obsolete wiring that used to sit here
    # searched for IndicTrans2 weights by `*.safetensors` and, finding none,
    # silently set the stage to `passthrough` - so translation was disabled in
    # the pipeline while the service worked perfectly on its own. NLLB ships
    # `pytorch_model.bin`, which that check could never match.
    translate_kwargs = dict(kwargs.get("translate", {}))

    return Stages(
        ocr=OcrStage(engine=ocr_engine, ai_base_url=ai_base_url, **ocr_kwargs),
        extract=ExtractStage(base_url=ai_base_url, prompt=prompt,
                             retry_supported=retry_supported,
                             **kwargs.get("extract", {})),
        validate=ValidateStage(**kwargs.get("validate", {})),
        translate=TranslateStage(engine=translator_engine, **translate_kwargs),
    )
