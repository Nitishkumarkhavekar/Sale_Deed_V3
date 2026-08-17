"""Application service layer.

Everything the UI can ask for, in one place. The bridge translates JSON to calls
here; this module owns the database session factory, the runner, the AI-server
client and the staged upload selection.

Deliberately free of Qt. That keeps the whole application testable from a script,
and it is why the bridge is thin enough to be obviously correct.
"""

from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core import paths
from core.csv_export import DocumentExport, FailedDocument, write_csv, write_failed_csv
from core.db.engine import build_engine, build_session_factory, check_connection, session_scope
from core.db.models import (
    BUSY_BATCH_STATES,
    RESUMABLE_BATCH_STATES,
    BatchState,
    DocumentState,
    StageState,
)
from core import failure_codes
from core.failure_codes import classify as _cause
from core.failure_codes import describe as _describe
from core.pdf_validation import CORRUPT_STATUSES
from core.db.repositories import (
    MAX_BATCH_BYTES,
    MAX_FILES_PER_BATCH,
    MAX_QUEUED_BATCHES,
    RepositoryError,
    UnitOfWork,
)
from core.pipeline.runner import BatchMode, BatchRunner, build_stages

from .status import Availability, ProbeResult, StatusService
from .ui.renderer import (
    Chrome,
    Renderer,
    dashboard_model,
    human_bytes,
    human_duration,
    local_time,
    machine_model,
    pager,
    percent,
    state_badge,
)

APP_VERSION = "3.0.0"

#: `processing_status` is a two-letter code in the database and the CSV, because
#: that is what the receiving system expects. The UI is read by a person, so the
#: code is spelled out there instead of shown raw.
#: How each validation verdict reads to a person, and how urgent it looks.
VALIDATION_LABELS = {
    "VALID": "PDF is valid",
    "PROCESSING_ERROR": "PDF fine - processing failed",
    "CORRUPTED_PDF": "Corrupted PDF",
    "PARTIALLY_CORRUPTED": "Partially corrupted",
    "INCOMPLETE_PDF": "Incomplete / truncated",
    "EMPTY_PDF": "Empty PDF",
    "PASSWORD_PROTECTED": "Password protected",
    "INVALID_PDF": "Not a valid PDF",
    "UNREADABLE_PDF": "Unreadable file",
    "PDF_PARSE_ERROR": "PDF parse error",
    "PDF_RENDER_ERROR": "Page render error",
    "UNKNOWN_FAILURE": "Cause not determined",
    "": "Not checked",
}


def _validation_class(status: str | None) -> str:
    """Badge class. A readable file is not an error, whatever else went wrong."""
    if not status:
        return ""
    if status in ("VALID", "PROCESSING_ERROR"):
        return "ok"
    if status in ("PARTIALLY_CORRUPTED", "UNKNOWN_FAILURE"):
        return "review"
    return "danger"


#: Batch-management operations log here, so every Run/Stop/Delete leaves a
#: trace next to the pipeline's own stage lines in runtime/logs/saledeed.log.
log = logging.getLogger("saledeed.batches")

#: Batches the management table shows. Everything that is not finished - a
#: stopped batch had no home before this and was unreachable from the UI.
LIVE_BATCH_STATES = (BatchState.QUEUED, BatchState.RUNNING, BatchState.STOPPING,
                     BatchState.STOPPED, BatchState.PAUSED)


def _log_corrupt_export(rows: int, path: Path) -> None:
    import logging

    logging.getLogger("saledeed.pdf_validator").info(
        "[PDF_VALIDATOR] corrupted report written: %d row(s) -> %s", rows,
        path.name, extra={"path": str(path)})


OCR_STATUS_LABELS = {
    "OCR_F": "OCR failed",
    "OCR_P": "OCR passed",
}
ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = paths.EXPORT_DIR
#: Cleaned copies land here. The source PDF is never modified, so a
#: separate directory keeps "original" and "cleaned" impossible to confuse.
WATERMARK_DIR = paths.WATERMARK_DIR


def _looks_like_pdf(path: Path) -> bool:
    """Check the file's own header, not just its name.

    An extension is a claim, not evidence. Without this, a mislabelled file
    reaches the pipeline and fails during OCR, where the error is attributed to
    the *document* rather than to the upload - the user sees "processing failed"
    on page 1 of 1 instead of "this is not a PDF" at the moment they chose it.

    `%PDF-` may sit a little way into the file: some producers emit a UTF-8 BOM
    or stray bytes first, and readers tolerate it, so a strict prefix test would
    reject files that open perfectly well elsewhere.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(1024)
    except OSError:
        return False
    return b"%PDF-" in head


@dataclass
class Selection:
    """PDFs staged for upload but not yet committed to a batch."""

    paths: list[Path] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.paths if p.is_file())

    def add(self, candidates: list[Path]) -> int:
        seen = {p.resolve() for p in self.paths}
        added = 0
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            if resolved.suffix.lower() != ".pdf" or not _looks_like_pdf(resolved):
                continue
            self.paths.append(resolved)
            seen.add(resolved)
            added += 1
        return added


class AiClient:
    """Thin HTTP client for the AI server. Standard library only."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, timeout: float = 5.0) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return {}

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def profile(self) -> dict[str, Any]:
        return self._get("/profile", timeout=8.0)

    def hardware(self) -> dict[str, Any]:
        return self._get("/hardware", timeout=8.0)


class AppService:
    """Owns application state for the desktop process."""

    def __init__(self, *, ai_base_url: str | None = None,
                 reload_templates: bool = False) -> None:
        self.renderer = Renderer(reload=reload_templates)
        self.ai = AiClient(ai_base_url or os.environ.get(
            "SALEDEED_AI_URL", "http://127.0.0.1:8077"))
        self.selection = Selection()
        self.errors: list[str] = []

        self.engine = build_engine()
        # One short attempt at startup so the first paint knows whether to gate
        # actions. Everything after this is probed in the background.
        self.db_ok, self.db_detail = check_connection(self.engine)
        self.sessions = build_session_factory(self.engine)

        # In-memory settings cache. Every _setting() call previously opened its
        # own session; the Validation page alone made 14 round trips.
        self._settings_cache: dict[str, str | None] = {}
        self._settings_loaded = False

        self.status_service = StatusService(self.ai.base_url, self._probe_database)

        # Watermark page state. Kept per-session rather than persisted: the
        # selection is a handful of files a user picked moments ago, and a stale
        # scan result surviving a restart would be worse than re-scanning.
        self.watermark_files = Selection()
        self._watermark_scans: dict[str, Any] = {}
        self._watermark_removals: dict[str, Any] = {}
        #: Source path -> where its cleaned copy was written. Kept so the page
        #: can show each file's destination and "Open Output Folder" can open
        #: the folder actually used rather than a guess.
        self._watermark_outputs: dict[str, Path] = {}

        # OCR tool page state, held the same way and for the same reason.
        #
        # Unlike watermark scanning, an OCR pass is minutes per document, so it
        # cannot run inside the bridge call - the UI would block and the
        # front-end's 120 s call timeout would fire long before a real deed
        # finished. It runs on a worker thread and this dictionary is what the
        # page renders from, refreshed by the status poll already running.
        self.ocr_files = Selection()
        self._ocr_results: dict[str, dict[str, Any]] = {}
        self._ocr_thread: threading.Thread | None = None
        self._ocr_cancel = threading.Event()
        self._ocr_detail = ""
        self._ocr_lock = threading.Lock()

        self.stages = build_stages(ai_base_url=self.ai.base_url)
        self.runner = BatchRunner(self.sessions, self.stages,
                                  mode=BatchMode.MANUAL, max_workers=1)

        #: Started in start_up(), not here - the constructor runs before the
        #: window is painted and must not spawn threads that could log or touch
        #: the database while the UI is still assembling.
        self.retention: Any = None

        self._profile: dict[str, Any] = {}
        self._last_batch_count = -1

    # -- lifecycle ---------------------------------------------------------

    def start_up(self) -> None:
        """Begin background probing, then recover stranded work.

        Called after the window is painted, so neither step delays first paint.
        """
        self.status_service.start(interval_s=2.0)
        self._apply_runner_settings()
        self._start_retention()
        if self.db_ok:
            try:
                recovered = self.runner.recover()
                if recovered:
                    self.errors.append(
                        f"Recovered {recovered} document stage(s) interrupted by a "
                        "previous shutdown; they will be reprocessed.")
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"Startup recovery failed: {exc}")

    def _apply_runner_settings(self) -> None:
        """Push the saved batch mode and cooldown onto the runner.

        The Settings page has offered manual/auto since the UI was built, but
        the runner was constructed with MANUAL and never consulted the stored
        value - so choosing "Auto" changed a row in the database and nothing
        else. Applied at startup and again whenever settings are saved, so the
        change takes effect without a restart.
        """
        try:
            mode = (self._setting("batch_mode", "manual") or "manual").strip().lower()
            self.runner.mode = BatchMode.AUTO if mode == "auto" else BatchMode.MANUAL
            cooldown = float(self._setting("auto_cooldown_seconds", "60") or 60)
            # A zero cooldown would start the next batch the instant the last
            # document lands, giving the GPU no chance to release memory.
            self.runner.auto_cooldown_s = max(5.0, cooldown)
        except Exception as exc:  # noqa: BLE001 - a bad setting must not stop startup
            self._record_error("batch mode", exc)

    def _start_retention(self) -> None:
        """Begin the nightly backup and purge timer.

        Off by default. Retention *deletes* data - old documents, expired OCR
        text, rotated logs - and a destructive job that starts itself on first
        launch is the wrong default for a records system. `SALEDEED_RETENTION`
        turns it on.

        `is_busy` defers a run while a batch is processing: a dump taken during
        heavy write activity is slower, larger, and competes for the disk the
        batch is writing to.
        """
        if os.environ.get("SALEDEED_RETENTION", "").strip().lower() not in (
                "1", "true", "yes", "on"):
            return
        if not self.db_ok:
            return
        try:
            from core.backup import RetentionScheduler
            from core.db.engine import dsn_from_env

            hours = float(self._setting("retention_interval_hours", "24") or 24)
            self.retention = RetentionScheduler(
                self.sessions, dsn_from_env(),
                backup_dir=paths.BACKUP_DIR, log_dir=paths.LOG_DIR,
                interval_hours=hours,
                is_busy=lambda: self.runner.state.value == "running")
            self.retention.start()
        except Exception as exc:  # noqa: BLE001 - never block startup
            self._record_error("retention scheduler", exc)

    def shut_down(self) -> None:
        try:
            if self.retention is not None:
                self.retention.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.status_service.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.runner.stop(timeout=30)
        except Exception:  # noqa: BLE001
            pass

    def _probe_database(self) -> ProbeResult:
        """Background database probe. One cheap query, short timeout."""
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            self.db_ok = False
            return ProbeResult(Availability.DOWN, "not reachable",
                               error=f"{type(exc).__name__}: {exc}")
        self.db_ok = True
        return ProbeResult(Availability.UP, "connected")

    def log_exception(self, where: str, exc: Exception, trace: str) -> None:
        self.errors.append(f"{where}: {type(exc).__name__}: {exc}")

    def _record_error(self, where: str, exc: Exception) -> None:
        """Record a non-fatal failure for the dashboard notice.

        Bounded: an unbounded list would grow without limit across a long
        session, and only the most recent entry is ever displayed.
        """
        self.errors.append(f"{where}: {type(exc).__name__}: {exc}")
        del self.errors[:-20]

    # -- rendering ---------------------------------------------------------

    def _chrome(self) -> Chrome:
        """Top-bar values, read entirely from cached probes.

        No network call happens here. This is what takes the dashboard from
        8.4 s to milliseconds when the AI server is unreachable.
        """
        snap = self.status_service.snapshot()
        gpu = snap["gpu"]
        vram_total = gpu.get("vram_total_bytes") or 0
        vram_free = gpu.get("vram_free_bytes") or 0
        return Chrome(
            ai_ready=snap["ai"]["state"] == "up",
            running=self.runner.state.value == "running",
            runner_state=self.runner.status().get("detail") or self.runner.state.value,
            gpu_util=gpu.get("util") or "-",
            vram_free=human_bytes(vram_free) if vram_total else "-",
            # The pill named a setting that nothing read. Everything is
            # translated *into* English; what varies is how Devanagari is
            # resolved, so that is what is worth showing.
            translation_language=self._devanagari_label(),
        )

    def render_page(self, page: str, params: dict[str, Any],
                    *, shell_html: bool = True) -> str:
        builder = {
            "dashboard": self._dashboard,
            "upload": self._upload,
            "processing": self._processing,
            "failed_ocr": self._failed_ocr_page,
            "data": self._data_view,
            "ocr": self._ocr_page,
            "watermark": self._watermark_page,
            "settings": self._settings,
            "validation": self._validation,
            "help": self._help,
        }.get(page, self._dashboard)
        model = builder(params)
        # Capabilities are merged into every page model rather than threaded
        # through each builder. Gating is a property of the *system*, not of a
        # screen, and doing it here means a new page cannot forget to ask.
        # A builder that sets these itself wins - some pages have a stricter
        # local reason (an empty selection, a full queue) than the global one.
        for key, value in self._capability_model().items():
            model.setdefault(key, value)
        return self.renderer.render_page(page, model, self._chrome(),
                                         shell_html=shell_html)

    def _capability_model(self) -> dict[str, Any]:
        """Flatten Capabilities for the templates.

        Both polarities are supplied. Mustache has no `not`, so a template that
        must render "greyed out when unavailable" needs the positive form for
        `{{^...}}` and the negative for a message block - deriving one from the
        other in the template is not possible.
        """
        caps = self.status_service.snapshot().get("capabilities") or {}
        reasons = caps.get("reasons") or {}
        model: dict[str, Any] = {}
        for name in ("browse", "export", "upload", "process"):
            allowed = bool(caps.get(f"can_{name}", False))
            model[f"can_{name}"] = allowed
            model[f"no_{name}"] = not allowed
            model[f"{name}_reason"] = reasons.get(name, "")
        model["degraded"] = not all(
            model[f"can_{n}"] for n in ("browse", "export", "upload", "process"))
        model["capability_reasons"] = [
            {"action": name.capitalize(), "reason": reason}
            for name, reason in sorted(reasons.items()) if reason]
        return model

    def render_fragment(self, template: str, params: dict[str, Any]) -> str:
        if template == "batch_detail":
            return self.renderer.render_fragment(
                "batch_detail", self._batch_detail(int(params.get("batch_id") or 0)))
        return self.renderer.render_fragment(template, params)

    # -- status poll -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Poll target. Cache only - must finish well inside the 2.5 s interval."""
        snap = self.status_service.snapshot()
        chrome = self._chrome()
        payload: dict[str, Any] = {
            "ai": snap["ai"],
            "database": snap["database"],
            "capabilities": snap["capabilities"],
            "ai_ready": chrome.ai_ready,
            "running": chrome.running,
            "runner_state": chrome.runner_state,
            "gpu_util": chrome.gpu_util,
            "vram_free": chrome.vram_free,
        }
        # The OCR tool page has no timer of its own; it refreshes off this poll.
        # Cheap enough to include unconditionally - three integers read from a
        # dictionary already in memory, no database and no filesystem.
        if self.ocr_files.paths or self._ocr_busy():
            with self._ocr_lock:
                finished = sum(1 for r in self._ocr_results.values()
                               if r.get("state") == "done")
            payload["ocr_tool"] = {
                "running": self._ocr_busy(),
                "processed": finished,
                "total": len(self.ocr_files.paths),
                "detail": self._ocr_detail,
            }
        if not self.db_ok:
            return payload

        try:
            with session_scope(self.sessions) as session:
                uow = UnitOfWork(session)
                # Counted on every poll, and before the early return below: a
                # failed document does not stop being failed because no batch is
                # currently active, and the nav badge is the only place an
                # operator finds out without going looking.
                payload["failed_ocr"] = uow.documents.failed_ocr_count()
                active = uow.batches.active()
                if active is None:
                    payload["batch_completed"] = self._last_batch_count >= 0
                    self._last_batch_count = -1
                    return payload
                progress = uow.batches.progress(active.id)
        except Exception:  # noqa: BLE001 - a failed poll must not break the UI
            return payload

        if progress is None:
            return payload

        payload["stages"] = {
            key: {"done": stage.done, "total": stage.total,
                  "percent": percent(stage.done, stage.total)}
            for key, stage in progress.stages.items()
        }
        payload["counts"] = {
            "completed": progress.completed,
            "needs_review": progress.needs_review,
            "failed": progress.failed,
        }
        finished = progress.completed + progress.failed + progress.needs_review
        payload["batch_completed"] = (self._last_batch_count >= 0
                                      and finished < self._last_batch_count)
        self._last_batch_count = finished
        return payload

    # -- control -----------------------------------------------------------

    def control(self, action: str) -> dict[str, Any]:
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])
        if action == "start":
            self.runner.start()
        elif action == "pause":
            self.runner.pause()
        elif action == "stop":
            self.runner.stop(timeout=5)
        else:
            raise ValueError(f"unknown action {action!r}")
        return {"status": self.status()}

    # -- upload ------------------------------------------------------------

    def pick_files(self) -> dict[str, Any]:
        """Native dialog. Injected by the shell so this module stays Qt-free."""
        picker = getattr(self, "file_picker", None)
        if picker is None:
            raise RuntimeError("no file picker is attached")
        added = self.selection.add([Path(p) for p in picker()])
        return {"count": added, "total": len(self.selection.paths)}

    def add_files(self, paths: list[str]) -> dict[str, Any]:
        added = self.selection.add([Path(p) for p in paths])
        return {"count": added, "total": len(self.selection.paths)}

    def clear_selection(self) -> dict[str, Any]:
        self.selection = Selection()
        return {"count": 0}

    def add_batch(self, username: str, name: str) -> dict[str, Any]:
        if not self.selection.paths:
            raise RepositoryError("No files selected.")
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create(username)
            batch = uow.batches.create(name, user, len(self.selection.paths),
                                       self.selection.total_bytes)
            uow.documents.add_many(batch, [
                {"document_id": p.stem, "source_filename": p.name,
                 "source_path": str(p), "size_bytes": p.stat().st_size}
                for p in self.selection.paths])
            result = {"batch_id": batch.id, "name": batch.name,
                      "files": len(self.selection.paths)}
        self.selection = Selection()
        return result

    # -- batches -----------------------------------------------------------

    def document_pdf(self, token: str) -> Path | None:
        """Resolve a document id to the PDF the viewer should show.

        **Always the prepared copy when one exists.** That is the document the
        pipeline actually read: overlays removed, text layer guaranteed. Showing
        the original would mean the operator selects text the extraction never
        saw, and any discrepancy would be impossible to explain.

        Takes an id, never a path. A path parameter here would turn a viewer
        into a file reader for anything the process can open.
        """
        if not str(token).isdigit() or not self.db_ok:
            return None
        try:
            with session_scope(self.sessions) as session:
                doc = UnitOfWork(session).documents.get(int(token))
                if doc is None:
                    return None
                for candidate in (doc.cleaned_path, doc.source_path):
                    if candidate and Path(candidate).is_file():
                        return Path(candidate)
        except Exception as exc:  # noqa: BLE001
            self._record_error("open document", exc)
        return None

    def document_view(self, document_pk: int) -> dict[str, Any]:
        """Everything the viewer page needs about one document."""
        if not self.db_ok:
            raise RepositoryError("The database is not reachable.")
        with session_scope(self.sessions) as session:
            doc = UnitOfWork(session).documents.get(int(document_pk))
            if doc is None:
                raise RepositoryError(f"Document {document_pk} not found.")
            cleaned = bool(doc.cleaned_path and Path(doc.cleaned_path).is_file())
            return {
                "document_pk": doc.id,
                "document_id": doc.document_id,
                # The registration number read off the deed. Distinct from
                # `document_id`, which is the internal handle - showing one in
                # place of the other is how the file name reached the report
                # (R-043). Blank when the deed did not yield one.
                "transaction_identity": doc.transaction_identity or "",
                "identity_confidence": (
                    f"{doc.transaction_identity_confidence:.2f}"
                    if doc.transaction_identity_confidence else ""),
                "filename": doc.source_filename,
                "url": f"app://ui/pdf/{doc.id}",
                "cleaned": cleaned,
                "pages": doc.page_count or 0,
                "state": doc.overall_state.value,
                # Said plainly in the UI: the operator is looking at a derived
                # file, and should know which.
                "source_note": ("Watermarks removed; text is selectable."
                                if cleaned else
                                "Original document - not yet prepared."),
            }

    def document_text(self, document_pk: int) -> dict[str, Any]:
        """The document's text, in page order, for copying.

        Served from the stored OCR rather than re-read from the PDF: that is the
        exact text the model was given, so what the operator copies is what the
        extraction saw. Re-reading the file could differ - a different reader,
        a different page order - and a discrepancy there would be very hard to
        explain to someone checking a flagged field.
        """
        if not self.db_ok:
            raise RepositoryError("The database is not reachable.")
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.get(int(document_pk))
            if doc is None:
                raise RepositoryError(f"Document {document_pk} not found.")
            text = uow.ocr.full_text(doc) or ""
        return {"document_pk": int(document_pk), "text": text,
                "chars": len(text),
                "detail": (f"{len(text):,} characters" if text
                           else "no text has been extracted yet")}

    def export_view(self, params: dict[str, Any]) -> dict[str, Any]:
        """Export what the Data View is currently showing.

        The page is scoped to one batch through its selector, so this resolves
        the same batch the table is displaying - including the default of the
        most recent batch when nothing is chosen - and hands off to
        `export_batch`. Exporting some *other* batch than the one on screen
        would be worse than the button doing nothing.
        """
        if not self.db_ok:
            raise RepositoryError("The database is not reachable.")

        batch_id = params.get("batch_id")
        target = int(batch_id) if str(batch_id or "").isdigit() else None
        if target is None:
            with session_scope(self.sessions) as session:
                batches, _ = UnitOfWork(session).batches.list_paginated(1, 1)
                target = batches[0].id if batches else None
        if target is None:
            raise RepositoryError("There are no batches to export.")
        result = self.export_batch(target, bool(params.get("failed_only")),
                                   destination=params.get("destination"))
        # Which batch was written, so the interface can say so rather than
        # leaving the operator to infer it from the file name.
        with session_scope(self.sessions) as session:
            batch = UnitOfWork(session).batches.get(target)
            result["batch_id"] = target
            result["batch_name"] = batch.name if batch else ""
        return result

    #: The extraction prompt, and the copy kept so an edit can be undone.
    PROMPT_PATH = paths.PROMPT_FILE

    def save_prompt(self, text: str) -> dict[str, Any]:
        """Persist an edited extraction prompt.

        The shipped prompt is copied to `.default` **before** the first
        overwrite. Without that, "Restore Default" has nothing to restore, and
        an operator who edits the prompt the model was finetuned on has no way
        back short of reinstalling.

        Stages are rebuilt because `build_stages` reads this file once at
        construction; skipping that would leave the running pipeline using the
        old prompt until a restart, which is the kind of gap that gets
        misdiagnosed as the edit having no effect.
        """
        body = (text or "").strip()
        if not body:
            raise RepositoryError("The prompt cannot be empty.")

        backup = self.PROMPT_PATH.with_suffix(self.PROMPT_PATH.suffix + ".default")
        if self.PROMPT_PATH.is_file() and not backup.is_file():
            backup.write_text(self.PROMPT_PATH.read_text(encoding="utf-8"),
                              encoding="utf-8", newline="\n")

        self.PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PROMPT_PATH.write_text(body, encoding="utf-8", newline="\n")
        self.stages = build_stages(ai_base_url=self.ai.base_url)
        return {"saved": True, "chars": len(body),
                "can_restore": backup.is_file()}

    def reset_prompt(self) -> dict[str, Any]:
        """Restore the extraction prompt from the shipped default.

        The prompt is what the model was finetuned against, so a hand-edited one
        that drifts is a silent accuracy problem. `.default` is written the first
        time the prompt is edited, which makes this recoverable.
        """
        prompt_path = self.PROMPT_PATH
        backup = prompt_path.with_suffix(prompt_path.suffix + ".default")
        if not backup.is_file():
            raise RepositoryError(
                "No saved default to restore - the prompt has not been edited.")
        prompt_path.write_text(backup.read_text(encoding="utf-8"),
                               encoding="utf-8", newline="\n")
        self.stages = build_stages(ai_base_url=self.ai.base_url)
        return {"restored": True, "path": str(prompt_path)}

    def check_updates(self) -> dict[str, Any]:
        """Report the version and where to look for a newer one.

        There is no update channel in this application, and inventing one that
        silently downloads would sit badly beside a fine-tuned model that must
        never be replaced. This tells the operator what they are running and
        opens the configured page if there is one.
        """
        url = self._setting("update_repo_url", "").strip()
        if not url:
            return {"version": APP_VERSION, "url": "",
                    "detail": f"Running version {APP_VERSION}. No update page is "
                              "configured - set one in Settings."}
        return {"version": APP_VERSION, "url": url,
                "detail": f"Running version {APP_VERSION}. Opening {url}"}

    def export_batch(self, batch_id: int, failed_only: bool = False,
                     destination: str | Path | None = None) -> dict[str, Any]:
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None:
                raise RepositoryError(f"Batch {batch_id} not found.")
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in batch.name)

            if failed_only:
                rows = [FailedDocument(
                    # Same rule as the main export: the extracted number or
                    # nothing. A failed document usually has nothing, which is
                    # honest - `source_filename` on the next line is how the
                    # operator identifies which file it was (R-043).
                    transaction_identity=d.transaction_identity or "",
                    source_filename=d.source_filename,
                    failed_stage=self._failed_stage(d),
                    processing_status=d.processing_status or "",
                    reason=d.failure_reason or "",
                ) for d in uow.documents.failed_for_batch(batch_id)]
                path = self._export_path(destination, f"{safe}_failed_{stamp}.csv")
                count = write_failed_csv(path, rows)
            else:
                docs, _ = uow.documents.list_for_batch(batch_id, per_page=100_000)
                exports = [self._document_export(d) for d in docs]
                path = self._export_path(destination, f"{safe}_{stamp}.csv")
                count = write_csv(path, exports)
        return {"rows": count, "path": str(path)}

    @staticmethod
    def _export_path(destination: str | Path | None, default_name: str) -> Path:
        """Where the export is written: the operator's choice, or the usual place.

        A destination naming an existing directory is treated as a directory -
        the Save As dialog returns a file, but a caller may reasonably pass a
        folder, and writing a file *called* `Documents` over their folder would
        be worse than either interpretation.
        """
        if destination is None or not str(destination).strip():
            return EXPORT_DIR / default_name
        chosen = Path(str(destination)).expanduser()
        if chosen.is_dir():
            return chosen / default_name
        # A bare name with no suffix is a file the operator named; give it one.
        if not chosen.suffix:
            chosen = chosen.with_suffix(".csv")
        return chosen

    def pick_save_path(self, suggested: str = "export.csv") -> dict[str, Any]:
        """Native Save As dialog. Injected by the shell so this stays Qt-free.

        An empty path means the operator cancelled. That is reported as an
        ordinary result, not an error - cancelling a save is a decision.
        """
        picker = getattr(self, "save_picker", None)
        if picker is None:
            raise RuntimeError("no save-file picker is attached")
        chosen = str(picker(suggested) or "")
        return {"path": chosen, "cancelled": not chosen}

    def reprocess_failed(self, batch_id: int) -> dict[str, Any]:
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            count = uow.documents.reprocess_failed(batch_id)
            batch = uow.batches.get(batch_id)
            if batch is not None and count:
                uow.batches.set_state(batch, BatchState.QUEUED)
        return {"count": count}

    # -- failed OCR --------------------------------------------------------
    #
    # OCR is the stage that fails for reasons an operator can actually do
    # something about - a scanner glitch, a page the model timed out on, VRAM
    # taken by something else - and it is also the most expensive to redo. So
    # rerunning it is a deliberate, per-document action rather than a side
    # effect of reprocessing a whole batch.
    #
    # There is no second OCR path here. `rerun_ocr` only returns documents to
    # PENDING; the runner then claims them through the same `claim_next("ocr")`
    # every other document goes through, and the same `_do_ocr` does the work.

    def failed_ocr(self, batch_id: int | None = None, page: int = 1,
                   per_page: int = 25) -> dict[str, Any]:
        """The failed-OCR list, ready to render."""
        if not self.db_ok:
            return {"documents": [], "total": 0, "page": 1, "pages": 1,
                    "db_offline": True}
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            docs, total = uow.documents.failed_ocr(batch_id, page=page,
                                                   per_page=per_page)
            batch_names = {}
            rows = []
            for doc in docs:
                if doc.batch_id not in batch_names:
                    batch = uow.batches.get(doc.batch_id)
                    batch_names[doc.batch_id] = batch.name if batch else ""
                rows.append({
                    "document_pk": doc.id,
                    "document_id": doc.document_id,
                    "source_filename": doc.source_filename,
                    "batch_id": doc.batch_id,
                    "batch_name": batch_names[doc.batch_id],
                    "pages": doc.page_count,
                    "size": human_bytes(doc.size_bytes or 0),
                    # "OCR_F" is what the CSV carries; spell it out for a person.
                    "status": doc.processing_status or "OCR_F",
                    "status_label": OCR_STATUS_LABELS.get(
                        doc.processing_status or "OCR_F", "OCR failed"),
                    "attempts": doc.ocr_attempts,
                    # The whole reason, not a truncation. An operator deciding
                    # whether a rerun is worth minutes of GPU time needs to read
                    # what actually went wrong.
                    "reason": doc.failure_reason or "no reason recorded",
                    "failed_at": local_time(doc.updated_at),
                    # The validator's verdict on the file itself. NULL means it
                    # was never asked - older rows, or a failure that predates
                    # the feature.
                    # Why it failed, in words. Classified from what is already
                    # stored, so nothing needs re-running and a document that
                    # failed before this existed is still explained.
                    "cause": (_cause(doc) or {}).get("reason", ""),
                    "cause_code": (_cause(doc) or {}).get("code", ""),
                    "cause_stage": (_cause(doc) or {}).get("stage", ""),
                    "cause_technical": (_cause(doc) or {}).get("technical", ""),
                    # The full sequence, oldest first. A retry appends rather
                    # than replacing, so "watermark failed, then OCR found no
                    # text" stays visible as two events.
                    "history": _describe(uow.documents.failure_history(doc)),
                    "has_history": len(doc.failure_events) > 1,
                    "validation_status": doc.validation_status or "",
                    "validation_label": VALIDATION_LABELS.get(
                        doc.validation_status or "", doc.validation_status or "Not checked"),
                    "validation_class": _validation_class(doc.validation_status),
                    "validation_error_code": doc.validation_error_code or "",
                    "validation_message": doc.validation_error_message or "",
                    "corrupted_pages": doc.corrupted_pages or "",
                    "validated_at": local_time(doc.validated_at) if doc.validated_at else "",
                    # A corrupt file fails again identically; offering Retry
                    # there would be a button that cannot work.
                    "retryable": doc.is_retryable is not False,
                    "is_corrupt": bool(doc.validation_status
                                       and doc.validation_status in CORRUPT_STATUSES),
                })
        pages = max(1, (total + per_page - 1) // per_page)
        return {"documents": rows, "total": total,
                "page": max(1, min(page, pages)), "pages": pages,
                "batch_id": batch_id or 0}

    def revalidate(self, document_pks: list[int] | None = None,
                   batch_id: int | None = None) -> dict[str, Any]:
        """Check the files again, after the operator has repaired or replaced them.

        This is how a corrupt document becomes processable again: the verdict is
        recomputed from the file as it is *now*, and a file that has been fixed
        stops being marked non-retryable. Nothing is queued here - revalidating
        answers a question, it does not start work, and automatically
        reprocessing a file the operator has just swapped in would take the
        decision away from them.
        """
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])

        from core.pdf_validation import validate_pdf

        targets = [int(pk) for pk in (document_pks or [])]
        checked: list[dict[str, Any]] = []
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            if not targets:
                docs, _ = uow.documents.corrupted(batch_id, page=1, per_page=1000)
            else:
                docs = [d for d in (uow.documents.get(pk) for pk in targets)
                        if d is not None]
            for doc in docs:
                path = doc.source_path or doc.source_filename
                try:
                    result = validate_pdf(path)
                except Exception as exc:  # noqa: BLE001 - a bad file must not raise
                    self._record_error("revalidate", exc)
                    continue
                uow.documents.record_validation(doc, result)
                checked.append({"document_pk": doc.id,
                                "file": doc.source_filename,
                                "status": result.status,
                                "now_valid": result.is_valid,
                                "retryable": result.retryable})

        repaired = [c for c in checked if c["now_valid"]]
        noun = "file" if len(checked) == 1 else "files"
        detail = f"{len(checked)} {noun} re-checked."
        if repaired:
            detail += (f" {len(repaired)} now valid and ready to reprocess.")
        return {"count": len(checked), "repaired": len(repaired),
                "documents": checked, "detail": detail}

    def corrupted_pdfs(self, batch_id: int | None = None, page: int = 1,
                       per_page: int = 25) -> dict[str, Any]:
        """The corrupted-PDF list, ready to render or export."""
        if not self.db_ok:
            return {"documents": [], "total": 0, "page": 1, "pages": 1,
                    "db_offline": True}
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            docs, total = uow.documents.corrupted(batch_id, page=page,
                                                  per_page=per_page)
            names: dict[int, str] = {}
            rows = []
            for doc in docs:
                if doc.batch_id not in names:
                    batch = uow.batches.get(doc.batch_id)
                    names[doc.batch_id] = batch.name if batch else ""
                rows.append({
                    "document_pk": doc.id,
                    "file": doc.source_filename,
                    "path": doc.source_path or "",
                    "batch_id": doc.batch_id,
                    "batch_name": names[doc.batch_id],
                    "status": doc.validation_status or "",
                    "label": VALIDATION_LABELS.get(doc.validation_status or "",
                                                   doc.validation_status or ""),
                    "error_code": doc.validation_error_code or "",
                    "message": doc.validation_error_message or "",
                    "page_count": doc.page_count or 0,
                    "corrupted_pages": doc.corrupted_pages or "",
                    "validated_at": local_time(doc.validated_at) if doc.validated_at else "",
                    "retryable": doc.is_retryable is not False,
                })
        pages = max(1, (total + per_page - 1) // per_page)
        return {"documents": rows, "total": total,
                "page": max(1, min(page, pages)), "pages": pages,
                "batch_id": batch_id or 0}

    def export_corrupted(self, batch_id: int | None = None,
                         destination: str | Path | None = None) -> dict[str, Any]:
        """Write the corrupted-PDF report.

        Its own writer rather than the 42-column deed export: this describes
        *files that failed*, not transactions, and forcing it into the deed
        schema would produce forty empty columns and hide the seven that matter.
        """
        import csv as _csv

        listing = self.corrupted_pdfs(batch_id, page=1, per_page=100_000)
        rows = listing["documents"]
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
        path = self._export_path(destination, f"corrupted_pdfs_{stamp}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        columns = ("File Name", "Full Path", "Batch Name", "Batch ID",
                   "Validation Status", "Error Code", "Error Message",
                   "Page Count", "Corrupted Pages", "Retryable",
                   "Validation Timestamp")
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = _csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([
                    row["file"], row["path"], row["batch_name"], row["batch_id"],
                    row["status"], row["error_code"], row["message"],
                    row["page_count"], row["corrupted_pages"],
                    "yes" if row["retryable"] else "no", row["validated_at"]])
        _log_corrupt_export(len(rows), path)
        return {"rows": len(rows), "path": str(path)}

    def rerun_ocr(self, document_pks: list[int] | None = None,
                  batch_id: int | None = None, *, all_failed: bool = False,
                  force: bool = False) -> dict[str, Any]:
        """Send failed documents back through OCR. Reuses the normal pipeline.

        `all_failed` reruns every failed document (optionally within one batch);
        otherwise only the ids given. Either way the batch is returned to QUEUED
        so the runner has something to pick up, and the runner is started if it
        is not already going - a rerun that sits waiting for someone to press
        Start looks exactly like a rerun that did nothing.
        """
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            if all_failed:
                docs, _ = uow.documents.failed_ocr(batch_id, page=1,
                                                   per_page=100_000)
                targets = [d.id for d in docs]
            else:
                targets = [int(pk) for pk in (document_pks or [])]
            if not force:
                # A bulk rerun must not spend GPU time on files already known to
                # be unopenable. Naming them individually still works - that is
                # the operator overriding a verdict deliberately, which is what
                # `force` records.
                blocked = [uow.documents.get(pk) for pk in targets]
                skipped = [d.source_filename for d in blocked
                           if d is not None and d.is_retryable is False]
                targets = [d.id for d in blocked
                           if d is not None and d.is_retryable is not False]
            else:
                skipped = []
            requeued = uow.documents.requeue_ocr(targets)
            names = [d.source_filename for d in requeued]
            batches = {d.batch_id for d in requeued}
            for bid in batches:
                batch = uow.batches.get(bid)
                if batch is not None:
                    uow.batches.set_state(batch, BatchState.QUEUED)

        # Asked for, but nothing was eligible - a stale page, or another
        # operator got there first. Reported rather than silently succeeding.
        if targets and not requeued:
            return {"count": 0, "started": False, "documents": [],
                    "detail": "Nothing to rerun: those documents are no longer "
                              "in the failed-OCR list."}

        started = False
        if requeued and not self.runner.status().get("running"):
            try:
                self.runner.start()
                started = True
            except Exception as exc:  # noqa: BLE001 - the requeue already stuck
                # The documents *are* queued; only the automatic start failed.
                # Saying so is more useful than raising and implying nothing
                # happened, because pressing Start now would work.
                self._record_error("rerun_ocr.start", exc)
                return {"count": len(requeued), "started": False,
                        "documents": names,
                        "detail": f"{len(requeued)} document(s) queued for OCR. "
                                  f"Press Start to process them ({exc})."}

        noun = "document" if len(requeued) == 1 else "documents"
        detail = f"{len(requeued)} {noun} queued for OCR."
        if skipped:
            detail += (f" {len(skipped)} skipped as a damaged file - repair or "
                       "replace it, then Revalidate.")
        return {"count": len(requeued), "started": started, "documents": names,
                "skipped": skipped,
                "detail": detail + (" Processing started." if started else "")}

    # -- per-batch management ---------------------------------------------
    #
    # The runner's start/pause/stop are global: they govern the worker threads,
    # not any one batch. These act on a single batch by changing its row, which
    # is what lets one batch be stopped while another keeps running - the runner
    # never learns about it, it simply finds a different batch to claim from.

    #: How long a forced delete waits for the in-flight document to be released.
    #: Short on purpose. Deleting rows a worker is holding would fail mid-stage
    #: with a stale-object error, so the wait is a safety interlock, not a
    #: convenience; when it expires the operator is told to try again rather
    #: than being made to watch a spinner for the several minutes a long deed
    #: can take.
    DELETE_SETTLE_TIMEOUT_S = 20.0

    def batch_action(self, batch_id: int, action: str,
                     confirm: bool = False) -> dict[str, Any]:
        """Run, stop or delete one batch.

        Every guard lives here rather than in the UI. The dashboard refreshes on
        a timer, so any page more than a moment old can offer an action that has
        since become invalid - and two windows can be open at once. The button
        being visible is never taken as evidence that the action is legal.
        """
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])
        if action not in ("run", "stop", "delete"):
            raise ValueError(f"unknown batch action {action!r}")
        if action == "delete":
            return self.delete_batch(batch_id, force=confirm)

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None:
                raise RepositoryError(f"Batch {batch_id} not found.")
            name, state = batch.name, batch.state

            if action == "run":
                result = self._run_batch(uow, batch)
            else:
                result = self._stop_batch(uow, batch)

        # `batch_name`, not `name`: `name` is a reserved LogRecord attribute and
        # passing it through `extra` raises at emit time - a logging call that
        # crashes the operation it was meant to record.
        log.info("batch %s: %s -> %s", action, state.value, result["state"],
                 extra={"batch": batch_id, "batch_name": name, "action": action})

        if action == "run":
            # Starting the runner is deliberately outside the session: it spawns
            # threads, and holding a transaction open across that would keep a
            # row lock for as long as the thread pool took to come up.
            self.runner.start()
        return {"batch_id": batch_id, "name": name, **result,
                "status": self.status()}

    @staticmethod
    def _manage_row(batch: Any, progress: Any) -> dict[str, Any]:
        """One row of the batch-management table.

        The action flags are computed from the same state machine the service
        enforces, so a button is offered only when pressing it would succeed.
        `needs_force` marks the case the specification singles out: deleting a
        batch that is mid-processing is allowed, but only behind a second,
        differently-worded confirmation.
        """
        total = progress.total if progress else 0
        done = progress.completed if progress else 0
        failed = progress.failed if progress else 0
        review = progress.needs_review if progress else 0
        finished = done + failed + review
        state = batch.state
        return {
            "id": batch.id,
            "name": batch.name,
            "state": state.value,
            # Title-cased for the badge; "stopping" reads as a status, "Stopping"
            # reads as a label.
            "state_label": state.value.title(),
            "username": batch.user.username if batch.user else "-",
            "created_at": local_time(batch.created_at),
            "started_at": local_time(batch.started_at) if batch.started_at else "-",
            "file_count": batch.file_count,
            "size": human_bytes(batch.total_bytes),
            # `total` is what the database holds; `file_count` is what was
            # uploaded. They agree unless a batch is still being written, and
            # showing both would only invite the question of why they differ.
            "processed": finished,
            "completed": done,
            "failed": failed,
            "needs_review": review,
            "pending": max(0, (total or batch.file_count) - finished),
            "percent": round(100.0 * finished / total, 1) if total else 0.0,
            "can_run": state in RESUMABLE_BATCH_STATES or state is BatchState.QUEUED,
            "run_label": "Run" if state is BatchState.QUEUED else "Resume",
            "can_stop": state in (BatchState.QUEUED, BatchState.RUNNING),
            "needs_force": state in BUSY_BATCH_STATES,
            "is_stopping": state is BatchState.STOPPING,
        }

    def _run_batch(self, uow: UnitOfWork, batch: Any) -> dict[str, Any]:
        """Queue a batch for processing, resuming a stopped one in place."""
        if batch.state is BatchState.RUNNING:
            raise RepositoryError(
                f"{batch.name!r} is already running.")
        if batch.state is BatchState.STOPPING:
            raise RepositoryError(
                f"{batch.name!r} is still stopping. Wait for the document in "
                "flight to finish, then run it again.")

        if batch.state is BatchState.QUEUED:
            # Already queued: the operator means "start now", so move it to the
            # head. Its position relative to nothing else changes if it is
            # already there.
            uow.batches.promote(batch)
            return {"state": BatchState.QUEUED.value,
                    "detail": f"{batch.name!r} is next in the queue."}

        if batch.state is BatchState.COMPLETED:
            # A completed batch is only ever marked so when every document has
            # reached a terminal state, so there is nothing for Run to do. The
            # documents that did *not* succeed are a different request, and the
            # message points at the button that serves it.
            raise RepositoryError(
                f"{batch.name!r} has already finished. Use Reprocess Failed to "
                "retry the documents that did not succeed.")

        uow.batches.resume(batch)
        pending = uow.batches.progress(batch.id)
        remaining = 0 if pending is None else max(
            0, pending.total - pending.completed - pending.failed
            - pending.needs_review)
        return {"state": BatchState.QUEUED.value,
                "detail": (f"{batch.name!r} resumed with {remaining} document(s) "
                           "left; work already done is kept.")}

    def _stop_batch(self, uow: UnitOfWork, batch: Any) -> dict[str, Any]:
        """Ask a batch to stop, allowing in-flight work to finish."""
        if batch.state is BatchState.STOPPING:
            raise RepositoryError(f"{batch.name!r} is already stopping.")
        if batch.state in (BatchState.STOPPED, BatchState.PAUSED):
            raise RepositoryError(f"{batch.name!r} is already stopped.")
        if batch.state in (BatchState.COMPLETED, BatchState.FAILED):
            raise RepositoryError(
                f"{batch.name!r} has already finished; there is nothing to stop.")

        in_flight = uow.batches.in_flight(batch.id)
        state = uow.batches.request_stop(batch)
        if state is BatchState.STOPPING:
            detail = (f"Stopping {batch.name!r}. {in_flight} document(s) "
                      "already in progress will finish first.")
        else:
            detail = f"{batch.name!r} stopped. No document was interrupted."
        return {"state": state.value, "detail": detail}

    def delete_batch(self, batch_id: int, force: bool = False) -> dict[str, Any]:
        """Remove a batch, its rows and its derived files.

        What is removed and what is not:

          * **Database rows** - the batch and, by cascade, its documents, OCR
            pages, extractions, people, properties, validations and failure
            events. Scoped by foreign key, so no other batch is touched.
          * **Prepared copies** - the overlay-stripped PDFs under
            `runtime/data/cleaned`. These are derived, regenerable and useless
            once the rows they belong to are gone. Deleted by the paths recorded
            on the documents rather than by sweeping the directory: two batches
            can hold a file of the same name, and a sweep would take the other
            batch's copy.
          * **Not the source PDFs.** They are the operator's own files, often
            the only copy, and were never ours to remove.
          * **Not exported CSVs.** They are deliverables that were explicitly
            asked for; deleting a batch is not a request to destroy a report
            already produced from it.
        """
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None:
                raise RepositoryError(f"Batch {batch_id} not found.")
            if batch.state in BUSY_BATCH_STATES and not force:
                raise RepositoryError(
                    f"{batch.name!r} is {batch.state.value}. Stop it first, or "
                    "confirm deletion to stop and delete it.")

        if force:
            self._settle_for_delete(batch_id)

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None:
                # Another window won the race. Deleting an already-deleted batch
                # is the outcome the operator asked for, so it is not an error.
                return {"deleted": batch_id, "files_removed": 0,
                        "detail": "Batch was already deleted."}
            name = batch.name
            stale = uow.batches.in_flight(batch_id)
            if stale:
                raise RepositoryError(
                    f"{name!r} still has {stale} document(s) in progress. It has "
                    "been asked to stop - try deleting again in a moment.")
            paths = uow.batches.cleaned_paths(batch_id)
            session.delete(batch)

        # Files are removed only after the rows are committed. Reversed, a
        # failed commit would leave rows pointing at files that no longer exist,
        # which is the one inconsistency the viewer cannot recover from.
        removed = self._remove_files(paths)
        log.info("batch deleted: %s (%d prepared file(s) removed)", name, removed,
                 extra={"batch": batch_id, "batch_name": name, "files": removed})
        return {"deleted": batch_id, "name": name, "files_removed": removed,
                "detail": f"Deleted {name!r} and {removed} prepared file(s)."}

    def _settle_for_delete(self, batch_id: int) -> None:
        """Stop a busy batch and wait, briefly, for its worker to let go."""
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None or batch.state not in BUSY_BATCH_STATES:
                return
            if batch.state is BatchState.RUNNING:
                uow.batches.request_stop(batch)

        deadline = time.monotonic() + self.DELETE_SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            with session_scope(self.sessions) as session:
                if UnitOfWork(session).batches.in_flight(batch_id) == 0:
                    return
            time.sleep(0.5)

    @staticmethod
    def _remove_files(paths: list[str]) -> int:
        removed = 0
        for raw in paths:
            try:
                path = Path(raw)
                if path.is_file():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                # A file left behind is untidy; a delete that half-succeeded and
                # then raised would leave the operator unable to tell whether
                # the batch is gone. The rows are already committed by now.
                log.warning("could not remove prepared file %s: %s", raw, exc)
        return removed

    def _devanagari_label(self) -> str:
        """What the header pill says about language handling."""
        setting = self._setting("translation_devanagari_as", "auto")
        return {"hin_Deva": "English (Devanagari as Hindi)",
                "mar_Deva": "English (Devanagari as Marathi)"}.get(
                    setting, "English (auto-detect)")

    # -- settings ----------------------------------------------------------

    def _setting(self, key: str, default: str = "") -> str:
        """Read a setting from an in-memory cache.

        The whole table loads once - it is a handful of rows - rather than one
        query per key. The Validation page reads 12 settings; that was 12
        sessions and 14 queries before this.
        """
        if not self._settings_loaded:
            self._load_settings()
        value = self._settings_cache.get(key)
        return default if value is None else value

    def _load_settings(self) -> None:
        self._settings_loaded = True
        if not self.db_ok:
            self._settings_cache = {}
            return
        try:
            with session_scope(self.sessions) as session:
                self._settings_cache = UnitOfWork(session).settings.all()
        except Exception:  # noqa: BLE001
            self._settings_cache = {}

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            for key, value in values.items():
                uow.settings.set(key, str(value))
        self._settings_loaded = False  # write-through invalidation
        # Mode and cooldown live on the runner object, so the saved value has to
        # be pushed onto it - reloading the cache alone changes nothing.
        self._apply_runner_settings()
        return {"saved": len(values)}

    def save_rules(self, rules: dict[str, Any],
                   thresholds: dict[str, Any]) -> dict[str, Any]:
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            for key, enabled in rules.items():
                uow.settings.set(f"rule_{key}", "true" if enabled else "false")
            for key, value in thresholds.items():
                uow.settings.set(key, str(value))
        self._settings_loaded = False
        return {"saved": len(rules) + len(thresholds)}

    def watermark(self, action: str) -> dict[str, Any]:
        """Single entry point for the Watermark Remover page.

        Scanning and removal both run inline. That is a deliberate choice for
        this page rather than an oversight: detection is a PDF structure walk of
        a handful of files chosen by hand, not a batch, and the bridge already
        dispatches every call off the UI thread. Putting it through the batch
        runner would buy queueing nobody asked for.
        """
        from core import watermark as wm

        if action == "browse":
            picker = getattr(self, "file_picker", None)
            if picker is None:
                raise RuntimeError("no file picker is attached")
            added = self.watermark_files.add([Path(p) for p in picker()])
            return {"added": added, "total": len(self.watermark_files.paths)}

        if action == "clear":
            self.watermark_files = Selection()
            self._watermark_scans.clear()
            self._watermark_removals.clear()
            self._watermark_outputs.clear()
            return {"total": 0}

        if action == "scan":
            scanned = 0
            for path in self.watermark_files.paths:
                try:
                    self._watermark_scans[str(path)] = wm.scan(path)
                    scanned += 1
                except Exception as exc:  # noqa: BLE001 - one bad PDF must not
                    # abort the rest of the selection.
                    self._record_error("watermark scan", exc)
            return {"scanned": scanned}

        if action == "remove":
            removed = failed = 0
            self._watermark_outputs.clear()
            for path in self.watermark_files.paths:
                result = self._watermark_scans.get(str(path))
                if result is None or not result.confirmed:
                    continue
                try:
                    target = self._cleaned_target(path)
                except OSError as exc:
                    # Creating the output folder is the first thing that can
                    # fail, and it fails for reasons the operator can act on -
                    # a read-only drive, a network share that has gone away.
                    # Reported per file rather than aborting the batch, because
                    # a selection can span several folders and only one of them
                    # may be the problem.
                    self._watermark_removals[str(path)] = wm.RemovalResult(
                        path, None,
                        error=self._filesystem_reason(exc, path.parent))
                    failed += 1
                    continue
                try:
                    # allow_lossy stays False. A raster watermark is burned into
                    # the scan, so the pixels beneath were never captured -
                    # "removing" it means inventing content on a legal document.
                    outcome = wm.remove(path, target, scan_result=result,
                                        allow_lossy=False)
                    self._watermark_removals[str(path)] = outcome
                    if outcome.ok:
                        removed += 1
                        self._watermark_outputs[str(path)] = target
                    else:
                        failed += 1
                except OSError as exc:
                    # Disk full, target locked by a viewer, permission revoked
                    # mid-run. `wm.remove` catches most of these itself, but not
                    # every path through PyMuPDF's save raises inside it.
                    self._watermark_removals[str(path)] = wm.RemovalResult(
                        path, None, error=self._filesystem_reason(exc, target))
                    failed += 1
                except Exception as exc:  # noqa: BLE001
                    self._record_error("watermark removal", exc)
                    self._watermark_removals[str(path)] = wm.RemovalResult(
                        path, None, error=f"{type(exc).__name__}: {exc}")
                    failed += 1
            folders = sorted({str(p.parent) for p in self._watermark_outputs.values()})
            return {"removed": removed, "failed": failed,
                    "output_dir": folders[0] if folders else "",
                    "output_dirs": folders}

        if action == "open":
            target = self._watermark_output_dir()
            target.mkdir(parents=True, exist_ok=True)
            return {"path": str(target)}

        raise ValueError(f"unknown watermark action {action!r}")

    #: Where cleaned copies go, relative to the folder the deeds came from.
    CLEANED_SUBFOLDER = "Cleaned Watermark"

    def _cleaned_target(self, source: Path) -> Path:
        """Where `source`'s cleaned copy is written.

        Beside the deed rather than in a shared runtime folder: an operator
        working through a folder of deeds wants the results with them, not
        somewhere under the installation directory. The subfolder is created if
        absent and reused if present - it is `mkdir(exist_ok=True)`, so a second
        run adds to the folder rather than making a second one.

        The filename is the original, unchanged. That is safe because the copy
        lands in a *different directory* from the source, so the input can never
        be overwritten - which is the one outcome that would be unrecoverable.

        The exception is a deed that is itself already inside a `Cleaned
        Watermark` folder - cleaning a cleaned file. There the subfolder would
        be the source's own directory and the original name would overwrite the
        input, so that case keeps the older `_clean` suffix instead. Nesting a
        second `Cleaned Watermark` inside the first would be tidier to look at
        and worse to use.
        """
        parent = source.parent
        if parent.name == self.CLEANED_SUBFOLDER:
            return parent / f"{source.stem}_clean{source.suffix}"
        folder = parent / self.CLEANED_SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
        return folder / source.name

    def _watermark_output_dir(self) -> Path:
        """What "Open Output Folder" should show.

        The folder the last run actually wrote to. Falls back to the shared
        runtime directory only when nothing has been cleaned this session, so
        the button is never dead.
        """
        for target in self._watermark_outputs.values():
            if target.parent.is_dir():
                return target.parent
        for path in self.watermark_files.paths:
            candidate = path.parent / self.CLEANED_SUBFOLDER
            if candidate.is_dir():
                return candidate
        return WATERMARK_DIR

    @staticmethod
    def _filesystem_reason(exc: OSError, where: Path) -> str:
        """A filesystem failure in words an operator can act on.

        `[Errno 28]` and `WinError 32` are not answers. Each of these has a
        different remedy - free space, close the file, ask for write access -
        and the whole point of naming them is that the operator knows which.
        """
        import errno

        winerror = getattr(exc, "winerror", None)
        if exc.errno == errno.ENOSPC:
            return (f"The disk holding {where.parent} is full. Free some space "
                    "and run the removal again.")
        if exc.errno == errno.EACCES or isinstance(exc, PermissionError):
            # On Windows a locked file and a permissions problem both surface as
            # PermissionError; WinError 32 distinguishes them, and the remedies
            # are completely different.
            if winerror == 32:
                return (f"{where.name} is open in another program. Close it and "
                        "run the removal again.")
            return (f"No permission to write to {where.parent}. Choose a folder "
                    "you can write to, or ask for access to this one.")
        if exc.errno == errno.EROFS:
            return f"{where.parent} is on a read-only drive."
        if exc.errno == errno.ENAMETOOLONG:
            return f"The path for {where.name} is too long for this filesystem."
        if exc.errno == errno.ENOENT:
            return f"{where.parent} no longer exists."
        return f"Could not write to {where.parent}: {exc.strerror or exc}"

    # -- OCR tool ----------------------------------------------------------
    #
    # A standalone page over the *same* `OcrStage` the pipeline uses -
    # `self.stages.ocr`, the configured instance, not a second one. Engine
    # choice, Surya interpreter, DPI, language list, timeout and the GPU lease
    # are therefore whatever the batch pipeline is using at that moment, and
    # there is no second implementation to keep in step.
    #
    # It exists because the pipeline's OCR is only reachable by committing a
    # batch. An operator who wants to know whether a scan is legible at all, or
    # to OCR a handful of files and feed the text straight to extraction, had no
    # way to do either.

    def ocr_tool(self, action: str) -> dict[str, Any]:
        """Single entry point for the OCR page, mirroring `watermark`."""
        if action == "browse":
            picker = getattr(self, "file_picker", None)
            if picker is None:
                raise RuntimeError("no file picker is attached")
            added = self.ocr_files.add([Path(p) for p in picker()])
            return {"added": added, "total": len(self.ocr_files.paths)}

        if action == "add":
            return {"total": len(self.ocr_files.paths)}

        if action == "clear":
            if self._ocr_busy():
                raise RepositoryError(
                    "OCR is still running. Stop it before clearing the list.")
            self.ocr_files = Selection()
            with self._ocr_lock:
                self._ocr_results.clear()
            self._ocr_detail = ""
            return {"total": 0}

        if action == "run":
            return self._start_ocr_run()

        if action == "stop":
            # Cancels between documents, never inside one: the page in flight
            # finishes so its text is not thrown away, exactly as a batch stop
            # behaves.
            self._ocr_cancel.set()
            self._ocr_detail = "stopping after the current file"
            return {"stopping": True}

        if action == "open":
            paths.OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
            return {"path": str(paths.OCR_TEXT_DIR)}

        if action == "queue":
            return self._queue_ocr_results()

        raise ValueError(f"unknown OCR action {action!r}")

    def _ocr_busy(self) -> bool:
        thread = self._ocr_thread
        return thread is not None and thread.is_alive()

    def _start_ocr_run(self) -> dict[str, Any]:
        if self._ocr_busy():
            raise RepositoryError("OCR is already running.")
        if not self.ocr_files.paths:
            raise RepositoryError("Choose some PDFs first.")

        ok, detail = self.stages.ocr.available()
        if not ok:
            # An unavailable engine is an environment problem, and saying so
            # here is far better than starting a run that fails identically on
            # every file.
            raise RepositoryError(f"OCR engine unavailable: {detail}")

        self._ocr_cancel.clear()
        with self._ocr_lock:
            self._ocr_results.clear()
        self._ocr_detail = "starting"
        self._ocr_thread = threading.Thread(
            target=self._ocr_worker, name="ocr-tool", daemon=True)
        self._ocr_thread.start()
        return {"started": len(self.ocr_files.paths), "engine": self.stages.ocr.engine}

    def _ocr_worker(self) -> None:
        """Run OCR over the selection, one file at a time.

        One at a time and under the pipeline's own GPU lease. Surya and the
        language model cannot both be resident on a 4 GB card, and this page can
        be used while a batch is running - without the lease the two would race
        for VRAM and whichever lost would OOM mid-document.
        """
        paths.OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        stage = self.stages.ocr
        targets = list(self.ocr_files.paths)

        for index, path in enumerate(targets, 1):
            if self._ocr_cancel.is_set():
                self._ocr_detail = f"stopped after {index - 1} of {len(targets)}"
                break
            self._ocr_detail = f"reading {path.name} ({index} of {len(targets)})"
            self._record_ocr(path, {"state": "running"})
            started = time.monotonic()
            try:
                with self.runner.gpu_lease(stage, "ocr-tool"):
                    outcome = stage.run(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not end
                # the run; the rest of the selection is still worth doing.
                self._record_error("ocr tool", exc)
                self._record_ocr(path, self._ocr_failure(path, str(exc), None))
                continue

            elapsed = time.monotonic() - started
            if not outcome.ok:
                self._record_ocr(path, self._ocr_failure(
                    path, outcome.detail, outcome, elapsed))
                continue

            pages = outcome.data.get("page_texts") or []
            text = "\n\n".join(t for _, t in pages).strip()
            if not text:
                # Succeeded mechanically and produced nothing. Reported as the
                # distinct condition it is: a scan needing a real OCR pass, not
                # a broken file.
                self._record_ocr(path, self._ocr_failure(
                    path, "OCR produced no text", outcome, elapsed,
                    code=failure_codes.OCR_NO_TEXT))
                continue

            out_file = paths.OCR_TEXT_DIR / f"{path.stem}.txt"
            try:
                out_file.write_text(text, encoding="utf-8")
            except OSError as exc:
                self._record_ocr(path, self._ocr_failure(
                    path, f"could not write {out_file.name}: {exc}", outcome,
                    elapsed, code=failure_codes.FILE_ACCESS_ERROR))
                continue

            self._record_ocr(path, {
                "state": "done", "ok": True,
                "pages": outcome.data.get("pages") or len(pages),
                "chars": len(text), "seconds": round(elapsed, 1),
                "text_path": str(out_file),
                "page_texts": pages,
            })
            log.info("ocr tool: %s -> %d chars in %.1fs", path.name, len(text),
                     elapsed, extra={"file": path.name, "chars": len(text)})

        else:
            self._ocr_detail = f"finished {len(targets)} file(s)"
        if self._ocr_cancel.is_set():
            self._ocr_cancel.clear()

    def _ocr_failure(self, path: Path, detail: str, outcome: Any,
                     seconds: float = 0.0, code: str | None = None
                     ) -> dict[str, Any]:
        """Turn a failure into the same structured reason the pipeline records.

        `failure_codes.classify` and `pdf_validation` are reused rather than
        re-deriving a message here, so a file that fails on this page and the
        same file failing inside a batch give the operator identical wording.
        """
        from core.pdf_validation import validate_pdf

        # The file itself is examined only now, when OCR has already failed -
        # the same order the pipeline uses, and for the same reason: "is the
        # file broken?" is worth a few KB of reads once something has gone
        # wrong, and the answer is better than whatever OCR said on its way out.
        validation_status = None
        if code is None:
            try:
                verdict = validate_pdf(path)
            except Exception:  # noqa: BLE001 - diagnosis must never raise
                verdict = None
            if verdict is not None and verdict.is_corrupt:
                detail = verdict.error_message or detail
                validation_status = getattr(verdict.status, "value", None)

        if code is None:
            code, message, retryable = failure_codes.classify_text(
                detail, validation_status)
        else:
            message, retryable = failure_codes.MESSAGES.get(
                code, failure_codes.MESSAGES[failure_codes.UNKNOWN_ERROR])

        log.warning("ocr tool failed: %s - %s", path.name, message,
                    extra={"file": path.name, "code": code})
        return {"state": "done", "ok": False, "seconds": round(seconds, 1),
                "code": code, "message": message,
                "stage": failure_codes.STAGES.get(code, "OCR"),
                "retryable": retryable,
                "detail": detail,
                "pages": (outcome.data.get("pages") if outcome is not None
                          and getattr(outcome, "data", None) else 0) or 0}

    def _record_ocr(self, path: Path, row: dict[str, Any]) -> None:
        with self._ocr_lock:
            self._ocr_results[str(path)] = row

    def _queue_ocr_results(self) -> dict[str, Any]:
        """Hand successful OCR results to the deed pipeline.

        This is the integration that makes the page more than a viewer. A batch
        is created from the files that produced text, their pages are written
        straight into `ocr_pages`, and the OCR stage is marked DONE - so the
        pipeline starts these documents at *extraction* rather than paying for
        an OCR pass that has already been done. The runner's `_claim_downstream`
        is written for exactly this case ("OCR ran in an earlier run"), so no
        pipeline change is needed to accept them.
        """
        if not self.db_ok:
            raise RuntimeError(self.db_detail.splitlines()[0])
        if self._ocr_busy():
            raise RepositoryError("Wait for the OCR run to finish first.")

        with self._ocr_lock:
            ready = [(Path(key), row) for key, row in self._ocr_results.items()
                     if row.get("ok") and row.get("page_texts")]
        if not ready:
            raise RepositoryError(
                "No file has produced OCR text yet. Run OCR first.")

        name = f"OCR {datetime.now().strftime('%d-%m-%Y %H:%M')}"
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("ocr_tool")
            batch = uow.batches.create(
                name, user, len(ready),
                sum(p.stat().st_size for p, _ in ready if p.is_file()))
            docs = uow.documents.add_many(batch, [
                {"document_id": p.stem, "source_filename": p.name,
                 "source_path": str(p),
                 "size_bytes": p.stat().st_size if p.is_file() else 0}
                for p, _ in ready])
            by_stem = {p.stem: row for p, row in ready}
            seeded = 0
            for doc in docs:
                row = by_stem.get(doc.document_id)
                if row is None:
                    continue
                uow.ocr.save_pages(doc, row["page_texts"])
                doc.page_count = row.get("pages") or doc.page_count
                # DONE, not PENDING: the text is already in the table, and
                # leaving it claimable would re-run minutes of GPU work over a
                # document that is finished with that stage.
                uow.documents.mark_stage(doc, "ocr", StageState.DONE)
                seeded += 1
            batch_id, batch_name = batch.id, batch.name

        log.info("ocr tool queued batch %s with %d pre-OCR'd document(s)",
                 batch_name, seeded,
                 extra={"batch": batch_id, "documents": seeded})
        return {"batch_id": batch_id, "name": batch_name, "documents": seeded,
                "detail": (f"Queued {seeded} document(s) as {batch_name!r}. "
                           "They start at extraction - the OCR text is already "
                           "stored.")}

    # -- page models -------------------------------------------------------

    def _machine(self) -> dict[str, Any]:
        """Machine panel, from cached probes only."""
        snap = self.status_service.snapshot()
        gpu_data = self.status_service.get("gpu").data
        backend = "PostgreSQL" if self.db_ok else "unreachable"
        if self.db_ok and self.db_detail:
            parts = self.db_detail.split()
            backend = f"{parts[0]} {parts[1]}" if len(parts) > 1 else backend
        health = {
            "pressure": gpu_data.get("pressure") or "unknown",
            "engine": self.status_service.get("ai").data.get("engine") or {},
            "resources": {
                "vram_free_bytes": gpu_data.get("vram_free_bytes") or 0,
                "vram_total_bytes": gpu_data.get("vram_total_bytes") or 0,
                "ram_available_bytes": gpu_data.get("ram_available_bytes") or 0,
                "ram_total_bytes": gpu_data.get("ram_total_bytes") or 0,
            },
        }
        return machine_model(snap["hardware"], snap["profile"], health, backend)

    def _dashboard(self, params: dict[str, Any]) -> dict[str, Any]:
        page = int(params.get("page") or 1)
        notice = None
        # The template renders `message` unescaped so these notices can carry
        # deliberate markup. Anything derived from an exception must therefore be
        # escaped here: `_record_error` formats `{exc}`, and an exception text
        # routinely contains a filename - which the user chose, and which can
        # hold angle brackets.
        if not self.db_ok:
            notice = {"level": "danger",
                      "message": "<strong>Database unreachable.</strong> "
                                 + html.escape(self.db_detail.splitlines()[0])}
        elif self.errors:
            notice = {"level": "warn", "message": html.escape(self.errors[-1])}

        active_model: dict[str, Any] | None = None
        manage: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []
        total_completed = 0

        if self.db_ok:
            with session_scope(self.sessions) as session:
                uow = UnitOfWork(session)
                active = uow.batches.active()
                if active is not None:
                    progress = uow.batches.progress(active.id)
                    stats = self.runner.stats
                    remaining = (progress.total - progress.completed
                                 - progress.failed - progress.needs_review)
                    active_model = {
                        "name": active.name, "state": active.state.value,
                        "total": progress.total, "completed": progress.completed,
                        "failed": progress.failed,
                        "needs_review": progress.needs_review,
                        "stages": {k: {"done": v.done} for k, v in progress.stages.items()},
                        "seconds_per_document": round(stats.seconds_per_document, 1),
                        "eta_seconds": stats.eta_seconds(max(0, remaining)),
                        "started_at": active.started_at,
                    }

                # Every batch that is not finished, in one table. Split across
                # a "queued" list and a separate running panel, an operator had
                # no single place showing what exists and what can be done to
                # it - and no way at all to reach a stopped batch.
                rows, _ = uow.batches.list_paginated(1, 20, LIVE_BATCH_STATES)
                live_progress = uow.batches.progress_many([b.id for b in rows])
                manage = [self._manage_row(b, live_progress.get(b.id))
                          for b in rows]

                done_rows, total_completed = uow.batches.list_paginated(
                    page, 5, [BatchState.COMPLETED, BatchState.FAILED])
                # One query for the whole page. This was `progress(b.id)` inside
                # the loop, and `progress` itself issued seven statements - so a
                # five-batch page cost about forty round trips.
                page_progress = uow.batches.progress_many([b.id for b in done_rows])
                for b in done_rows:
                    p = page_progress.get(b.id)
                    if p is None:
                        continue
                    completed.append({
                        "id": b.id, "name": b.name, "total": p.total,
                        "completed": p.completed, "needs_review": p.needs_review,
                        "failed": p.failed, "has_failed": p.failed > 0,
                        # A failed batch is resumable - it stopped short, and
                        # the documents it never reached are still pending.
                        "can_resume": b.state in RESUMABLE_BATCH_STATES,
                        "state": b.state.value,
                        "finished_at": local_time(b.finished_at)})

        return dashboard_model(
            active=active_model, manage=manage, completed=completed,
            page=page, completed_total=total_completed,
            max_queued=MAX_QUEUED_BATCHES, notice=notice)

    def _upload(self, _params: dict[str, Any]) -> dict[str, Any]:
        queued_count = 0
        next_position = 1
        if self.db_ok:
            with session_scope(self.sessions) as session:
                uow = UnitOfWork(session)
                queued_count = uow.batches.queued_count()
                next_position = queued_count + 1

        paths = self.selection.paths
        size = self.selection.total_bytes
        over_files = len(paths) > MAX_FILES_PER_BATCH
        over_size = size > MAX_BATCH_BYTES
        messages = []
        if over_files:
            messages.append(f"{len(paths):,} files exceeds the "
                            f"{MAX_FILES_PER_BATCH:,}-file limit.")
        if over_size:
            messages.append(f"{human_bytes(size)} exceeds the "
                            f"{human_bytes(MAX_BATCH_BYTES)} limit.")

        shown = paths[:200]
        return {
            "username": self._setting("last_username", ""),
            "suggested_name": f"Batch_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "selected_count": len(paths),
            "selected_size": human_bytes(size),
            "next_position": next_position,
            "max_files": f"{MAX_FILES_PER_BATCH:,}",
            "max_size": human_bytes(MAX_BATCH_BYTES),
            "queued_count": queued_count,
            "max_queued": MAX_QUEUED_BATCHES,
            "queue_full": queued_count >= MAX_QUEUED_BATCHES,
            "has_selection": bool(paths),
            "files": [{"index": i, "name": p.name, "size": human_bytes(p.stat().st_size),
                       "document_id": p.stem} for i, p in enumerate(shown, 1)],
            "truncated": len(paths) > len(shown),
            "shown": len(shown),
            "over_limit": bool(messages),
            "over_limit_message": " ".join(messages),
            "can_submit": bool(paths) and not messages
                          and queued_count < MAX_QUEUED_BATCHES and self.db_ok,
        }

    def _failed_ocr_page(self, params: dict[str, Any]) -> dict[str, Any]:
        """The Failed OCR screen.

        Reads the same `failed_ocr` the bridge serves to the live refresh, so
        the first paint and every later poll cannot disagree about what failed.
        """
        batch_id = int(params.get("batch_id") or 0) or None
        page = int(params.get("page") or 1)
        model = self.failed_ocr(batch_id, page)
        model["running"] = self.runner.state.value == "running"
        model["has_failures"] = bool(model["documents"])
        model["none_failed"] = not model["documents"] and not model.get("db_offline")
        model["plural"] = "" if model["total"] == 1 else "s"
        model["pager"] = pager(model["page"], 25, model["total"])
        return model

    def _processing(self, _params: dict[str, Any]) -> dict[str, Any]:
        pressure = str(self.status_service.get("gpu").data.get("pressure")
                       or "normal")
        model: dict[str, Any] = {
            "running": self.runner.state.value == "running",
            "runner_detail": self.runner.status().get("detail") or "idle",
            "pressure": pressure.title(),
            "pressure_warning": pressure in ("high", "critical"),
            "pressure_detail": (
                "Processing is throttled while memory is scarce. In-flight documents "
                "finish; new ones wait. Closing other applications will speed this up."),
        }
        if not self.db_ok:
            return model

        stage_meta = (("ocr", "OCR", ""), ("extract", "Extraction", "accent"),
                      ("translate", "Translation", "saffron"),
                      ("validate", "Validation", ""))
        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            active = uow.batches.active()
            if active is None:
                return model
            progress = uow.batches.progress(active.id)
            remaining = (progress.total - progress.completed - progress.failed
                         - progress.needs_review)
            finished = progress.completed + progress.failed + progress.needs_review
            model["active"] = {
                "name": active.name,
                "stages": [{"label": label, "bar_class": bar,
                            "done": progress.stages[key].done,
                            "failed": progress.stages[key].failed,
                            "total": progress.total,
                            "percent": percent(progress.stages[key].done, progress.total)}
                           for key, label, bar in stage_meta],
                "completed": progress.completed,
                "needs_review": progress.needs_review,
                "failed": progress.failed,
                "eta": human_duration(self.runner.stats.eta_seconds(max(0, remaining))),
                # The reference's headline readout: "998 / 998 (100.0%)".
                # One decimal place, because on a 1000-file batch the last
                # thirty documents all read "99%" without it.
                "total": progress.total,
                "finished": finished,
                "overall_percent": percent(finished, progress.total),
                "overall_readout": (
                    f"{finished} / {progress.total} "
                    f"({(100.0 * finished / progress.total) if progress.total else 0:.1f}%)"),
                "in_queue": max(0, remaining),
            }
            model["failed_ocr_count"] = uow.documents.failed_ocr_count()
            model.update(self._failure_panel(uow, active.id))
        return model

    #: How many failures the Processing page lists inline. A thousand-file batch
    #: can fail in bulk, and a page that renders every one of them stops being
    #: readable; the Failed OCR page is the place for the full list.
    PROCESSING_FAILURE_LIMIT = 25

    def _failure_panel(self, uow: UnitOfWork, batch_id: int) -> dict[str, Any]:
        """Why each document in this batch failed, in words.

        The Processing page showed a "Failed: 9" tile and nothing else - an
        operator could see *that* nine documents had failed and had to open
        another page, per document, to learn why. Worse, "failed" covers a
        corrupt file, an unreachable AI server and a database error, and those
        call for completely different responses.

        `failure_codes.classify` does the work, reading only what is already
        stored - so this needs no new column, no reprocessing, and explains
        documents that failed before any of this existed.
        """
        docs = uow.documents.failed_for_batch(batch_id)
        rows: list[dict[str, Any]] = []
        for doc in docs[:self.PROCESSING_FAILURE_LIMIT]:
            cause = _cause(doc) or {}
            rows.append({
                "document_pk": doc.id,
                "document_id": doc.document_id,
                "source_filename": doc.source_filename,
                # Never a bare "Failed": the classifier always yields a sentence,
                # falling back to "Processing failed for an unrecognised reason"
                # only when there genuinely is nothing recorded.
                "reason": cause.get("reason") or "No reason was recorded.",
                "code": cause.get("code") or "",
                "stage": cause.get("stage") or "Processing",
                # The raw text behind the sentence, for someone who wants it.
                "technical": cause.get("technical") or "",
                "retryable": cause.get("retryable", True),
                # An individual rerun goes through `requeue_ocr`, which by
                # design only touches documents whose *OCR* stage failed - it
                # ignores the rest rather than restarting healthy work. Offering
                # the button for an extraction or translation failure would
                # therefore produce a button that reports "0 queued" every time.
                # Those are handled by the batch-level Reprocess Failed, which
                # the card footer points at.
                "can_rerun": (cause.get("retryable", True)
                              and cause.get("failed_stage") == "ocr"),
                "failed_stage": cause.get("failed_stage") or "unknown",
                "attempts": doc.ocr_attempts,
            })
        return {
            "failures": rows,
            "has_failures": bool(rows),
            "failure_total": len(docs),
            "failures_truncated": len(docs) > len(rows),
            "failures_hidden": max(0, len(docs) - len(rows)),
        }

    def _data_view(self, params: dict[str, Any]) -> dict[str, Any]:
        page = int(params.get("page") or 1)
        query = str(params.get("query") or "").strip()
        model: dict[str, Any] = {
            "query": query,
            "statuses": [{"value": v, "label": lbl, "selected": params.get("status") == v}
                         for v, lbl in (("", "All"), ("processed", "Processed"),
                                        ("needs_review", "Needs Review"),
                                        ("failed", "Failed"))],
            "sorts": [{"value": v, "label": lbl, "selected": params.get("sort") == v}
                      for v, lbl in (("recent", "Most recent"), ("document", "Document ID"))],
            "batches": [], "rows": [], "has_rows": False,
            "pager": pager(page, 25, 0),
        }
        if not self.db_ok:
            return model

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batches, _ = uow.batches.list_paginated(1, 50)
            # Document counts come from one grouped query, not one per batch:
            # the Download selector shows a count beside every batch, and a
            # fifty-batch page would otherwise be fifty round trips to render a
            # dropdown.
            counts = {bid: p.total for bid, p in
                      uow.batches.progress_many([b.id for b in batches]).items()}
            # Nothing selected yet means the newest batch, which is what the
            # table below is already showing - so the Download selector and the
            # table can never disagree about which batch is in view.
            default_id = batches[0].id if batches else None
            chosen = params.get("batch_id")
            model["batches"] = [
                {"id": b.id, "name": b.name,
                 "document_count": counts.get(b.id, 0),
                 "state": b.state.value,
                 "created_at": local_time(b.created_at),
                 # Everything the operator needs to tell two similarly named
                 # batches apart, in the one line a <option> gives us.
                 "label": (f"{b.name} (ID {b.id}) - {counts.get(b.id, 0)} "
                           f"document(s) - {b.state.value} - "
                           f"{local_time(b.created_at)}"),
                 "selected": (str(b.id) == str(chosen) if chosen
                              else b.id == default_id)} for b in batches]

            batch_id = params.get("batch_id")
            target = int(batch_id) if str(batch_id or "").isdigit() else (
                batches[0].id if batches else None)
            if target is None:
                return model

            docs, total = uow.documents.list_for_batch(
                target, page=page, per_page=25, search=query or None)
            rows = []
            for doc in docs:
                prop = doc.property_
                flags = sorted({v.flag_code for v in doc.validations})
                for person in doc.persons or [None]:
                    rows.append({
                        "document_pk": doc.id, "document_id": doc.document_id,
                        "batch_name": doc.batch.name if doc.batch else "-",
                        "person_name": (person.name_translated or person.name
                                        if person else "-") or "-",
                        "relation": person.relation.value if person else "-",
                        "relation_class": "ok" if person and person.relation.value == "B" else "",
                        "pan": (person.pan_card_number if person else "") or "-",
                        "aadhaar": (person.aadhaar_number if person else "") or "-",
                        "property_address": (prop.schedule_c_address if prop else "") or "-",
                        "consideration": (f"{int(prop.sale_consideration):,}"
                                          if prop and prop.sale_consideration else "-"),
                        "stamp_value": (f"{int(prop.stamp_value):,}"
                                        if prop and prop.stamp_value else "-"),
                        "transaction_date": (prop.transaction_date.strftime("%d-%m-%Y")
                                             if prop and prop.transaction_date else "-"),
                        "state": doc.overall_state.value,
                        "state_class": state_badge(doc.overall_state.value),
                        "flags": [{"code": f, "class": ""} for f in flags],
                    })
            model["rows"] = rows
            model["has_rows"] = bool(rows)
            model["pager"] = pager(page, 25, total)
        return model

    def _batch_detail(self, batch_id: int) -> dict[str, Any]:
        from core.csv_export import CSV_COLUMNS, build_rows

        with session_scope(self.sessions) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is None:
                return {"name": "Not found", "has_documents": False}
            progress = uow.batches.progress(batch_id)
            docs, total = uow.documents.list_for_batch(batch_id, page=1, per_page=10)
            exports = [self._document_export(d) for d in docs]
            rows = build_rows(exports)

            # Every attribute is read here, inside the session. `user` is a
            # relationship and was never loaded by the query, so touching it
            # after the scope closes raises DetachedInstanceError - the object
            # is still usable for the columns already loaded, which is what made
            # this look correct until a batch was actually opened.
            fields = {
                "name": batch.name,
                "state": batch.state.value,
                "username": batch.user.username if batch.user else "-",
                "created_at": local_time(batch.created_at),
                "finished_at": local_time(batch.finished_at),
            }

        numeric = {"Report Serial Number", "Transaction Amount", "Stamp Value",
                   "Postal Code", "Pin Code (PC-L)"}
        return {
            "id": batch_id, "name": fields["name"], "state": fields["state"],
            "state_class": state_badge(fields["state"]),
            "total": progress.total, "completed": progress.completed,
            "needs_review": progress.needs_review, "failed": progress.failed,
            "has_failed": progress.failed > 0,
            "username": fields["username"],
            "created_at": fields["created_at"],
            "finished_at": fields["finished_at"],
            "ocr_status": f"{progress.stages['ocr'].done}/{progress.total}",
            "ocr_class": "ok" if progress.stages["ocr"].done == progress.total else "",
            "extract_status": f"{progress.stages['extract'].done}/{progress.total}",
            "extract_class": "ok" if progress.stages["extract"].done == progress.total else "",
            "translate_status": f"{progress.stages['translate'].done}/{progress.total}",
            "translate_class": "",
            "columns": list(CSV_COLUMNS),
            "has_documents": bool(rows),
            "documents": [{"cells": [
                {"value": row.get(col, ""),
                 "class": "num" if col in numeric else ("script" if "Name" in col
                                                        or "Address" in col else "")}
                for col in CSV_COLUMNS]} for row in rows],
            "pager": pager(1, 10, total),
        }

    def _settings(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._profile = self.status_service.snapshot()["profile"]
        # Machine details live here now rather than on the dashboard. The same
        # `_machine()` as before - cached probes only, so this adds no request.
        machine = self._machine()
        prompt_path = paths.PROMPT_FILE
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""

        def options(pairs, current):
            return [{"value": v, "label": lbl, "selected": v == current}
                    for v, lbl in pairs]

        return {
            **machine,
            # Devanagari is the one script that carries two languages, so it
            # is the one language choice that changes what the pipeline does.
            # Every other script identifies its language by itself, and Surya's
            # recognition model is multilingual - it takes no language hint at
            # all, which is why the OCR selector that used to sit here was
            # removed: it was saved, displayed, and read by nothing (R-038).
            "devanagari_languages": options(
                (("auto", "Detect automatically (recommended)"),
                 ("hin_Deva", "Always Hindi"),
                 ("mar_Deva", "Always Marathi")),
                self._setting("translation_devanagari_as", "auto")),
            "dpis": options((("200", "200 DPI"), ("300", "300 DPI (recommended)"),
                             ("400", "400 DPI")), self._setting("ocr_dpi", "300")),
            "batch_modes": options((("manual", "Manual"), ("auto", "Auto")),
                                   self._setting("batch_mode", "manual")),
            "llm_modes": [
                {"value": "medium", "label": "Medium (4B)",
                 "selected": self._setting("llm_mode", "medium") == "medium"},
                {"value": "high", "label": "High (12B) - not installed",
                 "selected": False, "unavailable": True},
            ],
            "cooldown": 60,
            # Cached probe, never a live call - this was the last blocking
            # network access left in a render path.
            "model_name": ((self.status_service.get("ai").data.get("engine") or {})
                           .get("model") or "-"),
            "quantisation": self._profile.get("quantisation") or "-",
            "lossy_model": not self._profile.get("lossless", True),
            "n_ctx": f"{self._profile.get('n_ctx') or 0:,}",
            "prompt_capacity": f"{self._profile.get('prompt_capacity_tokens') or 0:,}",
            "device": (self._profile.get("device") or "-").upper(),
            "stamp_multiplier": self._setting("stamp_value_multiplier", "1"),
            "stamp_disabled": self._setting("rule_stamp_value", "false") != "true",
            "prompt": prompt,
            "prompt_tokens": f"~{len(prompt) // 4}",
            "debug_logs": self._setting("debug_logs", "false") == "true",
            "db_summary": self.db_detail.split(",")[0] if self.db_ok else "unreachable",
            "db_state": "Connected" if self.db_ok else "Offline",
            "db_class": "ok" if self.db_ok else "danger",
            "update_url": self._setting("update_repo_url", ""),
            "app_version": APP_VERSION,
        }

    def _validation(self, _params: dict[str, Any]) -> dict[str, Any]:
        defs = (
            ("pan", "PAN Verification", "PM",
             "Checks PAN format and that the value appears in the OCR. "
             "Well-formed but absent values are discarded rather than exported."),
            ("aadhaar", "Aadhaar Verification", "WAN",
             "Requires exactly 12 digits and locates the number in the OCR, "
             "tolerating spaces and line wraps."),
            ("registration_fee", "Registration Fee Verification", "WSV",
             "Cross-checks the fee against the OCR and an independent regex sweep."),
            ("sale_consideration", "Sale Consideration Verification", "WSC",
             "Confirms the amount appears in the source, in plain or Indian grouping."),
            ("transaction_date", "Transaction Date Verification", "WTD",
             "Requires an ISO date that maps to a date present in the OCR."),
            ("ocr_cross_verify", "OCR Cross Verification", "",
             "Master switch for source-grounding. Turning this off disables every "
             "presence check and is not recommended on quantised weights."),
            ("confidence", "Confidence Validation", "",
             "Derives a per-field confidence from validator outcomes."),
            ("stamp_value", "Stamp Value Verification", "SSV",
             "Inactive: the reference export contradicts the written formula."),
        )
        retry_off = not getattr(self.stages.extract, "retry_supported", False)
        return {
            "rules": [{"key": k, "name": n, "flag": f, "description": d,
                       "enabled": self._setting(f"rule_{k}", "true") == "true",
                       "locked": k == "stamp_value"} for k, n, f, d in defs],
            "pan_coverage_threshold": self._setting("pan_coverage_threshold", "0.6"),
            "pan_min_unmatched": self._setting("pan_coverage_min_unmatched", "2"),
            "pan_split_threshold": self._setting("pan_split_threshold", "25"),
            "proximity_chars": self._setting("pan_aadhaar_proximity_chars", "250"),
            "stamp_value_disabled": self._setting("rule_stamp_value", "false") != "true",
            "retry_unavailable": retry_off,
            "retry_detail": (
                "The loaded weights do not support the split-prompt retry path, and a "
                "same-prompt rerun at temperature 0 is byte-identical. Documents that "
                "fail validation are routed to review instead of retried."),
        }

    def _ocr_page(self, _params: dict[str, Any]) -> dict[str, Any]:
        """OCR tool page model, shaped like the Watermark Remover's.

        Deliberately the same structure - dropzone, selected-file table,
        progress row, button row - so the two tool pages read as one feature
        family rather than two people's ideas of a tool page.
        """
        running = self._ocr_busy()
        with self._ocr_lock:
            results = dict(self._ocr_results)

        files: list[dict[str, Any]] = []
        succeeded = failed = 0
        for path in self.ocr_files.paths:
            row = results.get(str(path)) or {}
            state = row.get("state") or "pending"
            item: dict[str, Any] = {
                "name": path.name, "pages": row.get("pages") or "",
                "chars": f"{row['chars']:,}" if row.get("chars") else "",
                "seconds": f"{row['seconds']}s" if row.get("seconds") else "",
                "done": state == "done", "running": state == "running",
                "ok": bool(row.get("ok")),
                # The reason, not just "failed". Same wording the pipeline uses
                # for the same condition, because it is the same classifier.
                "reason": row.get("message") or "",
                "code": row.get("code") or "",
                "result": "", "result_class": "",
            }
            if state == "done":
                if row.get("ok"):
                    item["result"] = "text extracted"
                    item["result_class"] = "ok"
                    succeeded += 1
                else:
                    item["result"] = "failed"
                    item["result_class"] = "danger"
                    failed += 1
            files.append(item)

        total = len(self.ocr_files.paths)
        processed = succeeded + failed
        engine_ok, engine_detail = self.stages.ocr.available()
        return {
            "has_files": bool(files),
            "files": files,
            "engine": self.stages.ocr.engine,
            "engine_detail": engine_detail,
            "engine_ok": engine_ok,
            "engine_bad": not engine_ok,
            "languages": ", ".join(self.stages.ocr.languages),
            "running": running,
            "can_run": bool(files) and not running and engine_ok,
            "can_queue": succeeded > 0 and not running,
            "detail": self._ocr_detail,
            "total": total,
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "pending": max(0, total - processed),
            "percent": int(processed / total * 100) if total else 0,
            "has_output": (paths.OCR_TEXT_DIR.is_dir()
                           and any(paths.OCR_TEXT_DIR.glob("*.txt"))),
            "output_dir": str(paths.OCR_TEXT_DIR),
        }

    def _watermark_page(self, _params: dict[str, Any]) -> dict[str, Any]:
        from core.watermark import Fidelity

        files: list[dict[str, Any]] = []
        removable = 0
        for path in self.watermark_files.paths:
            key = str(path)
            scan = self._watermark_scans.get(key)
            removal = self._watermark_removals.get(key)

            row: dict[str, Any] = {"name": path.name, "pages": "", "detected": "",
                                   "detect_class": "", "method": "",
                                   "done": False, "result": "", "result_class": "",
                                   # Where this file's clean copy went, and why
                                   # it did not go anywhere if it failed.
                                   "output": "", "reason": ""}

            if scan is not None:
                row["pages"] = scan.page_count
                # `confirmed` rather than `findings`: a scanned page has no text
                # layer at all, which an earlier version reported as a detected
                # watermark on every scan.
                confirmed = scan.confirmed
                if scan.error:
                    row["detected"] = "unreadable"
                    row["detect_class"] = "danger"
                    row["method"] = scan.error[:80]
                elif confirmed:
                    row["detected"] = f"{len(confirmed)} found"
                    row["detect_class"] = "review"
                    row["method"] = ", ".join(
                        sorted({f.kind.value for f in confirmed}))
                    removable += 1
                else:
                    row["detected"] = "none"
                    row["detect_class"] = "ok"
                    row["method"] = "nothing to remove"

            if removal is not None:
                row["done"] = True
                out = self._watermark_outputs.get(key)
                if out is not None:
                    row["output"] = str(out)
                if removal.error:
                    row["result"] = "failed"
                    row["result_class"] = "danger"
                    # The existing reason display, unchanged in kind: whatever
                    # the remover said, in full, rather than a bare "failed".
                    row["reason"] = removal.error
                elif removal.fidelity is Fidelity.LOSSLESS and removal.removed:
                    row["result"] = "lossless"
                    row["result_class"] = "ok"
                elif removal.skipped and not removal.removed:
                    row["result"] = "skipped"
                    row["result_class"] = "review"
                else:
                    row["result"] = removal.fidelity.value
                    row["result_class"] = "review"
            files.append(row)

        done = len(self._watermark_removals)
        total = len(self.watermark_files.paths)
        cleaned = sum(1 for r in self._watermark_removals.values() if r.ok)
        failed = done - cleaned

        # Where the results will go, shown *before* the run as well as after -
        # an operator should not have to clean a folder of deeds to find out
        # where the copies landed. One folder per source folder, because a
        # selection can span several.
        destinations = sorted({
            str(self._destination_for(path)) for path in self.watermark_files.paths})
        return {
            "has_files": bool(files),
            "can_remove": removable > 0,
            "output_dir": destinations[0] if destinations else "",
            "output_dirs": [{"path": d} for d in destinations],
            "many_outputs": len(destinations) > 1,
            "subfolder": self.CLEANED_SUBFOLDER,
            "has_output": any(Path(d).is_dir() for d in destinations) or (
                WATERMARK_DIR.is_dir() and any(WATERMARK_DIR.glob("*.pdf"))),
            # Scanning and removal are synchronous, so a render never catches
            # them mid-flight. The block stays because the template models it.
            "running": False,
            "done": done, "total": total,
            "cleaned": cleaned, "failed": failed,
            "has_run": done > 0,
            "percent": int(done / total * 100) if total else 0,
            "files": files,
        }

    def _destination_for(self, source: Path) -> Path:
        """The folder a file's clean copy would go to. Creates nothing.

        Separate from `_cleaned_target` on purpose: rendering a page must not
        have the side effect of making directories all over an operator's disk
        for files they have merely selected.
        """
        parent = source.parent
        return parent if parent.name == self.CLEANED_SUBFOLDER             else parent / self.CLEANED_SUBFOLDER

    def _help(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "flags": [
                {"code": "OCR_P", "class": "ok", "meaning": "OCR succeeded",
                 "action": "Nothing to do."},
                {"code": "OCR_F", "class": "danger", "meaning": "OCR failed",
                 "action": "Usually a scanned PDF with no text layer. Needs a real OCR engine."},
                {"code": "PM", "class": "ok", "meaning": "PAN matched",
                 "action": "The PAN was found in the source document."},
                {"code": "WAN", "class": "review", "meaning": "Wrong Aadhaar number",
                 "action": "The Aadhaar could not be located in the OCR. Verify before filing."},
                {"code": "WSC", "class": "review", "meaning": "Wrong sale consideration",
                 "action": "The amount is not present in the source. Check the deed."},
                {"code": "WSV", "class": "review", "meaning": "Wrong stamp value",
                 "action": "The registration fee disagrees with the source."},
                {"code": "SSV", "class": "review", "meaning": "Stamp value exceeds consideration",
                 "action": "Inactive until the stamp-value formula is confirmed."},
                {"code": "WTD", "class": "review", "meaning": "Transaction date missing",
                 "action": "No usable date was found. Enter it manually."},
                {"code": "HPAN", "class": "danger", "meaning": "Discarded PAN",
                 "action": "A PAN-shaped value was not in the source, or was malformed, so it was dropped."},
                {"code": "XPAN", "class": "", "meaning": "Extra PANs in OCR",
                 "action": "The document contains PANs not attributed to a party - often witnesses."},
                {"code": "PAF", "class": "review", "meaning": "PAN and Aadhaar far apart",
                 "action": "The pair may belong to different people. Verify before filing."},
                {"code": "TRC", "class": "danger", "meaning": "Output truncated",
                 "action": "Generation hit the token ceiling, usually a repetition loop. Re-run."},
            ],
            "flow": [
                {"stage": "OCR", "what": "Each PDF is rendered and read, then cleaned "
                 "(line endings, spacing, page markers).",
                 "on_failure": "Retried once, then marked OCR_F."},
                {"stage": "Extraction", "what": "Cleaned text goes to the model, which "
                 "returns structured JSON.",
                 "on_failure": "Routed to review; the raw output is retained for audit."},
                {"stage": "Validation", "what": "Every value is checked against the OCR "
                 "source and flagged if absent.",
                 "on_failure": "The document is marked Needs Review, never silently exported."},
                {"stage": "Translation", "what": "Names are transliterated; addresses are "
                 "translated.",
                 "on_failure": "Skipped - the extracted data is still usable, just untranslated."},
            ],
            "states": [
                {"name": "Processing", "class": "accent",
                 "meaning": "Still moving through the pipeline."},
                {"name": "Processed", "class": "ok",
                 "meaning": "Completed and every enabled check passed."},
                {"name": "Needs Review", "class": "review",
                 "meaning": "Extracted, but at least one value could not be confirmed "
                            "against the source. A human should look."},
                {"name": "Failed", "class": "danger",
                 "meaning": "A stage could not complete. See the failed CSV for the reason."},
            ],
            "faqs": [
                {"q": "Why is a document marked Needs Review when the data looks right?",
                 "a": "A value was not found in the OCR text. That usually means a digit "
                      "was misread. The extracted data is kept so you can compare it "
                      "against the deed."},
                {"q": "Can I close the application mid-batch?",
                 "a": "Yes. Progress is committed after every document, so restarting "
                      "resumes exactly where it stopped and never reprocesses finished work."},
                {"q": "Why does processing slow down or pause by itself?",
                 "a": "The application watches memory and reduces concurrency when the "
                      "machine is under pressure, rather than risking a crash mid-batch. "
                      "Closing other applications speeds it up."},
                {"q": "Why are some Aadhaar numbers blank?",
                 "a": "Masked Aadhaars (XXXX XXXX 1234) are deliberately not extracted - "
                      "a partial number is worse than none."},
            ],
        }

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _failed_stage(doc: Any) -> str:
        for stage in ("ocr", "extract", "translate", "validate"):
            if getattr(doc, f"{stage}_state").value == "failed":
                return stage
        return "unknown"

    @staticmethod
    def _document_export(doc: Any) -> DocumentExport:
        prop = doc.property_

        def person(p: Any) -> dict[str, Any]:
            return {"name": p.name, "name_translated": p.name_translated,
                    "gender": p.gender, "father_name": p.father_name,
                    "father_name_translated": p.father_name_translated,
                    "aadhaar_number": p.aadhaar_number,
                    "pan_card_number": p.pan_card_number,
                    "address": p.address, "address_translated": p.address_translated,
                    "state": p.state}

        # The deed states the property's kind and its municipal status in
        # prose, and the extraction schema has no field for either. Reading the
        # stored OCR here recovers both without touching the model. Relationship
        # access, so it costs nothing extra when the pages are already loaded.
        try:
            source_text = "\n".join(
                page.text or "" for page in sorted(
                    doc.ocr_pages, key=lambda pg: pg.page_number))
        except Exception:  # noqa: BLE001 - classification is a bonus, not a gate
            source_text = ""

        return DocumentExport(
            # The extracted registration number, and nothing else. Blank when
            # the deed did not yield one - `transaction_id.extract` returns
            # empty on purpose rather than risk writing a previous owner's
            # number, and this column must honour that (R-043).
            transaction_identity=doc.transaction_identity or "",
            source_filename=doc.source_filename,
            source_text=source_text,
            stamp_value=str(int(prop.stamp_value)) if prop and prop.stamp_value else None,
            extraction={
                "buyer_details": [person(p) for p in doc.persons
                                  if p.relation.value == "B"],
                "seller_details": [person(p) for p in doc.persons
                                   if p.relation.value == "S"],
                "property_details": {
                    "schedule_c_property_address": prop.schedule_c_address if prop else None,
                    "state": prop.state if prop else None,
                    "sale_consideration": str(int(prop.sale_consideration))
                    if prop and prop.sale_consideration else None,
                    "registration_fee": str(int(prop.registration_fee))
                    if prop and prop.registration_fee else None,
                    "paid_in_cash": ("yes" if prop.paid_in_cash else "no")
                    if prop and prop.paid_in_cash is not None else None,
                },
                "document_details": {
                    "transaction_date": prop.transaction_date.isoformat()
                    if prop and prop.transaction_date else None,
                    "registration_office": prop.registration_office if prop else None,
                },
            })
