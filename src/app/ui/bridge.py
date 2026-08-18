"""QWebChannel bridge - the only path between the webview and the application.

Every method takes a JSON string and returns a JSON string through a callback.
That uniformity is deliberate: QWebChannel marshals a small set of types well and
everything else badly, so one string in / one string out avoids a class of silent
conversion bugs.

Two rules this file exists to enforce:

**The UI never touches a model.** Inference happens in a separate process reached
over HTTP; this bridge only ever talks to the database and to that HTTP API.

**The UI thread never blocks.** Anything that can take more than a few
milliseconds - a batch scan, an export, a control action - runs on a worker
thread, and the result is delivered to the callback when it is ready. A blocking
call here would freeze the window, which is exactly what the specification
forbids.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..services import AppService

#: Methods that only read cheap in-memory state and can answer inline. Anything
#: touching the database, the filesystem or HTTP goes to the pool.
_INLINE = {"status"}

#: Methods that **must** run on the GUI thread, whatever they cost.
#:
#: `pick_files` opens a native `QFileDialog`. Qt widgets may only be created and
#: shown on the thread that owns the QApplication; constructing one on a pool
#: worker is undefined behaviour and on Windows terminates the process
#: immediately - no Python exception, no traceback, nothing in the log, because
#: the abort happens below the interpreter. The symptom is the window vanishing
#: the instant "Upload PDF" is clicked.
#:
#: These are scheduled onto the next turn of the event loop rather than run
#: during the call, because `QFileDialog` spins a nested event loop and doing
#: that inside an unfinished `invokeMethod` corrupts QWebChannel's reply
#: bookkeeping. See the dispatch in `_run`.
_GUI_THREAD = {"pick_files", "pick_save_path", "open_path"}

#: Actions within a slot that also open a dialog, and so share that constraint.
#:
#: Every slot that can reach `AppService.file_picker` belongs here, keyed by the
#: action that does it. Omitting one is not a small mistake: the service raises
#: rather than opening the dialog, so the operator sees a RuntimeError banner and
#: the button simply does not work. `test_app_wiring` derives the required set
#: from the service's own source, because this list is exactly the kind that gets
#: forgotten when a page is added - as it was for `ocr_tool`.
_GUI_THREAD_ACTIONS = {"watermark": {"browse"}, "ocr_tool": {"browse"}}


class Bridge(QObject):
    """Exposed to JavaScript as `bridge`."""

    #: Emitted when Python needs to push a change the UI did not ask for.
    notify = Signal(str)

    #: Carries a finished result back to JavaScript: (request_id, json).
    #:
    #: Results come back on a signal rather than through a callback parameter
    #: because of how QWebChannel actually dispatches. When JavaScript calls
    #: `bridge.render(payload, fn)`, the channel treats a trailing function as
    #: the *reply handler*, removes it from the argument list, and invokes the
    #: slot with what remains. A slot declared `(str, QJSValue)` therefore never
    #: matches - Qt reports `No candidates found for "render" with 1 arguments`
    #: and the call is dropped.
    #:
    #: Taking the reply value instead would mean the slot has to *return* it,
    #: which forces the work onto the GUI thread and freezes the window during an
    #: export. A request id plus this signal keeps the work on the pool and still
    #: delivers on the GUI thread, because `Bridge` lives there and a signal
    #: emitted from any thread arrives through a queued connection.
    completed = Signal(str, str)

    def __init__(self, service: AppService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.service = service
        # Small pool: the work here is IO-bound and mostly serialised by the
        # database anyway. A large pool would just queue inside PostgreSQL.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bridge")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- plumbing ---------------------------------------------------------

    def _run(self, name: str, handler: Callable[[dict[str, Any]], dict[str, Any]],
             payload: str, request_id: str) -> None:
        try:
            params = json.loads(payload or "{}")
            if not isinstance(params, dict):
                params = {}
        except json.JSONDecodeError as exc:
            # Deferred for the same reason as the dispatch below: emitting here
            # would re-enter QWebChannel mid-message.
            message = json.dumps({"ok": False, "error": f"bad payload: {exc}"})
            QTimer.singleShot(0, lambda: self.completed.emit(request_id, message))
            return

        def work() -> None:
            try:
                result = handler(params)
                result.setdefault("ok", True)
                self.completed.emit(request_id, json.dumps(result, default=_encode))
            except BaseException as exc:  # noqa: BLE001
                # BaseException, not Exception: a worker that dies on
                # KeyboardInterrupt or SystemExit would otherwise leave the
                # promise unresolved and the UI waiting forever with no clue.
                # The window must survive anything one action can do to it.
                self.service.log_exception(name, exc, traceback.format_exc())
                self.completed.emit(request_id, json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}))

        if self._needs_gui_thread(name, params) or name in _INLINE:
            # Deferred to the next turn of the event loop rather than run here.
            #
            # Running it inline emits `completed` *during* QWebChannel's own
            # dispatch of this `invokeMethod`. The channel is then re-entered
            # while it is still mid-message, and its reply bookkeeping is
            # corrupted - JavaScript reports
            # `channel.execCallbacks[message.id] is not a function`, once per
            # affected call, and the reply never arrives.
            #
            # `pick_files` made it worse still: `QFileDialog` spins a nested
            # event loop, so the channel processed queued messages *inside* an
            # unfinished slot invocation. The visible symptom was
            # "pick_files timed out after 120s".
            #
            # A zero-delay timer keeps the work on the GUI thread - which is the
            # requirement for touching a widget - while letting the current
            # message finish first.
            QTimer.singleShot(0, work)
        else:
            self._pool.submit(work)

    @staticmethod
    def _needs_gui_thread(name: str, params: dict[str, Any]) -> bool:
        """True when this call would touch a Qt widget.

        Kept as an explicit list rather than inferred: getting it wrong does not
        raise, it kills the process, so the decision belongs somewhere a reader
        can check against `app/main.py`.
        """
        if name in _GUI_THREAD:
            return True
        actions = _GUI_THREAD_ACTIONS.get(name)
        return bool(actions and str(params.get("action") or "") in actions)

    # -- rendering --------------------------------------------------------

    @Slot(str, str)
    def render(self, request_id: str, payload: str) -> None:
        """Render a full page. Returns {'html': ...}."""
        # `shell` defaults false: navigation replaces the content area only,
        # so the QWebChannel that carries this very reply stays alive. The full
        # document is produced once, by MainWindow at startup.
        self._run("render", lambda p: {"html": self.service.render_page(
            p.get("page") or "dashboard", p.get("params") or {},
            shell_html=bool(p.get("shell")))}, payload, request_id)

    @Slot(str, str)
    def fragment(self, request_id: str, payload: str) -> None:
        """Render a modal or partial. Returns {'html': ...}."""
        self._run("fragment", lambda p: {"html": self.service.render_fragment(
            p.get("template") or "", p.get("params") or {})}, payload, request_id)

    # -- live state -------------------------------------------------------

    @Slot(str, str)
    def status(self, request_id: str, payload: str) -> None:
        """Poll target. Cheap by design - called every couple of seconds."""
        self._run("status", lambda _p: self.service.status(), payload, request_id)

    # -- processing control ------------------------------------------------

    @Slot(str, str)
    def control(self, request_id: str, payload: str) -> None:
        """start | pause | stop."""
        self._run("control", lambda p: self.service.control(p.get("action") or ""),
                  payload, request_id)

    # -- upload ------------------------------------------------------------

    @Slot(str, str)
    def pick_files(self, request_id: str, payload: str) -> None:
        """Open a native file dialog and stage the selection."""
        self._run("pick_files", lambda _p: self.service.pick_files(), payload, request_id)

    @Slot(str, str)
    def add_files(self, request_id: str, payload: str) -> None:
        """Stage paths dropped onto the window."""
        self._run("add_files",
                  lambda p: self.service.add_files(p.get("paths") or []),
                  payload, request_id)

    @Slot(str, str)
    def clear_selection(self, request_id: str, payload: str) -> None:
        self._run("clear_selection", lambda _p: self.service.clear_selection(),
                  payload, request_id)

    @Slot(str, str)
    def add_batch(self, request_id: str, payload: str) -> None:
        """Queue the staged files as a batch."""
        self._run("add_batch",
                  lambda p: self.service.add_batch(p.get("username") or "",
                                                   p.get("name") or ""),
                  payload, request_id)

    # -- batches -----------------------------------------------------------

    @Slot(str, str)
    def export(self, request_id: str, payload: str) -> None:
        self._run("export",
                  lambda p: self.service.export_batch(int(p.get("batch_id") or 0),
                                                      bool(p.get("failed_only"))),
                  payload, request_id)

    @Slot(str, str)
    def reprocess(self, request_id: str, payload: str) -> None:
        self._run("reprocess",
                  lambda p: self.service.reprocess_failed(int(p.get("batch_id") or 0)),
                  payload, request_id)

    # -- failed OCR --------------------------------------------------------

    @Slot(str, str)
    def failed_ocr(self, request_id: str, payload: str) -> None:
        """The list of documents whose OCR stage failed."""
        self._run("failed_ocr",
                  lambda p: self.service.failed_ocr(
                      int(p["batch_id"]) if p.get("batch_id") else None,
                      int(p.get("page") or 1)),
                  payload, request_id)

    @Slot(str, str)
    def rerun_ocr(self, request_id: str, payload: str) -> None:
        """Requeue one document, a selection, or every failed document.

        `document_pk` carries the single-file case and `document_pks` the
        multi-select one; both funnel into the same service call so there is one
        code path to reason about.
        """
        def call(p: dict[str, Any]) -> dict[str, Any]:
            pks = [int(x) for x in (p.get("document_pks") or [])]
            if p.get("document_pk"):
                pks.append(int(p["document_pk"]))
            return self.service.rerun_ocr(
                pks,
                int(p["batch_id"]) if p.get("batch_id") else None,
                all_failed=bool(p.get("all")),
                force=bool(p.get("force")))

        self._run("rerun_ocr", call, payload, request_id)

    @Slot(str, str)
    def export_corrupted(self, request_id: str, payload: str) -> None:
        self._run("export_corrupted",
                  lambda p: self.service.export_corrupted(
                      int(p["batch_id"]) if p.get("batch_id") else None,
                      p.get("destination")),
                  payload, request_id)

    @Slot(str, str)
    def revalidate(self, request_id: str, payload: str) -> None:
        """Re-check the files of documents the validator condemned."""
        def call(p: dict[str, Any]) -> dict[str, Any]:
            pks = [int(x) for x in (p.get("document_pks") or [])]
            if p.get("document_pk"):
                pks.append(int(p["document_pk"]))
            return self.service.revalidate(
                pks, int(p["batch_id"]) if p.get("batch_id") else None)

        self._run("revalidate", call, payload, request_id)

    @Slot(str, str)
    def corrupted_pdfs(self, request_id: str, payload: str) -> None:
        self._run("corrupted_pdfs",
                  lambda p: self.service.corrupted_pdfs(
                      int(p["batch_id"]) if p.get("batch_id") else None,
                      int(p.get("page") or 1)),
                  payload, request_id)

    @Slot(str, str)
    def delete_batch(self, request_id: str, payload: str) -> None:
        self._run("delete_batch",
                  lambda p: self.service.delete_batch(
                      int(p.get("batch_id") or 0), force=bool(p.get("force"))),
                  payload, request_id)

    @Slot(str, str)
    def ocr_tool(self, request_id: str, payload: str) -> None:
        """The OCR tool page, shaped like the `watermark` slot beside it.

        `run` returns as soon as the worker thread is started rather than when
        OCR finishes - a real deed takes minutes and the front end gives up on a
        call after two. Progress is read back by re-rendering the page.
        """
        self._run("ocr_tool",
                  lambda p: self.service.ocr_tool(str(p.get("action") or "")),
                  payload, request_id)

    @Slot(str, str)
    def retranslate(self, request_id: str, payload: str) -> None:
        """Repair documents whose stored rows were never translated."""
        self._run("retranslate",
                  lambda p: self.service.retranslate(
                      int(p["batch_id"]) if p.get("batch_id") else None),
                  payload, request_id)

    @Slot(str, str)
    def batch_action(self, request_id: str, payload: str) -> None:
        """Run, stop or delete one batch.

        One slot for all three rather than three slots: the guards are a single
        state machine in the service, and splitting the entry points would
        invite a caller that reaches one transition without passing the others.
        """
        self._run("batch_action",
                  lambda p: self.service.batch_action(
                      int(p.get("batch_id") or 0),
                      str(p.get("action") or ""),
                      confirm=bool(p.get("confirm"))),
                  payload, request_id)

    # -- configuration -----------------------------------------------------

    @Slot(str, str)
    def save_settings(self, request_id: str, payload: str) -> None:
        self._run("save_settings", lambda p: self.service.save_settings(p),
                  payload, request_id)

    @Slot(str, str)
    def pick_save_path(self, request_id: str, payload: str) -> None:
        """Native Save As dialog. Routed to the GUI thread by `_GUI_THREAD`."""
        self._run("pick_save_path",
                  lambda p: self.service.pick_save_path(
                      str(p.get("suggested") or "export.csv")),
                  payload, request_id)

    @Slot(str, str)
    def export_view(self, request_id: str, payload: str) -> None:
        """Export the batch the Data View is currently showing."""
        self._run("export_view", lambda p: self.service.export_view(p),
                  payload, request_id)

    @Slot(str, str)
    def save_prompt(self, request_id: str, payload: str) -> None:
        """Persist an edited extraction prompt."""
        self._run("save_prompt",
                  lambda p: self.service.save_prompt(p.get("text") or ""),
                  payload, request_id)

    @Slot(str, str)
    def reset_prompt(self, request_id: str, payload: str) -> None:
        """Restore the extraction prompt from the shipped default."""
        self._run("reset_prompt", lambda _p: self.service.reset_prompt(),
                  payload, request_id)

    @Slot(str, str)
    def document_view(self, request_id: str, payload: str) -> None:
        """Everything the PDF viewer needs about one document."""
        self._run("document_view",
                  lambda p: self.service.document_view(int(p.get("document_pk") or 0)),
                  payload, request_id)

    @Slot(str, str)
    def document_text(self, request_id: str, payload: str) -> None:
        """The full OCR text, for Copy All Text."""
        self._run("document_text",
                  lambda p: self.service.document_text(int(p.get("document_pk") or 0)),
                  payload, request_id)

    @Slot(str, str)
    def open_path(self, request_id: str, payload: str) -> None:
        """Open a folder, file or URL in the desktop's default handler.

        Lives on the bridge rather than in `AppService` because it needs Qt, and
        the service layer is deliberately Qt-free so it stays testable without a
        display. Routed to the GUI thread: `QDesktopServices` is a GUI class.
        """
        self._run("open_path", lambda p: self._open(p.get("path") or ""),
                  payload, request_id)

    @staticmethod
    def _open(target: str) -> dict[str, Any]:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if not target:
            raise ValueError("no path given")
        url = (QUrl(target) if target.startswith(("http://", "https://"))
               else QUrl.fromLocalFile(str(Path(target).resolve())))
        return {"opened": bool(QDesktopServices.openUrl(url)), "path": target}

    @Slot(str, str)
    def check_updates(self, request_id: str, payload: str) -> None:
        self._run("check_updates", lambda _p: self.service.check_updates(),
                  payload, request_id)

    @Slot(str, str)
    def save_rules(self, request_id: str, payload: str) -> None:
        self._run("save_rules",
                  lambda p: self.service.save_rules(p.get("rules") or {},
                                                    p.get("thresholds") or {}),
                  payload, request_id)

    # -- watermark ---------------------------------------------------------

    @Slot(str, str)
    def watermark(self, request_id: str, payload: str) -> None:
        self._run("watermark",
                  lambda p: self.service.watermark(p.get("action") or "scan"),
                  payload, request_id)


def _encode(value: Any) -> Any:
    """JSON fallback for types QWebChannel would otherwise choke on."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone().isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):        # Enum
        return value.value
    return str(value)
