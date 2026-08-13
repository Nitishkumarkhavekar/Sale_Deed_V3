"""QWebChannel bridge - threading and error propagation.

The bridge is the only path between the webview and the application, and its
hardest constraint is not logic but *threading*: work runs on a pool so the UI
never blocks, while the JavaScript callback belongs to the GUI thread.

Calling a `QJSValue` callback from a worker thread does not raise. It silently
never reaches JavaScript, the promise settles with an empty response, and the UI
shows a bare "failed" with no explanation - which is exactly what a user
reported. These tests run a real `QApplication` so the thread hop is exercised
rather than assumed.

Headless: `QT_QPA_PLATFORM=offscreen` is set before Qt is imported, so no display
is needed.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit

QtCore = pytest.importorskip("PySide6.QtCore", reason="PySide6 not installed")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _StubService:
    """Just enough AppService for the bridge."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.calls: list[str] = []

    def render_page(self, page: str, params: dict, *, shell_html: bool = True) -> str:
        # Signature mirrors AppService.render_page deliberately - a double that
        # drifts from the real thing turns a green suite into no evidence at
        # all. `test_the_double_matches_the_real_service` enforces it.
        self.calls.append(page)
        body = f"<div>{page}</div>"
        return f"<html><body>{body}</body></html>" if shell_html else body

    def render_fragment(self, template: str, params: dict) -> str:
        return f"<div>{template}</div>"

    def status(self) -> dict:
        return {"ok": True, "state": "idle"}

    def log_exception(self, where: str, exc: Exception, trace: str) -> None:
        self.errors.append(f"{where}: {exc}")

    def boom(self, *_a, **_k):
        raise RuntimeError("deliberate failure")


@pytest.fixture()
def bridge(qapp):
    from app.ui.bridge import Bridge

    b = Bridge(_StubService())
    yield b
    b.shutdown()


def _await(bridge, method: str, payload: dict, timeout: float = 10.0,
           request_id: str = "1") -> dict:
    """Invoke a slot the way QWebChannel does and pump until `completed` fires.

    Two things are deliberate. The slot is called with `(request_id, payload)` -
    exactly the argument list QWebChannel produces after it strips the trailing
    reply function - and the result is collected from the `completed` signal
    rather than from a callback parameter.

    The pump matters: a result delivered by a queued signal only arrives when
    the GUI thread processes events, so a test that merely slept would see
    nothing and wrongly conclude the bridge was broken.
    """
    received: list[tuple[str, str]] = []
    bridge.completed.connect(lambda rid, raw: received.append((rid, raw)))

    body = payload if isinstance(payload, str) else json.dumps(payload)
    getattr(bridge, method)(request_id, body)

    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while not received and deadline.elapsed() < timeout * 1000:
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

    assert received, (
        f"{method} never emitted `completed` - the result did not reach the "
        "GUI thread")
    assert received[0][0] == request_id, "reply carried the wrong request id"
    return json.loads(received[0][1])


class TestCallbackDelivery:
    """Regression for a user-visible defect: "Could not open dashboard: failed".

    `render` runs on the pool. Its callback was invoked from the worker thread,
    which Qt cannot marshal into JavaScript, so nothing came back.
    """

    def test_the_double_matches_the_real_service(self):
        """A stub that accepts arguments the real object rejects proves nothing.

        This suite has twice been green while the application was broken because
        a double was more permissive than what it stood in for.
        """
        import inspect

        from app.services import AppService

        for name in ("render_page", "render_fragment", "status"):
            real = inspect.signature(getattr(AppService, name))
            stub = inspect.signature(getattr(_StubService, name))
            assert set(real.parameters) == set(stub.parameters), (
                f"_StubService.{name}{stub} does not match "
                f"AppService.{name}{real}")

    def test_render_reaches_its_callback(self, bridge):
        result = _await(bridge, "render", {"page": "dashboard"})
        assert result["ok"] is True
        assert "dashboard" in result["html"]

    def test_fragment_reaches_its_callback(self, bridge):
        result = _await(bridge, "fragment", {"template": "batch_detail"})
        assert result["ok"] is True

    def test_inline_method_still_answers(self, bridge):
        """`status` runs on the GUI thread and must not regress."""
        result = _await(bridge, "status", {})
        assert result["ok"] is True

    def test_every_page_renders_through_the_bridge(self, bridge):
        for page in ("dashboard", "upload", "processing", "data", "watermark",
                     "settings", "validation", "help"):
            result = _await(bridge, "render", {"page": page})
            assert result["ok"] is True, f"{page} failed through the bridge"

    def test_results_are_delivered_on_the_gui_thread(self, bridge, qapp):
        """Where the callback runs is the whole defect, so assert on it."""
        seen: list[int] = []
        done = threading.Event()

        def callback(_raw: str) -> None:
            seen.append(threading.get_ident())
            done.set()

        gui_thread = threading.get_ident()
        bridge.completed.connect(lambda _rid, raw: callback(raw))
        bridge.render("t1", json.dumps({"page": "dashboard"}))

        timer = QtCore.QElapsedTimer()
        timer.start()
        while not done.is_set() and timer.elapsed() < 10_000:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

        assert seen, "callback never fired"
        assert seen[0] == gui_thread, \
            "callback ran on a worker thread - JavaScript would never see it"

    def test_work_does_not_run_on_the_gui_thread(self, bridge):
        """The other half of the contract: the UI must not block.

        Delivery is marshalled to the GUI thread, but the *work* has to stay off
        it or the window freezes during a long export.
        """
        worker_threads: list[int] = []
        gui_thread = threading.get_ident()

        service = bridge.service
        original = service.render_page

        def spy(page: str, params: dict, **kwargs) -> str:
            worker_threads.append(threading.get_ident())
            return original(page, params, **kwargs)

        service.render_page = spy
        try:
            _await(bridge, "render", {"page": "dashboard"})
        finally:
            service.render_page = original

        assert worker_threads and worker_threads[0] != gui_thread, \
            "render blocked the GUI thread"


class TestErrorPropagation:
    def test_failure_carries_a_readable_message(self, bridge):
        """A bare "failed" tells the user nothing. The type and text must
        survive the trip."""
        bridge.service.render_page = bridge.service.boom
        result = _await(bridge, "render", {"page": "dashboard"})
        assert result["ok"] is False
        assert "RuntimeError" in result["error"]
        assert "deliberate failure" in result["error"]

    def test_failures_are_logged_for_the_dashboard(self, bridge):
        bridge.service.render_page = bridge.service.boom
        _await(bridge, "render", {"page": "dashboard"})
        assert bridge.service.errors

    def test_malformed_payload_is_reported_not_dropped(self, bridge):
        """Answered on the next event-loop turn, not during the call.

        Replying synchronously would re-enter QWebChannel while it is still
        dispatching this very `invokeMethod`, which corrupts its reply
        bookkeeping - JavaScript then reports
        `channel.execCallbacks[message.id] is not a function`.
        """
        result = _await(bridge, "render", "{not json", request_id="t2")
        assert result["ok"] is False
        assert "bad payload" in result["error"]

    def test_a_reply_with_no_listener_is_harmless(self, bridge):
        """The page can navigate away before a slow result arrives. Emitting to
        nobody must be a no-op, not an error."""
        bridge.completed.emit("gone", "{}")  # must not raise


class TestQWebChannelCallingConvention:
    """The defect these guard against was invisible to every other test.

    QWebChannel dispatches `bridge.render(payload, fn)` by treating the trailing
    function as the reply handler, removing it, and invoking the slot with what
    remains - one argument. A slot declaring `(str, QJSValue)` never matches, and
    Qt logs `No candidates found for "render" with 1 arguments` before dropping
    the call. Nothing raises on either side.

    Python-callable tests could not see it: a plain function is callable from any
    thread and accepts any signature, so the stub happily played a role the real
    `QJSValue` never could.
    """

    #: Every slot JavaScript invokes, and the argument count it will arrive with.
    INVOKED_FROM_JS = (
        "render", "fragment", "status", "control", "pick_files", "add_files",
        "clear_selection", "add_batch", "export", "reprocess", "save_settings",
        "save_rules", "watermark", "delete_batch",
    )

    def test_slots_accept_the_arguments_qwebchannel_sends(self, bridge):
        """Two strings: the request id and the payload. No callback parameter."""
        meta = bridge.metaObject()
        missing: list[str] = []
        for name in self.INVOKED_FROM_JS:
            index = meta.indexOfSlot(f"{name}(QString,QString)")
            if index < 0:
                missing.append(name)
        assert not missing, (
            f"slots not invokable as (QString,QString): {missing} - "
            "QWebChannel would report 'No candidates found'")

    def test_no_slot_still_declares_a_qjsvalue_callback(self):
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "src" / "app" / "ui" / "bridge.py").read_text(encoding="utf-8")
        # The prose explains why; what must not survive is the declaration.
        assert '@Slot(str, "QJSValue")' not in source, (
            "a QJSValue callback parameter is never populated by QWebChannel")
        assert "callback: Callable[[str], None]) -> None" not in source, (
            "a slot still takes a callback QWebChannel will never supply")

    def test_results_are_published_on_a_signal(self, bridge):
        """The reply route JavaScript subscribes to."""
        meta = bridge.metaObject()
        assert meta.indexOfSignal("completed(QString,QString)") >= 0

    def test_javascript_calls_with_an_id_not_a_function(self):
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "src" / "app" / "ui" / "assets" / "app.js").read_text(encoding="utf-8")
        assert "api[method](id, JSON.stringify" in source, \
            "app.js still passes a trailing callback function"
        assert "api.completed.connect" in source, \
            "app.js does not subscribe to the reply signal"

    def test_every_slot_javascript_calls_actually_exists(self):
        """A `call("watermark", ...)` naming a slot that is not there fails only
        at runtime, on the page nobody opened during testing."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1]
        js = (root / "src" / "app" / "ui" / "assets" / "app.js").read_text(encoding="utf-8")
        source = (root / "src" / "app" / "ui" / "bridge.py").read_text(encoding="utf-8")

        called = set(re.findall(r'call\("(\w+)"', js))
        declared = set(re.findall(r"def (\w+)\(self, request_id: str", source))
        missing = called - declared
        assert not missing, f"app.js calls slots that do not exist: {missing}"

    def test_watermark_buttons_reach_the_watermark_slot(self):
        """Regression: these routed to `pick_files` and `clear_selection`, which
        drive the *upload* selection - so the page appeared wired and operated on
        the wrong list, and scan/remove/open did nothing at all."""
        import pathlib

        js = (pathlib.Path(__file__).resolve().parents[1] / "src" / "app" / "ui" /
              "assets" / "app.js").read_text(encoding="utf-8")
        assert 'call("watermark"' in js
        for button in ("btn-wm-browse", "btn-wm-scan", "btn-wm-remove",
                       "btn-wm-open", "btn-wm-clear"):
            assert button in js, f"{button} has no handler"

    def test_request_ids_are_matched_to_their_replies(self, bridge):
        """Two calls in flight must not resolve each other's promise."""
        seen: list[tuple[str, str]] = []
        bridge.completed.connect(lambda rid, raw: seen.append((rid, raw)))

        bridge.render("alpha", json.dumps({"page": "upload"}))
        bridge.render("beta", json.dumps({"page": "settings"}))

        timer = QtCore.QElapsedTimer()
        timer.start()
        while len(seen) < 2 and timer.elapsed() < 10_000:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

        assert len(seen) == 2
        replies = dict(seen)
        assert "upload" in replies["alpha"]
        assert "settings" in replies["beta"]


class TestGuiThreadAffinity:
    """Calls that touch a Qt widget must not be dispatched to the pool.

    `pick_files` opens a `QFileDialog`. Qt widgets may only be created on the
    thread owning the QApplication; doing it on a worker terminates the process
    natively - no exception, no traceback, nothing in any log. The reported
    symptom was the window disappearing the moment "Upload PDF" was clicked.

    These tests assert the *routing decision* rather than trying to provoke the
    crash, because a genuine reproduction would take the test runner down with
    it.
    """

    def test_pick_files_runs_on_the_gui_thread(self, bridge):
        from app.ui.bridge import Bridge

        assert Bridge._needs_gui_thread("pick_files", {}) is True

    def test_watermark_browse_runs_on_the_gui_thread(self, bridge):
        """It opens the same dialog through a different slot."""
        from app.ui.bridge import Bridge

        assert Bridge._needs_gui_thread("watermark", {"action": "browse"}) is True

    def test_watermark_scan_still_runs_on_the_pool(self, bridge):
        """Scanning is slow and touches no widget - it must not block the UI."""
        from app.ui.bridge import Bridge

        assert Bridge._needs_gui_thread("watermark", {"action": "scan"}) is False
        assert Bridge._needs_gui_thread("watermark", {"action": "remove"}) is False

    def test_ordinary_work_is_not_forced_onto_the_gui_thread(self, bridge):
        from app.ui.bridge import Bridge

        for name in ("render", "export", "add_batch", "reprocess"):
            assert Bridge._needs_gui_thread(name, {}) is False, \
                f"{name} would block the window"

    def test_the_picker_actually_executes_on_the_gui_thread(self, bridge):
        """End to end through `_run`: record where the handler body runs."""
        gui_thread = threading.get_ident()
        ran_on: list[int] = []

        bridge.service.pick_files = lambda: ran_on.append(threading.get_ident()) or {}
        _await(bridge, "pick_files", {})

        assert ran_on and ran_on[0] == gui_thread, (
            "the file dialog would be constructed on a worker thread and the "
            "process would terminate")

    def test_the_picker_guards_itself(self):
        """Defence in depth: even if the routing were wrong, the dialog should
        raise a readable error rather than kill the process."""
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "src" / "app" / "main.py").read_text(encoding="utf-8")
        assert "QThread.currentThread() is not app.thread()" in source
        assert "terminate the application" in source


class TestNothingCanKillTheWindow:
    """One action must never be able to take the application down."""

    def test_a_handler_raising_baseexception_is_contained(self, bridge):
        """`Exception` alone would let SystemExit escape a worker and leave the
        promise unresolved, with the UI waiting forever."""
        def suicidal(*_a, **_k):
            raise SystemExit("handler tried to exit")

        bridge.service.render_page = suicidal
        result = _await(bridge, "render", {"page": "dashboard"})
        assert result["ok"] is False
        assert "SystemExit" in result["error"]

    def test_a_keyboard_interrupt_in_a_worker_is_contained(self, bridge):
        def interrupted(*_a, **_k):
            raise KeyboardInterrupt()

        bridge.service.render_page = interrupted
        result = _await(bridge, "render", {"page": "dashboard"})
        assert result["ok"] is False

    def test_every_failure_is_logged_with_a_traceback(self, bridge):
        bridge.service.render_page = bridge.service.boom
        _await(bridge, "render", {"page": "dashboard"})
        assert bridge.service.errors, "the failure was not recorded anywhere"


class TestNoReentrancyIntoQWebChannel:
    """Nothing may reply while QWebChannel is still dispatching the call.

    Reported by the user, repeating at the status-poll interval:

        Uncaught TypeError: channel.execCallbacks[message.id] is not a function

    That is QWebChannel's own reply dispatcher. Emitting `completed` from inside
    the slot re-enters the channel mid-message and corrupts its bookkeeping, so
    the reply is lost - which then surfaced as `pick_files timed out after 120s`,
    because `QFileDialog` spins a nested event loop and made the window for
    re-entry enormous.
    """

    def test_inline_work_is_deferred_past_the_call(self, bridge):
        """`status` is inline. It must not reply during the invocation."""
        replies: list[str] = []
        bridge.completed.connect(lambda _rid, raw: replies.append(raw))

        bridge.status("s1", "{}")
        assert not replies, (
            "replied during the slot call - QWebChannel is still dispatching "
            "and would be re-entered")

        timer = QtCore.QElapsedTimer()
        timer.start()
        while not replies and timer.elapsed() < 5000:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        assert replies, "the deferred reply never arrived"

    def test_gui_thread_work_is_deferred_past_the_call(self, bridge):
        replies: list[str] = []
        bridge.completed.connect(lambda _rid, raw: replies.append(raw))
        bridge.service.pick_files = lambda: {"count": 0}

        bridge.pick_files("p1", "{}")
        assert not replies, "the file dialog would open inside the dispatch"

        timer = QtCore.QElapsedTimer()
        timer.start()
        while not replies and timer.elapsed() < 5000:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        assert replies

    def test_a_bad_payload_is_also_deferred(self, bridge):
        replies: list[str] = []
        bridge.completed.connect(lambda _rid, raw: replies.append(raw))
        bridge.render("b1", "{not json")
        assert not replies, "the error reply re-entered the channel"

    def test_deferred_work_still_lands_on_the_gui_thread(self, bridge):
        """Deferring must not move it off the GUI thread - that was the fix for
        the crash, and it has to survive this one."""
        ran_on: list[int] = []
        gui_thread = threading.get_ident()
        bridge.service.pick_files = lambda: ran_on.append(threading.get_ident()) or {}

        _await(bridge, "pick_files", {})
        assert ran_on and ran_on[0] == gui_thread

    def test_pool_work_is_unaffected(self, bridge):
        """Only GUI-thread and inline paths changed; the pool still answers."""
        assert _await(bridge, "render", {"page": "dashboard"})["ok"] is True
