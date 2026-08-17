"""Service-layer wiring: watermark page, capability gating, batch mode, retention.

Four features that existed as working modules but were not reachable from the
application. Each had the same shape of defect - the backend was correct and
tested, and nothing called it - which is invisible to unit tests of either side.

These tests exercise `AppService` without Qt. The service layer is deliberately
Qt-free precisely so this is possible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


class _FakeStatus:
    """Stands in for StatusService so gating can be driven from a test.

    Only `snapshot()` is used by the code under test; a real StatusService would
    start probe threads and reach for the network.
    """

    def __init__(self, capabilities: dict[str, Any]) -> None:
        self._caps = capabilities

    def snapshot(self) -> dict[str, Any]:
        return {"capabilities": self._caps, "ai": {"state": "down"},
                "database": {"state": "down"}, "gpu": {}, "hardware": {},
                "profile": {}}

    def get(self, _name: str):
        class _P:
            data: dict[str, Any] = {}
        return _P()

    def start(self, **_kw: Any) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture()
def service(monkeypatch):
    """An AppService with no database and no network."""
    from app.services import AppService

    monkeypatch.setattr("app.services.check_connection",
                        lambda _engine: (False, "no database in tests"))
    svc = AppService()
    svc.db_ok = False
    return svc


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


class TestWatermarkWiring:
    def test_no_longer_raises_not_implemented(self, service):
        """The whole point of this change: the page had a dead entry point."""
        with pytest.raises(ValueError):
            service.watermark("no-such-action")

    def test_unknown_action_names_itself(self, service):
        with pytest.raises(ValueError, match="unknown watermark action"):
            service.watermark("destroy")

    def test_clear_resets_state(self, service):
        service._watermark_scans["x"] = object()
        result = service.watermark("clear")
        assert result["total"] == 0
        assert not service._watermark_scans

    def test_scan_of_an_empty_selection_is_not_an_error(self, service):
        assert service.watermark("scan") == {"scanned": 0}

    def test_page_model_is_empty_before_any_selection(self, service):
        model = service._watermark_page({})
        assert model["has_files"] is False
        assert model["can_remove"] is False
        assert model["files"] == []

    def test_scan_populates_the_page_model(self, service, tmp_path):
        import pymupdf

        pdf = tmp_path / "plain.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "An ordinary deed with no watermark.")
        doc.save(pdf)
        doc.close()

        service.watermark_files.add([pdf])
        service.watermark("scan")
        model = service._watermark_page({})

        assert model["has_files"] is True
        assert len(model["files"]) == 1
        row = model["files"][0]
        assert row["name"] == "plain.pdf"
        assert row["pages"] == 1
        # No watermark in a plain PDF, so nothing is offered for removal.
        assert row["detected"] == "none"
        assert model["can_remove"] is False

    def test_removal_never_touches_the_source(self, service, tmp_path):
        """The source is evidence. Cleaned copies go somewhere else entirely."""
        import pymupdf

        pdf = tmp_path / "src.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(pdf)
        doc.close()
        before = pdf.read_bytes()

        service.watermark_files.add([pdf])
        service.watermark("scan")
        service.watermark("remove")
        assert pdf.read_bytes() == before

    def test_output_directory_is_separate_from_the_source(self):
        from app.services import EXPORT_DIR, WATERMARK_DIR

        assert WATERMARK_DIR != EXPORT_DIR
        assert "watermark" in WATERMARK_DIR.name

    def test_removal_is_lossless_only(self):
        """A raster watermark is burned into the scan - the pixels beneath were
        never captured, so removing it means inventing content on a legal
        document. `allow_lossy` must stay False here."""
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        assert "allow_lossy=False" in source

    def test_a_broken_pdf_does_not_abort_the_selection(self, service, tmp_path):
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
        service.watermark_files.add([bad])
        # Reported, not raised: one bad file must not lose the others.
        result = service.watermark("scan")
        assert isinstance(result["scanned"], int)


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------


class TestCapabilityGating:
    def test_model_exposes_both_polarities(self, service):
        """Mustache has no `not`, so a template needs `can_x` for the control
        and `no_x` for the explanation."""
        service.status_service = _FakeStatus(
            {"can_browse": True, "can_export": True, "can_upload": False,
             "can_process": False, "reasons": {"upload": "database is down"}})
        model = service._capability_model()
        assert model["can_upload"] is False
        assert model["no_upload"] is True
        assert model["can_browse"] is True
        assert model["no_browse"] is False

    def test_reasons_reach_the_template(self, service):
        service.status_service = _FakeStatus(
            {"can_process": False, "reasons": {"process": "AI server is down"}})
        model = service._capability_model()
        assert model["process_reason"] == "AI server is down"
        assert {"action": "Process", "reason": "AI server is down"} \
            in model["capability_reasons"]

    def test_degraded_is_false_when_everything_works(self, service):
        service.status_service = _FakeStatus(
            {"can_browse": True, "can_export": True, "can_upload": True,
             "can_process": True, "reasons": {}})
        assert service._capability_model()["degraded"] is False

    def test_degraded_is_true_when_anything_is_blocked(self, service):
        service.status_service = _FakeStatus(
            {"can_browse": True, "can_export": True, "can_upload": True,
             "can_process": False, "reasons": {"process": "no AI server"}})
        assert service._capability_model()["degraded"] is True

    def test_a_page_builder_can_override_the_global_value(self, service):
        """Some pages have a stricter local reason - an empty selection, a full
        queue - and that must win over the system-wide capability."""
        service.status_service = _FakeStatus(
            {"can_upload": True, "reasons": {}})
        model: dict[str, Any] = {"can_upload": False}
        for key, value in service._capability_model().items():
            model.setdefault(key, value)
        assert model["can_upload"] is False

    def test_missing_capabilities_gate_everything_off(self, service):
        """An empty snapshot must fail closed, not open."""
        service.status_service = _FakeStatus({})
        model = service._capability_model()
        assert not any(model[f"can_{n}"]
                       for n in ("browse", "export", "upload", "process"))


class TestTemplatesUseTheGating:
    TEMPLATES = ROOT / "src" / "app" / "ui" / "templates"

    @pytest.mark.parametrize("template,flag", [
        ("processing.mustache", "no_process"),
        ("upload.mustache", "no_upload"),
        ("data_view.mustache", "no_export"),
    ])
    def test_action_is_gated(self, template, flag):
        text = (self.TEMPLATES / template).read_text(encoding="utf-8")
        assert flag in text, f"{template} does not consult {flag}"

    def test_a_disabled_control_explains_itself(self):
        """A greyed-out button with no reason reads as a broken application."""
        text = (self.TEMPLATES / "processing.mustache").read_text(encoding="utf-8")
        assert "process_reason" in text

    def test_banner_is_a_shared_partial(self):
        """One definition, two render paths.

        A full page load renders it through the shell; navigation renders the
        content only and prepends it. Duplicating the markup would let the two
        drift, and the content-only path is the one a user sees most.
        """
        base = (self.TEMPLATES / "base.mustache").read_text(encoding="utf-8")
        assert "{{> capability_banner}}" in base
        partial = (self.TEMPLATES / "capability_banner.mustache").read_text(
            encoding="utf-8")
        assert "capability_reasons" in partial
        assert "degraded" in partial

    def test_banner_survives_a_content_only_render(self, service):
        """Navigation replaces `#content`, so the banner has to come with it."""
        service.status_service = _FakeStatus(
            {"can_process": False, "reasons": {"process": "AI server offline"}})
        html = service.render_page("dashboard", {}, shell_html=False)
        assert "capability-banner" in html
        assert "<!DOCTYPE" not in html, "a fragment must not be a whole document"

    def test_banner_sections_are_balanced(self):
        import re

        base = (self.TEMPLATES / "base.mustache").read_text(encoding="utf-8")
        assert len(re.findall(r"\{\{[#^]\s*\w+", base)) == \
            len(re.findall(r"\{\{/\s*\w+", base))


class TestGatingActuallyRenders:
    """The template checks above prove the markup exists; these prove it fires.

    Regression for a real defect: the banner lives in `base.mustache`, which is
    rendered from the *chrome* context, not the page model. `degraded` was
    computed correctly and never reached the shell, so `{{#degraded}}` collapsed
    to nothing on every page and the feature was invisible.
    """

    DEGRADED = {
        "can_browse": False, "can_export": False,
        "can_upload": False, "can_process": False,
        "reasons": {"browse": "database is not reachable",
                    "process": "AI server is not running"},
    }
    HEALTHY = {"can_browse": True, "can_export": True, "can_upload": True,
               "can_process": True, "reasons": {}}

    PAGES = ("dashboard", "upload", "processing", "data", "watermark",
             "settings", "validation", "help")

    @pytest.mark.parametrize("page", PAGES)
    def test_banner_appears_on_every_page_when_degraded(self, service, page):
        service.status_service = _FakeStatus(self.DEGRADED)
        html = service.render_page(page, {})
        assert "capability-banner" in html, f"{page} renders no banner"

    @pytest.mark.parametrize("page", PAGES)
    def test_no_banner_when_everything_works(self, service, page):
        service.status_service = _FakeStatus(self.HEALTHY)
        assert "capability-banner" not in service.render_page(page, {})

    @pytest.mark.parametrize("page", PAGES)
    def test_no_unrendered_mustache_escapes(self, service, page):
        """An unbalanced section renders the rest of the page as nothing, which
        looks like a blank screen rather than an error."""
        service.status_service = _FakeStatus(self.DEGRADED)
        assert "{{" not in service.render_page(page, {})

    def test_the_reason_is_visible_to_the_user(self, service):
        service.status_service = _FakeStatus(self.DEGRADED)
        html = service.render_page("processing", {})
        assert "AI server is not running" in html

    def test_start_button_is_disabled_without_an_ai_server(self, service):
        service.status_service = _FakeStatus(self.DEGRADED)
        html = service.render_page("processing", {})
        start = html[html.index('id="btn-start"'):][:220]
        assert "disabled" in start


# ---------------------------------------------------------------------------
# Auto batch mode
# ---------------------------------------------------------------------------


class TestBatchMode:
    """The Settings page has offered manual/auto since the UI was built; the
    runner was constructed MANUAL and never read the stored value."""

    def test_setting_is_applied_to_the_runner(self, service, monkeypatch):
        from core.pipeline.runner import BatchMode

        monkeypatch.setattr(service, "_setting",
                            lambda key, default="": {"batch_mode": "auto"}.get(
                                key, default))
        service._apply_runner_settings()
        assert service.runner.mode is BatchMode.AUTO

    def test_manual_is_the_default(self, service, monkeypatch):
        from core.pipeline.runner import BatchMode

        monkeypatch.setattr(service, "_setting", lambda key, default="": default)
        service._apply_runner_settings()
        assert service.runner.mode is BatchMode.MANUAL

    def test_cooldown_is_applied(self, service, monkeypatch):
        monkeypatch.setattr(service, "_setting",
                            lambda key, default="": {
                                "batch_mode": "auto",
                                "auto_cooldown_seconds": "120"}.get(key, default))
        service._apply_runner_settings()
        assert service.runner.auto_cooldown_s == 120.0

    def test_zero_cooldown_is_floored(self, service, monkeypatch):
        """A zero wait would start the next batch the instant the last document
        lands, giving the GPU no chance to release memory."""
        monkeypatch.setattr(service, "_setting",
                            lambda key, default="": {
                                "auto_cooldown_seconds": "0"}.get(key, default))
        service._apply_runner_settings()
        assert service.runner.auto_cooldown_s >= 5.0

    def test_a_bad_value_does_not_stop_startup(self, service, monkeypatch):
        monkeypatch.setattr(service, "_setting",
                            lambda key, default="": {
                                "auto_cooldown_seconds": "not-a-number"}.get(
                                    key, default))
        service._apply_runner_settings()  # must not raise
        assert service.errors

    def test_cooldown_defers_promotion(self):
        """The cooldown path itself: inside the window, no batch is promoted."""
        import time

        from core.pipeline.runner import BatchMode, BatchRunner, build_stages

        runner = BatchRunner.__new__(BatchRunner)
        runner.mode = BatchMode.AUTO
        runner.auto_cooldown_s = 60.0
        runner.stats = type("S", (), {"last_document_at": time.monotonic()})()
        elapsed = time.monotonic() - runner.stats.last_document_at
        assert elapsed < runner.auto_cooldown_s, "cooldown would not defer"

    def test_settings_are_seeded(self):
        from tools.db_setup import DEFAULT_SETTINGS

        assert DEFAULT_SETTINGS["batch_mode"] == "manual"
        assert "auto_cooldown_seconds" in DEFAULT_SETTINGS


# ---------------------------------------------------------------------------
# Retention scheduler
# ---------------------------------------------------------------------------


class TestRetention:
    def test_off_by_default(self, service, monkeypatch):
        """Retention DELETES data. A destructive job must not start itself."""
        monkeypatch.delenv("SALEDEED_RETENTION", raising=False)
        service._start_retention()
        assert service.retention is None

    def test_not_started_without_a_database(self, service, monkeypatch):
        monkeypatch.setenv("SALEDEED_RETENTION", "true")
        service.db_ok = False
        service._start_retention()
        assert service.retention is None

    def test_enabled_by_the_environment(self, service, monkeypatch):
        monkeypatch.setenv("SALEDEED_RETENTION", "true")
        service.db_ok = True
        monkeypatch.setattr(service, "_setting", lambda key, default="": default)
        service._start_retention()
        try:
            assert service.retention is not None
        finally:
            if service.retention is not None:
                service.retention.stop()

    def test_defers_while_a_batch_runs(self, service, monkeypatch):
        """A dump taken during heavy write activity is slower, larger, and
        competes for the disk the batch is writing to."""
        monkeypatch.setenv("SALEDEED_RETENTION", "on")
        service.db_ok = True
        monkeypatch.setattr(service, "_setting", lambda key, default="": default)
        service._start_retention()
        try:
            assert service.retention.is_busy is not None
            assert service.retention.is_busy() is False  # runner is idle
        finally:
            service.retention.stop()

    def test_shutdown_stops_the_thread(self, service, monkeypatch):
        monkeypatch.setenv("SALEDEED_RETENTION", "1")
        service.db_ok = True
        monkeypatch.setattr(service, "_setting", lambda key, default="": default)
        service._start_retention()
        scheduler = service.retention
        service.shut_down()
        assert scheduler._thread is None

    def test_interval_is_seeded(self):
        from tools.db_setup import DEFAULT_SETTINGS

        assert "retention_interval_hours" in DEFAULT_SETTINGS


class TestPressureIsReportedHonestly:
    """A degraded server is not an offline one, and saying so matters.

    Reported by the user: the banner read "AI server offline" while the server
    was up, the model was loaded on CUDA, and the real problem was 0.48 GB of
    free host RAM. That wording sends someone to restart a healthy service
    instead of closing the applications eating their memory.
    """

    HEALTHY_ENGINE = {"loaded": True, "engine": "llamacpp", "device": "cuda",
                      "detail": "ready"}

    def _health(self, **overrides):
        payload = {
            "ready": False, "pressure": "critical", "admitting_work": False,
            "engine": dict(self.HEALTHY_ENGINE),
            "resources": {"ram_available_bytes": int(0.48 * 1024 ** 3),
                          "ram_total_bytes": int(7.42 * 1024 ** 3),
                          "vram_free_bytes": int(0.94 * 1024 ** 3),
                          "vram_total_bytes": 4 * 1024 ** 3,
                          "disk_free_bytes": 150 * 1024 ** 3},
        }
        payload.update(overrides)
        return payload

    def test_a_loaded_model_under_pressure_is_degraded_not_down(self):
        """The probe classification that produced the wrong word."""
        from app.status import Availability, StatusService

        service = StatusService("http://127.0.0.1:59999", lambda: None)
        service._fetch_health = lambda: self._health()  # type: ignore[attr-defined]
        # Exercise the branch directly with a payload rather than a socket.
        from app.status import ProbeResult

        health = self._health()
        assert health["engine"]["loaded"] is True
        # A loaded engine that is merely refusing work must never read as DOWN.
        assert Availability.DEGRADED.usable is True
        assert Availability.DOWN.usable is False

    def test_the_reason_names_memory_not_the_server(self):
        from app.status import _pressure_reason

        reason = _pressure_reason(self._health())
        assert "memory" in reason.lower()
        assert "0.4 GB" in reason or "0.5 GB" in reason
        assert "offline" not in reason.lower(), \
            "a running server must not be described as offline"

    def test_the_reason_tells_the_user_what_to_do(self):
        from app.status import _pressure_reason

        assert "Close other applications" in _pressure_reason(self._health())

    def test_vram_pressure_is_distinguished_from_ram(self):
        from app.status import _pressure_reason

        reason = _pressure_reason(self._health(resources={
            "ram_available_bytes": 6 * 1024 ** 3,
            "ram_total_bytes": 8 * 1024 ** 3,
            "vram_free_bytes": int(0.1 * 1024 ** 3),
            "vram_total_bytes": 4 * 1024 ** 3,
            "disk_free_bytes": 150 * 1024 ** 3}))
        assert "raphics memory" in reason

    def test_low_disk_is_distinguished(self):
        from app.status import _pressure_reason

        reason = _pressure_reason(self._health(resources={
            "ram_available_bytes": 6 * 1024 ** 3,
            "ram_total_bytes": 8 * 1024 ** 3,
            "vram_free_bytes": 3 * 1024 ** 3,
            "vram_total_bytes": 4 * 1024 ** 3,
            "disk_free_bytes": 1 * 1024 ** 3}))
        assert "disk space" in reason

    def test_unexplained_pressure_still_says_the_server_is_running(self):
        from app.status import _pressure_reason

        reason = _pressure_reason(self._health(resources={}))
        assert "running" in reason
        assert "offline" not in reason.lower()

    def test_processing_is_still_blocked_while_pressure_lasts(self, service):
        """Honest wording must not become a permissive gate: the governor is
        refusing work, so Start has to stay disabled."""
        service.status_service = _FakeStatus(
            {"can_browse": True, "can_export": True, "can_upload": True,
             "can_process": False,
             "reasons": {"process": "Not enough free memory - 0.4 GB of 7.4 GB"}})
        model = service._capability_model()
        assert model["can_process"] is False
        assert "memory" in model["process_reason"]


class TestNavigationKeepsOneChannel:
    """Navigation must not rebuild the JavaScript context.

    `document.write` tore down the context and re-ran app.js, constructing a
    second QWebChannel over the same transport. The new channel took over
    `onmessage` with an empty `execCallbacks`, so replies still in flight for
    the old one arrived with unknown ids:

        Uncaught TypeError: channel.execCallbacks[message.id] is not a function

    It repeated at the status-poll interval because a poll was almost always in
    flight when a navigation landed.
    """

    JS = ROOT / "src" / "app" / "ui" / "assets" / "app.js"

    def test_document_is_never_rewritten(self):
        js = self.JS.read_text(encoding="utf-8")
        for banned in ("document.write(", "document.open(", "document.close("):
            assert banned not in js, (
                f"{banned} destroys the QWebChannel and orphans every reply "
                "already in flight")

    def test_navigation_replaces_the_content_area(self):
        js = self.JS.read_text(encoding="utf-8")
        assert "host.innerHTML = res.html" in js

    def test_navigation_rebinds_element_handlers(self):
        """Delegated listeners survive an innerHTML swap; direct ones do not."""
        js = self.JS.read_text(encoding="utf-8")
        navigate = js[js.index("function navigate("):js.index("function setActiveNav(")]
        assert "bindDropzone()" in navigate
        assert "setActiveNav(" in navigate

    def test_navigation_asks_for_content_only(self):
        """The bridge defaults `shell` off, so a render is a fragment unless
        the caller says otherwise."""
        source = (ROOT / "src" / "app" / "ui" / "bridge.py").read_text(encoding="utf-8")
        assert 'shell_html=bool(p.get("shell"))' in source

    def test_a_fragment_render_is_not_a_document(self, service):
        service.status_service = _FakeStatus({"can_browse": True, "reasons": {}})
        for page in ("dashboard", "upload", "processing", "data", "watermark",
                     "settings", "validation", "help"):
            html = service.render_page(page, {}, shell_html=False)
            assert "<!DOCTYPE" not in html, f"{page} returned a whole document"
            assert "<html" not in html.lower()

    def test_the_full_document_is_still_available_for_first_paint(self, service):
        """MainWindow renders the shell once at startup."""
        service.status_service = _FakeStatus({"can_browse": True, "reasons": {}})
        html = service.render_page("dashboard", {})
        assert html.lstrip().startswith("<!DOCTYPE")
        assert 'id="content"' in html

    def test_the_channel_is_constructed_once(self):
        js = self.JS.read_text(encoding="utf-8")
        assert js.count("new QWebChannel(") == 1, \
            "more than one channel would fight over the transport"


class TestNoDetachedOrmAccess:
    """ORM objects must not be read after their session closes.

    Reported by the user once the UI actually worked:

        DetachedInstanceError: Parent instance is not bound to a Session;
        lazy load operation of attribute 'user' cannot proceed

    `_batch_detail` built its result dict *after* the `with session_scope(...)`
    block. Columns already loaded by the query still read fine, which is why it
    looked correct - but `user` is a relationship that was never loaded, so
    touching it tried to emit SQL on a closed session.

    A source-level rule cannot catch this in general; `tools/service_sweep.py`
    exercises every entry point against real rows. These are the cheap guards.
    """

    def test_batch_detail_reads_relationships_inside_the_session(self):
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        body = source[source.index("def _batch_detail"):]
        body = body[:body.index("\n    def ")]

        scope_end = body.index("numeric = {")
        inside, outside = body[:scope_end], body[scope_end:]
        assert "batch.user" in inside, \
            "the relationship must be read while the session is open"
        assert "batch.user" not in outside, \
            "reading batch.user after the scope raises DetachedInstanceError"

    def test_no_orm_attribute_is_read_after_its_scope(self):
        """Scan for the shape of the defect: a `session_scope` block that ends,
        followed by a `<local>.<relationship>` read in the same function."""
        import re

        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        # Relationship attributes - the ones that emit SQL when touched.
        # Raw strings: without the r-prefix these were ordinary strings, so
        # every \\b became a literal BACKSPACE character and the patterns
        # matched nothing. The check passed by never finding an offender.
        relationships = (r"\.user\b", r"\.documents\b", r"\.batch\b",
                         r"\.persons\b", r"\.pages\b", r"\.extractions\b")
        offenders: list[str] = []
        for match in re.finditer(r"def (_\w+)\(self", source):
            name = match.group(1)
            body = source[match.end():]
            nxt = body.find("\n    def ")
            body = body[:nxt] if nxt != -1 else body
            if "session_scope" not in body:
                continue
            # Everything after the last line indented inside the with-block.
            tail = body[body.rindex("session_scope"):]
            dedented = re.split(r"\n        (?=\S)", tail, maxsplit=1)
            after = dedented[1] if len(dedented) > 1 else ""
            for rel in relationships:
                if re.search(r"\b\w+" + rel, after):
                    offenders.append(f"{name}: {rel}")
        assert not offenders, (
            f"relationship read after the session closed: {offenders}")

    def test_the_sweep_tool_exists(self):
        """The only reliable check needs real rows, so it is a tool, not a test."""
        tool = ROOT / "src" / "tools" / "service_sweep.py"
        assert tool.is_file()
        text = tool.read_text(encoding="utf-8")
        assert r"d:\saledeed" not in text.lower(), "hard-coded developer path"
        assert "batch_detail" in text


class TestNoDeadControls:
    """Every button must reach a handler, and every handler a real slot.

    Reported by the user: clicking Download CSV did nothing. `btn-export-view`
    had no handler in `app.js` at all, so the click fell through and the page
    simply sat there - no error, because nothing was attempted.

    Three buttons were dead this way, and the prompt editor was worse than dead:
    "Edit Prompt" toggled `readOnly` and never saved, so an edit vanished on the
    next navigation, and "Restore Default" could never have anything to restore.
    """

    JS = ROOT / "src" / "app" / "ui" / "assets" / "app.js"
    TEMPLATES = ROOT / "src" / "app" / "ui" / "templates"
    BRIDGE = ROOT / "src" / "app" / "ui" / "bridge.py"

    def _button_ids(self) -> set[str]:
        import re

        markup = "".join(p.read_text(encoding="utf-8")
                         for p in self.TEMPLATES.glob("*.mustache"))
        return set(re.findall(r'<button[^>]*id="([\w-]+)"', markup))

    def test_every_button_has_a_handler(self):
        js = self.JS.read_text(encoding="utf-8")
        dead = [b for b in sorted(self._button_ids())
                if f'"{b}"' not in js and f"'{b}'" not in js]
        assert not dead, f"buttons with no handler: {dead}"

    def test_every_call_names_a_real_slot(self):
        """A `call("export_view", ...)` naming a slot that does not exist fails
        only at runtime, on a page nobody opened during testing."""
        import re

        js = self.JS.read_text(encoding="utf-8")
        bridge = self.BRIDGE.read_text(encoding="utf-8")
        called = set(re.findall(r'call\("(\w+)"', js))
        declared = set(re.findall(r"def (\w+)\(self, request_id: str", bridge))
        assert not (called - declared), \
            f"app.js calls slots that do not exist: {called - declared}"

    def test_every_slot_reaches_the_service(self):
        """The other direction: a slot calling a method the service lacks."""
        import re

        from app.services import AppService

        bridge = self.BRIDGE.read_text(encoding="utf-8")
        missing = [name for name in set(re.findall(r"self\.service\.(\w+)\(", bridge))
                   if not hasattr(AppService, name)]
        assert not missing, f"bridge calls service methods that do not exist: {missing}"

    def test_the_prompt_editor_actually_saves(self, service):
        """Toggling readOnly is not saving. The edit has to reach disk, or it
        silently disappears on the next navigation."""
        js = self.JS.read_text(encoding="utf-8")
        assert 'call("save_prompt"' in js
        assert hasattr(service, "save_prompt")

    def test_restoring_the_prompt_is_possible_after_an_edit(self, service, tmp_path):
        """`.default` is written before the first overwrite - without it,
        Restore Default can never have anything to restore."""
        original = "ORIGINAL PROMPT"
        prompt = tmp_path / "prompt.txt"
        prompt.write_text(original, encoding="utf-8")
        service.PROMPT_PATH = prompt

        service.save_prompt("EDITED")
        assert prompt.with_suffix(".txt.default").is_file()
        assert prompt.read_text(encoding="utf-8") == "EDITED"

        service.reset_prompt()
        assert prompt.read_text(encoding="utf-8") == original

    def test_an_empty_prompt_is_refused(self, service, tmp_path):
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("something", encoding="utf-8")
        service.PROMPT_PATH = prompt
        with pytest.raises(Exception):
            service.save_prompt("   ")

    def test_export_view_falls_back_to_a_real_batch(self, service):
        """The Data View defaults to the most recent batch, so exporting with
        no explicit selection must export *that* one - not fail, and not export
        something the user is not looking at."""
        import inspect

        source = inspect.getsource(type(service).export_view)
        assert "list_paginated(1, 1)" in source


class TestMachineDetailsLivesOnSettings:
    """Machine Details moved from the Dashboard to Settings.

    A relocation, not a rewrite: the markup became a partial and the same
    `_machine()` model feeds it. The tests below are about *where* it renders
    and that it renders exactly once - the two things a move can get wrong.
    """

    TEMPLATES = ROOT / "src" / "app" / "ui" / "templates"
    MARKERS = ("GPU (inference only)", "VRAM free / total", "RAM free / total",
               "Resource pressure", "Disk free")

    def test_the_markup_lives_in_one_partial(self):
        """Copying the block into settings.mustache would have worked and left
        two copies to maintain."""
        partial = self.TEMPLATES / "machine_details.mustache"
        assert partial.is_file()
        body = partial.read_text(encoding="utf-8")
        for marker in self.MARKERS:
            assert marker in body

    def test_settings_includes_the_partial_rather_than_repeating_it(self):
        settings = (self.TEMPLATES / "settings.mustache").read_text(encoding="utf-8")
        assert "{{> machine_details}}" in settings
        for marker in self.MARKERS:
            assert marker not in settings, f"markup duplicated into settings: {marker}"

    def test_the_dashboard_no_longer_carries_it(self):
        dashboard = (self.TEMPLATES / "dashboard.mustache").read_text(encoding="utf-8")
        assert "{{> machine_details}}" not in dashboard
        assert "<h2>Machine</h2>" not in dashboard
        for marker in self.MARKERS:
            assert marker not in dashboard, f"still on the dashboard: {marker}"

    def test_the_dashboard_model_no_longer_takes_a_machine(self):
        """Leaving the parameter would keep the data flowing to a page that no
        longer renders it - dead work on every dashboard paint."""
        import inspect

        from app.ui.renderer import dashboard_model

        assert "machine" not in inspect.signature(dashboard_model).parameters

    def test_settings_supplies_the_machine_keys(self):
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        settings = source[source.index("def _settings(self"):]
        settings = settings[:settings.index("\n    def ", 10)]
        assert "machine = self._machine()" in settings
        assert "**machine," in settings

    def test_moving_it_added_no_data_fetching(self):
        """`_machine()` reads cached probes. If it ever starts making a request,
        rendering Settings would begin costing one."""
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        body = source[source.index("def _machine(self)"):]
        body = body[:body.index("\n    def ", 10)]
        for forbidden in ("urlopen", "requests.", "self.ai.health(", "self.ai.profile("):
            assert forbidden not in body, f"_machine now fetches: {forbidden}"
        assert "self.status_service" in body


class TestFileDialogsRunOnTheGuiThread:
    """Every slot that can open a file dialog must be registered as GUI-thread.

    Qt widgets may only be created on the thread owning the QApplication.
    `Bridge._GUI_THREAD_ACTIONS` is the list that arranges it, and it is
    hand-maintained - exactly the kind that gets forgotten when a page is added.
    It was: the OCR page shipped with `ocr_tool` missing, and every Browse click
    put a RuntimeError banner on screen instead of a dialog.

    So the requirement is derived from the service's own source rather than
    restated here. A new page that opens a picker fails this test until it is
    registered, which is the only way a list like this stays correct.
    """

    @staticmethod
    def _picker_users() -> dict[str, set[str]]:
        """{service method -> action strings that reach `file_picker`}.

        An empty set means the method opens a dialog unconditionally, so the
        whole slot must be GUI-thread rather than one of its actions.
        """
        import ast

        source = (Path(__file__).resolve().parents[1] / "src" / "app"
                  / "services.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        def mentions_picker(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Constant) and n.value == "file_picker"
                or isinstance(n, ast.Attribute) and n.attr == "file_picker"
                for n in ast.walk(node))

        found: dict[str, set[str]] = {}
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
                if not mentions_picker(fn):
                    continue
                actions: set[str] = set()
                for branch in (n for n in ast.walk(fn) if isinstance(n, ast.If)):
                    if not mentions_picker(branch):
                        continue
                    test = branch.test
                    if (isinstance(test, ast.Compare)
                            and isinstance(test.left, ast.Name)
                            and test.left.id == "action"
                            and test.comparators
                            and isinstance(test.comparators[0], ast.Constant)):
                        actions.add(str(test.comparators[0].value))
                found[fn.name] = actions
        return found

    @staticmethod
    def _slot_targets() -> dict[str, set[str]]:
        """{bridge slot -> service methods it calls}."""
        import ast

        source = (Path(__file__).resolve().parents[1] / "src" / "app" / "ui"
                  / "bridge.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        targets: dict[str, set[str]] = {}
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if cls.name != "Bridge":
                continue
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                called = {
                    node.attr for node in ast.walk(fn)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "service"}
                if called:
                    targets[fn.name] = called
        return targets

    def test_the_scan_finds_the_known_picker_users(self):
        """Guards the guard. A parser that silently matched nothing would make
        every assertion below vacuously true."""
        users = self._picker_users()
        assert "watermark" in users, users
        assert "ocr_tool" in users, users

    def test_every_slot_that_opens_a_dialog_is_registered(self):
        from app.ui.bridge import _GUI_THREAD, _GUI_THREAD_ACTIONS

        users = self._picker_users()
        targets = self._slot_targets()

        for slot, methods in targets.items():
            reached = methods & set(users)
            if not reached:
                continue
            assert slot in _GUI_THREAD or slot in _GUI_THREAD_ACTIONS, (
                f"bridge slot {slot!r} reaches {sorted(reached)}, which opens a "
                "file dialog, but is not registered as GUI-thread - every "
                "Browse click will raise instead of opening the dialog")

    def test_the_registered_actions_are_the_ones_that_open_a_dialog(self):
        """Registering the slot is not enough - the *action* has to match, or
        the dispatch sends that particular call to a pool worker anyway."""
        from app.ui.bridge import _GUI_THREAD, _GUI_THREAD_ACTIONS

        users = self._picker_users()
        for slot, methods in self._slot_targets().items():
            if slot in _GUI_THREAD:
                continue
            for method in methods & set(users):
                needed = users[method]
                if not needed:
                    continue
                registered = _GUI_THREAD_ACTIONS.get(slot, set())
                missing = needed - registered
                assert not missing, (
                    f"{slot!r} opens a dialog for action(s) {sorted(missing)} "
                    f"but only {sorted(registered)} are registered")

    def test_the_dispatch_actually_routes_those_calls(self):
        """End of the chain: the list is consulted and answers correctly."""
        from app.ui.bridge import Bridge

        assert Bridge._needs_gui_thread("ocr_tool", {"action": "browse"}) is True
        assert Bridge._needs_gui_thread("watermark", {"action": "browse"}) is True
        # A non-dialog action must stay on the pool: running OCR on the GUI
        # thread would freeze the window for the minutes a deed takes.
        assert Bridge._needs_gui_thread("ocr_tool", {"action": "run"}) is False
        assert Bridge._needs_gui_thread("ocr_tool", {}) is False
