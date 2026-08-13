"""API, logging, monitoring, backup and recovery.

Grouped because they share a property: none of them affect extraction accuracy,
and all of them decide whether an operator can tell what went wrong at 2am and
recover from it. They are the difference between a tool that fails and a tool
that fails *legibly*.

The API tests run against the AI server's HTTP surface. Note it is stdlib
`ThreadingHTTPServer`, not FastAPI - a deliberate choice recorded in the
architecture notes, because the process must start fast and add no dependency
that can break the one thing the server exists to do.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# API - contract and error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApiContractOffline:
    """What the endpoints promise, checked without starting a server."""

    def test_documented_routes_are_implemented(self):
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        for route in ("/health", "/hardware", "/profile", "/jobs",
                      "/extract", "/extract/batch", "/shutdown"):
            assert f'"{route}"' in source, f"{route} is not routed"

    def test_unknown_paths_are_rejected(self):
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        assert "HTTPStatus.NOT_FOUND" in source, "no not-found handling"

    def test_handler_never_leaks_a_traceback(self):
        """An exception must become a JSON error, not a stack trace on the wire."""
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        assert "INTERNAL_SERVER_ERROR" in source

    def test_server_is_threaded(self):
        """A blocking handler would stall the health probe the UI polls."""
        from ai_server.server import make_http_server  # noqa: F401
        from http.server import ThreadingHTTPServer

        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        assert "ThreadingHTTPServer" in source
        assert ThreadingHTTPServer is not None


@pytest.mark.gpu
class TestApiLive:
    """Against a running server. Skipped when there is not one."""

    def _get(self, base: str, path: str, timeout: float = 10.0):
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as response:
            return response.status, json.loads(response.read())

    def test_health_reports_readiness(self, ai_server):
        status, payload = self._get(ai_server, "/health")
        assert status == 200
        assert "ready" in payload

    def test_hardware_reports_the_gpu(self, ai_server):
        _, payload = self._get(ai_server, "/hardware")
        assert "gpus" in payload or "cpu_name" in payload

    def test_profile_explains_its_choice(self, ai_server):
        _, payload = self._get(ai_server, "/profile")
        assert payload, "profile endpoint returned nothing"

    def test_unknown_route_returns_404(self, ai_server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            self._get(ai_server, "/no-such-endpoint")
        assert caught.value.code == 404

    def test_malformed_json_is_rejected_not_crashed(self, ai_server):
        request = urllib.request.Request(
            f"{ai_server}/extract", data=b"{not json at all",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                assert response.status < 500
        except urllib.error.HTTPError as exc:
            assert exc.code < 500, "malformed input caused a server error"

    def test_health_is_fast_enough_to_poll(self, ai_server):
        """The UI polls this every 2 s; it must not cost more than a fraction
        of that."""
        started = time.perf_counter()
        self._get(ai_server, "/health")
        assert time.perf_counter() - started < 1.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogging:
    def test_configure_is_idempotent(self, tmp_path):
        from core import logging_setup

        logging_setup.configure(app_name="saledeed", log_dir=tmp_path,
                                console=False, debug=True)
        first = len(logging.getLogger("saledeed").handlers)
        logging_setup.configure(app_name="saledeed", log_dir=tmp_path,
                                console=False, debug=True)
        second = len(logging.getLogger("saledeed").handlers)
        assert first == second, "handlers accumulated across calls"
        logging_setup.shutdown()

    def test_debug_records_are_suppressed_when_debug_is_off(self, tmp_path):
        """INFO and above are always kept - they are what an operator reads
        after a failure. DEBUG is the volume that must stay off by default."""
        from core import logging_setup

        logging_setup.configure(app_name="saledeed", log_dir=tmp_path,
                                console=False, debug=False)
        logging_setup.get_logger("test").debug("verbose-NOISE")
        logging_setup.get_logger("test").warning("kept-SIGNAL")
        logging_setup.shutdown()
        blob = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in tmp_path.glob("*.log"))
        assert "verbose-NOISE" not in blob
        assert "kept-SIGNAL" in blob

    def test_records_are_written_when_debug_is_on(self, tmp_path):
        from core import logging_setup

        logging_setup.configure(app_name="saledeed", log_dir=tmp_path,
                                console=False, debug=True)
        logging_setup.get_logger("test").warning("marker-XYZZY")
        logging_setup.shutdown()
        blob = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in tmp_path.glob("*.log"))
        assert "marker-XYZZY" in blob

    def test_structured_extras_appear_in_the_record(self, tmp_path):
        from core import logging_setup

        logging_setup.configure(app_name="saledeed", log_dir=tmp_path,
                                console=False, debug=True)
        logging_setup.get_logger("test").info(
            "batch progress", extra={"batch_id": 4242, "stage": "ocr"})
        logging_setup.shutdown()
        blob = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in tmp_path.glob("*.log"))
        assert "4242" in blob, "structured extra was dropped"

    def test_context_filter_attaches_to_handlers_not_loggers(self):
        """Regression: a filter on a logger never sees propagated records, so
        context silently vanished for every child logger."""
        source = (ROOT / "src" / "core" / "logging_setup.py").read_text(encoding="utf-8")
        assert "handler.addFilter" in source or ".addFilter(" in source

    def test_purge_removes_only_old_files(self, tmp_path):
        from core.logging_setup import purge_old_logs

        old = tmp_path / "old.log"
        old.write_text("x", encoding="utf-8")
        import os

        ancient = time.time() - (60 * 60 * 24 * 90)
        os.utime(old, (ancient, ancient))
        fresh = tmp_path / "fresh.log"
        fresh.write_text("y", encoding="utf-8")

        removed = purge_old_logs(tmp_path, retention_days=30)
        assert removed >= 1
        assert fresh.exists(), "a current log was deleted"
        assert not old.exists()


# ---------------------------------------------------------------------------
# Monitoring - the status layer the UI polls
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatusMonitoring:
    def test_circuit_breaker_opens_after_repeated_failure(self):
        from app.status import CircuitBreaker

        breaker = CircuitBreaker(threshold=2, cooldown_s=60.0)
        assert not breaker.open
        breaker.record(False)
        breaker.record(False)
        assert breaker.open, "breaker stayed closed after 2 failures"

    def test_circuit_breaker_recovers_on_success(self):
        from app.status import CircuitBreaker

        breaker = CircuitBreaker(threshold=2, cooldown_s=60.0)
        breaker.record(False)
        breaker.record(True)
        assert not breaker.open

    def test_probe_of_a_closed_port_is_bounded(self):
        """Windows retransmits SYN for ~2 s on a closed local port. A probe that
        waits that long turns a 2 s poll interval into a stall."""
        from app.status import CONNECT_TIMEOUT_S

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        started = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(CONNECT_TIMEOUT_S)
            sock.connect_ex(("127.0.0.1", free_port))
        elapsed = time.perf_counter() - started
        assert elapsed < 1.5, f"probe took {elapsed:.2f}s - the UI would stall"

    def test_availability_states_are_distinct(self):
        from app.status import Availability

        values = {member.value for member in Availability}
        assert len(values) == len(list(Availability))
        assert "up" in values and "down" in values

    def test_capabilities_carry_reasons(self):
        """A disabled button must be able to say why."""
        from app.status import Capabilities

        caps = Capabilities()
        assert hasattr(caps, "can_process")
        assert hasattr(caps, "reasons")


# ---------------------------------------------------------------------------
# Backup and recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackupOffline:
    def test_pg_dump_is_located_or_reported(self):
        from core.backup import find_pg_dump

        found = find_pg_dump()
        assert found is None or Path(found).exists()

    def test_password_is_passed_through_the_environment(self):
        source = (ROOT / "src" / "core" / "backup.py").read_text(encoding="utf-8")
        assert "PGPASSWORD" in source
        assert "--password" not in source, "password would be visible in argv"

    def test_verify_rejects_a_missing_archive(self, tmp_path):
        from core.backup import verify_archive

        ok, detail = verify_archive(tmp_path / "absent.dump")
        assert not ok and detail

    def test_verify_rejects_an_empty_archive(self, tmp_path):
        """An empty file is the classic silent-failure mode: pg_dump exits 0
        after writing nothing when the connection dies mid-stream."""
        from core.backup import verify_archive

        empty = tmp_path / "empty.dump"
        empty.write_bytes(b"")
        ok, _ = verify_archive(empty)
        assert not ok

    def test_restore_instructions_name_the_archive(self, tmp_path):
        from core.backup import restore_instructions

        archive = tmp_path / "backup_2026.dump"
        text = restore_instructions(archive, "postgresql://u:p@h/db")
        assert archive.name in text
        assert "psql" in text or "pg_restore" in text
        # Restoring over the live database is irreversible; the instructions
        # must say so rather than assume the reader knows.
        assert "NEW database" in text or "verify" in text

    def test_restore_instructions_do_not_leak_the_password(self, tmp_path):
        from core.backup import restore_instructions

        text = restore_instructions(tmp_path / "b.dump",
                                    "postgresql://u:sup3rsecret@h/db")
        assert "sup3rsecret" not in text


@pytest.mark.integration
class TestBackupLive:
    def test_dump_produces_a_verifiable_archive(self, tmp_path):
        from core.backup import dump_database, find_pg_dump, verify_archive

        if find_pg_dump() is None:
            pytest.skip("pg_dump is not installed")
        from core.db.engine import dsn_from_env

        result = dump_database(dsn_from_env(), tmp_path)
        if not result.ok:
            pytest.skip(f"dump unavailable: {result.detail}")
        ok, detail = verify_archive(Path(result.path))
        assert ok, detail
        assert Path(result.path).stat().st_size > 0


@pytest.mark.unit
class TestCrashRecoveryContract:
    def test_requeued_sentinel_is_distinct_from_failure(self):
        """A requeued document must not be finalised. Conflating REQUEUED with
        None stranded documents permanently: claim_next only considers work
        whose overall state is still PROCESSING."""
        from core.pipeline.runner import REQUEUED

        assert REQUEUED is not None
        assert REQUEUED is not False

    def test_runner_exposes_recover(self):
        from core.pipeline.runner import BatchRunner

        assert hasattr(BatchRunner, "recover")


class TestReclaimableMemoryAdmission:
    """The governor must not refuse work because of memory that work would free.

    R-037. On a 7.4 GiB machine the language model's own footprint is the single
    largest consumer: unloading it returns 2.65 GiB, measured. The governor
    judged admission on free RAM alone, so with the model loaded it read
    `critical`, refused all new work, and nothing else was ever going to release
    that memory - the state never cleared. A safeguard that cannot be satisfied
    is a deadlock.

    The OCR stage releases the model before it starts (R-035), so the memory is
    genuinely reclaimable and admitting the work is what frees it.
    """

    GIB = 1024 ** 3

    def _snapshot(self, *, free_gib: float, reclaimable_gib: float = 0.0,
                  total_gib: float = 7.42):
        from ai_server.resources import ResourceSnapshot

        return ResourceSnapshot(
            ram_total_bytes=int(total_gib * self.GIB),
            ram_available_bytes=int(free_gib * self.GIB),
            cpu_busy=0.1, vram_free_bytes=0, vram_total_bytes=0,
            disk_free_bytes=100 * self.GIB,
            reclaimable_bytes=int(reclaimable_gib * self.GIB))

    def test_effective_free_counts_what_can_be_handed_back(self):
        snap = self._snapshot(free_gib=0.58, reclaimable_gib=2.33)
        assert snap.ram_free_fraction < 0.08          # critical by itself
        assert snap.effective_free_fraction > 0.08    # not once the model goes

    def test_work_is_admitted_when_the_model_can_be_released(self):
        """The exact state that stopped the application: 8% free, model loaded."""
        from ai_server.resources import ResourceGovernor

        gov = ResourceGovernor()
        plan = gov._compute(self._snapshot(free_gib=0.58, reclaimable_gib=2.33))
        assert plan.admit_new_work, "refused work that unloading the model would allow"

    def test_work_is_still_refused_when_nothing_can_be_released(self):
        """The protection has to survive the fix. With no model loaded there is
        nothing to reclaim, and genuinely exhausted memory must still hold work
        back."""
        from ai_server.resources import ResourceGovernor

        gov = ResourceGovernor()
        plan = gov._compute(self._snapshot(free_gib=0.30, reclaimable_gib=0.0))
        assert not plan.admit_new_work

    def test_the_reported_pressure_stays_honest(self):
        """Only the admission decision looks at what could be freed. The level
        itself describes the machine as it is - the UI and the worker counts
        both read it, and neither should be told the memory is already free."""
        from ai_server.resources import Pressure, ResourceGovernor

        gov = ResourceGovernor()
        plan = gov._compute(self._snapshot(free_gib=0.58, reclaimable_gib=2.33))
        assert plan.pressure is Pressure.CRITICAL
        assert plan.admit_new_work

    def test_worker_counts_are_not_inflated_by_reclaimable_memory(self):
        """Planning four workers against memory that is not free yet would
        thrash the machine the moment they started."""
        from ai_server.resources import ResourceGovernor

        gov = ResourceGovernor()
        tight = gov._compute(self._snapshot(free_gib=0.58, reclaimable_gib=2.33))
        roomy = gov._compute(self._snapshot(free_gib=6.0, reclaimable_gib=0.0))
        assert max(tight.workers.values()) <= max(roomy.workers.values())

    def test_the_engine_reports_what_it_would_return(self):
        source = (ROOT / "src" / "ai_server" / "engines" / "llamacpp.py").read_text(
            encoding="utf-8")
        assert "reclaimable_bytes=" in source
        assert "_model_bytes" in source

    def test_the_server_tells_the_governor(self):
        """A provider that is never wired leaves the governor blind and the
        deadlock in place."""
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        assert "reclaimable_provider" in source
        assert "reclaimable_bytes" in source

    def test_a_broken_probe_cannot_block_work(self):
        """If asking how much is reclaimable raises, the answer is zero, not an
        exception on the sampling thread."""
        from ai_server.resources import ResourceGovernor

        gov = ResourceGovernor()
        gov.reclaimable_provider = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        assert gov._reclaimable() == 0


class TestTerminalLogging:
    """Logs have to reach the terminal, not only the file.

    R-039. The console handler sat at WARNING with a comment explaining that
    the desktop UI is the status display and the terminal should stay quiet.
    That is right for the packaged application and wrong for a service run from
    a terminal: the operator sees nothing and reads it as a dead process.
    """

    def _capture(self, tmp_path, monkeypatch, level=None, debug=False):
        import io
        import logging as std
        from core import logging_setup

        buffer = io.StringIO()
        monkeypatch.setattr("sys.stderr", buffer)
        logging_setup.shutdown()
        logging_setup.configure(app_name="probe", log_dir=tmp_path,
                                console_level=level, debug=debug)
        std.getLogger("probe.svc").info("info-LINE")
        std.getLogger("probe.svc").warning("warn-LINE")
        std.getLogger("probe.svc").debug("debug-LINE")
        logging_setup.shutdown()
        return buffer.getvalue()

    def test_info_reaches_the_terminal(self, tmp_path, monkeypatch):
        """The defect, stated directly."""
        assert "info-LINE" in self._capture(tmp_path, monkeypatch)

    def test_warnings_and_errors_still_reach_it(self, tmp_path, monkeypatch):
        assert "warn-LINE" in self._capture(tmp_path, monkeypatch)

    def test_debug_stays_out_unless_asked(self, tmp_path, monkeypatch):
        out = self._capture(tmp_path, monkeypatch)
        assert "debug-LINE" not in out
        assert "debug-LINE" in self._capture(tmp_path, monkeypatch, debug=True)

    def test_the_quiet_console_is_still_available(self, tmp_path, monkeypatch):
        """The packaged desktop application wants the old behaviour."""
        out = self._capture(tmp_path, monkeypatch, level="WARNING")
        assert "info-LINE" not in out
        assert "warn-LINE" in out

    def test_the_environment_can_set_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SALEDEED_LOG_CONSOLE", "WARNING")
        out = self._capture(tmp_path, monkeypatch)
        assert "info-LINE" not in out and "warn-LINE" in out

    def test_a_typo_falls_back_rather_than_raising(self, tmp_path, monkeypatch):
        """A bad environment variable must not stop the application starting."""
        monkeypatch.setenv("SALEDEED_LOG_CONSOLE", "LOUD")
        assert "info-LINE" in self._capture(tmp_path, monkeypatch)

    def test_every_line_carries_time_level_module_and_function(self, tmp_path,
                                                               monkeypatch):
        out = self._capture(tmp_path, monkeypatch)
        line = next(l for l in out.splitlines() if "info-LINE" in l)
        assert re.match(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ", line), line
        assert "INFO" in line
        assert "probe.svc." in line and "()" in line, line

    def test_exceptions_carry_the_stack(self, tmp_path, monkeypatch):
        import io
        import logging as std
        from core import logging_setup

        buffer = io.StringIO()
        monkeypatch.setattr("sys.stderr", buffer)
        logging_setup.shutdown()
        logging_setup.configure(app_name="probe", log_dir=tmp_path)
        try:
            raise ValueError("database refused the connection")
        except ValueError:
            std.getLogger("probe.db").error("connect failed", exc_info=True)
        logging_setup.shutdown()

        out = buffer.getvalue()
        assert "Traceback (most recent call last)" in out
        assert "ValueError: database refused the connection" in out


class TestRequestLogging:
    """Every request and response, once each, with an id and a duration."""

    SOURCE = ROOT / "src" / "ai_server" / "server.py"

    def test_requests_are_logged_on_the_way_out(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "def _send" in source
        send = source[source.index("def _send"):]
        send = send[:send.index("\n    def ", 10)]
        assert "log.log(level" in send, "responses are not logged"
        assert "request_id" in send and "elapsed" in send

    def test_the_handler_does_not_log_twice(self):
        """`BaseHTTPRequestHandler` writes its own line to stderr. Leaving it
        on would double every entry, which is how a log stops being read."""
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "def log_message" in source
        body = source[source.index("def log_message"):]
        body = body[:body.index("\n    def ", 10)]
        assert "return" in body and "log.info" not in body

    def test_health_polling_does_not_drown_the_log(self):
        """The shell polls /health every few seconds. At INFO that buries
        everything else within a minute."""
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "QUIET_PATHS" in source and "/health" in source

    def test_failures_are_logged_above_info(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "if status >= HTTPStatus.BAD_REQUEST" in source
        assert "level = logging.WARNING" in source

    def test_startup_failure_logs_a_traceback(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "log.critical" in source and "exc_info=True" in source


class TestPipelineStageLogging:
    """One line per stage per document - and only one."""

    SOURCE = ROOT / "src" / "core" / "pipeline" / "runner.py"

    def test_every_stage_reports_itself(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "_log_stage" in source
        for stage in ("ocr", "extract", "validate", "translate"):
            assert f'self._log_stage("{stage}"' in source, f"{stage} is silent"

    def test_the_batch_reports_start_and_finish(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "batch runner started" in source
        assert "batch runner stopping" in source

    def test_a_stage_failure_is_a_warning_not_a_silent_return(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        body = source[source.index("def _log_stage"):]
        body = body[:body.index("\n    def ", 10)]
        assert "log.warning" in body and "outcome.detail" in body

    def test_the_export_reports_what_it_wrote(self):
        source = (ROOT / "src" / "core" / "csv_export.py").read_text(encoding="utf-8")
        assert "CSV written" in source


class TestLogExtrasAreSafe:
    """`extra={...}` may not use a name LogRecord reserves.

    `logging.makeRecord` raises `KeyError: "Attempt to overwrite 'name' in
    LogRecord"`, from inside logging, at the moment the line is emitted. So a
    log call written to report an unusual condition crashes on exactly the
    condition it reports - and only once logging is configured at that level,
    which is why it survived the unit tests and appeared in the full suite.
    """

    RESERVED = {"name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "taskName"}

    def test_no_log_call_uses_a_reserved_key(self):
        import ast

        offenders = []
        for path in (ROOT / "src").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg != "extra" or not isinstance(kw.value, ast.Dict):
                        continue
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and key.value in self.RESERVED:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} -> {key.value!r}")
        assert not offenders, "reserved LogRecord keys in extra: " + "; ".join(offenders)

    def test_the_reserved_key_really_does_raise(self):
        """Documents why the rule exists, so it is not relaxed later."""
        import logging

        logger = logging.getLogger("probe.reserved")
        logger.setLevel(logging.INFO)
        with pytest.raises(KeyError):
            logger.makeRecord("probe.reserved", logging.INFO, "f", 1, "m", (),
                              None, extra={"name": "x"})
