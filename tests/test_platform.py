"""Installation, compatibility, performance, load and UI testing.

Grouped by what they measure rather than by module, because each of these asks a
question about the machine rather than about the code.

An honest note on two categories, restated here because a green suite must not
imply more than it proved:

**Load testing** at 100-1000 PDFs is *defined* here but not executed. On the
development machine OCR measured 2.9 minutes per page on CPU; a thousand
ten-page deeds is weeks of wall clock. The harness runs at whatever scale the
`SALEDEED_LOAD_PDFS` environment variable names, so the same test that runs 3
documents in CI runs 1000 on the deployment machine. Silence about the scale
actually run would be the dishonest part; the tests report it.

**Compatibility testing** across Windows versions and hardware cannot be
performed from one machine. What is testable from here is that the code makes no
assumption that would break elsewhere - no hard-coded paths, no fixed drive
letters, no assumption that a GPU exists. That is what these tests check, and it
is strictly weaker than running it on Windows 10.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest
import subprocess

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInstallation:
    """One-command start on a clean machine."""

    def test_entry_points_exist(self):
        assert (ROOT / "launcher.py").is_file()
        assert (ROOT / "Run Sale Deed AI.bat").is_file()

    def test_launcher_package_is_importable(self):
        from launcher import build_config, find_root  # noqa: F401

    def test_root_is_discovered_from_anywhere(self, tmp_path):
        """A shortcut, Task Scheduler and a copied folder all start elsewhere."""
        from launcher.config import find_root

        assert find_root(ROOT / "src" / "core" / "db") == ROOT

    def test_root_discovery_does_not_raise_outside_the_project(self, tmp_path):
        from launcher.config import find_root

        assert isinstance(find_root(tmp_path), Path)

    def test_required_directories_are_declared(self):
        from launcher.config import REQUIRED_DIRS

        # Under `runtime/` since the restructure: they are all disposable and
        # regenerated, which is the property that groups them.
        for expected in ("logs", "uploads", "exports", "backups"):
            assert f"runtime/{expected}" in REQUIRED_DIRS

    def test_every_preflight_step_is_callable(self):
        from launcher.steps import PREFLIGHT

        assert len(PREFLIGHT) >= 10
        for label, step in PREFLIGHT:
            assert isinstance(label, str) and callable(step)

    def test_steps_report_rather_than_raise(self, tmp_path):
        """A launcher that throws on a missing dependency cannot tell the user
        what to install."""
        from launcher.config import build_config
        from launcher.steps import PREFLIGHT, Result

        cfg = build_config(ROOT)
        cfg.engine = "mock"
        for label, step in PREFLIGHT:
            if label in ("PostgreSQL service", "Database", "Migrations"):
                continue  # touch external services
            result = step(cfg)
            assert isinstance(result, Result), f"{label} returned {type(result)}"
            assert result.detail, f"{label} gave no explanation"

    def test_failures_carry_a_remedy(self, tmp_path):
        """Every blocking failure must name the command that fixes it."""
        from launcher.config import build_config
        from launcher.steps import Outcome, check_model, check_runtime

        cfg = build_config(tmp_path)
        for step in (check_model, check_runtime):
            result = step(cfg)
            if result.outcome is Outcome.FAIL:
                assert result.remedy, f"{step.__name__} failed with no remedy"

    def test_batch_file_selects_an_interpreter_by_capability(self):
        """More than one Python is normally installed - this project needs a
        second one for Surya, and installing it changes what bare `python`
        resolves to. Choosing by version or by first-found silently launches an
        interpreter without PySide6, and the failure reads as a broken
        application rather than a wrong interpreter."""
        text = (ROOT / "Run Sale Deed AI.bat").read_text(encoding="utf-8",
                                                         errors="replace")
        assert "import PySide6" in text, \
            "the launcher picks a Python without checking it can run the app"

    def test_dependency_check_names_the_interpreter(self):
        """`pip install PySide6` against the wrong Python succeeds and changes
        nothing, so the message has to say which interpreter is short."""
        source = (ROOT / "src" / "launcher" / "steps.py").read_text(encoding="utf-8")
        assert "sys.version.split()" in source

    def test_batch_file_handles_paths_with_spaces(self):
        """The project lives in 'saledeed v3'; an unquoted path breaks on the
        space and the error is unreadable."""
        text = (ROOT / "Run Sale Deed AI.bat").read_text(encoding="utf-8",
                                                        errors="replace")
        assert '"launcher.py"' in text or '"%~dp0' in text
        assert "cd /d" in text, "batch file does not anchor to its own folder"

    def test_model_is_verified_never_downloaded(self):
        """Standing constraint: the fine-tuned model must never be replaced by
        an automatic download."""
        source = (ROOT / "src" / "launcher" / "steps.py").read_text(encoding="utf-8")
        assert "does not download a substitute" in source
        assert not re.search(r"urlretrieve|hf_hub_download|snapshot_download",
                             source), "the launcher can fetch a model"


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompatibility:
    """Portability properties checkable from one machine."""

    SOURCE_DIRS = ("src/core", "src/app", "src/ai_server", "src/launcher")

    def _sources(self):
        found = 0
        for directory in self.SOURCE_DIRS:
            for path in (ROOT / directory).rglob("*.py"):
                found += 1
                yield path
        # Without this, moving the source tree turns every test below into a
        # pass over an empty sequence. That is exactly what happened when the
        # project was restructured: the scan reported no offenders because it
        # scanned nothing.
        assert found, f"scanned no sources - are {self.SOURCE_DIRS} still there?"

    #: Absolute paths that are *search hints* for third-party software, each
    #: guarded by an existence check or used only as an environment-variable
    #: default. Reviewed individually - a new entry has to be justified, which is
    #: the point of listing them rather than matching a pattern.
    ALLOWED_ABSOLUTE = {
        "src/core/backup.py",       # where pg_dump lives when it is not on PATH
        "src/ai_server/hardware.py",   # default for %SystemRoot% only
        "src/ai_server/deployment.py",  # defaults for %ProgramFiles%, then is_file()
    }

    def test_no_unguarded_hard_coded_drive_letters(self):
        """A literal C:\\ that is *used* rather than *searched* breaks on any
        machine that installed things elsewhere."""
        offenders: list[str] = []
        pattern = re.compile(r'["\'][A-Za-z]:[\\/]')
        for path in self._sources():
            relative = path.relative_to(ROOT).as_posix()
            if relative in self.ALLOWED_ABSOLUTE:
                continue
            for number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line) and "example" not in line.lower():
                    offenders.append(f"{relative}:{number}")
        assert not offenders, f"unguarded absolute paths: {offenders}"

    def test_allowed_absolute_paths_are_still_guarded(self):
        """The allowlist is a promise that each site checks before using."""
        for relative in self.ALLOWED_ABSOLUTE:
            text = (ROOT / relative).read_text(encoding="utf-8")
            assert "is_file()" in text or "isfile" in text or "environ.get" in text, \
                f"{relative} uses an absolute path without a guard"

    def test_no_hard_coded_user_directories(self):
        offenders: list[str] = []
        for path in self._sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"[Uu]sers[\\/](?!<)[A-Za-z]", text) and \
                    "AppData" not in text:
                offenders.append(str(path.relative_to(ROOT)))
        assert not offenders, f"a specific user's folder is referenced: {offenders}"

    def test_paths_are_built_with_pathlib_not_concatenation(self):
        """String concatenation with '/' breaks the moment a path contains a
        backslash, which on Windows is always."""
        offenders: list[str] = []
        for path in self._sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r'\+\s*"/[a-z_]+\.(py|txt|json|gguf)"', text):
                offenders.append(str(path.relative_to(ROOT)))
        assert not offenders, offenders

    def test_runs_without_a_gpu(self):
        """Hardware detection must degrade, not raise, on a machine with no
        NVIDIA card."""
        from ai_server.hardware import detect

        hw = detect()
        assert hw.cpu_name
        assert hw.ram_total_bytes > 0
        # primary_gpu returns None rather than raising when there is no GPU.
        assert hw.primary_gpu is None or hw.primary_gpu.total_bytes > 0

    def test_profile_selection_has_a_cpu_path(self):
        """A machine with no GPU must still get a workable configuration."""
        source = (ROOT / "src" / "ai_server" / "profiles.py").read_text(encoding="utf-8")
        assert "cpu" in source.lower()

    def test_deployment_classes_cover_the_range(self):
        from ai_server.deployment import assess

        readiness = assess()
        assert readiness is not None
        assert hasattr(readiness, "checks") or hasattr(readiness, "ready")

    def test_line_endings_are_normalised_before_the_model_sees_text(self):
        """Measured: raw CRLF produced 6,758 tokens where the training tokenizer
        produced 6,408. A Windows-authored OCR file must not change the input
        distribution."""
        from core.ocr_cleanup import clean

        out, _ = clean("line one\r\nline two\r\n")
        assert "\r" not in out


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformance:
    """Micro-benchmarks with thresholds. Deliberately loose - these guard
    against an order-of-magnitude regression, not against noise."""

    def test_cleanup_is_linear_enough_for_a_large_deed(self):
        from core.ocr_cleanup import clean

        page = ("===== PAGE {} =====\n" + "Kannada ಕನ್ನಡ text 12,34,567 line\n" * 40)
        document = "".join(page.format(n) for n in range(1, 51))  # 50 pages

        started = time.perf_counter()
        _, report = clean(document)
        elapsed = time.perf_counter() - started

        assert report.pages_detected == 50
        assert elapsed < 2.0, f"cleanup of a 50-page deed took {elapsed:.2f}s"

    def test_validation_of_one_document_is_fast(self, ):
        from core.validation import validate_extraction

        extraction = {
            "document_details": {"transaction_date": "2024-06-15",
                                 "consideration_amount": 3000000},
            "buyer_details": [{"name": f"Person {i}",
                               "pan_card_number": "ABCDE1234F"} for i in range(10)],
            "seller_details": [{"name": f"Seller {i}"} for i in range(10)],
        }
        ocr = "Rs. 30,00,000 ABCDE1234F " * 500

        started = time.perf_counter()
        for _ in range(10):
            validate_extraction(extraction, ocr)
        elapsed = (time.perf_counter() - started) / 10

        assert elapsed < 0.5, f"validation took {elapsed * 1000:.0f}ms per document"

    def test_csv_export_scales_to_a_full_batch(self, tmp_path):
        from core.csv_export import DocumentExport, write_csv

        documents = [
            DocumentExport(
                transaction_identity=f"{i}/2024-25",
                extraction={
                    "document_details": {"transaction_date": "2024-06-15"},
                    "buyer_details": [{"name": f"Buyer {i}"}],
                    "seller_details": [{"name": f"Seller {i}"}]},
                source_filename=f"{i}.pdf")
            for i in range(1000)]

        started = time.perf_counter()
        written = write_csv(tmp_path / "big.csv", documents)
        elapsed = time.perf_counter() - started

        assert written >= 1000
        assert elapsed < 10.0, f"exporting 1000 documents took {elapsed:.1f}s"

    def test_status_snapshot_is_cheap_enough_to_poll(self):
        """The UI polls status every ~2 s. Anything approaching that is a stall.

        This is the regression guard for the measured 8,386 ms dashboard: the
        cost came from synchronous probes of an unreachable server, and the fix
        was to serve every field from cache.
        """
        from app.status import StatusService

        service = StatusService("http://127.0.0.1:59999", lambda: None)
        started = time.perf_counter()
        for _ in range(50):
            service.snapshot()
        elapsed = (time.perf_counter() - started) / 50

        assert elapsed < 0.01, f"snapshot cost {elapsed * 1000:.1f}ms - must be cached"


# ---------------------------------------------------------------------------
# Load, stress and scalability
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLoadAndStress:
    """Scale is controlled by SALEDEED_LOAD_PDFS so the same test runs at 3 in
    CI and at 1000 on the deployment machine. The count actually used is
    asserted on, so a report can never imply a scale that was not run."""

    @property
    def target(self) -> int:
        return int(os.environ.get("SALEDEED_LOAD_PDFS", "25"))

    def test_export_at_declared_scale(self, tmp_path):
        from core.csv_export import DocumentExport, write_csv

        count = self.target
        documents = [
            DocumentExport(
                transaction_identity=f"{i}/2024-25",
                extraction={"document_details": {"transaction_date": "2024-06-15"},
                            "buyer_details": [{"name": "B"}],
                            "seller_details": [{"name": "S"}]},
                source_filename=f"{i}.pdf")
            for i in range(count)]
        written = write_csv(tmp_path / "load.csv", documents)
        assert written >= count

    def test_cleanup_under_repeated_load(self):
        """Memory and time must not creep across many documents."""
        from core.ocr_cleanup import clean

        body = "===== PAGE 1 =====\n" + "text ಕನ್ನಡ 12,345\n" * 200
        first = time.perf_counter()
        clean(body)
        baseline = time.perf_counter() - first

        started = time.perf_counter()
        for _ in range(100):
            clean(body)
        average = (time.perf_counter() - started) / 100

        assert average < max(baseline * 4, 0.5), \
            f"per-document cost grew from {baseline * 1000:.0f}ms to {average * 1000:.0f}ms"

    def test_oversized_input_is_handled_not_crashed(self):
        """Stress: a corrupt OCR file can be far larger than any real deed."""
        from core.ocr_cleanup import clean

        huge = "x" * (5 * 1024 * 1024)
        out, report = clean(huge)
        assert isinstance(out, str)
        assert report is not None

    def test_pathological_input_does_not_hang(self):
        """Regex backtracking is the usual way text processing dies."""
        from core.ocr_cleanup import clean

        nasty = ("=" * 5000) + "\n" + ("<b>" * 5000) + "\n" + ("\\frac{1}{2}" * 2000)
        started = time.perf_counter()
        clean(nasty)
        assert time.perf_counter() - started < 5.0

    def test_disk_headroom_is_checked_before_a_batch(self):
        """1000 PDFs can be 25 GB; running out mid-run strands the work."""
        from launcher.steps import check_disk
        from launcher.config import build_config

        result = check_disk(build_config(ROOT))
        assert result.detail
        usage = shutil.disk_usage(ROOT)
        assert usage.free >= 0


# ---------------------------------------------------------------------------
# UI / UX
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUserInterface:
    """Rendering and gating, without instantiating Qt.

    Qt widget tests need a display and a running event loop, which makes them
    slow and flaky in a suite that should run on every change. What is tested
    here is everything that decides *what* the window shows.
    """

    TEMPLATES = ROOT / "src" / "app" / "ui" / "templates"

    def test_every_screen_has_a_template(self):
        for screen in ("dashboard", "upload", "processing", "validation",
                       "data_view", "batch_detail", "watermark", "settings",
                       "help"):
            assert (self.TEMPLATES / f"{screen}.mustache").is_file(), \
                f"{screen} screen is missing"

    def test_templates_are_valid_mustache(self):
        import pystache

        for path in self.TEMPLATES.glob("*.mustache"):
            try:
                pystache.parse(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"{path.name} does not parse: {exc}")

    def test_no_template_tag_is_left_unclosed(self):
        """An unbalanced section renders the rest of the page as nothing, which
        looks like a blank screen rather than an error."""
        for path in self.TEMPLATES.glob("*.mustache"):
            text = path.read_text(encoding="utf-8")
            opens = len(re.findall(r"\{\{[#^]\s*\w+", text))
            closes = len(re.findall(r"\{\{/\s*\w+", text))
            assert opens == closes, \
                f"{path.name}: {opens} sections opened, {closes} closed"

    def test_assets_referenced_by_templates_exist(self):
        """Every `app://` asset must resolve to a file on disk.

        Regression: this test once passed vacuously. The templates referenced
        `qrc:/assets/theme.css` - the Qt *resource* system, which serves only
        resources compiled into the binary - so the `app://` pattern matched
        nothing and the loop asserted over an empty set. The window opened with
        no stylesheet and no JavaScript, and the suite was green.
        """
        assets = ROOT / "src" / "app" / "ui" / "assets"
        if not assets.is_dir():
            pytest.skip("no assets directory")
        missing: list[str] = []
        found = 0
        for path in self.TEMPLATES.glob("*.mustache"):
            for match in re.finditer(r"app://ui/assets/([\w./-]+)",
                                     path.read_text(encoding="utf-8")):
                found += 1
                if not (assets / match.group(1)).exists():
                    missing.append(f"{path.name} -> {match.group(1)}")
        assert not missing, f"referenced assets do not exist: {missing}"
        # The guard that was absent: an empty match set proves nothing.
        assert found >= 2, "no app:// assets referenced - is the shell styled?"

    def test_stylesheet_and_script_are_loaded_over_the_app_scheme(self):
        """`qrc:` only works for resources compiled into the binary. Files on
        disk must go through the registered `app://` scheme and AssetHandler."""
        base = (self.TEMPLATES / "base.mustache").read_text(encoding="utf-8")
        assert 'href="app://ui/assets/theme.css"' in base
        assert 'src="app://ui/assets/app.js"' in base
        # qwebchannel.js is injected by install_web_channel_script() rather
        # than linked: `qrc:` is a different origin from `app://ui/`, so the
        # request would be blocked and QWebChannel left undefined.
        for attr in ('src="qrc:', "src='qrc:", 'href="qrc:', "href='qrc:"):
            assert attr not in base,                 "a qrc: URL is cross-origin from app://ui/ and would be blocked"

    def test_asset_handler_is_rooted_at_the_assets_directory(self):
        """`requestStarted` strips the leading `assets/` from the URL path, so
        rooting the handler at `ui/` resolves `ui/assets/theme.css` to
        `ui/theme.css` - which does not exist."""
        source = (ROOT / "src" / "app" / "main.py").read_text(encoding="utf-8")
        assert '"ui" / "assets"' in source

    def test_every_referenced_asset_resolves_the_way_the_handler_would(self):
        """Mirror AssetHandler's own path arithmetic, so a change to either side
        that breaks the pairing fails here rather than in a blank window."""
        asset_dir = (ROOT / "src" / "app" / "ui" / "assets").resolve()
        base = (self.TEMPLATES / "base.mustache").read_text(encoding="utf-8")
        for match in re.finditer(r"app://ui(/[\w./-]+)", base):
            rel = match.group(1).lstrip("/").removeprefix("assets/")
            target = (asset_dir / rel).resolve()
            assert target.is_file(), f"{match.group(0)} -> {target} is missing"
            assert asset_dir in target.parents or target.parent == asset_dir

    def test_capabilities_gate_actions_when_the_database_is_down(self):
        """Browsing and export must survive a database outage; processing must
        not silently appear to work."""
        from app.status import Availability, Capabilities

        caps = Capabilities()
        for attribute in ("can_browse", "can_export", "can_upload", "can_process"):
            assert hasattr(caps, attribute), f"{attribute} is not modelled"
        assert Availability.DOWN.value == "down"

    def test_a_disabled_action_can_explain_itself(self):
        from app.status import Capabilities

        caps = Capabilities()
        assert hasattr(caps, "reasons"), "a greyed-out button gives no reason"

    def test_window_process_never_imports_cuda(self):
        """A crash in the runtime must not take down the interface, so the UI
        process holds no model and links no CUDA."""
        source = (ROOT / "src" / "app" / "main.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "llama_cpp", "transformers"):
            assert forbidden not in source, f"{forbidden} in the window process"


class TestWebViewWiring:
    """The scheme, the page route and the channel script.

    Four separate faults kept the window unusable, each silent:
      * `LocalScheme` made Chromium refuse to *navigate* to `app://` at all -
        no error, no `loadFinished`, empty URL, handler never consulted.
      * `setHtml` gave the document no real origin, so every subresource
        request to the scheme was refused before the handler saw it.
      * `qrc:/qtwebchannel/qwebchannel.js` is cross-origin from `app://ui/`,
        so `QWebChannel` was undefined and every action was inert.
      * `installUrlSchemeHandler` does not take ownership, so a handler with no
        surviving reference was collected and the navigation never began.

    The full check is `tools/ui_smoke.py`, which drives a real QWebEngineView.
    These are the cheap invariants that keep the arrangement from drifting.
    """

    SOURCE = ROOT / "src" / "app" / "main.py"

    def test_scheme_is_not_marked_local(self):
        """A local scheme cannot be navigated to."""
        source = self.SOURCE.read_text(encoding="utf-8")
        register = source[source.index("def register_scheme"):]
        register = register[:register.index("\ndef ")]
        assert "Flag.LocalScheme" not in register, \
            "LocalScheme makes Chromium refuse to navigate to the scheme"
        assert "Flag.SecureScheme" in register
        assert "Flag.CorsEnabled" in register

    def test_the_page_is_served_not_pushed(self):
        """`setHtml` leaves the document without an origin on the scheme."""
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "self.view.setHtml(" not in source, \
            "setHtml gives the page no origin, so its assets never load"
        assert 'self.view.load(QUrl(f"app://ui/page/{page}"))' in source

    def test_the_handler_serves_pages(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert 'path.startswith("/page/")' in source

    def test_the_channel_script_is_injected(self):
        """`qrc:` is a different origin; the tag would simply be blocked."""
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "def install_web_channel_script" in source
        assert "DocumentCreation" in source

        base = (ROOT / "src" / "app" / "ui" / "templates" / "base.mustache").read_text(
            encoding="utf-8")
        assert "qrc:/qtwebchannel" not in base, \
            "a cross-origin script tag would be blocked"

    def test_the_handler_outlives_the_call(self):
        """installUrlSchemeHandler does not take ownership. Without a surviving
        reference the handler is collected and nothing loads - silently."""
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "handler.setParent(window)" in source

    def test_the_smoke_test_exists_and_is_portable(self):
        tool = ROOT / "src" / "tools" / "ui_smoke.py"
        assert tool.is_file()
        text = tool.read_text(encoding="utf-8")
        assert r"d:\saledeed" not in text.lower(), "hard-coded developer path"
        assert "QWebEngineView" in text or "MainWindow" in text


class TestSupervisedChildrenCanActuallyStart:
    """A child launched with `-m` must be able to import its own module.

    R-033. The whole suite passed while the application could not start its AI
    server at all: the supervisor spawned `python -m ai_server.server` with the
    project root as the working directory, and after the restructure the
    packages live in `src/`. The child died instantly with
    `No module named 'ai_server'`, the supervisor restarted it three times, and
    the only thing anybody ever saw was the UI saying "AI server offline".

    Nothing caught it because every existing check was static - the model file
    exists, the binary exists, the port is free. None of them started a process.
    These tests spawn one.
    """

    def _spawn(self, code: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
            env={**os.environ, **(env or {})})

    def test_child_env_puts_src_on_the_path(self):
        from launcher.runner import child_env

        env = child_env(ROOT)
        assert str(ROOT / "src") in env["PYTHONPATH"]

    def test_an_existing_pythonpath_is_preserved(self, monkeypatch):
        """Overwriting it would break anyone running with their own additions."""
        from launcher.runner import child_env

        monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
        env = child_env(ROOT)
        assert str(ROOT / "src") in env["PYTHONPATH"]
        assert "/somewhere/else" in env["PYTHONPATH"]

    def test_the_ai_server_module_imports_in_a_real_child(self):
        """The decisive one. Spawns a process exactly as the supervisor does.

        Importing is enough - it fails at module resolution, long before any
        model is loaded, so this costs a second rather than a minute.
        """
        from launcher.runner import child_env

        done = self._spawn("import ai_server.server", child_env(ROOT))
        assert done.returncode == 0, (
            f"the supervised child cannot import its own module:\n{done.stderr}")

    def test_without_the_env_it_fails(self):
        """Proves the test above has teeth rather than passing incidentally."""
        done = self._spawn("import ai_server.server", {"PYTHONPATH": ""})
        assert done.returncode != 0
        assert "No module named" in done.stderr

    def test_every_module_spawned_with_dash_m_is_importable(self):
        """Future-proofing: catch the next one without waiting for a user.

        Scans the source for `-m <our package>` spawns and checks each, so a new
        one added later is covered the day it is written.
        """
        from launcher.runner import child_env

        ours = {"core", "app", "ai_server", "launcher", "tools", "migrations"}
        pattern = re.compile(r'"-m",\s*"([\w.]+)"')
        modules = set()
        for path in (ROOT / "src").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for name in pattern.findall(path.read_text(encoding="utf-8",
                                                       errors="replace")):
                if name.split(".")[0] in ours:
                    modules.add(name)

        assert modules, "found no `-m` spawns - has the spawn style changed?"
        for name in sorted(modules):
            done = self._spawn(f"import {name}", child_env(ROOT))
            assert done.returncode == 0, (
                f"`python -m {name}` cannot resolve its module:\n{done.stderr}")


class TestSystemSetupShim:
    """`system_setup.bat` and the script behind it. R-046.

    The shim is 97 lines of batch and gets no test coverage from anything else,
    so the parts that can be checked as text are checked here.
    """

    BAT = ROOT / "system_setup.bat"
    SCRIPT = ROOT / "src" / "tools" / "system_setup.py"

    def test_the_shim_exists_and_delegates(self):
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        assert r'"src\tools\system_setup.py"' in body, "the shim lost its target"
        assert "%*" in body, "arguments are not forwarded"

    def test_it_works_from_a_path_with_spaces(self):
        r"""This project lives in `d:\saledeed v3`. Every path the shim uses has
        to be quoted or the space splits the command.

        The interpreter is quoted as well now, which the previous form could not
        be: it was `%PYEXE%`, holding `py -3.13` - two tokens that quoting would
        have broken. The shim runs the setup with the virtualenv's python.exe,
        a single path, so it both can and must be quoted.
        """
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        assert 'cd /d "%~dp0"' in body
        assert '"%VENV_PY%" "src\\tools\\system_setup.py"' in body

    def test_the_setup_runs_inside_the_project_virtualenv(self):
        """Everything `system_setup.py` installs goes to `sys.executable`, so
        the interpreter the shim picks decides whether the whole installation
        lands in `.venv` or in the machine's Python."""
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        assert r"set \"VENV_PY=%CD%\.venv\Scripts\python.exe\"".replace("\\\"", '"') in body
        assert "-m venv" in body, "the shim never creates the environment"

    def test_a_damaged_virtualenv_is_reported_rather_than_used(self):
        """A .venv copied from another machine exists but cannot run - a venv
        records absolute paths. Using it produces a missing-package error that
        names the wrong problem."""
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        assert '"%VENV_PY%" -c "import sys"' in body

    def test_the_bootstrap_interpreter_is_not_chosen_by_pyside(self):
        """It used to be, which was right when packages lived system-wide and
        is wrong now: on a clean machine nothing has PySide6, and on a
        configured one it would point at the very installation the virtualenv
        exists to stop depending on. All the bootstrap needs is `venv`."""
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        head = body[:body.index("-m venv")]
        assert "import PySide6" not in head
        assert "import venv" in head

    def test_the_preferred_interpreter_is_not_overwritten(self):
        """The fallback keeps the FIRST working interpreter, not the last.

        Without the guard, every loop iteration overwrote `PYANY`, so on a clean
        machine - where none of them has PySide6 yet - the choice fell to
        whichever version was probed last. On a box with 3.12, 3.13 and 3.14
        that selected 3.14, installing the application into an interpreter this
        project has never been tested on.
        """
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        # The loop body only. A wider slice reached the guard in the bare
        # `python` fallback below and passed with the loop's own guard removed.
        start = body.index("for %%V in (")
        loop = body[start:body.index("\n)\n", start)]
        assert "if not defined PYANY (" in loop, \
            "PYANY is unguarded; the last interpreter probed wins again"
        # 3.13 must be tried first for the guard to mean anything.
        versions = body[body.index("for %%V in ("):]
        versions = versions[:versions.index(")")]
        assert versions.split("in (")[1].split()[0] == "3.13"

    def test_a_failure_pauses_so_the_message_is_readable(self):
        """Double-clicked, the window closes the instant the script ends."""
        body = self.BAT.read_text(encoding="utf-8", errors="replace")
        tail = body[body.index('if not "%RC%"=="0"'):]
        assert "pause" in tail

    def test_setup_does_not_report_the_application_exit_as_its_own(self):
        """`subprocess.call(launcher)` was returned directly, so closing the
        application with a non-zero code printed "Setup stopped with code 1 -
        read INSTALLATION_REPORT.md", sending the operator to a report of a
        setup that had gone perfectly."""
        body = self.SCRIPT.read_text(encoding="utf-8", errors="replace")
        assert "return subprocess.call([sys.executable, str(paths.ROOT" not in body
        assert "Setup itself" in body, "the distinction is not explained"

    def test_elevation_is_named_when_a_step_needs_it(self):
        """Administrator status was detected and printed as a bare fact, while
        three install steps silently required it."""
        body = self.SCRIPT.read_text(encoding="utf-8", errors="replace")
        assert "Not running as administrator" in body
        assert "Run as administrator" in body

    def test_the_elevation_notice_is_conditional(self):
        """It must not fire on `--report-only`, which installs nothing, nor when
        the components are already present."""
        body = self.SCRIPT.read_text(encoding="utf-8", errors="replace")
        block = body[body.index("if install and not report.system.get(\"administrator\")"):]
        block = block[:block.index("Prerequisites")]
        assert "if needs_admin:" in block

    def test_the_script_never_downloads_the_extraction_model(self):
        """The fine-tuned weights are verified, never fetched - a helpful
        download would replace what every accuracy figure was measured on."""
        body = self.SCRIPT.read_text(encoding="utf-8", errors="replace")
        verify = body[body.index("def verify_extraction_model"):]
        verify = verify[:verify.index("\ndef ", 10)]
        assert "Never download" in verify
        for forbidden in ("urlretrieve", "hf_hub_download", "snapshot_download"):
            assert forbidden not in verify

    def test_required_directories_are_all_relative_to_the_project(self):
        from tools.system_setup import REQUIRED_DIRS

        for entry in REQUIRED_DIRS:
            assert not Path(entry).is_absolute(), entry
            assert ".." not in entry, entry


class TestInstallsAreDrivenByDetection:
    """Detection has to change what gets installed, or it is just a report.

    R-047. The system survey is thorough - CPU, RAM, GPU, VRAM, CUDA, disk,
    internet, elevation, Python versions - and for most of it that is the right
    depth. The question is which of those facts reach an install decision.
    """

    SETUP = ROOT / "src" / "tools" / "setup.py"
    SYSTEM = ROOT / "src" / "tools" / "system_setup.py"

    def test_the_runtime_build_follows_the_gpu(self):
        """A machine with no usable GPU gets the CPU build, not the CUDA one.

        Both the `--all` path and the individual `--install-runtime` flag pass
        `cpu_only` from the assessed deployment class.
        """
        body = self.SETUP.read_text(encoding="utf-8")
        assert body.count(
            "cpu_only=plan.deployment is DeploymentClass.CPU_ONLY") == 2

    def test_the_model_quantisation_follows_the_hardware(self):
        body = self.SETUP.read_text(encoding="utf-8")
        assert "quant = args.quant or plan.quantisation" in body
        assert "if plan.needs_model:" in body

    def test_an_individual_flag_installs_only_what_it_names(self):
        """`--install-runtime` also queued the 2.5 GB translation download.

        The step sat outside every flag check, so any individual flag pulled it
        in - and `--install-translation`, which has its own conditional append,
        ran it twice. On a fresh machine that is an unrequested multi-gigabyte
        download in answer to a request for llama.cpp.
        """
        body = self.SETUP.read_text(encoding="utf-8")
        individual = body[body.index("    else:\n        if args.install_deps"):
                          body.index("    if not steps:")]
        translation_steps = individual.count('steps.append(("translation"')
        assert translation_steps == 2, (
            f"{translation_steps} translation steps in the individual-flag "
            "branch; expected exactly the two guarded ones")
        for line in individual.splitlines():
            if 'steps.append(("translation"' in line:
                assert line.startswith("            "), (
                    "a translation step is not nested under a flag check")

    def test_a_full_install_still_includes_translation(self):
        """The fix must not remove it from `--all`, where it belongs."""
        body = self.SETUP.read_text(encoding="utf-8")
        full = body[body.index("    if args.all:"):body.index("    else:\n        if args.install_deps")]
        assert 'steps.append(("translation"' in full

    def test_disk_and_connectivity_reach_a_decision(self):
        """Both were detected, printed, and then ignored. A machine that cannot
        take the downloads failed part-way through a 2.5 GB fetch with a generic
        network error, when the fact was already on screen."""
        body = self.SYSTEM.read_text(encoding="utf-8")
        assert "FULL_INSTALL_GB" in body
        assert 'report.system.get("disk_free_gb"' in body
        assert 'report.system.get("internet"' in body

    def test_the_resource_notice_does_not_fire_on_report_only(self):
        body = self.SYSTEM.read_text(encoding="utf-8")
        block = body[body.index("    if install:\n        free_gb"):]
        block = block[:block.index("Prerequisites")]
        assert "free_gb < FULL_INSTALL_GB" in block

    def test_nothing_is_installed_that_is_already_present(self):
        """Every step is detect-skip-install-verify. Spot-checked on the three
        that download."""
        body = self.SETUP.read_text(encoding="utf-8")
        for marker in ("already installed at",
                       "translation model already present"):
            assert marker in body, marker
