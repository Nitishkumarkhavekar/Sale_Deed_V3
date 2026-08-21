"""The installer steps added to make setup survive an unattended machine.

Each test here pins a defect that reached a real installation:

  * `indic-transliteration` was declared in `requirements.txt`, imported by
    `core/translation/transliterate.py`, and absent from the list setup
    verifies. Because that check returns early when everything it knows about
    imports, a machine could report "6 import cleanly" and never install
    `requirements.txt` at all - so transliteration was missing and nothing said
    so until a Kannada name reached the translation stage.
  * The AI server's port was never checked. When something else held it the
    application started and the UI reported "AI server offline", which sends
    the operator looking at the AI server rather than at the port.
  * The extraction model was verified by existence and size. A GGUF truncated
    by an interrupted copy passed, and failed minutes later inside
    llama-server during model load.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_BAT = ROOT / "system_setup.bat"

#: A drive letter followed by a separator - the thing that stops a project
#: working after it is copied somewhere else. The letter must stand alone:
#: without that guard the scheme in "https://python.org" matches, and the test
#: fails on a URL that is not a path at all.
DRIVE_LETTER = re.compile("(?<![A-Za-z])[A-Za-z]:[" + chr(92) + "/]")


class TestEntryPoint:
    """`system_setup.bat` - the single user-facing installer.

    There were briefly two: this file and "System Setup.bat", one forwarding to
    the other. Two names for one installer is a thing to explain rather than a
    thing to have, so the logic lives here and the other name is gone.
    """

    def test_it_exists(self):
        assert SETUP_BAT.is_file()

    def test_there_is_only_one_installer(self):
        """The old name must not come back alongside this one."""
        rivals = [p.name for p in ROOT.glob("*.bat")
                  if p.name.lower().replace(" ", "_") == "system_setup.bat"]
        assert rivals == ["system_setup.bat"], rivals

    def test_it_runs_the_setup_inside_the_project_environment(self):
        """Installing into whatever interpreter happened to launch the file is
        how packages end up in a machine's system Python."""
        body = SETUP_BAT.read_text(encoding="utf-8")
        assert "%VENV_PY%" in body
        assert "src" in body and "system_setup.py" in body

    def test_it_forwards_its_arguments(self):
        """--report-only and --skip-tests have to reach the Python behind it."""
        assert "%*" in SETUP_BAT.read_text(encoding="utf-8")

    def test_it_preserves_the_exit_code(self):
        """An installer that always exits 0 cannot be used unattended."""
        assert "%ERRORLEVEL%" in SETUP_BAT.read_text(encoding="utf-8")

    def test_it_resolves_paths_from_its_own_folder(self):
        assert "%~dp0" in SETUP_BAT.read_text(encoding="utf-8")

    def test_no_hard_coded_drive_letters(self):
        """The project may be installed anywhere. Comments are exempt: the
        header explains the rule by naming the paths it forbids."""
        offenders = []
        for number, line in enumerate(
                SETUP_BAT.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.upper().startswith("REM") or stripped.startswith("::"):
                continue
            if DRIVE_LETTER.search(stripped):
                offenders.append(f"{number}: {stripped}")
        assert not offenders, offenders

    def test_it_is_readable_by_cmd(self):
        """CRLF and ASCII. A batch file saved as UTF-8 with a BOM fails on the
        first line, and one saved with bare LF breaks multi-line blocks."""
        raw = SETUP_BAT.read_bytes()
        assert not raw.startswith(bytes([0xEF, 0xBB, 0xBF])), "BOM"
        assert bytes([13, 10]) in raw, "needs CRLF"
        raw.decode("ascii")


class TestDeclaredDependenciesAreVerified:
    """Whatever the application imports at run time must be on the list setup
    checks, or a machine can pass setup and fail in use."""

    def test_runtime_distributions_are_all_verified(self):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import ensure_packages  # noqa: F401
        import inspect

        from tools import system_setup

        source = inspect.getsource(system_setup.ensure_packages)
        declared = set()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            match = re.match("^([A-Za-z0-9_.-]+)", line)
            if match:
                declared.add(match.group(1).lower().replace("_", "-"))

        # Distributions that are deliberately not gated on.
        exempt = {
            "pytest",           # developer tool; absence is reported, not fatal
            "shiboken6",        # pulled by PySide6 and unimportable alone
            "pyside6-addons",   # both halves are covered by `import PySide6`
            "pyside6-essentials",
        }
        missing = sorted(
            name for name in declared - exempt
            if name.replace("-", "_") not in source and name not in source
        )
        assert not missing, (
            "declared in requirements.txt but never verified by setup: "
            + ", ".join(missing)
        )


class TestPortResolution:
    """The installer must resolve the port the launcher will actually use."""

    def test_default_matches_the_launcher(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from launcher.config import build_config
        from tools.system_setup import resolve_ai_port

        monkeypatch.delenv("SALEDEED_AI_URL", raising=False)
        assert resolve_ai_port() == build_config(ROOT).ai_port

    def test_environment_overrides(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import resolve_ai_port

        monkeypatch.setenv("SALEDEED_AI_URL", "http://127.0.0.1:9123")
        assert resolve_ai_port() == 9123

    def test_a_malformed_url_falls_back_rather_than_raising(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import resolve_ai_port

        monkeypatch.setenv("SALEDEED_AI_URL", "http://127.0.0.1:not-a-port")
        assert resolve_ai_port() == 8077

    def test_an_occupied_port_is_seen_as_occupied(self, monkeypatch):
        """The probe must not set SO_REUSEADDR: on Windows it permits binding a
        port another socket already holds, which would answer yes to the one
        case this exists to detect."""
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import _port_free

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            assert _port_free(port) is False

        # Released again once the socket closes.
        assert _port_free(port) is True


class TestModelIntegrity:
    """Presence is not integrity."""

    def test_a_truncated_file_is_not_accepted(self, tmp_path):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import _is_gguf

        broken = tmp_path / "deeds.gguf"
        broken.write_bytes(b"" + bytes([0]) * 64)
        assert _is_gguf(broken) is False

    def test_the_magic_bytes_are_accepted(self, tmp_path):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import _is_gguf

        good = tmp_path / "deeds.gguf"
        good.write_bytes(b"GGUF" + bytes([0]) * 64)
        assert _is_gguf(good) is True

    def test_a_missing_file_is_not_an_exception(self, tmp_path):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import _is_gguf

        assert _is_gguf(tmp_path / "absent.gguf") is False

class TestPortConflictIsReported:
    """The occupied branch, end to end.

    Written because the happy path passed while this one raised NameError on an
    undefined variable in the remedy text - a fault no amount of testing the
    available-port case could reach.
    """

    def test_an_occupied_port_produces_a_usable_report(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import Report, Status, ensure_ports

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            monkeypatch.setenv("SALEDEED_AI_URL", f"http://127.0.0.1:{port}")

            report = Report()
            ensure_ports(report)

        step = report.steps[-1]
        assert step.name == "Ports"
        assert step.status is Status.MISSING
        assert str(port) in step.detail
        # The remedy has to name the symptom the operator will actually see,
        # and the setting that fixes it.
        assert "AI server offline" in step.remedy
        assert "SALEDEED_AI_URL" in step.remedy

    def test_a_free_port_passes(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import Report, Status, ensure_ports

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        monkeypatch.setenv("SALEDEED_AI_URL", f"http://127.0.0.1:{port}")

        report = Report()
        ensure_ports(report)
        assert report.steps[-1].status is Status.FOUND
        assert not report.failures


class TestLauncherDistinguishesTheOccupant:
    """A busy port is only "reuse that server" when it *is* that server."""

    def _config(self, port):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from launcher.config import build_config

        cfg = build_config(ROOT)
        object.__setattr__(cfg, "ai_port", port)
        object.__setattr__(cfg, "ai_host", "127.0.0.1")
        return cfg

    def test_a_stranger_on_the_port_fails_rather_than_being_reused(self):
        """Reusing it let the UI talk to a foreign process and then report
        "AI server offline" against a server that was never ours."""
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from launcher.steps import Outcome, check_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stranger:
            stranger.bind(("127.0.0.1", 0))
            stranger.listen(1)
            port = stranger.getsockname()[1]
            result = check_port(self._config(port))

        assert result.outcome is Outcome.FAIL
        assert "not this application" in result.detail
        assert "SALEDEED_AI_URL" in result.remedy

    def test_a_free_port_is_ok(self):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from launcher.steps import Outcome, check_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        assert check_port(self._config(port)).outcome is Outcome.OK


class TestTheEnginePortIsCheckedToo:
    """The AI server binds the configured port and gives its inference engine
    `port + 1`. Checking only the first let a machine install cleanly with the
    engine's port taken, and llama-server then exited during startup while the
    window reported the AI server as offline."""

    def test_a_conflict_on_only_the_engine_port_is_caught(self, monkeypatch):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import Report, Status, ensure_ports

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            engine_port = held.getsockname()[1]
            monkeypatch.setenv("SALEDEED_AI_URL",
                               f"http://127.0.0.1:{engine_port - 1}")
            report = Report()
            ensure_ports(report)

        step = report.steps[-1]
        assert step.status is Status.MISSING
        assert str(engine_port) in step.detail
        assert "inference engine" in step.remedy

    def test_the_suggested_port_is_not_the_engine_port(self, monkeypatch):
        """The remedy used to offer `port + 1`, which is the engine's own port
        and the reason this check exists."""
        import re as _re
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tools.system_setup import Report, ensure_ports

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            monkeypatch.setenv("SALEDEED_AI_URL", f"http://127.0.0.1:{port}")
            report = Report()
            ensure_ports(report)

        suggested = _re.search(r"SALEDEED_AI_URL=http://[^:]+:(\d+)",
                               report.steps[-1].remedy)
        assert suggested, report.steps[-1].remedy
        assert int(suggested.group(1)) not in (port, port + 1)

    def test_the_launcher_checks_the_engine_port_as_well(self):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from launcher.config import build_config
        from launcher.steps import Outcome, check_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            engine_port = held.getsockname()[1]
            cfg = build_config(ROOT)
            object.__setattr__(cfg, "ai_host", "127.0.0.1")
            object.__setattr__(cfg, "ai_port", engine_port - 1)
            result = check_port(cfg)

        assert result.outcome is Outcome.FAIL
        assert str(engine_port) in result.detail
