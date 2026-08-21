"""The OCR environment, and rebuilding it on the machine that will run it.

A virtualenv records the absolute paths it was created at. Copying the project
folder to another machine therefore produces the one failure everybody hits and
nobody can diagnose: the interpreter starts, and then cannot import `surya`.
Setup used to report that and stop. It now rebuilds, because on a machine set up
by copying this is the normal state rather than an exotic fault.

Nothing here performs a real install - the rebuild downloads about 2.5 GB. The
subprocess layer is replaced so the decisions can be checked without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OCR_REQS = ROOT / "requirements-ocr.txt"


class TestTheOcrEnvironmentIsDeclared:
    """The pins have to be version-controlled. The environment is not: it is
    gitignored, and it cannot be copied even if it were committed."""

    def test_the_requirements_file_ships(self):
        assert OCR_REQS.is_file()

    def test_it_pins_the_pair_that_has_to_agree(self):
        """Surya 0.17.1 reads attributes a later transformers removed."""
        text = OCR_REQS.read_text(encoding="utf-8")
        assert "surya-ocr==0.17.1" in text
        assert "transformers==4.57.1" in text

    def test_it_can_reach_cuda_wheels(self):
        text = OCR_REQS.read_text(encoding="utf-8")
        assert "download.pytorch.org" in text

    def test_it_is_not_merged_into_the_application_requirements(self):
        """Merging them is what makes an OCR upgrade break extraction."""
        lines = [line.split("#")[0].strip()
                 for line in (ROOT / "requirements.txt").read_text(
                     encoding="utf-8").splitlines()]
        installed = [line for line in lines if line]
        assert not [line for line in installed if "surya" in line.lower()], installed


class TestReportOnlyChangesNothing:
    def test_a_broken_environment_is_reported_not_rebuilt(self, tmp_path, monkeypatch):
        from tools import system_setup as ss

        fake_venv = tmp_path / "venv_new"
        (fake_venv / "Scripts").mkdir(parents=True)
        (fake_venv / "Scripts" / "python.exe").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(ss.paths, "SURYA_DIR", tmp_path)

        called = []
        monkeypatch.setattr(ss, "_rebuild_surya",
                            lambda *a, **k: called.append(True))
        monkeypatch.setattr(ss, "_run",
                            lambda *a, **k: (1, "ModuleNotFoundError: surya"))

        report = ss.Report()
        ss.ensure_ocr(report, install=False)

        assert not called, "--report-only must not rebuild anything"
        step = report.steps[-1]
        assert step.status is ss.Status.MISSING
        assert "rebuilt" in step.remedy.lower()


class TestRebuildIsTriggered:
    def test_an_unimportable_surya_is_rebuilt_during_an_install(
            self, tmp_path, monkeypatch):
        from tools import system_setup as ss

        fake_venv = tmp_path / "venv_new"
        (fake_venv / "Scripts").mkdir(parents=True)
        (fake_venv / "Scripts" / "python.exe").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(ss.paths, "SURYA_DIR", tmp_path)
        monkeypatch.setattr(ss, "_run",
                            lambda *a, **k: (1, "ModuleNotFoundError: surya"))

        reasons = []
        monkeypatch.setattr(ss, "_rebuild_surya",
                            lambda report, started, why: reasons.append(why))

        ss.ensure_ocr(ss.Report(), install=True)
        assert reasons == ["surya would not import"]

    def test_a_missing_interpreter_is_built_during_an_install(
            self, tmp_path, monkeypatch):
        from tools import system_setup as ss

        monkeypatch.setattr(ss.paths, "SURYA_DIR", tmp_path)
        reasons = []
        monkeypatch.setattr(ss, "_rebuild_surya",
                            lambda report, started, why: reasons.append(why))

        ss.ensure_ocr(ss.Report(), install=True)
        assert reasons == ["no interpreter"]


class TestRebuildRefusesRatherThanDestroys:
    """Whatever else it does, it must not delete a working environment it
    cannot then replace."""

    def test_it_stops_before_deleting_when_the_pins_are_missing(
            self, tmp_path, monkeypatch):
        from tools import system_setup as ss

        victim = tmp_path / "venv_new"
        victim.mkdir()
        (victim / "marker.txt").write_text("still here", encoding="utf-8")

        monkeypatch.setattr(ss, "SURYA_VENV", victim)
        monkeypatch.setattr(ss, "OCR_REQUIREMENTS", tmp_path / "absent.txt")

        report = ss.Report()
        ss._rebuild_surya(report, 0.0, "test")

        assert (victim / "marker.txt").is_file(), "deleted an environment it could not rebuild"
        assert report.steps[-1].status is ss.Status.MISSING

    def test_it_stops_before_deleting_when_python_312_is_absent(
            self, tmp_path, monkeypatch):
        from tools import system_setup as ss

        victim = tmp_path / "venv_new"
        victim.mkdir()
        (victim / "marker.txt").write_text("still here", encoding="utf-8")
        reqs = tmp_path / "requirements-ocr.txt"
        reqs.write_text("surya-ocr==0.17.1", encoding="utf-8")

        monkeypatch.setattr(ss, "SURYA_VENV", victim)
        monkeypatch.setattr(ss, "OCR_REQUIREMENTS", reqs)
        monkeypatch.setattr(ss, "_run", lambda *a, **k: (1, "not found: py"))

        report = ss.Report()
        ss._rebuild_surya(report, 0.0, "test")

        assert (victim / "marker.txt").is_file()
        assert "3.12" in report.steps[-1].detail + report.steps[-1].remedy


class TestTheBatchFileRebuildsItsOwnEnvironment:
    BAT = ROOT / "System Setup.bat"

    def test_it_rebuilds_rather_than_telling_the_operator_to(self):
        """"Delete the .venv folder and run this file again" is a step that
        gets skipped, and the folder is hidden."""
        body = self.BAT.read_text(encoding="utf-8")
        assert "Rebuilding it for this one" in body
        assert "rmdir /s /q" in body

    def test_it_verifies_the_rebuilt_environment_before_continuing(self):
        body = self.BAT.read_text(encoding="utf-8")
        after = body.split("Rebuilding it for this one", 1)[1]
        assert "-m venv" in after
        assert "import sys" in after, "must confirm the new one actually runs"

    def test_it_stops_if_the_old_environment_cannot_be_removed(self):
        """A locked .venv must not be followed by a half-built one."""
        body = self.BAT.read_text(encoding="utf-8")
        after = body.split("Rebuilding it for this one", 1)[1]
        assert "could not be removed" in after
