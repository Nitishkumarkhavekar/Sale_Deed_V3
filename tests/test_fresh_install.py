"""What a clone actually needs before the application will run.

The failure this guards: a fresh clone had every line of source, passed setup,
passed preflight, and then produced nothing on the first deed - because the
extraction prompt lives under `models/` and was swept up by the rule that keeps
37 GB of checkpoints out of the repository. Absent at run time it reads as a
model problem, which is the most expensive place to look.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPT = ROOT / "models" / "saledeed main" / "prompt_v6_short.txt"


def _tracked(relative: str) -> bool:
    done = subprocess.run(["git", "ls-files", "--error-unmatch", relative],
                          cwd=ROOT, capture_output=True, text=True)
    return done.returncode == 0


class TestTheExtractionPromptShips:
    """It is 1.4 KB of application logic, not a weight."""

    def test_it_exists(self):
        assert PROMPT.is_file()

    def test_it_is_tracked_in_git(self):
        """Setup cannot download it and no substitute will do - the model was
        fine-tuned against this exact text. If it is not in the repository it
        is not on the next machine."""
        assert _tracked("models/saledeed main/prompt_v6_short.txt")

    def test_it_is_not_empty(self):
        assert PROMPT.stat().st_size > 100


class TestTheWeightsStayOut:
    """Narrowing the ignore rule must not let 37 GB in behind it."""

    @pytest.mark.parametrize("relative", [
        "models/AI server/gguf/deeds-v6_7-Q4_K_M.gguf",
        "models/AI server/gemma4b-text/config.json",
        "models/SuryaOCR/venv_new/pyvenv.cfg",
        "models/vllm-env/pyvenv.cfg",
    ])
    def test_large_assets_are_still_ignored(self, relative):
        done = subprocess.run(["git", "check-ignore", relative],
                              cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, f"{relative} is no longer ignored"

    def test_nothing_enormous_got_committed(self):
        """A tracked file over 5 MB means the ignore rule slipped."""
        listing = subprocess.run(["git", "ls-files", "-s", "models/"],
                                 cwd=ROOT, capture_output=True, text=True)
        offenders = []
        for line in listing.stdout.splitlines():
            name = line.split("	", 1)[-1].strip().strip('"')
            path = ROOT / name
            if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
                offenders.append((name, path.stat().st_size))
        assert not offenders, offenders


class TestSetupCatchesAMissingPrompt:
    """At install time, where it can be acted on - not at first extraction."""

    def test_absent_is_blocking(self, monkeypatch):
        from tools import system_setup as ss

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(ss.paths, "PROMPT_FILE",
                                pathlib.Path(tmp) / "absent.txt")
            report = ss.Report()
            ss.verify_prompt(report)

        step = report.steps[-1]
        assert step.status is ss.Status.FAILED
        assert step.blocking, "a missing prompt must stop the install"

    def test_empty_is_blocking(self, monkeypatch):
        """A truncated copy is a file of the right name and no use at all."""
        from tools import system_setup as ss

        with tempfile.TemporaryDirectory() as tmp:
            empty = pathlib.Path(tmp) / "prompt.txt"
            empty.write_text("", encoding="utf-8")
            monkeypatch.setattr(ss.paths, "PROMPT_FILE", empty)
            report = ss.Report()
            ss.verify_prompt(report)

        assert report.steps[-1].status is ss.Status.FAILED

    def test_the_real_one_passes(self, monkeypatch):
        from tools import system_setup as ss

        report = ss.Report()
        ss.verify_prompt(report)
        assert report.steps[-1].status is ss.Status.FOUND

    def test_the_remedy_says_it_is_not_downloadable(self, monkeypatch):
        """Setup deliberately never fetches a substitute - the remedy has to
        say so, or someone will go looking for a download."""
        from tools import system_setup as ss

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(ss.paths, "PROMPT_FILE",
                                pathlib.Path(tmp) / "absent.txt")
            report = ss.Report()
            ss.verify_prompt(report)

        assert "not downloadable" in report.steps[-1].remedy.lower()
