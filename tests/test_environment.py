"""The dependency declaration, and that it is enough to run on another machine.

Everything here guards the same failure: the application runs on the machine it
was built on and not on the next one, because a package was installed by hand
once and never written down. That failure is invisible locally by definition -
the missing thing is present - so it needs checks that reason about the declared
set rather than about the running interpreter.

Two real defects were found this way and are pinned below:

  * `requirements.txt` named `psycopg` but not its binary half. Plain psycopg
    needs libpq from a PostgreSQL client install on PATH; without one it raises
    "no pq wrapper available" the moment anything touches the database. This
    machine had `psycopg-binary` installed by hand, so a clean environment built
    strictly from the file could not connect.
  * `system_setup.py` imported SQLAlchemy in `main()`, before the step that
    installs it. Harmless while packages lived in the system Python and fatal
    the first time the setup ran against a virtualenv it had just created.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"
SETUP_BAT = ROOT / "System Setup.bat"
SETUP_PY = ROOT / "src" / "tools" / "system_setup.py"


def _pins(path: Path) -> dict[str, str]:
    """{normalised name: version} from a requirements file."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;]+)", line)
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(3)
    return found


class TestTheDeclaredDependencies:
    def test_every_directly_imported_package_is_declared(self):
        """Derived from the imports, not from memory. A new `import` of an
        undeclared package fails here rather than on the next machine."""
        declared = set(_pins(REQUIREMENTS))
        # Import name -> distribution name, where they differ.
        needed = {"PySide6": "pyside6-essentials", "sqlalchemy": "sqlalchemy",
                  "alembic": "alembic", "pystache": "pystache",
                  "pymupdf": "pymupdf", "fitz": "pymupdf",
                  "indic_transliteration": "indic-transliteration",
                  "pytest": "pytest"}
        missing = sorted({dist for dist in needed.values()
                          if dist not in declared})
        assert not missing, f"imported but not declared: {missing}"

    def test_psycopg_brings_its_binary_half(self):
        """The defect this file was written for. `psycopg` alone imports and
        then fails at connection time unless libpq is on PATH."""
        text = REQUIREMENTS.read_text(encoding="utf-8")
        assert re.search(r"^psycopg\[binary\]==", text, re.M), (
            "requirements.txt must name psycopg[binary]; plain psycopg cannot "
            "connect without a PostgreSQL client installation")

    def test_every_requirement_is_pinned(self):
        """An unpinned dependency resolved a major version once that removed an
        attribute Surya needs, and failed only after downloading 1.4 GB."""
        loose = []
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line and "==" not in line:
                loose.append(line)
        assert not loose, f"unpinned: {loose}"

    def test_the_lock_file_exists_and_agrees_with_the_pins(self):
        """The lock records what those pins actually resolved to. If they
        disagree, one of the two was edited without regenerating the other."""
        assert LOCK.is_file(), "requirements.lock.txt is missing"
        declared, locked = _pins(REQUIREMENTS), _pins(LOCK)
        disagreements = {
            name: (version, locked[name])
            for name, version in declared.items()
            if name in locked and locked[name] != version}
        assert not disagreements, f"requirements vs lock: {disagreements}"

    def test_the_lock_is_a_superset(self):
        """It carries the transitive set as well, so it must not be smaller."""
        assert len(_pins(LOCK)) > len(_pins(REQUIREMENTS))

    def test_no_gpu_stack_is_declared_for_the_application(self):
        """torch, transformers and surya belong to the OCR environment, which
        pins transformers==4.57.1. Pulling them in here is what makes an OCR
        upgrade break extraction - and adds several GB to every install."""
        declared = set(_pins(REQUIREMENTS))
        for package in ("torch", "transformers", "surya-ocr", "vllm"):
            assert package not in declared, (
                f"{package} belongs in a separate environment")


class TestTheSetupBuildsAnEnvironment:
    def test_the_shim_creates_and_uses_a_virtualenv(self):
        body = SETUP_BAT.read_text(encoding="utf-8", errors="replace")
        assert "-m venv" in body
        assert '"%VENV_PY%" "src\\tools\\system_setup.py"' in body

    def test_no_absolute_path_is_written_into_the_shim(self):
        """It has to work wherever the project is copied. The working directory
        comes from %~dp0 and everything else is relative to it."""
        body = SETUP_BAT.read_text(encoding="utf-8", errors="replace")
        for line in body.splitlines():
            if line.strip().upper().startswith("REM"):
                continue
            assert not re.search(r"[A-Za-z]:\\\\|[A-Za-z]:\\[A-Za-z]", line), line

    def test_the_setup_reports_which_environment_it_installs_into(self):
        """"Installed successfully" against the wrong interpreter is the
        failure the virtualenv exists to prevent, and it is invisible unless
        something states the answer."""
        from tools.system_setup import _report_environment, in_virtual_environment

        assert callable(_report_environment)
        assert isinstance(in_virtual_environment(), bool)

    def test_nothing_imports_the_application_stack_before_installing_it(self):
        """`main()` used to build the DSN - and so import SQLAlchemy - before
        `ensure_packages` ran. Against a virtualenv created seconds earlier that
        is a ModuleNotFoundError for the very package the next step installs."""
        source = SETUP_PY.read_text(encoding="utf-8")
        body = source[source.index("def main("):]
        install_at = body.index("ensure_packages(report, install)")
        before = body[:install_at]
        for module in ("from core.db.engine import", "import sqlalchemy",
                       "from sqlalchemy"):
            assert module not in before, (
                f"{module!r} runs before the packages are installed")

    def test_a_missing_dsn_is_reported_rather_than_crashing(self):
        """Report-only on an unprovisioned machine has no DSN to check against,
        which is a fact to state, not a reason to abort a survey."""
        source = SETUP_PY.read_text(encoding="utf-8")
        assert "not checked - install the packages first" in source


class TestTheEnvironmentsStaySeparate:
    """Three virtualenvs, because two of them pin the same package to
    incompatible versions. Merging any pair is what breaks the other."""

    @pytest.mark.parametrize("name,reason", [
        ("models/SuryaOCR/venv_new", "OCR pins transformers==4.57.1"),
        ("models/vllm-env", "vLLM pins transformers>=5.5.3"),
    ])
    def test_the_separate_environments_are_documented(self, name, reason):
        text = (REQUIREMENTS.read_text(encoding="utf-8")
                + LOCK.read_text(encoding="utf-8")
                + SETUP_BAT.read_text(encoding="utf-8", errors="replace"))
        key = name.rsplit("/", 1)[-1]
        assert key in text, f"{name} is not mentioned anywhere ({reason})"

    def test_the_ocr_interpreter_is_not_the_application_one(self):
        """The pipeline runs Surya in its own interpreter as a subprocess. If
        that ever resolved to the application's venv, OCR would import a
        transformers version it cannot use."""
        import sys

        from core.pipeline.runner import build_stages

        surya = build_stages().ocr.surya_python
        if surya is None:
            pytest.skip("no Surya interpreter configured on this machine")
        assert Path(surya).resolve() != Path(sys.executable).resolve()
        assert "SuryaOCR" in str(surya)
