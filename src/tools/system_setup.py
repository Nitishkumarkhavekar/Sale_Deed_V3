"""One-command system setup: detect, install what is missing, verify, launch.

    "System Setup.bat"                 everything, then start the application
    py -3.13 tools/system_setup.py     the same, directly
    ... --report-only                  detect and report, change nothing
    ... --no-launch                    set up but do not start

Every step follows the same contract: **detect, skip if present, install if
missing, verify afterwards** - and say which of those three happened. A step
that cannot report whether it changed anything is impossible to trust on a
second run.

Design notes worth keeping:

**Nothing pre-existing is ever removed.** If PostgreSQL was on this machine
before the installer ran, a failure here must not uninstall it. The installer
owns only what it created.

**Failure is contained, not fatal.** A missing optional component should not
stop the rest of the setup; the report says what is missing and what that costs.
Only genuinely blocking failures stop the run.

**Idempotent.** Running it twice is safe and skips completed work. That is not a
nicety - the realistic use is running it again after fixing one problem.

**It orchestrates; it does not reimplement.** `tools/setup.py`,
`tools/db_setup.py` and `launcher.py` already install the runtime, the database
and the models, and already validate them. This adds the parts that were
missing - system detection, a Python environment, reports - and calls the rest.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `alembic.ini` sits beside the five top-level folders, not inside `src`, and
#: its `script_location` is written relative to itself. Alembic must therefore be
#: launched from the project root - run from `src` it reports "No 'script_location'
#: key found in configuration" and exits non-zero, which is exactly what every
#: --upgrade path here did until this was noticed while adding a migration.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core import paths

LOG_DIR = paths.LOG_DIR / "setup"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: Rough size of a full install: llama.cpp with its CUDA DLLs (~0.4 GB),
#: the translation model (~2.5 GB), Python packages, PostgreSQL, and room
#: for the database and cleaned copies. Used to warn, never to refuse -
#: a machine with everything already present needs none of it.
FULL_INSTALL_GB = 8

#: Directories the application expects. Created if absent, never cleaned - they
#: hold user data.
REQUIRED_DIRS = (
    "runtime/logs", "runtime/logs/setup", "runtime/uploads", "runtime/exports",
    "runtime/backups", "runtime/cache", "runtime/temp", "runtime/config",
    "runtime/data", "runtime/data/cleaned", "runtime/data/exports",
    "models/AI server", "models/AI server/gguf", "models/AI server/translator",
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Status:
    FOUND = "found"        # already present, nothing done
    INSTALLED = "installed"  # this run installed it
    MISSING = "missing"    # absent and not installable here
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    remedy: str = ""

    @property
    def blocking(self) -> bool:
        return self.status is Status.FAILED


@dataclass
class Report:
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    system: dict = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        _emit(step)
        return step

    @property
    def failures(self) -> list[Step]:
        return [s for s in self.steps if s.status is Status.FAILED]

    @property
    def missing(self) -> list[Step]:
        return [s for s in self.steps if s.status is Status.MISSING]


_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_MARK = {
    Status.FOUND: ("[ ok ]", "\033[32m"),
    Status.INSTALLED: ("[ new]", "\033[36m"),
    Status.MISSING: ("[miss]", "\033[33m"),
    Status.SKIPPED: ("[skip]", "\033[90m"),
    Status.FAILED: ("[FAIL]", "\033[31m"),
}


def _emit(step: Step) -> None:
    mark, colour = _MARK.get(step.status, ("[ ?? ]", ""))
    if _COLOUR and colour:
        mark = f"{colour}{mark}\033[0m"
    timing = f"{step.seconds:>6.1f}s" if step.seconds >= 0.1 else "       "
    print(f"  {mark} {step.name:<26} {step.detail}   {timing}", flush=True)
    if step.remedy and step.status in (Status.FAILED, Status.MISSING):
        for line in step.remedy.splitlines():
            print(f"          -> {line}", flush=True)
    _log(f"{step.status:<10} {step.name:<26} {step.detail}")


def _log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (LOG_DIR / "installation.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}  {message}\n")


def _run(command: list[str], timeout: float = 1800.0,
         cwd: Path | None = None) -> tuple[int, str]:
    """Run a command, capture everything, never raise."""
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, check=False, cwd=cwd,
                              creationflags=NO_WINDOW)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 1. System detection
# ---------------------------------------------------------------------------


def detect_system() -> dict:
    """Everything worth knowing about this machine, for the report."""
    info: dict = {
        "windows": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "64bit": sys.maxsize > 2 ** 32,
        "hostname": platform.node(),
        "python_running": sys.version.split()[0],
    }

    try:
        info["administrator"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        info["administrator"] = False

    # Hardware detection already exists and is NVIDIA-aware, pinned by UUID.
    try:
        from ai_server.hardware import detect

        hw = detect()
        info.update({
            "cpu": hw.cpu_name,
            "cores_physical": hw.physical_cores,
            "cores_logical": hw.logical_cores,
            "ram_total_gb": round(hw.ram_total_gib, 1),
            "ram_available_gb": round(hw.ram_available_gib, 1),
            "cuda_available": hw.cuda_available,
            "cuda_version": hw.cuda_version,
            "driver_version": hw.driver_version,
        })
        gpu = hw.primary_gpu
        info["gpu"] = f"{gpu.name} ({gpu.total_gib:.1f} GiB)" if gpu else "none"
        info["gpu_vram_gb"] = round(gpu.total_gib, 1) if gpu else 0
        info["excluded_adapters"] = hw.other_adapters
    except Exception as exc:  # noqa: BLE001
        info["hardware_error"] = f"{type(exc).__name__}: {exc}"

    usage = shutil.disk_usage(ROOT)
    info["disk_free_gb"] = round(usage.free / 1024 ** 3, 1)
    info["disk_total_gb"] = round(usage.total / 1024 ** 3, 1)

    # Internet reachability, bounded so a firewalled machine does not hang the
    # installer for the OS default of tens of seconds.
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=3):
            info["internet"] = True
    except OSError:
        info["internet"] = False

    code, out = _run(["git", "--version"], timeout=20)
    info["git"] = out.strip().splitlines()[0] if code == 0 and out.strip() else "not found"

    # PostgreSQL by *reachability*, not by whether `psql` is on PATH. The
    # standard Windows installer does not add it, so a PATH check reports "not
    # found" on a machine where the database is running and serving queries -
    # which is exactly what it did here.
    info["postgresql"] = _postgres_status()

    info["python_versions"] = _installed_pythons()
    return info


def _postgres_status() -> str:
    """Whether the database answers, and the service state behind it."""
    try:
        from core.db.engine import build_engine, check_connection

        ok, detail = check_connection(build_engine(connect_timeout_s=4))
        if ok:
            return detail.split(",")[0].strip()
    except Exception:  # noqa: BLE001
        pass
    code, out = _run(["sc", "query", "state=", "all"], timeout=20)
    if code == 0 and "postgresql" in out.lower():
        return "service installed, not reachable"
    return "not found"


def _installed_pythons() -> dict[str, str]:
    """Registered interpreters and whether each can run the application."""
    found: dict[str, str] = {}
    for version in ("3.14", "3.13", "3.12"):
        code, _ = _run(["py", f"-{version}", "-c", "import sys"], timeout=25)
        if code != 0:
            continue
        ok, _ = _run(["py", f"-{version}", "-c", "import PySide6"], timeout=40)
        found[version] = "can run the application" if ok == 0 else "present"
    return found


def print_system(info: dict) -> None:
    print("\n  System")
    print("  " + "-" * 66)
    rows = (
        ("Windows", info.get("windows")),
        ("Architecture", f"{info.get('architecture')} "
                         f"{'64-bit' if info.get('64bit') else '32-bit'}"),
        ("Administrator", "yes" if info.get("administrator") else
                          "no - some steps need elevation"),
        ("CPU", f"{info.get('cpu', '?')} "
                f"({info.get('cores_physical', '?')} cores / "
                f"{info.get('cores_logical', '?')} threads)"),
        ("RAM", f"{info.get('ram_available_gb', '?')} GB free of "
                f"{info.get('ram_total_gb', '?')} GB"),
        ("Disk", f"{info.get('disk_free_gb', '?')} GB free of "
                 f"{info.get('disk_total_gb', '?')} GB"),
        ("GPU", info.get("gpu", "unknown")),
        ("CUDA", info.get("cuda_version") or "not available"),
        ("Driver", info.get("driver_version") or "-"),
        ("Internet", "reachable" if info.get("internet") else "unreachable"),
        ("Git", info.get("git")),
        ("PostgreSQL", info.get("postgresql")),
        ("Python", ", ".join(f"{v} ({d})" for v, d
                             in info.get("python_versions", {}).items()) or "none"),
    )
    for label, value in rows:
        print(f"    {label:<16} {value}")
    if info.get("excluded_adapters"):
        print(f"    {'Excluded':<16} {', '.join(info['excluded_adapters'])} "
              "(NVIDIA only, by design)")


# ---------------------------------------------------------------------------
# 2. Prerequisites
# ---------------------------------------------------------------------------


#: Failure text that means "try again", not "this will never work". A refused
#: connection or a truncated download is worth one more attempt; a package that
#: does not exist is not, and retrying it just makes the operator wait twice.
_TRANSIENT = ("network", "timed out", "timeout", "connection", "temporarily",
              "0x8a15", "could not be resolved", "downloading", "hash",
              "server", "retry")


def _looks_transient(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _TRANSIENT)


def _winget_install(package: str, label: str,
                    attempts: int = 2) -> tuple[bool, str]:
    """Install through winget, retrying a transient failure once.

    An installer that gives up on a dropped connection sends the operator back
    to the start of a twenty-minute run for a fault that would have cleared on
    its own. Only failures that read as transient are retried: a wrong package
    id fails identically the second time and retrying it wastes their time.
    """
    if not shutil.which("winget"):
        return False, "winget is unavailable on this machine"

    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        code, out = _run(["winget", "install", "-e", "--id", package,
                          "--accept-source-agreements",
                          "--accept-package-agreements", "--silent"],
                         timeout=2400)
        if code == 0:
            suffix = "" if attempt == 1 else f" (attempt {attempt})"
            return True, f"{label} installed{suffix}"

        last = (out.strip().splitlines()[-1] if out.strip() else f"exit {code}")
        if attempt >= attempts or not _looks_transient(out):
            break
        _log(f"{label}: transient failure, retrying - {last[:90]}")
        print(f"          retrying {label} after a transient failure ...",
              flush=True)
        time.sleep(5)

    return False, last[:90]


def ensure_python_312(report: Report, install: bool) -> None:
    """Python 3.12 hosts OCR and translation.

    A second interpreter is deliberate: Surya pins `transformers==4.57.1`
    against the rest of the project.
    """
    started = time.monotonic()
    code, _ = _run(["py", "-3.12", "-c", "import sys"], timeout=25)
    if code == 0:
        report.add(Step("Python 3.12 (OCR)", Status.FOUND, "present",
                        time.monotonic() - started))
        return
    if not install:
        report.add(Step("Python 3.12 (OCR)", Status.MISSING,
                        "needed for OCR and translation",
                        time.monotonic() - started,
                        "winget install Python.Python.3.12"))
        return
    ok, detail = _winget_install("Python.Python.3.12", "Python 3.12")
    report.add(Step("Python 3.12 (OCR)",
                    Status.INSTALLED if ok else Status.MISSING, detail,
                    time.monotonic() - started,
                    "" if ok else "winget install Python.Python.3.12"))


def ensure_vcredist(report: Report, install: bool) -> None:
    """The C++ runtime PyMuPDF and PySide6 link against."""
    started = time.monotonic()
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    if (system32 / "vcruntime140.dll").is_file():
        report.add(Step("Visual C++ runtime", Status.FOUND, "present",
                        time.monotonic() - started))
        return
    if not install:
        report.add(Step("Visual C++ runtime", Status.MISSING, "absent",
                        time.monotonic() - started,
                        "winget install Microsoft.VCRedist.2015+.x64"))
        return
    ok, detail = _winget_install("Microsoft.VCRedist.2015+.x64", "VC++ runtime")
    report.add(Step("Visual C++ runtime",
                    Status.INSTALLED if ok else Status.MISSING, detail,
                    time.monotonic() - started))


def ensure_directories(report: Report, install: bool) -> None:
    started = time.monotonic()
    missing = [rel for rel in REQUIRED_DIRS if not (paths.ROOT / rel).exists()]
    if not install:
        report.add(Step("Folders",
                        Status.MISSING if missing else Status.FOUND,
                        f"{len(missing)} would be created" if missing
                        else f"{len(REQUIRED_DIRS)} present",
                        time.monotonic() - started))
        return
    created = []
    for relative in REQUIRED_DIRS:
        target = paths.ROOT / relative
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            created.append(relative)
    report.add(Step(
        "Folders",
        Status.INSTALLED if created else Status.FOUND,
        f"{len(created)} created" if created else f"{len(REQUIRED_DIRS)} present",
        time.monotonic() - started))


# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------


def in_virtual_environment() -> bool:
    """True when the running interpreter is a virtual environment.

    `base_prefix` differs from `prefix` inside a venv and matches outside it.
    This is the documented test and it works for `venv` and `virtualenv` alike.
    """
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _report_environment(report: Report) -> None:
    """Say which interpreter the installation is going into.

    Worth a line of its own in the report. "Installed successfully" against the
    wrong interpreter is the failure this whole arrangement exists to prevent,
    and it is invisible unless something states the answer.
    """
    expected = paths.ROOT / ".venv"
    inside = in_virtual_environment()
    here = Path(sys.prefix).resolve()

    if inside and here == expected.resolve():
        report.add(Step("Environment", Status.FOUND,
                        f"project virtualenv {expected.name}"))
        return
    if inside:
        # A venv, but not this project's. Usable, and not something to refuse -
        # a developer may well have their own - but it must be visible.
        report.add(Step("Environment", Status.FOUND,
                        f"virtualenv at {here}"))
        return
    # `MISSING` rather than a new status: `_MARK` has no entry for anything
    # else and would render "[ ?? ]", and this is exactly what MISSING means
    # here - the project environment is absent from this run, and the remedy
    # line says how to get it.
    report.add(Step(
        "Environment", Status.MISSING,
        "not a virtual environment - packages would go system-wide",
        remedy='run "System Setup.bat", which builds .venv and uses it'))


def ensure_packages(report: Report, install: bool) -> None:
    """Install `requirements.txt` into the interpreter running this script.

    Which is the project's own `.venv`: `System Setup.bat` creates it and runs
    this file with it, so every install here - and every migration, check and
    subprocess below, all of which use `sys.executable` - lands inside it and
    the machine's Python is left as it was found.

    That the two batch files must agree about the interpreter was the original
    argument against a venv, and it was a real one: disagreeing about it cost
    this project a defect (R-015). They agree by construction now. Both resolve
    `.venv\\Scripts\\python.exe` relative to their own folder and neither
    accepts anything else, so there is one answer rather than two guesses.

    The other two environments stay separate on purpose. `SuryaOCR/venv_new`
    pins `transformers==4.57.1` and vLLM pins `>=5.5.3`; merging either into
    this one is what makes an OCR upgrade break extraction.
    """
    started = time.monotonic()
    _report_environment(report)
    required = {"PySide6": "PySide6", "sqlalchemy": "SQLAlchemy",
                # `import psycopg` covers the binary half as well: without a pq
                # wrapper it does not import at all, it raises "no pq wrapper
                # available". Checking for `psycopg_binary` separately looks
                # more thorough and is wrong - that module refuses to import
                # unless `psycopg` was imported first, so the probe reports it
                # missing on a machine where it is installed and working.
                "psycopg": "psycopg[binary]", "alembic": "alembic",
                "pystache": "pystache", "pymupdf": "PyMuPDF"}
    missing = [name for module, name in required.items()
               if not _importable(module)]

    if not missing:
        report.add(Step("Python packages", Status.FOUND,
                        f"{len(required)} import cleanly",
                        time.monotonic() - started))
        return

    if not install:
        report.add(Step("Python packages", Status.MISSING, ", ".join(missing),
                        time.monotonic() - started,
                        f'"{sys.executable}" -m pip install -r requirements.txt'))
        return

    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], timeout=600)
    code, out = _run([sys.executable, "-m", "pip", "install", "-r",
                      str(paths.ROOT / "requirements.txt")], timeout=3600)

    still = [name for module, name in required.items() if not _importable(module)]
    if not still:
        report.add(Step("Python packages", Status.INSTALLED,
                        f"{len(missing)} installed",
                        time.monotonic() - started))
        return

    # Retry the stragglers individually: one bad wheel should not lose the rest.
    for name in list(still):
        _run([sys.executable, "-m", "pip", "install", name], timeout=1200)
    still = [name for module, name in required.items() if not _importable(module)]
    report.add(Step(
        "Python packages",
        Status.FAILED if still else Status.INSTALLED,
        ", ".join(still) + " would not install" if still
        else f"{len(missing)} installed",
        time.monotonic() - started,
        (out.strip().splitlines()[-1][:100] if still and out.strip() else "")))


def _importable(module: str) -> bool:
    code, _ = _run([sys.executable, "-c", f"import {module}"], timeout=90)
    return code == 0


# ---------------------------------------------------------------------------
# 4. Database, runtime, models
# ---------------------------------------------------------------------------


def ensure_database(report: Report, install: bool, password: str) -> bool:
    """Returns True only when *this run* created the database."""
    started = time.monotonic()
    code, out = _run([sys.executable, str(ROOT / "tools" / "db_setup.py"),
                      "--check"], timeout=180)
    if code == 0:
        _apply_migrations(report)
        return False

    if not install:
        report.add(Step("PostgreSQL", Status.MISSING, "not reachable",
                        time.monotonic() - started,
                        "py -3.13 tools/setup.py --install-database"))
        return False

    code, out = _run([sys.executable, str(ROOT / "tools" / "setup.py"),
                      "--install-database", "--db-password", password],
                     timeout=3600)
    ok, _ = _run([sys.executable, str(ROOT / "tools" / "db_setup.py"), "--check"],
                 timeout=180)
    if ok == 0:
        report.add(Step("PostgreSQL", Status.INSTALLED, "installed and reachable",
                        time.monotonic() - started))
        _apply_migrations(report)
        return True
    report.add(Step("PostgreSQL", Status.FAILED, "not reachable after install",
                    time.monotonic() - started,
                    "py -3.13 tools/db_setup.py --check"))
    return False


def _apply_migrations(report: Report) -> None:
    started = time.monotonic()
    code, out = _run([sys.executable, "-m", "alembic", "upgrade", "head"],
                     timeout=600, cwd=PROJECT_ROOT)
    applied = [ln for ln in out.splitlines() if "Running upgrade" in ln]
    if code == 0:
        report.add(Step("Migrations",
                        Status.INSTALLED if applied else Status.FOUND,
                        f"{len(applied)} applied" if applied else "already at head",
                        time.monotonic() - started))
        _run([sys.executable, str(ROOT / "tools" / "db_setup.py"), "--seed"],
             timeout=300)
    else:
        tail = out.strip().splitlines()[-1] if out.strip() else f"exit {code}"
        report.add(Step("Migrations", Status.FAILED, tail[:80],
                        time.monotonic() - started,
                        "py -3.13 -m alembic upgrade head"))


#: The PyTorch index. vLLM pins `torch==2.11.0+cu130`, and PyPI carries plain
#: 2.11.0 but not the CUDA-13 build - so a bare `pip install <wheel>` fails with
#: "no matching distribution" on a wheel that is perfectly good.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def ensure_vllm(report: Report, install: bool, system: dict) -> None:
    """vLLM, in its own environment, on hardware that can use it.

    Skipped rather than installed on a small card. vLLM allocates its KV cache
    pool up front against the *unquantised* checkpoint, so a machine that
    cannot host it gains nothing from several GB of torch and CUDA libraries -
    and llama.cpp already serves the quantised model there.

    A separate virtualenv is not tidiness: vLLM pins `transformers>=5.5.3` and
    Surya pins `==4.57.1`. They cannot share an interpreter, and installing
    vLLM into Surya's would break OCR - the stage that does most of the work.
    """
    from ai_server.engines.vllm import VENV_DIR

    started = time.monotonic()
    vram = float(system.get("gpu_vram_gb") or 0.0)
    env_dir = ROOT.parent / VENV_DIR if (ROOT.parent / VENV_DIR).exists() else         paths.ROOT / VENV_DIR
    python = env_dir / "Scripts" / "python.exe"

    if python.is_file():
        code, _ = _run([str(python), "-c", "import vllm"], timeout=180)
        if code == 0:
            report.add(Step("vLLM engine", Status.FOUND,
                            f"ready in {VENV_DIR}", time.monotonic() - started))
            return

    if vram < 16.0:
        report.add(Step(
            "vLLM engine", Status.SKIPPED,
            f"{vram:.0f} GB VRAM - llama.cpp serves this card",
            time.monotonic() - started))
        return

    wheels = sorted((paths.MODELS).glob("vllm-*.whl"))
    if not wheels:
        report.add(Step("vLLM engine", Status.MISSING, "no vllm wheel in models/",
                        time.monotonic() - started,
                        remedy="Optional. llama.cpp serves the quantised model."))
        return

    if not install:
        report.add(Step("vLLM engine", Status.MISSING,
                        f"would install {wheels[-1].name}",
                        time.monotonic() - started))
        return

    if not python.is_file():
        code, out = _run([sys.executable, "-m", "venv", str(env_dir)], timeout=600)
        if code != 0:
            report.add(Step("vLLM engine", Status.FAILED,
                            f"could not create {VENV_DIR}: {out.strip()[:70]}",
                            time.monotonic() - started))
            return

    code, out = _run([str(python), "-m", "pip", "install", str(wheels[-1]),
                      "--extra-index-url", TORCH_INDEX], timeout=5400)
    ok = code == 0
    report.add(Step(
        "vLLM engine", Status.INSTALLED if ok else Status.FAILED,
        f"{wheels[-1].name} into {VENV_DIR}" if ok
        else (out.strip().splitlines() or ["install failed"])[-1][:70],
        time.monotonic() - started,
        remedy="" if ok else
        "Optional - llama.cpp still serves the quantised model."))


def ensure_runtime(report: Report, install: bool) -> None:
    """llama.cpp, which ships the CUDA DLLs it needs.

    The full CUDA Toolkit is deliberately not installed: it is ~3 GB of
    compiler, and the three runtime DLLs are all that inference requires.
    """
    started = time.monotonic()
    binary = ROOT / "tools" / "llamacpp" / "llama-server.exe"
    if binary.is_file():
        report.add(Step("Inference runtime", Status.FOUND, binary.name,
                        time.monotonic() - started))
        return
    if not install:
        report.add(Step("Inference runtime", Status.MISSING, "llama-server absent",
                        time.monotonic() - started,
                        "py -3.13 tools/setup.py --install-runtime"))
        return
    _run([sys.executable, str(ROOT / "tools" / "setup.py"), "--install-runtime"],
         timeout=3600)
    report.add(Step("Inference runtime",
                    Status.INSTALLED if binary.is_file() else Status.MISSING,
                    binary.name if binary.is_file() else "download failed",
                    time.monotonic() - started))


def verify_extraction_model(report: Report) -> None:
    """Verify the fine-tuned model. **Never download it.**

    This application is built around one specific fine-tuned Gemma-3-4B. An
    installer that helpfully fetched "a Gemma 3 4B" would silently replace the
    weights every accuracy figure in this project was measured against, and
    nothing downstream would detect the substitution. Absent means report and
    stop, not fetch.
    """
    started = time.monotonic()
    gguf = sorted(paths.GGUF_DIR.glob("*.gguf"))
    if gguf:
        largest = max(gguf, key=lambda p: p.stat().st_size)
        size = largest.stat().st_size / 1024 ** 3
        report.add(Step("Extraction model", Status.FOUND,
                        f"{largest.name} ({size:.2f} GB)",
                        time.monotonic() - started))
        return
    report.add(Step(
        "Extraction model", Status.MISSING, "no GGUF present",
        time.monotonic() - started,
        "Copy the fine-tuned model into 'AI server/gguf/'.\n"
        "This installer never downloads a substitute model."))


def ensure_translation(report: Report, install: bool) -> None:
    started = time.monotonic()
    base = paths.TRANSLATOR_DIR
    weights = list(base.glob("*/*.safetensors")) + list(base.glob("*/pytorch_model*.bin"))
    if weights:
        size = sum(w.stat().st_size for w in weights) / 1024 ** 3
        report.add(Step("Translation model", Status.FOUND,
                        f"{weights[0].parent.name} ({size:.1f} GB)",
                        time.monotonic() - started))
        return
    if not install:
        report.add(Step("Translation model", Status.MISSING, "absent (~2.5 GB)",
                        time.monotonic() - started,
                        "py -3.13 tools/setup.py --install-translation"))
        return
    _run([sys.executable, str(ROOT / "tools" / "setup.py"),
          "--install-translation"], timeout=5400)
    weights = list(base.glob("*/*.safetensors")) + list(base.glob("*/pytorch_model*.bin"))
    report.add(Step("Translation model",
                    Status.INSTALLED if weights else Status.MISSING,
                    "downloaded" if weights else "download failed",
                    time.monotonic() - started))


def ensure_ocr(report: Report) -> None:
    """Surya OCR, in its own interpreter.

    Tesseract, Poppler and Ghostscript are **not** used and are not installed.
    Surya handles Kannada, which is the reason it was chosen; PyMuPDF covers
    everything Poppler and Ghostscript would have.
    """
    started = time.monotonic()
    interpreter = next(
        (paths.SURYA_DIR / rel
         for rel in ("venv_new/Scripts/python.exe", "venv/Scripts/python.exe")
         if (paths.SURYA_DIR / rel).is_file()), None)
    if interpreter is None:
        report.add(Step("OCR (Surya)", Status.MISSING,
                        "no interpreter - scanned pages will be skipped",
                        time.monotonic() - started,
                        "See docs/TRANSLATION.md for the OCR environment"))
        return
    code, _ = _run([str(interpreter), "-c", "import surya"], timeout=180)
    report.add(Step("OCR (Surya)",
                    Status.FOUND if code == 0 else Status.MISSING,
                    f"via {interpreter.parent.parent.name}" if code == 0
                    else "interpreter present, surya not importable",
                    time.monotonic() - started))


# ---------------------------------------------------------------------------
# 5. Configuration
# ---------------------------------------------------------------------------


def ensure_configuration(report: Report, dsn: str, install: bool,
                         created_database: bool) -> None:
    """Write `.env` if absent. Never overwrite: it holds the live password.

    Writing one is also **not** safe when the database already exists with a
    different password. A generated password only matches a database this
    installer created; on a machine where PostgreSQL was set up beforehand it
    would replace a working connection string with a wrong one - which is
    exactly what happened the first time this ran here.
    """
    started = time.monotonic()
    env_path = paths.ROOT / ".env"
    if not dsn:
        # No DSN could be built - the packages are not in yet. Saying so beats
        # reporting the file's presence as though it had been validated.
        report.add(Step("Configuration", Status.SKIPPED,
                        "not checked - install the packages first",
                        time.monotonic() - started))
        return
    if not install:
        report.add(Step("Configuration",
                        Status.FOUND if env_path.is_file() else Status.MISSING,
                        ".env exists" if env_path.is_file() else ".env would be written",
                        time.monotonic() - started))
        return
    if env_path.is_file():
        report.add(Step("Configuration", Status.FOUND,
                        ".env exists - left untouched",
                        time.monotonic() - started))
        return

    # If the database already answers on the default credentials, this installer
    # did not create it. Writing a freshly generated password would replace a
    # working connection string with a wrong one - which is precisely what
    # happened the first time this ran on a machine with an existing database.
    if not created_database:
        report.add(Step(
            "Configuration", Status.SKIPPED,
            "database already configured - .env left for you to write",
            time.monotonic() - started))
        return

    env_path.write_text(
        "# Sale Deed AI - written by System Setup.\n"
        "# Holds the database password. Never commit this file.\n\n"
        f"SALEDEED_DB_URL={dsn}\n"
        "SALEDEED_AI_URL=http://127.0.0.1:8077\n"
        "SALEDEED_DEBUG=false\n"
        "# Nightly backup and purge. Off by default: retention DELETES data.\n"
        "SALEDEED_RETENTION=false\n",
        encoding="utf-8", newline="\n")

    # Restrict to the current user. A world-readable file holding a database
    # password is worth one command to avoid.
    _run(["icacls", str(env_path), "/inheritance:r", "/grant:r",
          f"{os.environ.get('USERNAME', 'Users')}:F"], timeout=60)
    report.add(Step("Configuration", Status.INSTALLED, ".env written",
                    time.monotonic() - started))


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------


def validate(report: Report) -> bool:
    """Run the checks that already exist rather than inventing new ones."""
    print("\n  Validation")
    print("  " + "-" * 66)
    started = time.monotonic()
    code, out = _run([sys.executable, str(paths.ROOT / "launcher.py"), "--check"],
                     timeout=900)
    for line in out.splitlines():
        if line.strip().startswith(("[ ok ]", "[warn]", "[FAIL]")):
            print(f"  {line.strip()}")
    passed = code == 0
    report.add(Step("Preflight (13 checks)",
                    Status.FOUND if passed else Status.FAILED,
                    "all passed" if passed else "see above",
                    time.monotonic() - started))

    started = time.monotonic()
    code, out = _run([sys.executable, "-m", "pytest", str(paths.ROOT / "tests"), "-q"],
                     timeout=1800)
    summary = next((ln for ln in reversed(out.splitlines())
                    if "passed" in ln or "failed" in ln), "no result")
    report.add(Step("Test suite",
                    Status.FOUND if code == 0 else Status.FAILED,
                    summary.strip()[:60], time.monotonic() - started))
    return passed


def write_reports(report: Report) -> Path:
    """A file the operator can send when something is wrong."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "system.json").write_text(
        json.dumps(report.system, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Installation Report", "",
        f"**Generated:** {report.started}",
        f"**Duration:** {report.seconds:.0f}s", "",
        "## System", "",
        "| Property | Value |", "|---|---|",
    ]
    for key, value in report.system.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Steps", "",
              "| Step | Result | Detail | Seconds |", "|---|---|---|---|"]
    for step in report.steps:
        lines.append(f"| {step.name} | {step.status} | {step.detail} | "
                     f"{step.seconds:.1f} |")

    if report.failures or report.missing:
        lines += ["", "## Needs attention", ""]
        for step in report.failures + report.missing:
            lines.append(f"- **{step.name}** — {step.detail}")
            if step.remedy:
                for remedy in step.remedy.splitlines():
                    lines.append(f"  - `{remedy}`")

    target = paths.DOCS / "INSTALLATION_REPORT.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="System Setup",
        description="Prepare this machine and start Sale Deed AI.")
    parser.add_argument("--report-only", action="store_true",
                        help="detect and report, change nothing")
    parser.add_argument("--no-launch", action="store_true",
                        help="set up but do not start the application")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the test suite during validation")
    parser.add_argument("--db-password", default="",
                        help="database password; generated if omitted")
    # Per-machine. A target system may differ from this one in exactly one
    # respect - a busy 5432, a shared server, a renamed database - and having
    # to hand-edit a DSN to change a port is how credentials end up pasted
    # into scripts.
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="saledeed")
    parser.add_argument("--db-user", default="saledeed")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    install = not args.report_only
    started = time.monotonic()
    report = Report()

    password = args.db_password or _generate_password()
    sys.path.insert(0, str(ROOT))
    db_settings = {"SALEDEED_DB_HOST": args.db_host,
                   "SALEDEED_DB_PORT": args.db_port,
                   "SALEDEED_DB_NAME": args.db_name,
                   "SALEDEED_DB_USER": args.db_user,
                   "SALEDEED_DB_PASSWORD": password}

    print()
    print("  Sale Deed AI - System Setup")
    print(f"  {ROOT}")
    _log(f"=== setup started ({'install' if install else 'report only'}) ===")

    report.system = detect_system()
    print_system(report.system)

    # Elevation is only a problem when something actually has to be installed.
    # Reporting it as a bare fact - which is all `print_system` did - leaves an
    # operator to work out for themselves why three steps failed, when the fix
    # is one right-click. Named here, before any of them runs.
    if install and not report.system.get("administrator"):
        needs_admin = [
            label for label, present in (
                ("PostgreSQL", report.system.get("postgresql", "").startswith("not")),
                ("Python 3.12 for OCR", "3.12" not in report.system.get(
                    "python_versions", {})),
            ) if present
        ]
        if needs_admin:
            print("\n  Not running as administrator, and these need it:")
            for label in needs_admin:
                print(f"    - {label}")
            print("  Right-click \"System Setup.bat\" and choose "
                  "\"Run as administrator\", or install those two by hand.")
            _log(f"not elevated; installs needing admin: {', '.join(needs_admin)}")

    # Disk and connectivity were detected, printed, and then forgotten. The
    # downloads below total several GB, so a machine that cannot take them
    # fails part-way through a 2.5 GB fetch with a generic network error - when
    # the fact that made it inevitable was already on screen. Said once, here,
    # before anything starts. Not fatal: a machine that already has the
    # components needs neither disk nor network.
    if install:
        free_gb = report.system.get("disk_free_gb", 0)
        if free_gb and free_gb < FULL_INSTALL_GB:
            print(f"\n  Only {free_gb} GB free on this drive. A full install "
                  f"needs about {FULL_INSTALL_GB} GB - llama.cpp is ~0.4 GB and "
                  "the translation model ~2.5 GB. Anything already present is "
                  "skipped, so this may still be enough.")
            _log(f"low disk: {free_gb} GB free, {FULL_INSTALL_GB} GB wanted")
        if not report.system.get("internet", True):
            print("\n  No internet. Anything already installed still works; "
                  "what is missing cannot be fetched and will be reported "
                  "as missing rather than installed.")
            _log("no internet reachable; downloads will fail")

    print("\n  Prerequisites")
    print("  " + "-" * 66)
    ensure_directories(report, install)
    ensure_vcredist(report, install)
    ensure_python_312(report, install)
    ensure_packages(report, install)

    # The DSN is assembled here, not at the top of `main`, and the reason is
    # ordering rather than tidiness: `build_dsn` lives in `core.db.engine`,
    # which imports SQLAlchemy. On a machine whose virtualenv was created
    # moments ago nothing is installed yet, so building it any earlier ends the
    # setup with `ModuleNotFoundError: sqlalchemy` before `ensure_packages` has
    # had its chance to fix exactly that. It only ever worked because the
    # interpreter running the setup already had the packages - which is the
    # assumption the virtualenv exists to remove.
    #
    # Assembled once, from this machine's answers, and never rebuilt from
    # literals further down.
    try:
        from core.db.engine import build_dsn

        dsn = build_dsn(db_settings)
    except ImportError:
        # Report-only, on a machine whose virtualenv was created seconds ago:
        # nothing is installed, so the module that builds the DSN cannot be
        # imported. That is a fact to report, not a reason to abort - the whole
        # point of --report-only is to survey a machine before touching it.
        #
        # Not rebuilt from a local copy of the URL logic. A second
        # implementation of a connection string is how the two drift and a
        # password ends up quoted one way here and another way there; the
        # steps that need it say they cannot check yet instead.
        dsn = ""

    print("\n  Application")
    print("  " + "-" * 66)
    created_database = ensure_database(report, install, password)
    ensure_configuration(report, dsn, install, created_database)
    ensure_runtime(report, install)
    verify_extraction_model(report)
    ensure_vllm(report, install, report.system)
    ensure_translation(report, install)
    ensure_ocr(report)

    ready = True
    if install and not args.skip_tests:
        ready = validate(report)

    report.seconds = time.monotonic() - started
    path = write_reports(report)

    print()
    print("  " + "-" * 66)
    print(f"  Report: docs/{path.name}      Logs: runtime/logs/setup/")

    if report.failures:
        print(f"\n  {len(report.failures)} blocking problem(s). "
              "Fix them and run this file again - completed steps are skipped.")
        _log("=== setup finished with failures ===")
        return 1

    if report.missing:
        print(f"\n  Set up, with {len(report.missing)} optional component(s) "
              "missing. The application will start; see the report.")

    if args.report_only:
        print("\n  Report only - nothing was changed.")
        return 0

    if args.no_launch:
        print("\n  Ready. Start with:  Run Sale Deed AI.bat")
        return 0

    print("\n  Starting the application ...\n")
    _log("=== setup finished, launching ===")

    # Setup has succeeded by this point. The launcher's exit code describes the
    # *application*, and returning it made a normal-but-non-zero exit from the
    # program print "Setup stopped with code 1 - read INSTALLATION_REPORT.md",
    # sending the operator to a report of a setup that went perfectly.
    code = subprocess.call([sys.executable, str(paths.ROOT / "launcher.py")])
    if code != 0:
        print(f"\n  The application exited with code {code}. Setup itself "
              "completed - see runtime/logs/launcher.log for the application.")
        _log(f"application exited with code {code} (setup was successful)")
    return 0


def _generate_password() -> str:
    """A password nobody has to remember and nobody can guess.

    `secrets`, not `random`: the latter is seeded predictably and this ends up
    in a file that grants access to every deed the system holds.
    """
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(20))


if __name__ == "__main__":
    raise SystemExit(main())
