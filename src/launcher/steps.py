"""Startup steps, each independently testable.

A step reports one of three outcomes:

    OK        the requirement is satisfied
    WARN      degraded, but the application can still open
    FAIL      the application cannot start; the message says what to do

Steps never raise for an expected condition. A missing database is an ordinary
outcome on a fresh machine, not an exception - the launcher must be able to
report every problem it found, not just the first one.

Adding a requirement means adding a function and listing it, so the sequence is
open for extension without editing the runner (SOLID, open/closed).
"""

from __future__ import annotations

import os
import shutil
import socket
import urllib.request
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import REQUIRED_DIRS, LauncherConfig


class Outcome(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(slots=True)
class Result:
    outcome: Outcome
    detail: str
    #: Shown under the failure. A command the user can copy, or a plain
    #: instruction. Blank when there is nothing actionable.
    remedy: str = ""
    facts: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is not Outcome.FAIL


def _ok(detail: str, **facts: object) -> Result:
    return Result(Outcome.OK, detail, facts=facts)


def _warn(detail: str, remedy: str = "", **facts: object) -> Result:
    return Result(Outcome.WARN, detail, remedy, facts)


def _fail(detail: str, remedy: str = "", **facts: object) -> Result:
    return Result(Outcome.FAIL, detail, remedy, facts)


Step = Callable[[LauncherConfig], Result]


# --------------------------------------------------------------------------
# 1. project layout
# --------------------------------------------------------------------------

def check_project(cfg: LauncherConfig) -> Result:
    missing = [name for name in ("src/core", "src/app", "src/ai_server",
                                "alembic.ini")
               if not (cfg.root / name).exists()]
    if missing:
        return _fail(f"project files missing at {cfg.root}: {', '.join(missing)}",
                     "Run the launcher from inside the project folder.")
    return _ok(f"project root {cfg.root}", root=str(cfg.root))


def check_python(cfg: LauncherConfig) -> Result:
    version = sys.version_info
    if version < (3, 12):
        return _fail(
            f"Python {version.major}.{version.minor} is too old (3.12+ required)",
            "winget install Python.Python.3.12")
    in_venv = sys.prefix != sys.base_prefix
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    if not in_venv:
        return _warn(f"{detail} (not in a virtual environment)",
                     "python -m venv .venv  then re-run the launcher",
                     python=sys.executable)
    return _ok(f"{detail} in {Path(sys.prefix).name}", python=sys.executable)


def check_dependencies(cfg: LauncherConfig) -> Result:
    """Import the packages the application cannot open without.

    Checked by import rather than by pip metadata: a package can be recorded as
    installed and still fail to load, which is exactly what a broken PySide6 or
    a mismatched psycopg does.
    """
    required = {
        "PySide6": "PySide6",
        "sqlalchemy": "SQLAlchemy",
        "psycopg": "psycopg",
        "alembic": "Alembic",
        "pystache": "Pystache",
        "fitz": "PyMuPDF",
    }
    missing: list[str] = []
    for module, name in required.items():
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - a broken install must read as missing
            missing.append(name)
    if missing:
        # Name the interpreter, not just the packages. With more than one Python
        # installed - and this project needs a second one for Surya - the usual
        # cause is that the wrong interpreter was launched, not that anything is
        # genuinely uninstalled. "pip install PySide6" against the wrong Python
        # succeeds and changes nothing.
        return _fail(
            f"packages unavailable to {Path(sys.executable).parent.name}"
            f" ({sys.version.split()[0]}): {', '.join(missing)}",
            f'"{sys.executable}" -m pip install ' + " ".join(missing))
    return _ok(f"{len(required)} core packages import cleanly "
               f"(Python {sys.version.split()[0]})")


def check_directories(cfg: LauncherConfig) -> Result:
    created: list[str] = []
    for rel in REQUIRED_DIRS:
        target = cfg.root / rel
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    if created:
        return _ok(f"{len(created)} folder(s) created: {', '.join(created)}",
                   created=created)
    return _ok(f"{len(REQUIRED_DIRS)} folders present")


def check_disk(cfg: LauncherConfig) -> Result:
    """A batch can write 25 GB. Running out mid-run corrupts nothing, but it
    strands work that then has to be re-processed."""
    usage = shutil.disk_usage(cfg.root)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 2:
        return _fail(f"only {free_gb:.1f} GB free on {cfg.root.drive}",
                     "Free up space before processing.")
    if free_gb < 10:
        return _warn(f"{free_gb:.1f} GB free - large batches may not fit",
                     free_gb=round(free_gb, 1))
    return _ok(f"{free_gb:.0f} GB free", free_gb=round(free_gb, 1))


# --------------------------------------------------------------------------
# 2. database
# --------------------------------------------------------------------------

def _postgres_service_name() -> str | None:
    """Find the installed PostgreSQL service, whatever its version suffix."""
    if os.name != "nt":
        return None
    try:
        proc = subprocess.run(["sc", "query", "state=", "all"],
                              capture_output=True, text=True, timeout=15,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.upper().startswith("SERVICE_NAME:") and "postgresql" in line.lower():
            return line.split(":", 1)[1].strip()
    return None


def _service_running(name: str) -> bool:
    try:
        proc = subprocess.run(["sc", "query", name], capture_output=True,
                              text=True, timeout=15,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        return False
    return "RUNNING" in proc.stdout.upper()


def start_postgres(cfg: LauncherConfig) -> Result:
    """Start the PostgreSQL service if it is installed but stopped.

    Starting a Windows service needs elevation. When that is refused the step
    warns rather than fails: the connectivity check that follows produces the
    message the user can act on, and a remote database needs no local service
    at all.
    """
    from core.db.engine import dsn_from_env

    dsn = cfg.db_url or dsn_from_env()
    if "localhost" not in dsn and "127.0.0.1" not in dsn:
        return _ok("database is remote - no local service to start")

    name = _postgres_service_name()
    if not name:
        return _warn("no local PostgreSQL service found",
                     "Install PostgreSQL, or set SALEDEED_DB_URL to a remote server.")
    if _service_running(name):
        return _ok(f"service {name} already running", service=name)

    try:
        proc = subprocess.run(["net", "start", name], capture_output=True,
                              text=True, timeout=90,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        return _warn(f"could not start {name}: {exc}")
    if proc.returncode == 0 or _service_running(name):
        return _ok(f"service {name} started", service=name)
    hint = "Run the launcher as Administrator to start it automatically."
    return _warn(f"service {name} is stopped and could not be started", hint,
                 service=name)


def check_database(cfg: LauncherConfig) -> Result:
    from core.db.engine import build_engine, check_connection, dsn_from_env

    dsn = cfg.db_url or dsn_from_env()
    safe = dsn.split("@")[-1] if "@" in dsn else dsn
    try:
        engine = build_engine(dsn)
        ok, detail = check_connection(engine)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"database unreachable: {type(exc).__name__}: {exc}",
                     f'"{sys.executable}" tools/db_setup.py --check')
    if not ok:
        return _fail(f"database unreachable at {safe}",
                     f'"{sys.executable}" tools/db_setup.py --check\n'
                     f"       {detail.splitlines()[0] if detail else ''}")
    return _ok(f"connected to {safe}", dsn=safe)


def run_migrations(cfg: LauncherConfig) -> Result:
    """Bring the schema to head.

    Runs even when already current - `alembic upgrade head` is a no-op then, and
    checking first would cost the same connection round trip.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(cfg.root), capture_output=True, text=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return _fail("alembic is not installed",
                     f'"{sys.executable}" -m pip install alembic')
    except subprocess.TimeoutExpired:
        return _fail("migrations timed out after 3 minutes",
                     "Check for a lock held by another connection.")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = tail[-1] if tail else f"exit {proc.returncode}"
        return _fail(f"migration failed: {message}",
                     f'"{sys.executable}" -m alembic upgrade head')
    applied = [ln for ln in (proc.stderr or "").splitlines() if "Running upgrade" in ln]
    if applied:
        return _ok(f"{len(applied)} migration(s) applied", applied=len(applied))
    return _ok("schema already at head")


# --------------------------------------------------------------------------
# 3. models and OCR
# --------------------------------------------------------------------------

def check_model(cfg: LauncherConfig) -> Result:
    """Verify the fine-tuned model is present.

    Deliberately verify-only. This application is built around one specific
    fine-tuned checkpoint; downloading a replacement when the file is missing
    would silently swap the model the extraction accuracy was measured against.
    A missing model is reported, never substituted.
    """
    if cfg.engine == "mock":
        return _ok("mock engine - no model required")

    if cfg.model_gguf.is_file():
        size_gb = cfg.model_gguf.stat().st_size / (1024 ** 3)
        if size_gb < 0.5:
            return _fail(f"{cfg.model_gguf.name} is only {size_gb:.2f} GB - truncated",
                         "Re-copy the model file; it is incomplete.")
        return _ok(f"{cfg.model_gguf.name} ({size_gb:.2f} GB)",
                   model=str(cfg.model_gguf), size_gb=round(size_gb, 2))

    if cfg.model_dir.is_dir() and any(cfg.model_dir.glob("*.safetensors")):
        return _warn(f"no GGUF built; safetensors found in {cfg.model_dir.name}",
                     f'"{sys.executable}" tools/setup.py --build-model --quant Q4_K_M')

    return _fail(f"model not found: {cfg.model_gguf}",
                 "Copy the fine-tuned model into 'AI server/gguf/'.\n"
                 "       This application does not download a substitute model.")


def check_runtime(cfg: LauncherConfig) -> Result:
    if cfg.engine == "mock":
        return _ok("mock engine - no runtime required")
    if not cfg.llama_binary.is_file():
        return _fail(f"llama-server not found: {cfg.llama_binary}",
                     f'"{sys.executable}" tools/setup.py --install-runtime')
    return _ok(f"runtime {cfg.llama_binary.name}", binary=str(cfg.llama_binary))


def check_ocr(cfg: LauncherConfig) -> Result:
    """OCR is optional, so this warns rather than fails.

    Most registered deeds are digitally generated and carry a text layer, which
    the pipeline reads directly. Surya is needed for scanned deeds and for
    Kannada pages that were photographed rather than typed.
    """
    try:
        import fitz  # noqa: F401
    except Exception:  # noqa: BLE001
        return _fail("PyMuPDF is unavailable - no PDF can be read",
                     f'"{sys.executable}" -m pip install PyMuPDF')

    if cfg.surya_python and cfg.surya_python.is_file():
        script = cfg.surya_script
        if not script or not script.is_file():
            return _warn("Surya interpreter found but tools/surya_runner.py is missing")
        return _ok(f"Surya OCR via {cfg.surya_python.parent.parent.name}",
                   interpreter=str(cfg.surya_python))
    return _warn("Surya OCR unavailable - scanned pages will be skipped",
                 f'"{sys.executable}" tools/setup.py --install-ocr')


def check_hardware(cfg: LauncherConfig) -> Result:
    try:
        from ai_server.hardware import detect
    except Exception as exc:  # noqa: BLE001
        return _warn(f"hardware detection unavailable: {exc}")
    hw = detect()
    gpu = hw.primary_gpu
    if gpu is None:
        return _warn(f"no NVIDIA GPU - running on CPU "
                     f"({hw.ram_total_gib:.0f} GB RAM)",
                     "Processing will be several times slower.")
    return _ok(f"{gpu.name}, {gpu.free_gib:.1f} of {gpu.total_gib:.0f} GiB free",
               gpu=gpu.name, vram_total=gpu.total_bytes, vram_free=gpu.free_bytes)


def check_port(cfg: LauncherConfig) -> Result:
    """The ports the application binds, and who holds them.

    Two ports, not one. The AI server binds `ai_port` and hands its inference
    engine `ai_port + 1`; checking only the first let a machine start with the
    engine's port taken, and llama-server then exited during startup while the
    window reported the AI server as offline.

    A busy port is also only "reuse that server" when it *is* that server. This
    reported reuse for anything listening at all, so a stranger on the port meant
    the UI talked to a foreign process and the failure surfaced as "AI server
    offline" against a server that was never ours. One request to /health
    separates the two cases.
    """
    engine_port = cfg.ai_port + 1

    def listening(number: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            return sock.connect_ex((cfg.ai_host, number)) == 0

    def ours(number: int) -> bool:
        try:
            with urllib.request.urlopen(
                    f"http://{cfg.ai_host}:{number}/health", timeout=2) as response:
                return b"status" in response.read(400).lower()
        except Exception:  # noqa: BLE001
            return False

    # Probed once each and remembered. Re-probing filled the listen backlog of
    # whatever was on the other end, so the third connect was refused and the
    # port that had just been reported busy read as free - a check that
    # disagreed with itself inside one call.
    busy = {number: listening(number) for number in (cfg.ai_port, engine_port)}

    if not any(busy.values()):
        return _ok(f"ports {cfg.ai_port} and {engine_port} available")

    if busy[cfg.ai_port] and ours(cfg.ai_port):
        return _warn(f"port {cfg.ai_port} already serves this application "
                     "- reusing that server",
                     "Close the other instance if this is unexpected.",
                     reuse=True)

    taken = [str(number) for number, is_busy in busy.items() if is_busy]
    return _fail(
        "port " + " and ".join(taken)
        + " held by something that is not this application",
        "Whatever is listening there did not answer /health, so the AI server "
        "cannot start and the window would report it as offline.\n"
        "Find the process:\n"
        f"    Get-NetTCPConnection -LocalPort {','.join(taken)} -State Listen\n"
        "Close it, or move this application to a free pair in .env - the "
        "engine takes the next port up:\n"
        f"    SALEDEED_AI_URL=http://{cfg.ai_host}:{cfg.ai_port + 13}",
        reuse=False)


#: Order matters: cheap local checks first, so a broken checkout fails in
#: milliseconds instead of after a database timeout.
PREFLIGHT: tuple[tuple[str, Step], ...] = (
    ("Project files", check_project),
    ("Python runtime", check_python),
    ("Dependencies", check_dependencies),
    ("Folders", check_directories),
    ("Disk space", check_disk),
    ("Hardware", check_hardware),
    ("PostgreSQL service", start_postgres),
    ("Database", check_database),
    ("Migrations", run_migrations),
    ("Model", check_model),
    ("Inference runtime", check_runtime),
    ("OCR", check_ocr),
    ("Port", check_port),
)
