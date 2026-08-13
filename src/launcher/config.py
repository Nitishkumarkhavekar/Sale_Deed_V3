"""Launcher configuration: project root discovery and `.env` loading.

No path in this package is hard-coded. The root is found by walking up from this
file until a directory containing the project's markers appears, so the launcher
works from any working directory, from a shortcut, from Task Scheduler, and from
a copied folder on another machine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Files that together identify the project root. `alembic.ini` alone is not
#: enough - a nested virtualenv can contain one.
ROOT_MARKERS = ("alembic.ini", "src", "models")

#: Directories the application expects to exist. Created if absent; never
#: cleaned, because they hold user data.
REQUIRED_DIRS = ("runtime/logs", "runtime/uploads", "runtime/exports",
                 "runtime/backups", "runtime/cache", "runtime/temp",
                 "models/AI server", "models/AI server/gguf")


def find_root(start: Path | None = None) -> Path:
    """Walk up until every marker is present.

    Falls back to the launcher's own parent rather than raising: a partially
    assembled checkout should still reach the validation step, which produces a
    readable diagnosis instead of a traceback from import time.
    """
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).exists() for m in ROOT_MARKERS):
            return candidate
    return Path(__file__).resolve().parents[2]


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Minimal `.env` parser - no dependency on python-dotenv.

    Supports `KEY=value`, `export KEY=value`, `#` comments, and quoted values.
    A real process environment variable wins over the file unless `override`,
    so an operator can point one run at a different database without editing
    anything on disk.
    """
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(slots=True)
class LauncherConfig:
    """Everything the launcher needs, resolved once at startup."""

    root: Path
    python: Path
    ai_host: str = "127.0.0.1"
    ai_port: int = 8077
    model_gguf: Path = field(default_factory=Path)
    model_dir: Path = field(default_factory=Path)
    llama_binary: Path = field(default_factory=Path)
    surya_python: Path | None = None
    surya_script: Path | None = None
    engine: str = "llamacpp"
    db_url: str = ""
    env_loaded: dict[str, str] = field(default_factory=dict)
    #: Skip the desktop window - used by `--headless` for servers and CI.
    headless: bool = False
    #: Seconds to wait for the AI server's HTTP endpoint to answer at all. This
    #: is not the model load time; see `wait_for_http` in supervisor.py.
    ai_http_timeout_s: float = 45.0

    @property
    def ai_base_url(self) -> str:
        return f"http://{self.ai_host}:{self.ai_port}"

    @property
    def log_dir(self) -> Path:
        return self.root / "runtime" / "logs"


def _first_existing(root: Path, *relative: str) -> Path | None:
    for rel in relative:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def build_config(root: Path | None = None, *, headless: bool = False) -> LauncherConfig:
    """Resolve configuration from `.env`, the environment, and the layout on disk."""
    root = root or find_root()
    env_loaded = load_env_file(root / ".env")

    def env(key: str, default: str) -> str:
        return os.environ.get(key, default).strip() or default

    ai_url = env("SALEDEED_AI_URL", "http://127.0.0.1:8077")
    host, _, port_text = ai_url.removeprefix("http://").removeprefix("https://").partition(":")
    try:
        port = int(port_text.split("/")[0]) if port_text else 8077
    except ValueError:
        port = 8077

    # A Surya interpreter is optional: the OCR stage falls back to the embedded
    # text layer when it is absent, and most registered deeds carry one.
    surya_python = _first_existing(
        root,
        "models/SuryaOCR/venv_new/Scripts/python.exe",
        "models/SuryaOCR/venv/Scripts/python.exe",
    )

    return LauncherConfig(
        root=root,
        python=Path(sys.executable),
        ai_host=host or "127.0.0.1",
        ai_port=port,
        model_gguf=root / env("SALEDEED_MODEL_GGUF",
                              "models/AI server/gguf/deeds-v6_7-Q4_K_M.gguf"),
        model_dir=root / env("SALEDEED_MODEL_DIR", "models/AI server/gemma4b-text"),
        llama_binary=root / env("SALEDEED_LLAMA_BINARY",
                                "src/tools/llamacpp/llama-server.exe"),
        surya_python=surya_python,
        surya_script=root / "src/tools/surya_runner.py",
        engine=env("SALEDEED_ENGINE", "llamacpp"),
        db_url=env("SALEDEED_DB_URL", ""),
        env_loaded=env_loaded,
        headless=headless,
    )
