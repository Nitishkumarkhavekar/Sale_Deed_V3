"""Every filesystem location the project uses, in one place.

Before this module each of about twenty files worked out the project root for
itself with `Path(__file__).resolve().parents[N]` and then appended a directory
name. That is fine until the tree moves, at which point `N` is wrong in twenty
places and every one of them fails somewhere different and late - a missing
model reads as "no GPU", a missing prompt reads as "extraction failed".

So: one module knows the layout, everything else asks. Moving a folder is now a
one-line change here.

    src/        the code - this file lives in it
    models/     weights, the Surya installation, prompts. Large, static.
    runtime/    everything written while running. Disposable; recreated on demand.
    docs/       documentation and notes
    tests/      the suite and the sample corpus

`runtime/` is separated from `models/` deliberately: one can be deleted to
reclaim space and will rebuild itself, the other is 36 GB that must never be
touched by a cleanup script.
"""

from __future__ import annotations

from pathlib import Path

#: The project root. This file is `<root>/src/core/paths.py`, so two levels up
#: from `src`. Everything else is derived, never recomputed.
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
DOCS = ROOT / "docs"
TESTS = ROOT / "tests"

# -- models and weights -----------------------------------------------------
#: The folder name keeps its space for the same reason the model keeps its
#: filename: paths are baked into configuration written before this refactor.
AI_SERVER = MODELS / "AI server"
GGUF_DIR = AI_SERVER / "gguf"
CHECKPOINT_DIR = AI_SERVER / "gemma4b"
REPACKED_DIR = AI_SERVER / "gemma4b-text"
TRANSLATOR_DIR = AI_SERVER / "translator"
SURYA_DIR = MODELS / "SuryaOCR"
PROMPT_DIR = MODELS / "saledeed main"
PROMPT_FILE = PROMPT_DIR / "prompt_v6_short.txt"

# -- written while running --------------------------------------------------
DATA_DIR = RUNTIME / "data"
LOG_DIR = RUNTIME / "logs"
UPLOAD_DIR = RUNTIME / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
TEMP_DIR = RUNTIME / "temp"
CACHE_DIR = RUNTIME / "cache"
BACKUP_DIR = RUNTIME / "backups"
CONFIG_DIR = RUNTIME / "config"
CLEANED_DIR = DATA_DIR / "cleaned"
WATERMARK_DIR = DATA_DIR / "watermark_cleaned"

# -- code -------------------------------------------------------------------
MIGRATIONS_DIR = SRC / "migrations"
TOOLS_DIR = SRC / "tools"
UI_DIR = SRC / "app" / "ui"

#: Directories created on demand at startup. Models are absent from this list on
#: purpose - a missing model is a problem to report, not a directory to invent.
RUNTIME_DIRS = (DATA_DIR, LOG_DIR, UPLOAD_DIR, EXPORT_DIR, TEMP_DIR,
                CACHE_DIR, BACKUP_DIR, CONFIG_DIR, CLEANED_DIR, WATERMARK_DIR)


def ensure_runtime_dirs() -> None:
    """Create the working directories. Safe to call repeatedly."""
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
