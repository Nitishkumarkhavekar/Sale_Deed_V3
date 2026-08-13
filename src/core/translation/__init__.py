"""Multilingual translation: detection, orchestration, and the model runner.

    detect.py    script-based language identification, no model required
    config.py    resolved settings, and the reasoning behind the model choice
    service.py   the single entry point every caller uses

The model itself runs in `tools/translate_runner.py`, inside the OCR virtual
environment, because it needs torch and transformers and because ~2.5 GB of VRAM
should not be held for the lifetime of the window.
"""

from .config import DEFAULT_MODEL, DEFAULT_MODEL_REPO, MODEL_ROOT, TranslationConfig, build_config
from .detect import (
    ENGLISH,
    LANGUAGE_NAMES,
    Detection,
    Script,
    detect,
    needs_translation,
    summarise,
)
from .service import TranslationItem, TranslationResult, TranslationService

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_MODEL_REPO", "MODEL_ROOT",
    "ENGLISH", "LANGUAGE_NAMES",
    "Detection", "Script", "TranslationConfig", "TranslationItem",
    "TranslationResult", "TranslationService",
    "build_config", "detect", "needs_translation", "summarise",
]
