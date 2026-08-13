"""Translation configuration.

Resolved from three places, in increasing precedence: the shipped defaults here,
the `settings` table (what the operator chose in the UI), and the environment
(what an operator overrode for a single run without editing anything).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .. import paths

ROOT = Path(__file__).resolve().parents[2]

#: Where a downloaded model lands. One directory per model so a future upgrade
#: sits beside the current one rather than overwriting it mid-batch.
MODEL_ROOT = paths.TRANSLATOR_DIR

#: The model this application ships against.
#:
#: **facebook/nllb-200-distilled-600M**, chosen over the alternatives for
#: reasons that are specific rather than general:
#:
#: * **Ungated.** IndicTrans2 scores better on Indic->English, but both its
#:   variants sit behind a HuggingFace licence gate, so a fresh install cannot
#:   fetch them without an account and a token. A translation system that
#:   requires the operator to create a third-party account before it works is
#:   not an offline system in any useful sense.
#: * **One model, every language in scope.** NLLB-200 covers all 200 languages
#:   including Kannada, Hindi, Marathi, Telugu, Tamil, Malayalam, Gujarati,
#:   Bengali, Punjabi, Odia and Urdu. The alternative was a per-language model
#:   set, which multiplies download size and gives the detector something to get
#:   wrong.
#: * **It fits.** The distilled 600M is ~2.5 GB, against 5.5 GB for the 1.3B and
#:   17 GB for the 3.3B. On a 4 GB card already holding 3.2 GB of language
#:   model, only the distilled variant has any chance of the GPU, and it runs
#:   acceptably on CPU when it does not.
#: * **Runs entirely locally.** Deed text is a legal record and must not leave
#:   the machine, which rules out every hosted API regardless of quality.
DEFAULT_MODEL = "nllb-200-distilled-600M"
DEFAULT_MODEL_REPO = "facebook/nllb-200-distilled-600M"

#: Interpreters that can host the model, in order. Shared with Surya: both need
#: torch and transformers, and duplicating ~3 GB of torch for a 600M model would
#: be wasteful.
INTERPRETERS = (
    "models/SuryaOCR/venv_new/Scripts/python.exe",
    "models/SuryaOCR/venv/Scripts/python.exe",
    "models/SuryaOCR/venv_new/bin/python",
    "models/SuryaOCR/venv/bin/python",
)


def _find_interpreter(root: Path) -> Path | None:
    for relative in INTERPRETERS:
        candidate = root.parent / relative
        if candidate.is_file():
            return candidate
    return None


def _find_model(root: Path, name: str) -> Path | None:
    """The configured model if present, else any usable one in the directory.

    Falling back means a model downloaded under a different name still works
    rather than the system reporting itself unavailable next to 2.5 GB of
    perfectly good weights.
    """
    base = root.parent / "models" / "AI server" / "translator"
    preferred = base / name
    if preferred.is_dir():
        return preferred
    if not base.is_dir():
        return None
    for candidate in sorted(base.iterdir()):
        if candidate.is_dir() and (list(candidate.glob("*.safetensors"))
                                   or list(candidate.glob("pytorch_model*.bin"))):
            return candidate
    return None


@dataclass
class TranslationConfig:
    """Everything the translation service needs."""

    #: Master switch. Off means values pass through untouched and the export
    #: reports what it could not render.
    enabled: bool = True

    #: Output language. English throughout this application; kept configurable
    #: because the requirement says "unless explicitly requested".
    target_language: str = "eng_Latn"

    #: Source language. "auto" means detect per field - correct for a deed,
    #: which is a mixed document. A fixed code forces every field.
    source_language: str = "auto"

    #: Hindi and Marathi share Devanagari. "auto" reads each field and decides
    #: from the vocabulary and the letters Hindi does not use; where a field
    #: offers no evidence it falls back to Hindi. `hin_Deva` or `mar_Deva`
    #: forces every Devanagari field, which is right for an operator working a
    #: single jurisdiction - a Maharashtra registry should set `mar_Deva`.
    devanagari_as: str = "auto"

    model_dir: Path | None = None
    python: Path | None = None
    script: Path | None = None

    #: "auto" picks CUDA only when enough VRAM is genuinely free.
    device: str = "auto"

    #: Sentences per forward pass. 16 is comfortable for deed fields, which are
    #: short; raising it helps throughput and costs memory.
    batch_size: int = 16

    timeout_s: float = 600.0

    #: Retries on a *transient* failure. One is deliberate: a model that fails
    #: twice is broken, and a batch of 500 documents must not spend its time
    #: retrying.
    max_retries: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_language": self.target_language,
            "source_language": self.source_language,
            "devanagari_as": self.devanagari_as,
            "model": self.model_dir.name if self.model_dir else "",
            "model_dir": str(self.model_dir or ""),
            "device": self.device,
            "batch_size": self.batch_size,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
        }


def _flag(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def build_config(setting: Callable[[str, str], str] | None = None,
                 root: Path | None = None) -> TranslationConfig:
    """Resolve configuration from defaults, stored settings and environment.

    `setting` is `AppService._setting` when the database is reachable, and None
    otherwise - translation must still work on a machine whose database is down,
    because the export does.
    """
    root = root or ROOT

    def stored(key: str, default: str) -> str:
        if setting is None:
            return default
        try:
            return setting(key, default) or default
        except Exception:  # noqa: BLE001 - a bad setting must not disable translation
            return default

    def env(key: str, default: str) -> str:
        return os.environ.get(key, "").strip() or default

    model_name = env("SALEDEED_TRANSLATION_MODEL",
                     stored("translation_model", DEFAULT_MODEL))

    configured = env("SALEDEED_TRANSLATION_MODEL_DIR", "")
    model_dir = Path(configured) if configured else _find_model(root, model_name)

    return TranslationConfig(
        enabled=_flag(env("SALEDEED_TRANSLATION",
                          stored("translation_enabled", "true")), True),
        target_language=env("SALEDEED_TRANSLATION_TARGET",
                            stored("translation_target", "eng_Latn")),
        source_language=stored("translation_source", "auto"),
        devanagari_as=stored("translation_devanagari_as", "auto"),
        model_dir=model_dir,
        python=_find_interpreter(root),
        script=root / "tools" / "translate_runner.py",
        device=env("SALEDEED_TRANSLATION_DEVICE",
                   stored("translation_device", "auto")),
        batch_size=int(stored("translation_batch_size", "16") or 16),
        timeout_s=float(stored("translation_timeout_s", "600") or 600),
        max_retries=int(stored("translation_max_retries", "1") or 1),
    )
