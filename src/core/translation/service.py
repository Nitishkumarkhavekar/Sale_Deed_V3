"""The one place translation happens.

Everything that needs a value rendered in English goes through
`TranslationService`. Before this existed the logic lived inside
`TranslateStage`, which meant the export, the API and the UI each had to hope the
pipeline had already run - and any new caller would have reimplemented detection
and batching slightly differently.

Design notes worth keeping:

**The model runs in a subprocess.** It needs torch and transformers, which live
in the OCR virtual environment, not the application's. Loading it in-process
would also hold ~2.5 GB of VRAM for the lifetime of the window. The service is
therefore pure orchestration: detect, cache, batch, spawn, collect.

**Failure is never fatal.** A deed is a legal record; losing it because a
translation model is missing, slow or broken would be a far worse outcome than
an untranslated field. Every failure path returns the original text and records
why, and the caller can see exactly which fields did not make it.

**The cache is content-addressed.** Deeds repeat: the same village, the same
district, the same registration office across a whole batch. Caching by
`(text, source, target)` turns a 500-document batch into a few hundred distinct
strings.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import TranslationConfig
from .detect import ENGLISH, LANGUAGE_NAMES, Detection, Script, detect, needs_translation
from .postprocess import tidy
from .transliterate import (
    has_indic as _has_indic,
    transliterate_mixed,
    transliterate_supported,
)

_log = logging.getLogger("saledeed.translation")

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TranslationItem:
    """One value to render in English."""

    key: str
    text: str
    #: "translate" for meaning, "transliterate" for a proper noun. See the
    #: runner's docstring for why the distinction is load-bearing.
    kind: str = "translate"
    detection: Detection | None = None
    translated: str | None = None
    error: str = ""
    from_cache: bool = False

    @property
    def output(self) -> str:
        """What should be written. Falls back to the original, never blank."""
        return self.translated or self.text


@dataclass
class TranslationResult:
    """What happened to a whole request."""

    items: list[TranslationItem] = field(default_factory=list)
    engine: str = "disabled"
    model: str = ""
    device: str = ""
    seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)
    error: str = ""

    @property
    def translated(self) -> int:
        return sum(1 for i in self.items if i.translated)

    @property
    def untranslated(self) -> list[TranslationItem]:
        """Items that still hold non-English text. This is the honest answer to
        "is the export clean"."""
        return [i for i in self.items
                if not i.translated and needs_translation(i.text)]

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine, "model": self.model, "device": self.device,
            "seconds": round(self.seconds, 2), "languages": self.languages,
            "translated": self.translated,
            "untranslated": len(self.untranslated),
            "error": self.error,
        }


class TranslationService:
    """Detect, cache, batch and translate. Thread-safe."""

    def __init__(self, config: TranslationConfig | None = None) -> None:
        self.config = config or TranslationConfig()
        self._cache: dict[tuple[str, str, str], str] = {}
        self._lock = threading.Lock()
        self._probe: dict[str, Any] | None = None
        self.hits = 0
        self.misses = 0
        #: Asked only whether the language model is resident, so this stage can
        #: keep off a card it would OOM on. No work is ever sent here.
        self.ai_base_url = os.environ.get(
            "SALEDEED_AI_URL", "http://127.0.0.1:8077").rstrip("/")

    # -- availability -----------------------------------------------------

    def available(self) -> tuple[bool, str]:
        """Whether translation can actually run, and why not otherwise."""
        cfg = self.config
        if not cfg.enabled:
            return False, "translation is disabled in settings"
        if not cfg.model_dir or not cfg.model_dir.is_dir():
            return False, f"model directory not found: {cfg.model_dir}"
        if not (list(cfg.model_dir.glob("*.safetensors"))
                or list(cfg.model_dir.glob("pytorch_model*.bin"))):
            return False, f"no model weights in {cfg.model_dir}"
        if not cfg.python or not cfg.python.is_file():
            return False, f"translator interpreter not found: {cfg.python}"
        if not cfg.script or not cfg.script.is_file():
            return False, f"runner script not found: {cfg.script}"
        return True, f"{cfg.model_dir.name} via {cfg.python.parent.parent.name}"

    def probe(self) -> dict[str, Any]:
        """Ask the runner what it would do, without loading the model.

        Cached: it spawns an interpreter, and the Settings page would otherwise
        pay that on every render.
        """
        if self._probe is not None:
            return self._probe
        ok, detail = self.available()
        if not ok:
            self._probe = {"ready": False, "detail": detail}
            return self._probe
        try:
            proc = subprocess.run(
                [str(self.config.python), str(self.config.script),
                 "--model", str(self.config.model_dir), "--probe"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._probe = json.loads(proc.stdout or "{}")
        except Exception as exc:  # noqa: BLE001
            self._probe = {"ready": False, "detail": f"{type(exc).__name__}: {exc}"}
        return self._probe

    # -- the work ---------------------------------------------------------

    def translate(self, items: list[TranslationItem]) -> TranslationResult:
        """Render every item in the target language.

        Never raises. A failure leaves `item.translated` unset, so `item.output`
        yields the original and the caller can still write a complete record.
        """
        started = time.monotonic()
        result = TranslationResult(items=items)
        cfg = self.config

        for item in items:
            item.detection = detect(item.text, devanagari_as=cfg.devanagari_as)

        # Devanagari is the one script that does not name its own language, so
        # record what was decided and on what grounds. A wrong Hindi/Marathi
        # call produces fluent, plausible, wrong English - the kind of failure
        # that is invisible in the output and obvious in the log.
        devanagari = [i for i in items
                      if i.detection and i.detection.script is Script.DEVANAGARI]
        if devanagari:
            decided: dict[str, int] = {}
            grounds: dict[str, int] = {}
            for item in devanagari:
                lang = item.detection.language  # type: ignore[union-attr]
                why = item.detection.reason or "unknown"  # type: ignore[union-attr]
                decided[LANGUAGE_NAMES.get(lang, lang)] = decided.get(
                    LANGUAGE_NAMES.get(lang, lang), 0) + 1
                grounds[why] = grounds.get(why, 0) + 1
            _log.info("Devanagari: %d field(s) resolved to %s",
                      len(devanagari),
                      ", ".join(f"{n} {name}" for name, n in sorted(decided.items())),
                      extra={"setting": cfg.devanagari_as, "languages": decided,
                             "evidence": grounds})

        # A value qualifies when its language differs from the target *or* when
        # it merely contains a non-target script. The second case is the mixed
        # one - `KRISHNAPPA ರಾಜು` detects as English on character count, and
        # filtering on language alone let the Kannada half through to the CSV.
        def _wanted(item: TranslationItem) -> bool:
            if not item.detection or item.detection.script.value == "neutral":
                return False
            if item.detection.language != cfg.target_language:
                return True
            return _has_indic(item.text)

        pending = [i for i in items if _wanted(i)]
        result.languages = {}
        for item in pending:
            lang = item.detection.language  # type: ignore[union-attr]
            result.languages[lang] = result.languages.get(lang, 0) + 1

        if not pending:
            result.engine = "none"
            result.seconds = time.monotonic() - started
            return result

        # Cache first. A batch of deeds repeats villages and offices heavily.
        remaining: list[TranslationItem] = []
        for item in pending:
            key = (item.text, item.detection.language, cfg.target_language)  # type: ignore[union-attr]
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                item.translated = cached
                item.from_cache = True
                self.hits += 1
            else:
                remaining.append(item)
                self.misses += 1

        # Proper nouns are rendered by rule, never by the model.
        #
        # NLLB translates *meaning*, and on Indian names the meaning is often a
        # word: measured on this project's own model, a party named
        # `ಲಕ್ಷ್ಮಿ ದೇವಿ` came back as "Goddess Lakshmi" and `ವೆಂಕಟೇಶ್` as
        # "What is Venkatesh?". On a record identifying parties to a property
        # transfer that is a corrupted document, not a quality issue.
        # Rule-based transliteration is deterministic and cannot invent.
        # Gated on `enabled` like everything else. Transliteration is cheap and
        # deterministic, but an operator who switched translation off asked for
        # values to be left alone, and silently rewriting names would be exactly
        # the surprise the switch exists to prevent.
        still_pending: list[TranslationItem] = []
        for item in (remaining if cfg.enabled else []):
            script = item.detection.script if item.detection else Script.NEUTRAL
            # A value may be part English and part Indic - `KRISHNAPPA ರಾಜು`.
            # `transliterate_mixed` converts only the Indic runs, so a name
            # already written in English comes back byte-identical, spelling
            # and capitalisation intact. That is a requirement of the report,
            # not a nicety: the deed's spelling of a name is the record.
            if item.kind == "transliterate" and (
                    transliterate_supported(script) or _has_indic(item.text)):
                rendered = transliterate_mixed(item.text, script=script)
                if rendered and rendered != item.text:
                    item.translated = rendered
                    with self._lock:
                        self._cache[(item.text,
                                     item.detection.language,  # type: ignore[union-attr]
                                     cfg.target_language)] = rendered
                    _log.debug("transliterated field", extra={
                        "field": item.key, "operation": "transliterate",
                        "source_language": item.detection.language,  # type: ignore[union-attr]
                        "original": item.text, "translation": rendered,
                        "engine": "rule"})
                    continue
            still_pending.append(item)
        if cfg.enabled:
            remaining = still_pending
        if not remaining:
            result.engine = "rule"
            result.seconds = time.monotonic() - started
            return result

        ok, detail = self.available()
        if not ok:
            result.engine = "unavailable"
            result.error = detail
            result.seconds = time.monotonic() - started
            if remaining:
                _log.warning(
                    "translation unavailable - %d field(s) stay in the source "
                    "language", len(remaining),
                    extra={"reason": detail,
                           "languages": {LANGUAGE_NAMES.get(k, k): v
                                         for k, v in result.languages.items()}})
            return result

        if remaining:
            self._run(remaining, result)

        result.seconds = time.monotonic() - started
        _log.info(
            "translated %d/%d field(s) in %.2fs", result.translated,
            len(pending), result.seconds,
            extra={"engine": result.engine, "model": result.model,
                   "device": result.device, "cache_hits": len(pending) - len(remaining),
                   "languages": {LANGUAGE_NAMES.get(k, k): v
                                 for k, v in result.languages.items()},
                   "target": cfg.target_language})
        return result

    def _run(self, items: list[TranslationItem], result: TranslationResult) -> None:
        """Spawn the runner. Retries once on a transient failure."""
        cfg = self.config
        payload = {"items": [
            {"id": item.key, "text": item.text, "kind": item.kind,
             "src": item.detection.language}  # type: ignore[union-attr]
            for item in items]}

        last_error = ""
        for attempt in range(1, cfg.max_retries + 2):
            try:
                data = self._spawn(payload)
            except subprocess.TimeoutExpired:
                last_error = f"timed out after {cfg.timeout_s:.0f}s"
            except Exception as exc:  # noqa: BLE001 - never fatal
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                by_key = {r.get("id"): r for r in data.get("results") or []}
                for item in items:
                    entry = by_key.get(item.key)
                    text = (entry or {}).get("text", "").strip()
                    if not text:
                        item.error = "the model returned nothing"
                        continue
                    text = tidy(text, kind=item.kind)
                    item.translated = text
                    with self._lock:
                        self._cache[(item.text,
                                     item.detection.language,  # type: ignore[union-attr]
                                     cfg.target_language)] = text
                    _log.debug("translated field", extra={
                        "field": item.key, "operation": item.kind,
                        "source_language": item.detection.language,  # type: ignore[union-attr]
                        "original": item.text, "translation": text})
                result.engine = "nllb"
                result.model = data.get("model", cfg.model_dir.name if cfg.model_dir else "")
                result.device = data.get("device", "")
                return

            if attempt <= cfg.max_retries:
                _log.warning("translation attempt %d failed, retrying", attempt,
                             extra={"error": last_error, "fields": len(items)})

        result.engine = "failed"
        result.error = last_error
        for item in items:
            item.error = last_error
        _log.error("translation failed after %d attempt(s)", cfg.max_retries + 1,
                   extra={"error": last_error, "fields": len(items)})

    def _device(self) -> str:
        """CPU while the language model is resident, otherwise as configured.

        The runner picks its own device from `torch.cuda.mem_get_info()`, and on
        Windows that number cannot be trusted: with `llama-server` holding
        3.06 GiB of a 4 GiB card, a fresh CUDA context still reports "3.2 GiB
        free of 4.0 GiB". WDDM lets the driver over-promise and the allocation
        fails later instead. Surya believed that number and died with `free: 0`
        mid-batch (R-035); this stage would have followed.

        So the question asked is not "how much is free" but "is the other model
        loaded", which has a definite answer.
        """
        if self.config.device != "auto":
            return self.config.device
        try:
            with urllib.request.urlopen(
                    f"{self.ai_base_url}/health", timeout=3) as resp:
                engine = json.loads(resp.read()).get("engine") or {}
        except Exception:  # noqa: BLE001 - no server means no contention
            return "auto"
        return "cpu" if engine.get("loaded") else "auto"

    def _spawn(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="saledeed_tr_") as tmp:
            in_path = Path(tmp) / "in.json"
            out_path = Path(tmp) / "out.json"
            in_path.write_text(json.dumps(payload, ensure_ascii=False),
                               encoding="utf-8")
            proc = subprocess.run(
                [str(cfg.python), str(cfg.script),
                 "--model", str(cfg.model_dir),
                 "--in", str(in_path), "--out", str(out_path),
                 "--tgt", cfg.target_language,
                 "--device", self._device(),
                 "--batch-size", str(cfg.batch_size)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=cfg.timeout_s, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if proc.returncode != 0:
                raise RuntimeError(
                    f"runner exited {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[:300]}")
            if not out_path.is_file():
                raise RuntimeError("runner reported success but wrote nothing")
            return json.loads(out_path.read_text(encoding="utf-8"))

    # -- convenience -------------------------------------------------------

    def translate_text(self, text: str, *, kind: str = "translate") -> str:
        """One value in, English out. Returns the original if it cannot."""
        item = TranslationItem(key="text", text=text, kind=kind)
        self.translate([item])
        return item.output

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._cache), "hits": self.hits,
                    "misses": self.misses}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = self.misses = 0
