"""Pipeline stages - document processing logic, no database.

Each stage is a small object with one `run` method. Stages take and return plain
data, so the whole chain can be exercised without PostgreSQL, without the queue,
and (for everything but extraction) without a GPU.

Retry policy, per the specification:

    OCR         exactly one retry
    Extraction  one retry, triggered by parse failure or PAN coverage below 0.6

An important caveat on extraction retry: the documented retry is *split-prompt*,
and that path is broken on the v6.7 weights actually present - the model was
trained only on the full schema. A same-prompt retry is useless because
temperature is 0, so a rerun is byte-identical. `ExtractStage` therefore reports
`retry_supported = False` on v6.7 and routes failures to review instead of
pretending to retry. See docs/DOCUMENTATION.md.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .. import paths
from ..ocr_cleanup import CleanupOptions, clean, page_texts
from ..translation import LANGUAGE_NAMES
from ..validation import (
    Disposition,
    RuleToggles,
    ValidationReport,
    extract_json,
    ocr_pans,
    pan_coverage,
    validate_extraction,
)


#: Stage-level logging. Quiet by default; DEBUG carries the
#: per-field original and translation.
_log = logging.getLogger("saledeed.pipeline")


class StageName(str, Enum):
    OCR = "ocr"
    EXTRACT = "extract"
    TRANSLATE = "translate"
    VALIDATE = "validate"


@dataclass
class StageOutcome:
    """Result of one stage on one document."""

    stage: StageName
    ok: bool
    detail: str = ""
    #: True when a retry could plausibly change the result. False for
    #: deterministic failures - retrying those only wastes GPU time.
    retryable: bool = False
    duration_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, stage: StageName, **data: Any) -> StageOutcome:
        return cls(stage=stage, ok=True, data=data)

    @classmethod
    def failure(cls, stage: StageName, detail: str, *, retryable: bool = False,
                **data: Any) -> StageOutcome:
        return cls(stage=stage, ok=False, detail=detail, retryable=retryable, data=data)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


class OcrStage:
    """PDF -> page images -> OCR text -> cleanup.

    Two backends:

    `surya` is the specified engine and the one that handles Kannada properly. It
    runs in a **separate interpreter** because it pins a different transformers
    version than the model side historically required; the repack removed the
    model's pin, but Surya keeps its own.

    `textlayer` reads the PDF's embedded text with PyMuPDF. Zero dependencies
    beyond PyMuPDF and no GPU, but Kannada quality is poor and it returns nothing
    at all for pure scans. It exists as a fallback and for development, not as a
    production substitute.
    """

    def __init__(
        self,
        *,
        engine: str = "textlayer",
        surya_python: str | Path | None = None,
        surya_script: str | Path | None = None,
        dpi: int = 300,
        languages: tuple[str, ...] = ("kn", "en"),
        cleanup: CleanupOptions | None = None,
        timeout_s: float = 900.0,
        min_chars_per_page: int = 40,
        device: str = "auto",
        #: Used only to ask whether the language model is resident, so OCR can
        #: keep off a card it would OOM on. Never used to send work.
        ai_base_url: str = "http://127.0.0.1:8077",
    ) -> None:
        self.engine = engine
        #: "auto" lets the runner pick CPU or CUDA from free VRAM. Forcing "cpu"
        #: is useful on a small card where the language model already holds most
        #: of the VRAM.
        self.device = device
        self.ai_base_url = ai_base_url.rstrip("/")
        self.surya_python = Path(surya_python) if surya_python else None
        self.surya_script = Path(surya_script) if surya_script else None
        self.dpi = dpi
        self.languages = languages
        self.cleanup = cleanup or CleanupOptions()
        self.timeout_s = timeout_s
        #: Below this many characters per page the text layer is assumed absent.
        #: 011.pdf in the corpus had 20 pages and 141 characters - a pure scan.
        self.min_chars_per_page = min_chars_per_page

    @property
    def uses_gpu(self) -> bool:
        """Surya is a GPU model; the text-layer backend is pure CPU.

        The runner takes the GPU lease only when this is True. Taking it for
        text-layer extraction would serialise CPU work behind GPU work for no
        reason and cut throughput.
        """
        return self.engine == "surya"

    def available(self) -> tuple[bool, str]:
        if self.engine == "textlayer":
            try:
                import pymupdf  # noqa: F401
            except ImportError:
                return False, "PyMuPDF is not installed"
            return True, "embedded text layer (PyMuPDF)"
        if self.engine == "surya":
            if not self.surya_python or not self.surya_python.is_file():
                return False, f"Surya interpreter not found: {self.surya_python}"
            if not self.surya_script or not self.surya_script.is_file():
                return False, f"Surya runner script not found: {self.surya_script}"
            return True, f"Surya via {self.surya_python}"
        return False, f"unknown OCR engine {self.engine!r}"

    def run(self, pdf_path: str | Path) -> StageOutcome:
        started = time.monotonic()
        path = Path(pdf_path)
        if not path.is_file():
            return StageOutcome.failure(StageName.OCR, f"file not found: {path}")

        ok, detail = self.available()
        if not ok:
            # Missing engine is an environment problem, not a document problem:
            # retrying this document will not help.
            return StageOutcome.failure(StageName.OCR, detail, retryable=False)

        try:
            lines: list[list] = []
            if self.engine == "surya":
                raw, pages, lines = self._run_surya(path)
            else:
                raw, pages = self._run_textlayer(path)
        except subprocess.TimeoutExpired:
            return StageOutcome.failure(
                StageName.OCR, f"OCR timed out after {self.timeout_s:.0f}s",
                retryable=True)
        except Exception as exc:  # noqa: BLE001 - one document must not kill the batch
            return StageOutcome.failure(
                StageName.OCR, f"{type(exc).__name__}: {exc}", retryable=True)

        if not raw.strip():
            return StageOutcome.failure(
                StageName.OCR, "OCR produced no text", retryable=False)

        if pages and len(raw) / pages < self.min_chars_per_page:
            return StageOutcome.failure(
                StageName.OCR,
                f"only {len(raw)} characters across {pages} pages - the PDF has no "
                "usable text layer (a pure scan). A real OCR engine is required.",
                retryable=False, pages=pages, chars=len(raw))

        cleaned, report = clean(raw, self.cleanup)
        return StageOutcome.success(
            StageName.OCR, text=cleaned, pages=report.pages_detected,
            page_texts=page_texts(cleaned), chars=len(cleaned),
            cleanup=report.summary(),
            # Normalised line boxes, when the engine reported them. The caller
            # uses these to give a scanned page an invisible, selectable text
            # layer; an engine that cannot supply them simply sends none.
            lines=lines,
            duration_s=round(time.monotonic() - started, 2))

    def ocr_pages(self, pdf_path: str | Path, pages: list[int]) -> str:
        """OCR a named handful of pages and return their text. Never raises.

        A second, cheap read for one specific purpose: recovering the
        registration number when the embedded text layer did not carry it.
        Kaveri pastes its registration certificate onto the scan as an image,
        so a deed that is otherwise digital text can be missing the one field
        that identifies it - and `_run_textlayer` returns nothing for a picture.

        Only the pages asked for are rendered, because the alternative is
        re-reading the whole deed through the GPU model to find a box on page
        one. Returns "" for anything that goes wrong, including "this build has
        no real OCR engine": the caller's document is already extractable and
        must not fail over a recovery attempt that could not be made.
        """
        path = Path(pdf_path)
        if not pages or not path.is_file():
            return ""
        if not self.surya_python or not self.surya_script:
            return ""
        if not (self.surya_python.is_file() and self.surya_script.is_file()):
            return ""

        try:
            import pymupdf
        except ImportError:
            return ""

        try:
            with tempfile.TemporaryDirectory(prefix="saledeed_identity_") as tmp:
                shots = Path(tmp) / "pages"
                shots.mkdir()
                rendered = 0
                with pymupdf.open(path) as doc:
                    for number in pages:
                        if not 1 <= number <= doc.page_count:
                            continue
                        pixmap = doc[number - 1].get_pixmap(dpi=self.dpi)
                        # Zero-padded so the runner's sorted glob keeps them in
                        # page order; it reads *.png by name, not by mtime.
                        pixmap.save(shots / f"{number:04d}.png")
                        rendered += 1
                if not rendered:
                    return ""

                out_path = Path(tmp) / "ocr.txt"
                result = subprocess.run(
                    [str(self.surya_python), str(self.surya_script),
                     "--images", str(shots),
                     "--out", str(out_path),
                     "--langs", ",".join(self.languages),
                     "--device", self.device,
                     "--json"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=self.timeout_s, check=False)

                if result.returncode != 0 or not out_path.is_file():
                    _log.info("identity re-read produced nothing", extra={
                        "path": str(path), "pages": pages,
                        "exit": result.returncode,
                        "detail": (result.stderr or "").strip()[:200]})
                    return ""

                raw = out_path.read_text(encoding="utf-8")
                try:
                    return json.loads(raw).get("text", "")
                except (ValueError, TypeError):
                    return raw
        except Exception as exc:  # noqa: BLE001 - a recovery must not fail a document
            _log.info("identity re-read could not run", extra={
                "path": str(path), "pages": pages,
                "error": f"{type(exc).__name__}: {exc}"})
            return ""

    def _run_textlayer(self, path: Path) -> tuple[str, int]:
        import pymupdf

        chunks: list[str] = []
        with pymupdf.open(path) as doc:
            for number, page in enumerate(doc, start=1):
                chunks.append(f"===== PAGE {number} =====")
                chunks.append(page.get_text())
            pages = doc.page_count
        return "\n".join(chunks), pages

    def _set_model(self, loaded: bool) -> bool:
        """Ask the AI server to release or reload its weights. True if it did.

        Best effort: no server, an old build without the route, or any error at
        all means carry on. This is an optimisation, never a precondition.
        """
        try:
            body = json.dumps({"loaded": loaded}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.ai_base_url}/model", data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as resp:
                return bool(json.loads(resp.read()).get("changed"))
        except Exception:  # noqa: BLE001
            return False

    def _free_the_gpu(self) -> bool:
        """Release the language model so Surya can have the card.

        The two cannot co-reside: `llama-server` holds ~3.2 GiB of a 4 GiB card
        and roughly as much host RAM, and Surya needs about the same of either.
        Before this, OCR OOMed on the GPU and then found no room on the CPU
        either - the document simply failed (R-035).

        Reloading the model afterwards costs ~50 s, against the ~22 minutes of
        GPU time this buys back on a 14-page deed. `ExtractStage` reloads it when
        it next needs it, so nothing here has to remember to.
        """
        return self._set_model(False)

    def _run_surya(self, path: Path) -> tuple[str, int, list[list]]:
        """Hand the PDF to Surya in its own interpreter.

        Isolation is required: Surya and the LLM side pin incompatible
        dependencies, so they must never share a process.

        The PDF goes across as a path rather than as pre-rendered images. Surya's
        own loader renders at `IMAGE_DPI_HIGHRES`, which is what produced the
        corpus this model was finetuned on; rendering here at some other DPI
        would change both the recognised text and the bounding boxes the layout
        reconstruction depends on.
        """
        import pymupdf

        if self._free_the_gpu():
            _log.info("released the language model so OCR can use the GPU")

        with pymupdf.open(path) as doc:
            pages = doc.page_count

        # A temporary file rather than stdout: a 30-page deed in Kannada is
        # hundreds of KB, and a full pipe buffer deadlocks a subprocess that is
        # still writing.
        with tempfile.TemporaryDirectory(prefix="saledeed_ocr_") as tmp:
            out_path = Path(tmp) / "ocr.txt"
            result = subprocess.run(
                [str(self.surya_python), str(self.surya_script),
                 "--pdf", str(path),
                 "--out", str(out_path),
                 "--langs", ",".join(self.languages),
                 # The runner decides CPU vs CUDA from free VRAM at the moment it
                 # starts. That has to be its call, not ours: it runs in a
                 # different interpreter and only it can see what torch reports.
                 "--device", self.device,
                 # JSON carries the per-line boxes as well as the text. The
                 # `text` field is byte-identical to the plain-text output, so
                 # nothing the extraction model sees changes.
                 "--json"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.timeout_s, check=False)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Surya exited {result.returncode}: "
                    f"{(result.stderr or '').strip()[:400]}")
            if not out_path.is_file():
                raise RuntimeError("Surya reported success but wrote no output")

            body = out_path.read_text(encoding="utf-8")
            try:
                payload = json.loads(body)
            except (ValueError, TypeError):
                # An older runner without `--json` wrote plain text. Take it and
                # go without a text layer rather than failing the document.
                return body, pages, []
            return payload.get("text", ""), pages, payload.get("lines") or []


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


#: Characters of OCR text the model can be sent. The served context is 16,384
#: tokens and the prompt template takes a share of it; Kannada-and-English deed
#: text measured at ~3.55 characters per token on this corpus, so this leaves
#: real headroom rather than sitting on the boundary.
#:
#: Without a bound the server simply refuses: a 59,012-character deed came back
#: `HTTP 400 ... request (16,6xx tokens) exceeds the available context size`,
#: three times, and the document extracted nothing at all.
MAX_INPUT_CHARS = 40_000


def fit_to_context(text: str, budget: int = MAX_INPUT_CHARS) -> tuple[str, bool]:
    """Bring `text` within `budget`, dropping from the middle.

    The middle is what goes. A deed opens with the parties and closes with the
    schedule of the property and the boundaries - the two places every field
    this application extracts actually lives. What sits between them is
    recitals: chains of "WHEREAS" clauses reciting prior title, which carry
    none of the sixteen fields.

    Head and tail are kept in a 3:2 split because the parties block is the
    longer of the two. A marker is left in place of what was removed, so the
    model is not silently handed a document that appears to jump.
    """
    if len(text) <= budget:
        return text, False

    marker = "\n\n[... recitals omitted to fit the model context ...]\n\n"
    room = budget - len(marker)
    head = int(room * 0.6)
    tail = room - head
    return text[:head] + marker + text[-tail:], True


class ExtractStage:
    """Send cleaned OCR to the AI server and parse the response.

    Talks HTTP to the AI server rather than loading a model, so this process never
    links CUDA. Standard library only.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8077",
        prompt: str = "",
        #: Upper bound on the OCR text sent in one request. See MAX_INPUT_CHARS.
        max_input_chars: int = MAX_INPUT_CHARS,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        #: 1.0 = disabled, and it must stay disabled. A penalty suppresses the
        #: repeated key tokens that JSON array elements share, truncating the
        #: party list. Measured: at 1.1 the model emitted 3 of 5 persons and
        #: nulled paid_in_cash. See docs/DOCUMENTATION.md.
        repetition_penalty: float = 1.0,
        grammar: str | None = None,
        pan_coverage_threshold: float = 0.6,
        min_unmatched_pans: int = 2,
        retry_supported: bool = False,
        retry_prompt: str | None = None,
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        backpressure_wait_s: float = 10.0,
        max_backpressure_retries: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.prompt = prompt
        self.max_input_chars = max_input_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.grammar = grammar
        self.pan_coverage_threshold = pan_coverage_threshold
        self.min_unmatched_pans = min_unmatched_pans
        #: False on v6.7: split-prompt is broken there and a same-prompt rerun at
        #: temperature 0 is byte-identical, so no retry can change the outcome.
        self.retry_supported = retry_supported
        self.retry_prompt = retry_prompt
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.backpressure_wait_s = backpressure_wait_s
        self.max_backpressure_retries = max_backpressure_retries

    def health(self) -> tuple[bool, str]:
        # A reachable server with a loaded model is still useless without the
        # prompt: the model answers in prose and nothing parses. Checked here so
        # the pipeline refuses to start rather than failing document by document
        # with a message that points at the response instead of the cause.
        if not self.prompt:
            return False, ("extraction prompt is empty - the model cannot "
                           "produce JSON without it")
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            return False, f"AI server unreachable at {self.base_url}: {exc}"
        engine = payload.get("engine") or {}
        if engine.get("loaded") is False:
            # OCR releases the weights so Surya can have the card (R-035).
            # Bring them back rather than reporting "not ready" at the one
            # moment the pipeline actually needs them.
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/model",
                    data=json.dumps({"loaded": True}).encode("utf-8"),
                    method="POST", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=180):
                    pass
                with urllib.request.urlopen(f"{self.base_url}/health",
                                            timeout=10) as resp:
                    payload = json.loads(resp.read())
                engine = payload.get("engine") or {}
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                pass
        if not payload.get("ready"):
            # Name the actual obstacle. `ready` is the AND of two independent
            # things - the model being loaded, and the governor admitting work -
            # and reporting the engine's own detail conflated them into
            # "not ready: ready (pressure critical)", which reads as nonsense
            # and tells an operator nothing they can act on.
            if not engine.get("loaded"):
                return False, (f"AI server is still loading "
                               f"{engine.get('model') or 'the model'}")
            pressure = payload.get("pressure") or "resource"
            return False, (f"AI server is up but not admitting work - "
                           f"{pressure} resource pressure. This clears on its "
                           f"own; loading the model briefly causes it.")
        return True, f"{engine.get('engine')} on {engine.get('device')}"

    def run(self, ocr_text: str, document_id: str = "",
            attempt: int = 1) -> StageOutcome:
        started = time.monotonic()
        prompt = self.retry_prompt if (attempt > 1 and self.retry_prompt) else self.prompt

        sent, trimmed = fit_to_context(ocr_text, self.max_input_chars)
        if trimmed:
            _log.warning(
                "deed trimmed to fit the model context: %d -> %d characters",
                len(ocr_text), len(sent),
                extra={"document": document_id, "dropped": len(ocr_text) - len(sent)})

        try:
            job = self._submit(sent, document_id, prompt)
        except _Backpressure as exc:
            return StageOutcome.failure(StageName.EXTRACT, str(exc), retryable=True)
        except Exception as exc:  # noqa: BLE001
            return StageOutcome.failure(
                StageName.EXTRACT, f"{type(exc).__name__}: {exc}", retryable=True)

        if job.get("state") != "done":
            # Named as a server problem rather than left as a bare state.
            # A job accepted and then never finished is what a restart mid-batch
            # looks like from here: the id is gone with the process that held
            # it. "job ended in state None" classified as UNKNOWN_ERROR and the
            # operator was told "Processing failed for an unrecognised reason",
            # when the actual answer - the server went away, try again - is one
            # they can act on.
            return StageOutcome.failure(
                StageName.EXTRACT,
                job.get("error")
                or (f"AI server did not finish the job (state "
                    f"{job.get('state')!r}) - it may have restarted"),
                retryable=True)

        raw = job.get("result") or ""
        parsed = extract_json(raw)
        coverage = pan_coverage(parsed, ocr_text) if parsed else 0.0
        common = {
            "input_trimmed": trimmed,
            "raw_output": raw,
            "parsed": parsed,
            "pan_coverage": round(coverage, 3),
            "prompt_tokens": job.get("prompt_tokens") or 0,
            "completion_tokens": job.get("completion_tokens") or 0,
            "truncated": bool(job.get("truncated")),
            "duration_s": round(time.monotonic() - started, 2),
        }

        if parsed is None:
            return StageOutcome.failure(
                StageName.EXTRACT, "response contained no parseable JSON",
                retryable=self.retry_supported, **common)

        if job.get("truncated"):
            # On this workload a token-limit stop almost always means a repetition
            # loop, not a genuinely long answer: legitimate outputs average ~664
            # tokens against a 2048 cap.
            return StageOutcome.failure(
                StageName.EXTRACT,
                "generation hit the token ceiling - likely a repetition loop",
                retryable=self.retry_supported, **common)

        # Coverage alone is too coarse at small denominators, so the shortfall
        # must also be meaningful. With 2 PANs in a document the ratio can only be
        # 0.0/0.5/1.0, and one witness PAN would fail every such deed forever.
        unmatched = len(ocr_pans(ocr_text) - _extracted_pans(parsed))
        if coverage < self.pan_coverage_threshold and unmatched >= self.min_unmatched_pans:
            return StageOutcome.failure(
                StageName.EXTRACT,
                f"PAN coverage {coverage:.2f} below {self.pan_coverage_threshold:.2f} "
                f"with {unmatched} OCR PANs unmatched",
                retryable=self.retry_supported, **common)

        return StageOutcome.success(StageName.EXTRACT, **common)

    def _submit(self, ocr_text: str, document_id: str, prompt: str) -> dict[str, Any]:
        payload = {
            "ocr_text": ocr_text,
            "document_id": document_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "wait": True,
            "timeout_s": self.timeout_s,
        }
        if prompt:
            payload["prompt"] = prompt
        if self.grammar:
            payload["grammar"] = self.grammar

        body = json.dumps(payload).encode("utf-8")
        for _ in range(self.max_backpressure_retries):
            request = urllib.request.Request(
                f"{self.base_url}/extract", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s + 60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 503:
                    # The governor is refusing new work under resource pressure.
                    # Waiting is correct - the condition is transient by design.
                    time.sleep(self.backpressure_wait_s)
                    continue
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"AI server HTTP {exc.code}: {detail}") from exc
        raise _Backpressure(
            f"AI server refused work for "
            f"{self.max_backpressure_retries * self.backpressure_wait_s:.0f}s "
            "(sustained resource pressure)")


def _extracted_pans(parsed: dict[str, Any] | None) -> set[str]:
    return {
        (p.get("pan_card_number") or "").strip().upper()
        for side in ("buyer_details", "seller_details")
        for p in (parsed or {}).get(side) or []
        if isinstance(p, dict) and p.get("pan_card_number")
    }


class _Backpressure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidateStage:
    """Run the layer 2-7 validation stack.

    Load-bearing on quantised weights: Q4_K_M produced a sale consideration of
    1500000 where the OCR reads 1,50,000. Only cross-checking against the source
    catches that, so this stage is what makes the output trustworthy.
    """

    def __init__(self, toggles: RuleToggles | None = None) -> None:
        self.toggles = toggles or RuleToggles()

    def run(self, extraction: dict[str, Any] | str, ocr_text: str, *,
            truncated: bool = False, ocr_succeeded: bool = True) -> StageOutcome:
        started = time.monotonic()
        try:
            report = validate_extraction(
                extraction, ocr_text, self.toggles,
                truncated=truncated, ocr_succeeded=ocr_succeeded)
        except Exception as exc:  # noqa: BLE001
            return StageOutcome.failure(
                StageName.VALIDATE, f"{type(exc).__name__}: {exc}")

        data = {
            "report": report,
            "disposition": report.disposition.value,
            "confidence": report.confidence,
            "pan_coverage": report.pan_coverage,
            "document_remarks": report.remarks,
            "duration_s": round(time.monotonic() - started, 2),
        }
        # A REVIEW verdict is not a stage failure - the stage did its job. The
        # document is flagged and routed, not marked failed.
        return StageOutcome.success(StageName.VALIDATE, **data)

    @staticmethod
    def flag_rows(report: ValidationReport,
                  person_ids: dict[tuple[str, int], int] | None = None
                  ) -> list[dict[str, Any]]:
        """Flatten a report into rows for `validation_results`."""
        ids = person_ids or {}
        rows: list[dict[str, Any]] = []
        for flag in report.document_flags:
            rows.append({"flag_code": flag.value, "person_id": None,
                         "confidence": report.confidence})
        for check in report.document_checks:
            if check.suspect:
                rows.append({"flag_code": "FLD", "field": check.field,
                             "detail": check.detail or f"value {check.value!r}",
                             "confidence": check.confidence, "person_id": None})
        for person in report.persons:
            pid = ids.get((person.relation, person.index))
            for flag in person.flags:
                rows.append({"flag_code": flag.value, "person_id": pid,
                             "confidence": person.confidence})
        return rows


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

#: Kannada block, U+0C80-U+0CFF. Detection is by *script*, not by a language
#: classifier: a deed is a mixed document - Kannada prose around Latin PANs,
#: digits and English legal terms - so "which language is this document" is the
#: wrong question. The right one is "does this field contain characters the
#: English column cannot carry", and that is exactly a script range.
_KANNADA = re.compile(r"[ಀ-೿]")

#: Person fields that can hold Kannada, and the operation each needs.
#:
#: The distinction is load-bearing. A name must be **transliterated** - ರಮೇಶ್
#: becomes "Ramesh", the sound - because translating a proper noun produces
#: nonsense. An address must be **translated** - ಮುಖ್ಯ ರಸ್ತೆ becomes "Main
#: Road", the meaning - because transliterating it gives "Mukhya Raste", which
#: is useless to an English reader. Getting these the wrong way round produces
#: output that reads plausibly and is wrong in half the columns.
PERSON_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "transliterate"),
    ("father_name", "transliterate"),
    ("address", "translate"),
    # Reaches the "Gender (PC)" column. Audited into this list after a Kannada
    # value was found leaking into the export with no translation path at all.
    ("gender", "translate"),
    ("occupation", "translate"),
    # A state name is a proper noun with a settled English form, so it is
    # transliterated rather than translated - "Karnataka", not a rendering of
    # what the name means. Added because `State (PC-L)` had no translation path
    # at all and relied on the model choosing English of its own accord.
    ("state", "transliterate"),
)

#: Document-level free text. The registration office and document type are
#: routinely recorded in the regional language.
DOCUMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("registration_office", "translate"),
    ("document_type", "translate"),
    ("sub_registrar_office", "translate"),
)

#: Property fields that can hold Kannada. Place names are transliterated - a
#: village is a proper noun, and "Bengaluru" is what a reader expects, not a
#: literal rendering of what the name means.
PROPERTY_FIELDS: tuple[tuple[str, str], ...] = (
    ("schedule_c_property_address", "translate"),
    ("property_description", "translate"),
    ("village", "transliterate"),
    ("district", "transliterate"),
    ("taluk", "transliterate"),
    ("state", "transliterate"),
)


class TranslateStage:
    """Render every non-English field of an extraction into English.

    The stage no longer performs translation itself. Detection, caching,
    batching, the subprocess and the retry policy all live in
    `core.translation.TranslationService`, so the export, the API and any future
    caller share one implementation rather than each reinventing it slightly
    differently.

    What remains here is the part that is genuinely about *deeds*: which fields
    exist, and whether each is a proper noun or prose.
    """

    def __init__(
        self,
        *,
        engine: str = "auto",
        service: Any | None = None,
        config: Any | None = None,
        model_dir: str | Path | None = None,
        translator_python: str | Path | None = None,
        translator_script: str | Path | None = None,
        source_lang: str = "auto",
        target_lang: str = "eng_Latn",
        mapping: dict[str, str] | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        from ..translation import TranslationService, build_config

        self.engine = engine
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.mapping = mapping or {}

        if service is not None:
            self.service = service
        else:
            cfg = config or build_config()
            # Explicit arguments win, so a test can point the stage anywhere.
            if model_dir:
                cfg.model_dir = Path(model_dir)
            if translator_python:
                cfg.python = Path(translator_python)
            if translator_script:
                cfg.script = Path(translator_script)
            if timeout_s:
                cfg.timeout_s = timeout_s
            if engine == "passthrough":
                cfg.enabled = False
            self.service = TranslationService(cfg)

    @property
    def config(self) -> Any:
        return self.service.config

    @property
    def model_dir(self) -> Path | None:
        return self.service.config.model_dir

    @property
    def uses_gpu(self) -> bool:
        """The translator may take the GPU, so the runner must lease it.

        True whenever translation is enabled: whether it *actually* lands on
        CUDA is decided inside the runner from free VRAM, and the lease has to
        be held either way or two models could try for the card at once.
        """
        return bool(self.service.config.enabled)

    def available(self) -> tuple[bool, str]:
        return self.service.available()

    @staticmethod
    def needs_translation(value: str | None) -> bool:
        """Kept as a static method: several call sites and tests use it."""
        from ..translation import needs_translation as _needs

        return bool(value) and _needs(str(value))

    def _pending(self, extraction: dict[str, Any]) -> list[Any]:
        """Every field of this deed that is not already English."""
        from ..translation import TranslationItem

        items: list[TranslationItem] = []
        for side in ("buyer_details", "seller_details"):
            for i, person in enumerate(extraction.get(side) or [], start=1):
                if not isinstance(person, dict):
                    continue
                for field_name, kind in PERSON_FIELDS:
                    value = person.get(field_name)
                    if self.needs_translation(value):
                        items.append(TranslationItem(
                            key=f"{side[0]}{i}.{field_name}",
                            text=str(value), kind=kind))

        prop = extraction.get("property_details") or {}
        for field_name, kind in PROPERTY_FIELDS:
            value = prop.get(field_name)
            if self.needs_translation(value):
                items.append(TranslationItem(
                    key=f"property.{field_name}", text=str(value), kind=kind))

        meta = extraction.get("document_details") or {}
        for field_name, kind in DOCUMENT_FIELDS:
            value = meta.get(field_name)
            if self.needs_translation(value):
                items.append(TranslationItem(
                    key=f"document.{field_name}", text=str(value), kind=kind))
        return items

    def run(self, extraction: dict[str, Any]) -> StageOutcome:
        started = time.monotonic()
        items = self._pending(extraction)

        if not items:
            return StageOutcome.success(
                StageName.TRANSLATE, translated=0, pending=0, fields=[],
                detail="nothing to translate - every field is already English",
                source_language="eng_Latn", engine="none",
                duration_s=round(time.monotonic() - started, 2))

        try:
            result = self.service.translate(items)
        except Exception as exc:  # noqa: BLE001 - translation must never lose a deed
            _log.error("translation raised", exc_info=exc)
            return StageOutcome.success(
                StageName.TRANSLATE, translated=0, pending=len(items),
                fields=[i.key for i in items],
                detail=f"translation failed: {type(exc).__name__}: {exc}",
                engine="failed",
                duration_s=round(time.monotonic() - started, 2))

        written = 0
        for item in items:
            if not item.translated:
                continue
            container, key = self._field(extraction, item.key)
            if container is None:
                continue
            container[f"{key}_translated"] = item.translated
            written += 1

        detail = (f"{written} field(s) via {result.model or result.engine}"
                  if written else (result.error or "no field was translated"))
        return StageOutcome.success(
            StageName.TRANSLATE,
            translated=written,
            pending=len(result.untranslated),
            fields=[i.key for i in result.untranslated],
            languages={LANGUAGE_NAMES.get(k, k): v
                       for k, v in result.languages.items()},
            source_language=next(iter(result.languages), "eng_Latn"),
            engine=result.engine, model=result.model, device=result.device,
            detail=detail,
            duration_s=round(time.monotonic() - started, 2))

    def _field(self, extraction: dict[str, Any], path: str) -> Any:
        """Resolve `b1.name` / `property.village` to a (container, key) pair."""
        prefix, _, field_name = path.partition(".")
        if prefix == "property":
            return extraction.get("property_details") or {}, field_name
        if prefix == "document":
            return extraction.get("document_details") or {}, field_name
        side = "buyer_details" if prefix[0] == "b" else "seller_details"
        try:
            index = int(prefix[1:]) - 1
        except ValueError:
            return None, field_name
        people = extraction.get(side) or []
        if 0 <= index < len(people) and isinstance(people[index], dict):
            return people[index], field_name
        return None, field_name


def surya_available(model_dir: str | Path) -> bool:
    path = Path(model_dir)
    return path.is_dir() and any(path.iterdir()) and shutil.which("python") is not None


#: Interpreters searched for Surya, in order. `venv_new` first: it is the one
#: built against a Python that exists on this machine. The original `venv` is
#: retained because a working install should not be discarded, but it was
#: created on a different machine and its path references may be stale.
SURYA_INTERPRETERS = (
    "models/SuryaOCR/venv_new/Scripts/python.exe",
    "models/SuryaOCR/venv/Scripts/python.exe",
    "models/SuryaOCR/venv_new/bin/python",
    "models/SuryaOCR/venv/bin/python",
)


def find_surya(root: str | Path | None = None) -> tuple[Path | None, Path | None]:
    """Locate the Surya interpreter and the runner script.

    Returns `(None, None)` when Surya is not installed. That is an ordinary
    outcome, not an error: deeds with an embedded text layer - most registered
    ones - never need it.
    """
    base = Path(root) if root else paths.SRC
    project = base.parent if root is None else Path(root)
    script = base / "tools" / "surya_runner.py"
    if not script.is_file():
        return None, None
    for relative in SURYA_INTERPRETERS:
        candidate = project / relative
        if candidate.is_file():
            return candidate, script
    return None, None
