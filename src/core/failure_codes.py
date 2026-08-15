"""Why a deed failed, in words an operator can act on.

The pipeline already records *that* a document failed, *which stage* failed, and
a technical detail. What it did not do was say **why** in language anyone but
the author could use: `failure_reason` held strings like

    EngineNotReadyError: llama-server HTTP 400: {"error":{"code":400,...

which names the component that gave up, not the problem the operator has to fix.

This module classifies an existing failure into a stable code and a sentence.
It reads what is already stored - the stage columns, `failure_reason`, and the
PDF validator's verdict - so nothing new has to be captured and no migration is
needed. Classification happens at read time, which also means a document that
failed before this existed is explained retrospectively.

The technical text is never discarded; it is returned separately so the
interface can keep it behind "Technical details" instead of putting a Python
exception in front of a clerk.
"""

from __future__ import annotations

import re
from typing import Any

#: Stable codes. The interface maps these to messages; logs and exports carry
#: the code, so a change of wording never breaks a downstream reader.
PDF_CORRUPTED = "PDF_CORRUPTED"
PDF_INVALID = "PDF_INVALID"
PDF_ENCRYPTED = "PDF_ENCRYPTED"
PDF_EMPTY = "PDF_EMPTY"
PDF_INCOMPLETE = "PDF_INCOMPLETE"
WATERMARK_REMOVAL_FAILED = "WATERMARK_REMOVAL_FAILED"
OCR_FAILED = "OCR_FAILED"
OCR_NO_TEXT = "OCR_NO_TEXT"
OCR_INSUFFICIENT_TEXT = "OCR_INSUFFICIENT_TEXT"
OCR_TIMEOUT = "OCR_TIMEOUT"
AI_SERVER_UNAVAILABLE = "AI_SERVER_UNAVAILABLE"
AI_INPUT_TOO_LARGE = "AI_INPUT_TOO_LARGE"
AI_EXTRACTION_FAILED = "AI_EXTRACTION_FAILED"
AI_PROCESSING_FAILED = "AI_PROCESSING_FAILED"
TRANSLATION_FAILED = "TRANSLATION_FAILED"
DATABASE_ERROR = "DATABASE_ERROR"
FILE_ACCESS_ERROR = "FILE_ACCESS_ERROR"
MEMORY_ERROR = "MEMORY_ERROR"
TIMEOUT = "TIMEOUT"
UNKNOWN_ERROR = "UNKNOWN_ERROR"

#: What the operator is told, and whether trying again is honest.
#: `retryable=False` means the file itself must change first - repeating the
#: stage would fail identically and cost another GPU pass.
MESSAGES: dict[str, tuple[str, bool]] = {
    PDF_CORRUPTED: ("PDF file is corrupted or cannot be read.", False),
    PDF_INVALID: ("The file is not a valid PDF.", False),
    PDF_ENCRYPTED: ("PDF is password protected and cannot be opened.", False),
    PDF_EMPTY: ("PDF is empty - it contains no pages.", False),
    PDF_INCOMPLETE: ("PDF is incomplete or truncated - the copy did not finish.",
                     False),
    WATERMARK_REMOVAL_FAILED: ("Watermark was not removed.", True),
    OCR_FAILED: ("OCR was not completed.", True),
    OCR_NO_TEXT: ("OCR completed but no readable text was extracted.", True),
    OCR_INSUFFICIENT_TEXT: (
        "OCR found too little text to be a readable deed - the pages are "
        "probably scans that need a real OCR pass.", True),
    OCR_TIMEOUT: ("OCR took too long and was stopped.", True),
    AI_SERVER_UNAVAILABLE: ("The AI server is not reachable.", True),
    AI_INPUT_TOO_LARGE: (
        "The deed is too long for the model to read in one pass.", True),
    AI_EXTRACTION_FAILED: (
        "The AI returned no usable data for this deed.", True),
    AI_PROCESSING_FAILED: ("AI processing failed for this deed.", True),
    TRANSLATION_FAILED: ("Translation of the extracted values failed.", True),
    DATABASE_ERROR: ("The result could not be saved to the database.", True),
    FILE_ACCESS_ERROR: ("The file could not be read from disk.", True),
    MEMORY_ERROR: ("The machine ran out of memory or GPU memory.", True),
    TIMEOUT: ("Processing timed out.", True),
    UNKNOWN_ERROR: ("Processing failed for an unrecognised reason.", True),
}

#: Which stage each code belongs to, for the "Stage:" line.
STAGES: dict[str, str] = {
    PDF_CORRUPTED: "PDF Validation", PDF_INVALID: "PDF Validation",
    PDF_ENCRYPTED: "PDF Validation", PDF_EMPTY: "PDF Validation",
    PDF_INCOMPLETE: "PDF Validation",
    WATERMARK_REMOVAL_FAILED: "Watermark Removal",
    OCR_FAILED: "OCR", OCR_NO_TEXT: "OCR", OCR_INSUFFICIENT_TEXT: "OCR",
    OCR_TIMEOUT: "OCR",
    AI_SERVER_UNAVAILABLE: "AI Extraction",
    AI_INPUT_TOO_LARGE: "AI Extraction",
    AI_EXTRACTION_FAILED: "AI Extraction",
    AI_PROCESSING_FAILED: "AI Extraction",
    TRANSLATION_FAILED: "Translation",
    DATABASE_ERROR: "Database Save",
    FILE_ACCESS_ERROR: "File Access",
    MEMORY_ERROR: "Resources", TIMEOUT: "Processing",
    UNKNOWN_ERROR: "Processing",
}

#: Matched against the stored technical text, most specific first. Order is the
#: whole design: "exceeds the available context size" must be recognised as a
#: length problem before the generic "HTTP 400" makes it look like a server
#: fault, because the two need completely different actions from the operator.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"exceeds the available context size|n_ctx|too many tokens", AI_INPUT_TOO_LARGE),
    (r"could not be made because the target machine actively refused"
     r"|EngineNotReadyError|connection refused|URLError|llama-server (exited|did not)",
     AI_SERVER_UNAVAILABLE),
    (r"no parseable JSON|no usable JSON|unparseable|empty response",
     AI_EXTRACTION_FAILED),
    (r"out of memory|CUDA out of memory|OutOfMemoryError|MemoryError", MEMORY_ERROR),
    (r"timed out|TimeoutExpired|timeout", TIMEOUT),
    (r"password|encrypted", PDF_ENCRYPTED),
    (r"incomplete|truncated", PDF_INCOMPLETE),
    (r"corrupt|structure is invalid|cannot be opened|PdfiumError|FileDataError",
     PDF_CORRUPTED),
    (r"not a valid PDF|does not begin with a PDF header|invalid header", PDF_INVALID),
    (r"empty \(0 bytes\)|contains no pages|zero pages", PDF_EMPTY),
    (r"no usable text|produced no text|no text layer", OCR_NO_TEXT),
    (r"only \d+ characters across|pure scan|insufficient text",
     OCR_INSUFFICIENT_TEXT),
    (r"watermark", WATERMARK_REMOVAL_FAILED),
    (r"OCR", OCR_FAILED),
    (r"translat", TRANSLATION_FAILED),
    (r"IntegrityError|OperationalError|database|psycopg", DATABASE_ERROR),
    (r"PermissionError|FileNotFoundError|OSError|could not read", FILE_ACCESS_ERROR),
)

#: When the text says nothing recognisable, the stage that failed still does.
_STAGE_FALLBACK = {
    "ocr": OCR_FAILED, "extract": AI_PROCESSING_FAILED,
    "translate": TRANSLATION_FAILED, "validate": AI_EXTRACTION_FAILED,
}


def classify(document: Any) -> dict[str, Any] | None:
    """Explain one failed document, or None if it did not fail.

    Reads only what is already stored, so nothing needs re-running and
    documents that failed before this existed are still explained.
    """
    technical = str(getattr(document, "failure_reason", "") or "").strip()
    validation_status = getattr(document, "validation_status", None)
    validation_code = getattr(document, "validation_error_code", None)
    validation_message = getattr(document, "validation_error_message", None)

    stage = _failed_stage(document)
    overall = getattr(getattr(document, "overall_state", None), "value", "")
    if not technical and not validation_status and stage == "":
        return None
    if overall not in ("failed", "needs_review") and stage == "":
        return None

    # The PDF validator's verdict wins when it condemned the file: it examined
    # the bytes, while every other stage only saw a symptom.
    code = _from_validation(validation_status)
    if code is None:
        code = _from_text(technical) or _STAGE_FALLBACK.get(stage, UNKNOWN_ERROR)

    message, retryable = MESSAGES.get(code, MESSAGES[UNKNOWN_ERROR])
    detail = validation_message or technical
    stored_retryable = getattr(document, "is_retryable", None)
    if stored_retryable is not None:
        retryable = bool(stored_retryable)

    return {
        "code": code,
        "reason": message,
        "stage": STAGES.get(code, "Processing"),
        # Kept apart from `reason` on purpose: a Python exception belongs behind
        # "Technical details", not in front of a clerk.
        "technical": _sanitise(detail),
        "retryable": retryable,
        "failed_stage": stage or "unknown",
    }


def _failed_stage(document: Any) -> str:
    for stage in ("ocr", "extract", "translate", "validate"):
        state = getattr(document, f"{stage}_state", None)
        if getattr(state, "value", "") == "failed":
            return stage
    return ""


def _from_validation(status: str | None) -> str | None:
    return {
        "CORRUPTED_PDF": PDF_CORRUPTED, "INVALID_PDF": PDF_INVALID,
        "PASSWORD_PROTECTED": PDF_ENCRYPTED, "EMPTY_PDF": PDF_EMPTY,
        "INCOMPLETE_PDF": PDF_INCOMPLETE, "UNREADABLE_PDF": FILE_ACCESS_ERROR,
        "PDF_PARSE_ERROR": PDF_CORRUPTED, "PDF_RENDER_ERROR": PDF_CORRUPTED,
    }.get(status or "")


def _from_text(text: str) -> str | None:
    if not text:
        return None
    for pattern, code in _PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return None


def _sanitise(text: str) -> str:
    """Trim the technical detail to something safe to show on request.

    A stack trace is removed outright - it names internal paths and adds
    nothing an operator can use - and the remainder is bounded so one runaway
    message cannot fill the page.
    """
    cleaned = str(text or "").strip()
    if "Traceback" in cleaned:
        cleaned = cleaned.split("Traceback")[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:400]
