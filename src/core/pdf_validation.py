"""PDF corruption detection, run before a document reaches OCR.

Why this exists: a corrupt PDF used to be discovered *by Surya*, minutes into a
GPU stage, as `PdfiumError: Failed to load document`. That spends the most
expensive resource in the system on a file that could never have worked, and the
failure arrives attributed to OCR rather than to the file.

Three levels, cheapest first, stopping at the first that settles the question:

  1. **File** - exists, non-empty, readable, `%PDF-` header, `%%EOF` present.
     Pure bytes; catches a renamed JPEG or a truncated download for the cost of
     reading a few KB.
  2. **Structure** - the document opens, its page count and metadata are
     readable, the cross-reference resolves. One `pymupdf.open`.
  3. **Pages** - each page object loads and reports usable dimensions.

Level 3 deliberately does **not** rasterise. Rendering every page of a 46-page
deed costs seconds and is the same work OCR is about to do anyway; loading the
page object and reading its geometry catches a broken page object without
paying for pixels. `deep=True` opts into rendering when a caller genuinely
wants it.

One library, not several. `pymupdf` is already the project's PDF reader, and a
second parser would mean two definitions of "valid" that could disagree - the
independent check here is the raw-bytes level, which shares no code with it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger("saledeed.pdf_validator")

#: Bumped when the rules change, and stored beside each result. A file marked
#: valid by an older validator can then be re-checked deliberately rather than
#: trusted forever.
VALIDATOR_VERSION = "1.0"

#: How far into the file the header may sit. Some producers emit a BOM or a
#: stray byte or two first, and every reader tolerates it.
_HEADER_WINDOW = 1024
#: How much of the tail to search for the end marker.
_TAIL_WINDOW = 4096

_HEADER = re.compile(rb"%PDF-\d\.\d")


class Status:
    """The vocabulary stored in the database and shown in the interface.

    `PROCESSING_ERROR` is the one that matters most. This validator runs only
    after a job has already failed, so "the file is fine" is a real and common
    answer - and it is the answer that decides whether a retry is worth
    offering. Calling a healthy PDF corrupt because something downstream broke
    would send the operator to repair a file that was never the problem.
    """

    VALID = "VALID"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    CORRUPTED_PDF = "CORRUPTED_PDF"
    PARTIALLY_CORRUPTED = "PARTIALLY_CORRUPTED"
    INCOMPLETE_PDF = "INCOMPLETE_PDF"
    EMPTY_PDF = "EMPTY_PDF"
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"
    INVALID_PDF = "INVALID_PDF"
    UNREADABLE_PDF = "UNREADABLE_PDF"
    PDF_PARSE_ERROR = "PDF_PARSE_ERROR"
    PDF_RENDER_ERROR = "PDF_RENDER_ERROR"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

    #: Every status a stored row may hold.
    ALL = (VALID, PROCESSING_ERROR, CORRUPTED_PDF, PARTIALLY_CORRUPTED,
           INCOMPLETE_PDF, EMPTY_PDF, PASSWORD_PROTECTED, INVALID_PDF,
           UNREADABLE_PDF, PDF_PARSE_ERROR, PDF_RENDER_ERROR, UNKNOWN_FAILURE)


#: Statuses that must never reach OCR. `VALIDATION_FAILED` is included: the
#: validator itself broke, so nothing is known about the file, and spending GPU
#: time on an unknown is the same gamble as spending it on a known-bad one.
#: Statuses where the file itself is the problem, so a plain retry would fail
#: again in exactly the same way. These are marked non-retryable: repeating a
#: GPU stage on a file that cannot be opened wastes the most expensive resource
#: in the system and tells the operator nothing new.
CORRUPT_STATUSES = frozenset({
    Status.CORRUPTED_PDF, Status.INCOMPLETE_PDF, Status.EMPTY_PDF,
    Status.PASSWORD_PROTECTED, Status.INVALID_PDF, Status.UNREADABLE_PDF,
    Status.PDF_PARSE_ERROR, Status.PDF_RENDER_ERROR,
})

#: Retrying is worth offering: either the file is fine and something else
#: broke, or enough of it is readable to be worth another pass.
RETRYABLE_STATUSES = frozenset({
    Status.VALID, Status.PROCESSING_ERROR, Status.PARTIALLY_CORRUPTED,
    Status.UNKNOWN_FAILURE,
})

#: `PARTIALLY_CORRUPTED` is *not* blocking. A 25-page deed with one unreadable
#: page still carries the parties and the schedule on the other 24, and refusing
#: to read it would lose a recoverable document. It is flagged loudly instead.


@dataclass
class ValidationResult:
    is_valid: bool
    status: str
    page_count: int = 0
    corrupted_pages: list[int] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    validator_version: str = VALIDATOR_VERSION
    validated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_corrupt(self) -> bool:
        """The file itself is the problem."""
        return self.status in CORRUPT_STATUSES

    @property
    def retryable(self) -> bool:
        """Whether offering Retry is honest. A corrupt file fails again
        identically, so the operator is offered Revalidate instead."""
        return self.status in RETRYABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "page_count": self.page_count,
            "corrupted_pages": list(self.corrupted_pages),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "validator_version": self.validator_version,
            "validated_at": self.validated_at,
        }


def _fail(status: str, code: str, message: str, **extra: Any) -> ValidationResult:
    return ValidationResult(is_valid=False, status=status, error_code=code,
                            error_message=message, **extra)


# ---------------------------------------------------------------------------
# Level 1 - the file itself
# ---------------------------------------------------------------------------


def validate_file(path: Path) -> ValidationResult | None:
    """Bytes only. Returns a failure, or None when the file looks like a PDF."""
    try:
        if not path.exists():
            return _fail(Status.UNREADABLE_PDF, "FILE_NOT_FOUND",
                         "The file no longer exists at its recorded location.")
        if not path.is_file():
            return _fail(Status.UNREADABLE_PDF, "NOT_A_FILE",
                         "The path is a directory, not a file.")
        size = path.stat().st_size
        if size == 0:
            return _fail(Status.EMPTY_PDF, "EMPTY_FILE",
                         "The file is empty (0 bytes).")
        if path.suffix.lower() != ".pdf":
            return _fail(Status.INVALID_PDF, "NOT_A_PDF_EXTENSION",
                         f"The file has a {path.suffix or 'missing'} extension, "
                         "not .pdf.")
        with path.open("rb") as handle:
            head = handle.read(_HEADER_WINDOW)
            if size > _TAIL_WINDOW:
                handle.seek(-_TAIL_WINDOW, 2)
            else:
                handle.seek(0)
            tail = handle.read()
    except PermissionError:
        return _fail(Status.UNREADABLE_PDF, "PERMISSION_DENIED",
                     "The file cannot be read; permission was refused.")
    except OSError as exc:
        return _fail(Status.UNREADABLE_PDF, "FILE_READ_ERROR",
                     f"The file could not be read: {exc.strerror or exc}.")

    if not _HEADER.search(head):
        # The commonest real case: something renamed to .pdf.
        return _fail(Status.INVALID_PDF, "INVALID_HEADER",
                     "The file does not begin with a PDF header, so it is not a "
                     "PDF regardless of its name.")
    if b"%%EOF" not in tail:
        # Not fatal on its own - some files carry trailing bytes - but combined
        # with a structural failure it names the cause precisely.
        _log.debug("no %%EOF marker", extra={"file": path.name})
    return None


# ---------------------------------------------------------------------------
# Levels 2 and 3 - structure and pages
# ---------------------------------------------------------------------------


def validate_pdf(file_path: str | Path, *, deep: bool = False,
                 max_pages: int | None = None) -> ValidationResult:
    """Validate one PDF. Never raises - every failure becomes a result.

    That total-catch is the point: this runs over operator-supplied files, and
    an exception escaping here would take down the batch that a corrupt file was
    supposed to be isolated from.
    """
    path = Path(file_path)
    _log.info("[PDF_VALIDATOR] starting validation: %s", path.name,
              extra={"file": str(path)})

    early = validate_file(path)
    if early is not None:
        _log.warning("[PDF_VALIDATOR] file validation failed: %s (%s)",
                     path.name, early.error_code,
                     extra={"file": str(path), "status": early.status,
                            "error_code": early.error_code})
        return early
    _log.debug("[PDF_VALIDATOR] file validation passed: %s", path.name)

    try:
        import pymupdf
    except ImportError:                                   # pragma: no cover
        return _fail(Status.UNKNOWN_FAILURE, "NO_PDF_LIBRARY",
                     "The PDF library is unavailable, so the file could not be "
                     "checked.")

    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                # Readable as a container, but nothing inside can be reached.
                return _fail(Status.PASSWORD_PROTECTED, "PASSWORD_PROTECTED",
                             "The PDF is password protected and cannot be read.")

            # MuPDF silently reconstructs a damaged file rather than refusing
            # it, so a truncated deed opens and reports pages as though nothing
            # were wrong. `is_repaired` is the only signal that it had to. A
            # validator that missed this would call a half-downloaded file
            # VALID - the worst answer it can give, because the operator then
            # has no reason to look at it.
            repaired = bool(getattr(doc, "is_repaired", False))

            page_count = doc.page_count
            if page_count <= 0:
                return _fail(Status.EMPTY_PDF, "ZERO_PAGES",
                             "The PDF opens but contains no pages.")

            # Touching the metadata and the xref forces the structure to be
            # parsed; a broken cross-reference table surfaces here rather than
            # halfway through OCR.
            _ = doc.metadata
            _ = doc.xref_length()
            _log.debug("[PDF_VALIDATOR] structure validation passed: %s (%d pages)",
                       path.name, page_count)

            corrupted = _bad_pages(doc, page_count, deep=deep,
                                   max_pages=max_pages)

    except Exception as exc:  # noqa: BLE001 - any parser error is a result
        message = str(exc).strip() or type(exc).__name__
        _log.warning("[PDF_VALIDATOR] validation failed: %s - %s", path.name,
                     message,
                     extra={"file": str(path), "error_code": "STRUCTURE_UNREADABLE"})
        truncated = _looks_truncated(path)
        if truncated:
            return _fail(Status.INCOMPLETE_PDF, "TRUNCATED_FILE",
                         "The PDF is incomplete - it ends before the document "
                         "does, which usually means the copy or download did "
                         "not finish.")
        return _fail(Status.CORRUPTED_PDF, "STRUCTURE_UNREADABLE",
                     "The PDF cannot be opened because its document structure "
                     f"is invalid ({message[:160]}).")

    if repaired:
        if _looks_truncated(path):
            return _fail(Status.INCOMPLETE_PDF, "TRUNCATED_FILE",
                         "The PDF is incomplete - it ends before the document "
                         "does, which usually means the copy or download did "
                         "not finish. Replacing the file should fix it.",
                         page_count=page_count, corrupted_pages=corrupted)
        return _fail(Status.CORRUPTED_PDF, "STRUCTURE_REPAIRED",
                     "The PDF's structure is damaged; it could only be read by "
                     "reconstructing it, so parts of the document may be "
                     "missing.",
                     page_count=page_count, corrupted_pages=corrupted)

    if corrupted:
        listed = ", ".join(str(n) for n in corrupted[:10])
        more = "" if len(corrupted) <= 10 else f" and {len(corrupted) - 10} more"
        # Every page unreadable is a broken document, not a partial one.
        if len(corrupted) >= page_count:
            _log.warning("[PDF_VALIDATOR] every page unreadable: %s", path.name)
            return _fail(Status.PDF_RENDER_ERROR, "ALL_PAGES_UNREADABLE",
                         "No page of this PDF could be read.",
                         page_count=page_count, corrupted_pages=corrupted)
        _log.warning("[PDF_VALIDATOR] status: PARTIALLY_CORRUPTED %s pages=%s",
                     path.name, listed,
                     extra={"file": str(path), "corrupted_pages": corrupted})
        return ValidationResult(
            is_valid=False, status=Status.PARTIALLY_CORRUPTED,
            page_count=page_count, corrupted_pages=corrupted,
            error_code="PAGE_RENDER_FAILED",
            error_message=(f"Page{'s' if len(corrupted) > 1 else ''} {listed}"
                           f"{more} could not be read."))

    _log.info("[PDF_VALIDATOR] status: VALID %s (%d pages)", path.name, page_count,
              extra={"file": str(path), "pages": page_count})
    return ValidationResult(is_valid=True, status=Status.VALID,
                            page_count=page_count)


def _looks_truncated(path: Path) -> bool:
    """No end-of-file marker in the tail.

    Only consulted once the parser has already failed: plenty of intact PDFs
    carry bytes after `%%EOF`, so its absence alone proves nothing - but a file
    that will not parse *and* has no terminator is an unfinished copy, and that
    is worth saying plainly because re-copying fixes it.
    """
    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(-min(size, _TAIL_WINDOW), 2 if size > _TAIL_WINDOW else 0)
            return b"%%EOF" not in handle.read()
    except OSError:
        return False


def _bad_pages(doc: Any, page_count: int, *, deep: bool,
               max_pages: int | None) -> list[int]:
    """Page numbers (1-based) that cannot be loaded.

    Loading the page and reading its rectangle is the cheap check: it forces the
    page object and its resources to resolve without rasterising. `deep` adds a
    tiny render - a 20-dpi pixmap, which is roughly a hundredth of the pixels
    OCR will ask for - for callers that want the stronger guarantee.
    """
    limit = page_count if max_pages is None else min(page_count, max_pages)
    bad: list[int] = []
    for index in range(limit):
        try:
            page = doc.load_page(index)
            rect = page.rect
            if rect.width <= 0 or rect.height <= 0:
                bad.append(index + 1)
                continue
            if deep:
                page.get_pixmap(dpi=20)
        except Exception:  # noqa: BLE001 - one bad page is a finding, not a crash
            bad.append(index + 1)
    return bad


def summarise(results: list[ValidationResult]) -> dict[str, int]:
    """Counts per status, plus the totals the dashboard shows."""
    counts = {name: 0 for name in Status.ALL}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    counts["TOTAL"] = len(results)
    counts["CORRUPT"] = sum(1 for r in results if r.is_corrupt)
    counts["RETRYABLE"] = sum(1 for r in results if r.retryable)
    return counts
