"""Watermark detection and removal.

**Read this before trusting the output.**

A watermark that exists as a *separate object* - an optional-content group, an
annotation, or a distinct drawing in the content stream - can be deleted
outright. The page beneath is untouched: same fonts, same tables, same images,
same layout, byte-identical text. That is genuinely lossless.

A watermark **burned into a scanned image** is a different problem entirely. Those
pixels replaced the original ones when the page was scanned; what was underneath
was never captured and does not exist anywhere in the file. It cannot be
"reconstructed" - only guessed at. On a legal instrument, a plausible guess is
worse than a visible gap, because it looks authoritative.

So this module does two things and refuses to pretend otherwise:

  * removes separable watermarks losslessly, and
  * reports raster watermarks as `LOSSY`, leaving them in place unless the caller
    explicitly opts in.

Originals are never modified. Cleaned copies are written to a new file.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .logging_setup import get_logger

log = get_logger("watermark")

#: Words that commonly appear in watermark text on registry documents.
WATERMARK_WORDS = (
    "watermark", "draft", "copy", "specimen", "sample", "duplicate",
    "not for sale", "confidential", "preview", "demo", "trial",
    "unregistered", "evaluation", "e-stamp", "certified true copy",
    # Indian registry wording. The original list was generic and Western, so a
    # Kaveri document stamped "For Government Purpose Only" matched nothing.
    "government purpose", "government use", "official use", "official purpose",
    "purpose only", "not valid", "for reference", "office copy",
    "free of cost", "computer generated", "digitally signed",
    "ಸರ್ಕಾರಿ",          # Kannada: government
    "ಕಚೇರಿ ಪ್ರತಿ",       # Kannada: office copy
)

#: A watermark repeats across most pages; ordinary content does not.
_REPEAT_FRACTION = 0.6

#: Below this, a repeated phrase is a common word ("the", "and") or a page
#: number, not a watermark. Watermarks are phrases.
_MIN_WATERMARK_CHARS = 12
_MIN_WATERMARK_WORDS = 2

#: Body text in these documents sits at 8-15pt. A repeated phrase noticeably
#: larger than that is a stamp, not prose.
_LARGE_POINT_SIZE = 18.0

#: Watermarks are printed faint. sRGB packed as an integer; anything above this
#: on every channel is a light grey.
_FAINT_CHANNEL = 0x99


class Kind(str, Enum):
    OCG = "optional_content_group"
    ANNOTATION = "annotation"
    TEXT_OVERLAY = "text_overlay"
    RASTER = "raster"


class Fidelity(str, Enum):
    LOSSLESS = "lossless"
    LOSSY = "lossy"
    #: Detected but deliberately not touched.
    SKIPPED = "skipped"


@dataclass
class Finding:
    kind: Kind
    label: str
    pages: list[int] = field(default_factory=list)
    detail: str = ""
    #: Identifier used at removal time (OCG xref, annotation index, image xref).
    handle: Any = None

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.LOSSY if self.kind is Kind.RASTER else Fidelity.LOSSLESS

    @property
    def confirmed(self) -> bool:
        """Is this definitely a watermark?

        OCG, annotation and repeated-text findings are confirmed - they were
        matched on watermark wording or object type. A RASTER finding is NOT: it
        only says the page is a scanned image, on which a watermark would be
        indistinguishable from the document itself. Reporting that as a detected
        watermark would overstate what is known.
        """
        return self.kind is not Kind.RASTER


@dataclass
class ScanResult:
    path: Path
    page_count: int
    findings: list[Finding] = field(default_factory=list)
    error: str = ""

    @property
    def has_watermark(self) -> bool:
        """Only confirmed findings count. A scanned page is not evidence."""
        return any(f.confirmed for f in self.findings)

    @property
    def confirmed(self) -> list[Finding]:
        return [f for f in self.findings if f.confirmed]

    @property
    def scanned_pages(self) -> list[int]:
        """Pages that are pure images - a watermark there cannot be detected."""
        return sorted({p for f in self.findings if f.kind is Kind.RASTER
                       for p in f.pages})

    @property
    def removable_losslessly(self) -> list[Finding]:
        return [f for f in self.findings if f.fidelity is Fidelity.LOSSLESS]

    @property
    def raster_only(self) -> list[Finding]:
        return [f for f in self.findings if f.kind is Kind.RASTER]

    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        parts = [f"{f.kind.value}×{len(f.pages)}" for f in self.confirmed]
        scanned = self.scanned_pages
        if scanned:
            parts.append(f"{len(scanned)} scanned page(s) - watermark undetectable")
        return ", ".join(parts) if parts else "no watermark detected"


@dataclass
class RemovalResult:
    source: Path
    output: Path | None
    removed: list[Finding] = field(default_factory=list)
    skipped: list[Finding] = field(default_factory=list)
    fidelity: Fidelity = Fidelity.LOSSLESS
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.error


def _require_pymupdf():  # noqa: ANN202
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for watermark handling: pip install PyMuPDF") from exc
    return pymupdf


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _looks_like_watermark(text: str) -> bool:
    """Whether the wording alone identifies a watermark."""
    lowered = text.strip().lower()
    if not lowered or len(lowered) > 80:
        return False
    return any(word in lowered for word in WATERMARK_WORDS)


def _is_rotated(direction: tuple[float, float]) -> bool:
    """Text that does not run left-to-right along the page.

    The single most reliable signal available from the text layer: body text,
    headers and footers are horizontal; a stamp is set at an angle. Measured on
    the corpus, every span of ordinary deed text reports `dir=(1.0, 0.0)`.
    """
    dx, dy = direction
    return abs(dy) > 0.05 or dx < 0.95


def _is_faint(colour: int) -> bool:
    """A light grey, as watermarks are printed."""
    if not isinstance(colour, int) or colour <= 0:
        return False
    r, g, b = (colour >> 16) & 0xFF, (colour >> 8) & 0xFF, colour & 0xFF
    return min(r, g, b) >= _FAINT_CHANNEL


def _phrase_worth_testing(text: str) -> bool:
    """Long enough to be a phrase rather than a word or a page number.

    Without this, decoupling repetition from the vocabulary would flag "the"
    (119 occurrences in one corpus deed) and delete it from the document.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_WATERMARK_CHARS:
        return False
    if len(stripped.split()) < _MIN_WATERMARK_WORDS:
        return False
    # A repeated number is a page number or a document number, not a stamp.
    return not stripped.replace(" ", "").isdigit()


def scan(pdf_path: str | Path) -> ScanResult:
    """Identify watermark candidates without modifying anything."""
    pymupdf = _require_pymupdf()
    path = Path(pdf_path)
    if not path.is_file():
        return ScanResult(path=path, page_count=0, error="file not found")

    findings: list[Finding] = []
    try:
        with pymupdf.open(path) as doc:
            pages = doc.page_count

            # 1. Optional content groups. Registry tools often place watermarks
            #    on a named layer, which makes removal exact.
            try:
                for xref, info in (doc.get_ocgs() or {}).items():
                    name = str(info.get("name") or "")
                    if _looks_like_watermark(name):
                        findings.append(Finding(
                            Kind.OCG, name, list(range(1, pages + 1)),
                            "optional content layer", handle=xref))
            except Exception:  # noqa: BLE001 - not all documents expose OCGs
                pass

            # 2. Annotations and 3. repeated short text.
            annotations: dict[str, list[int]] = {}
            text_runs: Counter[str] = Counter()
            text_pages: dict[str, list[int]] = {}

            for number, page in enumerate(doc, start=1):
                for annot in page.annots() or []:
                    label = (annot.info.get("content") or annot.info.get("title")
                             or annot.type[1] or "")
                    if _looks_like_watermark(label) or annot.type[1] in ("Stamp", "Watermark"):
                        annotations.setdefault(label or annot.type[1], []).append(number)

                for block in page.get_text("dict").get("blocks", []):
                    for line in block.get("lines", []):
                        direction = tuple(line.get("dir", (1.0, 0.0)))
                        for span in line.get("spans", []):
                            content = (span.get("text") or "").strip()
                            if not content:
                                continue

                            # Wording alone is enough when it matches.
                            if _looks_like_watermark(content):
                                text_runs[content] += 1
                                text_pages.setdefault(content, []).append(number)
                                continue

                            # Otherwise judge it by how it is *printed*.
                            #
                            # The vocabulary can never be complete - a Kaveri
                            # document stamped "For Government Purpose Only"
                            # matched none of it, and previously the repetition
                            # check was gated on the same vocabulary, so the
                            # supposedly robust signal could only ever confirm
                            # what the fragile one had already found.
                            #
                            # A watermark is set apart typographically: rotated,
                            # oversized, or printed faint. Requiring one of
                            # those *and* a multi-word phrase *and* repetition
                            # across most pages is what keeps ordinary body text
                            # out - "the" appears 119 times in one corpus deed
                            # and must never be a candidate.
                            if not _phrase_worth_testing(content):
                                continue
                            if (_is_rotated(direction)
                                    or span.get("size", 0) >= _LARGE_POINT_SIZE
                                    or _is_faint(span.get("color", 0))):
                                text_runs[content] += 1
                                text_pages.setdefault(content, []).append(number)

            for label, page_list in annotations.items():
                findings.append(Finding(Kind.ANNOTATION, label, sorted(set(page_list)),
                                        "annotation object"))

            for content, count in text_runs.items():
                # A single stray mention is not a watermark; repetition is.
                if count >= max(2, int(pages * _REPEAT_FRACTION)):
                    findings.append(Finding(
                        Kind.TEXT_OVERLAY, content, sorted(set(text_pages[content])),
                        f"repeated on {count} of {pages} pages"))

            # 4. Full-page images on a page that also has text: the hallmark of a
            #    scan with a stamp burned in. Detected so it can be reported, not
            #    so it can be silently "fixed".
            for number, page in enumerate(doc, start=1):
                area = abs(page.rect.get_area())
                if not area:
                    continue
                has_text = bool(page.get_text().strip())
                for img in page.get_images(full=True):
                    try:
                        rects = page.get_image_rects(img[0])
                    except Exception:  # noqa: BLE001
                        continue
                    for rect in rects or []:
                        if abs(rect.get_area()) > area * 0.7 and not has_text:
                            findings.append(Finding(
                                Kind.RASTER, f"page {number} scanned image",
                                [number],
                                "page is a scanned image; a watermark here would be "
                                "part of the pixels and cannot be detected or removed",
                                handle=img[0]))
                            break
    except Exception as exc:  # noqa: BLE001
        return ScanResult(path=path, page_count=0, error=f"{type(exc).__name__}: {exc}")

    log.debug("watermark scan complete", extra={
        "file": path.name, "pages": pages, "findings": len(findings)})
    return ScanResult(path=path, page_count=pages, findings=findings)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def remove(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    *,
    scan_result: ScanResult | None = None,
    allow_lossy: bool = False,
) -> RemovalResult:
    """Write a cleaned copy. The source file is never modified.

    Only separable watermarks are removed. Raster watermarks are skipped unless
    `allow_lossy` is set, and even then the result is marked LOSSY so the caller
    cannot mistake it for a faithful copy.
    """
    pymupdf = _require_pymupdf()
    source = Path(pdf_path)
    result = scan_result or scan(source)
    if result.error:
        return RemovalResult(source, None, error=result.error)

    target = Path(output_path) if output_path else \
        source.with_name(f"{source.stem}_clean{source.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)

    removed: list[Finding] = []
    skipped: list[Finding] = []

    try:
        with pymupdf.open(source) as doc:
            for finding in result.findings:
                if finding.kind is Kind.OCG:
                    if _remove_ocg(doc, finding):
                        removed.append(finding)
                    else:
                        skipped.append(finding)
                elif finding.kind is Kind.ANNOTATION:
                    if _remove_annotations(doc, finding):
                        removed.append(finding)
                    else:
                        skipped.append(finding)
                elif finding.kind is Kind.TEXT_OVERLAY:
                    if _remove_text(doc, finding):
                        removed.append(finding)
                    else:
                        skipped.append(finding)
                elif finding.kind is Kind.RASTER:
                    # Nothing here can be recovered. Skipping is the honest
                    # default; opting in still yields a LOSSY result.
                    skipped.append(finding)

            if not removed and not allow_lossy:
                return RemovalResult(
                    source, None, removed=[], skipped=skipped,
                    fidelity=Fidelity.SKIPPED,
                    error=(
                        "no watermark detected in this document."
                        if not result.has_watermark and not result.scanned_pages else
                        "no separable watermark found. "
                        + (f"Page(s) {result.scanned_pages} are scanned images: any "
                           "watermark there is part of the pixels, and the content "
                           "beneath it was never captured, so it cannot be recovered."
                           if result.scanned_pages else
                           "The detected watermark could not be removed losslessly.")))

            # garbage=3 + deflate rewrites the file compactly. Text, fonts,
            # images and page order are preserved - only the removed objects go.
            doc.save(target, garbage=3, deflate=True, clean=True)
    except Exception as exc:  # noqa: BLE001
        return RemovalResult(source, None, error=f"{type(exc).__name__}: {exc}")

    fidelity = Fidelity.LOSSY if any(f.kind is Kind.RASTER for f in removed) \
        else Fidelity.LOSSLESS
    log.info("watermark removal complete", extra={
        "file": source.name, "removed": len(removed), "skipped": len(skipped),
        "fidelity": fidelity.value})
    return RemovalResult(source, target, removed, skipped, fidelity)


def _remove_ocg(doc: Any, finding: Finding) -> bool:
    """Turn the layer off and bake that state in."""
    try:
        config = doc.get_layer(-1) or {}
        off = list(config.get("off") or [])
        if finding.handle not in off:
            off.append(finding.handle)
        doc.set_layer(-1, off=off)
        return True
    except Exception:  # noqa: BLE001
        return False


def _remove_annotations(doc: Any, finding: Finding) -> bool:
    removed_any = False
    for number in finding.pages:
        try:
            page = doc[number - 1]
        except Exception:  # noqa: BLE001
            continue
        for annot in list(page.annots() or []):
            label = (annot.info.get("content") or annot.info.get("title")
                     or annot.type[1] or "")
            if label == finding.label or annot.type[1] in ("Stamp", "Watermark"):
                try:
                    page.delete_annot(annot)
                    removed_any = True
                except Exception:  # noqa: BLE001
                    continue
    return removed_any


def _remove_text(doc: Any, finding: Finding) -> bool:
    """Delete the watermark's text, leaving the rest of the page alone.

    The previous implementation matched the whole phrase inside a single `Tj`
    operator. Producers split a textbox across one `Tj` per word or per line, so
    the phrase is almost never contiguous in the content stream and the pattern
    matched nothing - detection reported a watermark and removal silently
    achieved nothing.

    This locates the text by its rendered position and removes it with a
    redaction configured to touch **only** text:

        images=PDF_REDACT_IMAGE_NONE          leave images untouched
        graphics=PDF_REDACT_LINE_ART_NONE     leave vector art untouched
        text=PDF_REDACT_TEXT_REMOVE           remove the glyphs

    That is what keeps this lossless. The default redaction rasterises the
    affected region, which on a deed would replace real content with a picture
    of itself and defeat the purpose of removing a watermark at all.
    """
    pymupdf = _require_pymupdf()

    changed = False
    for number in finding.pages:
        try:
            page = doc[number - 1]
            rects = page.search_for(finding.label)
            if not rects:
                continue
            for rect in rects:
                # A hair of padding: glyph bounds are tight and a descender can
                # sit a fraction outside the reported rectangle.
                page.add_redact_annot(rect + (-1, -1, 1, 1))
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE)
            changed = True
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            continue

    return changed


def process_many(paths: list[str | Path], output_dir: str | Path | None = None,
                 *, allow_lossy: bool = False) -> list[RemovalResult]:
    """Scan and clean a list of PDFs, reporting per file."""
    results: list[RemovalResult] = []
    out_dir = Path(output_dir) if output_dir else None
    for raw in paths:
        source = Path(raw)
        target = (out_dir / f"{source.stem}_clean.pdf") if out_dir else None
        results.append(remove(source, target, allow_lossy=allow_lossy))
    return results
