"""Prepare a deed for processing: remove overlays, guarantee a text layer.

One step, called once, between upload and OCR. Everything downstream - OCR,
extraction, validation, export and the viewer - then works from the same
prepared file, so there is exactly one notion of "the document".

    original.pdf -> [remove separable overlays] -> [ensure a text layer] -> cleaned.pdf

**What this can and cannot do**, because the difference is the whole design:

A watermark, seal or stamp that exists as a *separate object* - an
optional-content layer, an annotation, a distinct run in the content stream -
can be deleted outright. The page beneath is untouched: same fonts, same
tables, same spacing. That is genuinely lossless, and it measurably helps OCR,
because Surya reads a *rendered image* of the page and a diagonal stamp across
the text degrades recognition.

A seal, signature or handwritten note **burned into a scan** is a different
thing entirely. Those pixels replaced the original ones when the page was
scanned; what was underneath was never captured and exists nowhere in the file.
"Removing" it means inpainting - inventing content - on a document that
establishes ownership of property. This module refuses. It reports the page as
carrying an unremovable overlay and leaves the pixels alone.

Refusing is not a limitation to be worked around later. A plausible-looking
guess on a legal instrument is worse than a visible mark, because the mark is
obviously a mark and the guess is not.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .watermark import Fidelity, ScanResult, remove, scan

_log = logging.getLogger("saledeed.pdf_prepare")

#: Below this many characters a page is treated as having no usable text layer -
#: a scan. Matches `OcrStage.min_chars_per_page`, deliberately: the two must
#: agree about what "scanned" means or they will disagree about the same file.
MIN_CHARS_PER_PAGE = 40

#: Invisible text render mode. The glyphs are placed and selectable but not
#: drawn, which is what makes a scan searchable without altering its appearance.
_INVISIBLE = 3


@dataclass
class PrepareResult:
    """What preparation did, and what it deliberately did not do."""

    source: Path
    output: Path
    #: True when the output is a new file rather than a copy of the source.
    changed: bool = False
    watermarks_removed: list[str] = field(default_factory=list)
    watermarks_skipped: list[str] = field(default_factory=list)
    #: Pages that are pure images - an overlay there cannot be separated.
    scanned_pages: list[int] = field(default_factory=list)
    text_layer_pages: int = 0
    fidelity: Fidelity = Fidelity.LOSSLESS
    seconds: float = 0.0
    error: str = ""
    scan_result: ScanResult | None = None

    @property
    def ok(self) -> bool:
        return not self.error and self.output.is_file()

    @property
    def searchable(self) -> bool:
        """Whether every page can be selected and copied from."""
        return not self.scanned_pages or self.text_layer_pages > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.name,
            "output": self.output.name,
            "changed": self.changed,
            "watermarks_removed": self.watermarks_removed,
            "watermarks_skipped": self.watermarks_skipped,
            "scanned_pages": self.scanned_pages,
            "text_layer_pages": self.text_layer_pages,
            "fidelity": self.fidelity.value,
            "seconds": round(self.seconds, 2),
            "searchable": self.searchable,
            "error": self.error,
        }


def _require_pymupdf():  # noqa: ANN202
    import pymupdf

    return pymupdf


def pages_without_text(pdf_path: Path) -> list[int]:
    """Page numbers with no usable text layer."""
    pymupdf = _require_pymupdf()
    empty: list[int] = []
    try:
        with pymupdf.open(pdf_path) as doc:
            for number, page in enumerate(doc, start=1):
                if len(page.get_text().strip()) < MIN_CHARS_PER_PAGE:
                    empty.append(number)
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not inspect text layer", extra={
            "file": pdf_path.name, "error": f"{type(exc).__name__}: {exc}"})
    return empty


#: Base-14 font. Needs no embedding, so a Latin-only page stays small. Used only
#: for lines it can actually encode - see `_font_for`.
_FONT = "helv"

#: Font files that cover the Indic scripts this project handles. Nirmala UI
#: ships with Windows 8 and later and covers Kannada, Devanagari, Tamil, Telugu,
#: Malayalam, Bengali and Gujarati in one file - every script the pipeline can
#: encounter. The Linux entries are fallbacks for a non-Windows checkout.
#:
#: Names only. The Windows font directory is read from the environment rather
#: than assumed to be on C:, because it is not on every machine and a wrong
#: absolute path fails as "no Kannada font" rather than as "wrong drive".
_WINDOWS_FONTS = ("Nirmala.ttc", "NirmalaS.ttf")
_OTHER_FONTS = (
    "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf",
)


@lru_cache(maxsize=1)
def _unicode_font() -> str:
    """Path to a font that can carry non-Latin text, or "" if none is installed."""
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        for name in _WINDOWS_FONTS:
            candidate = Path(windir) / "Fonts" / name
            if candidate.is_file():
                return str(candidate)
    for path in _OTHER_FONTS:
        if Path(path).is_file():
            return path
    return ""


@lru_cache(maxsize=8)
def _glyphs(fontfile: str):  # noqa: ANN202
    pymupdf = _require_pymupdf()
    try:
        return pymupdf.Font(fontfile=fontfile) if fontfile else pymupdf.Font(_FONT)
    except Exception:  # noqa: BLE001
        return None


def _font_for(text: str) -> tuple[str, str] | None:
    """Choose `(fontname, fontfile)` for one line, or None if nothing can encode it.

    Returning None matters more than it looks. A font without the glyphs does not
    fail - it writes null bytes or question marks, and those become the
    "searchable" text of a legal document. Silence is the correct outcome there:
    the OCR text is already in the database, and a page without a layer is
    honestly unsearchable rather than dishonestly wrong.
    """
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        pass
    else:
        return _FONT, ""

    fontfile = _unicode_font()
    if not fontfile:
        return None
    font = _glyphs(fontfile)
    if font is None:
        return None
    if not all(font.has_glyph(ord(c)) for c in text if not c.isspace()):
        return None
    return "uni", fontfile


def _fit(pymupdf, text: str, box, fontname: str = _FONT) -> float:
    """Largest font size at which `text` fits `box`, in points.

    `insert_textbox` writes nothing and returns a negative number when the
    string is too wide - a failure that is easy to miss because it raises
    nothing. Measuring first turns that into a size, not a silent no-op.
    """
    try:
        unit = pymupdf.get_text_length(text, fontname=fontname, fontsize=1.0)
    except Exception:  # noqa: BLE001
        # Embedded fonts are not measurable by name. Half the point size per
        # character is a serviceable estimate; the fit is checked afterwards.
        unit = 0.5 * len(text)
    by_width = box.width / unit if unit > 0 else 24.0
    # 1.35 leaves room for the ascender and descender; sizing to the box height
    # alone overflows vertically and fails the same silent way.
    by_height = box.height / 1.35
    return max(2.0, min(24.0, by_width, by_height))


def add_text_layer(pdf_path: Path, page_text: dict[int, list[tuple[Any, str]]],
                   *, output: Path | None = None) -> int:
    """Write an invisible, selectable text layer onto scanned pages.

    `page_text` maps a page number to `(rect, text)` pairs - typically the line
    boxes an OCR engine reported. The glyphs are placed at those positions in
    render mode 3, so they are selectable and copyable but never drawn: the page
    looks exactly as it did, and the text lands where a reader would expect it
    when they drag a selection across it.

    Returns the number of pages given a layer.
    """
    pymupdf = _require_pymupdf()
    written = 0
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        _log.error("could not open for text layer", extra={
            "file": pdf_path.name, "error": str(exc)})
        return 0

    try:
        for number, lines in page_text.items():
            if number < 1 or number > doc.page_count or not lines:
                continue
            page = doc[number - 1]
            placed = skipped = 0
            for rect, text in lines:
                content = (text or "").strip()
                if not content:
                    continue
                box = rect if isinstance(rect, pymupdf.Rect) else pymupdf.Rect(*rect)
                if box.is_empty or box.is_infinite:
                    continue
                # Size the glyphs to the line box so a selection drag tracks the
                # visible ink rather than sitting above or below it. Both
                # dimensions matter: sizing by height alone produces a string
                # wider than its box, and `insert_textbox` then silently writes
                # nothing and returns a negative number.
                chosen = _font_for(content)
                if chosen is None:
                    skipped += 1
                    continue
                fontname, fontfile = chosen
                size = _fit(pymupdf, content, box, fontname)
                try:
                    written_height = page.insert_textbox(
                        box, content, fontsize=size, fontname=fontname,
                        fontfile=fontfile or None,
                        render_mode=_INVISIBLE, align=pymupdf.TEXT_ALIGN_LEFT)
                except Exception:  # noqa: BLE001 - one bad line must not lose the page
                    written_height = -1.0

                if written_height < 0:
                    # Still would not fit - a very short box, or a long line. Put
                    # the text on the baseline instead: it may overrun the box
                    # slightly, but invisible text that is present and roughly
                    # positioned beats no text at all.
                    try:
                        page.insert_text(
                            (box.x0, box.y1 - box.height * 0.2), content,
                            fontsize=size, fontname=fontname,
                            fontfile=fontfile or None,
                            render_mode=_INVISIBLE)
                    except Exception:  # noqa: BLE001
                        continue
                placed += 1
            if placed:
                written += 1
            if skipped:
                # Named, not hidden: an operator who cannot select part of a page
                # should be able to find out why.
                _log.warning("%d line(s) had no font that could encode them",
                             skipped, extra={"file": pdf_path.name,
                                             "page": number, "lines": skipped})

        if written:
            # A full Indic font is ~1.5 MB, and it would be embedded whole in
            # every cleaned deed. Subsetting keeps only the glyphs actually used.
            try:
                doc.subset_fonts()
            except Exception:  # noqa: BLE001 - size is a nicety, the text is not
                pass

        target = output or pdf_path
        doc.save(target, incremental=(target == pdf_path and doc.can_save_incrementally()),
                 encryption=pymupdf.PDF_ENCRYPT_KEEP)
    except Exception as exc:  # noqa: BLE001
        _log.error("text layer failed", extra={
            "file": pdf_path.name, "error": f"{type(exc).__name__}: {exc}"})
        return 0
    finally:
        doc.close()

    _log.info("text layer written to %d page(s)", written,
              extra={"file": pdf_path.name, "pages": written})
    return written


def prepare(source: str | Path, output_dir: str | Path, *,
            remove_watermarks: bool = True,
            allow_lossy: bool = False) -> PrepareResult:
    """Produce the cleaned document everything downstream will use.

    Always returns a usable output path. If nothing can be removed the source is
    copied unchanged, so the caller has one file to reason about rather than a
    conditional. Never raises: a deed that cannot be cleaned must still be
    processed.
    """
    started = time.monotonic()
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source.stem}_clean{source.suffix or '.pdf'}"

    result = PrepareResult(source=source, output=target)

    if not source.is_file():
        result.error = f"file not found: {source}"
        result.output = source
        _log.error("prepare: source missing", extra={"file": str(source)})
        return result

    scan_result: ScanResult | None = None
    if remove_watermarks:
        try:
            scan_result = scan(source)
            result.scan_result = scan_result
            result.scanned_pages = scan_result.scanned_pages
        except Exception as exc:  # noqa: BLE001
            _log.warning("watermark scan failed", extra={
                "file": source.name, "error": f"{type(exc).__name__}: {exc}"})

    confirmed = scan_result.confirmed if scan_result else []
    if confirmed:
        try:
            removal = remove(source, target, scan_result=scan_result,
                             allow_lossy=allow_lossy)
            if removal.ok and removal.removed:
                result.changed = True
                result.fidelity = removal.fidelity
                result.watermarks_removed = [f.label for f in removal.removed]
                result.watermarks_skipped = [f.label for f in removal.skipped]
            else:
                result.watermarks_skipped = [f.label for f in confirmed]
        except Exception as exc:  # noqa: BLE001 - never lose the document
            _log.error("watermark removal failed", extra={
                "file": source.name, "error": f"{type(exc).__name__}: {exc}"})
            result.watermarks_skipped = [f.label for f in confirmed]

    if not result.changed:
        # No separable overlay, or removal could not act. Copy so the rest of
        # the pipeline has exactly one file to work from either way.
        try:
            shutil.copy2(source, target)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            result.output = source
            _log.error("prepare: copy failed", extra={
                "file": source.name, "error": result.error})
            return result

    result.scanned_pages = result.scanned_pages or pages_without_text(target)
    result.seconds = time.monotonic() - started

    _log.info(
        "prepared %s", source.name,
        extra={"file": source.name, "output": target.name,
               "watermarks_removed": len(result.watermarks_removed),
               "watermarks_skipped": len(result.watermarks_skipped),
               "scanned_pages": len(result.scanned_pages),
               "fidelity": result.fidelity.value,
               "seconds": round(result.seconds, 2)})

    if result.watermarks_skipped:
        _log.warning(
            "%d overlay(s) could not be removed - they are part of the page "
            "image and cannot be separated without inventing content",
            len(result.watermarks_skipped),
            extra={"file": source.name, "labels": result.watermarks_skipped})

    return result
