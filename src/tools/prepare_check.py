"""End-to-end check of document preparation.

    py -3.13 tools/prepare_check.py

Builds a watermarked deed, prepares it, and measures what preparation actually
achieved: is the overlay gone, is the content intact, is the page still
selectable, and does the rendered image - which is what Surya reads - improve.

Real corpus deeds are the control: preparation must not alter a document that
has nothing to remove.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf

from core.pdf_prepare import pages_without_text, prepare
from core import paths

WATERMARK = "For Government Purpose Only"
DEED_LINES = ("SALE DEED", "Consideration Rs. 30,00,000/-",
              "Seller PAN ABCDE1234F", "Survey No 455/1", "Bengaluru 560001")


def build_watermarked(path: pathlib.Path, pages: int = 6) -> pathlib.Path:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        for i, line in enumerate(DEED_LINES):
            page.insert_text((72, 90 + i * 22), line, fontsize=11)
        page.insert_textbox(pymupdf.Rect(50, 300, 560, 430), WATERMARK,
                            fontsize=30, color=(0.78, 0.78, 0.78),
                            align=pymupdf.TEXT_ALIGN_CENTER)
    doc.save(path)
    doc.close()
    return path


def text_of(path: pathlib.Path) -> str:
    with pymupdf.open(path) as doc:
        return "".join(page.get_text() for page in doc)


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="prepare_"))
    out = tmp / "cleaned"
    failures = 0

    print("=== a watermarked deed ===")
    source = build_watermarked(tmp / "deed.pdf")
    before = source.read_bytes()
    result = prepare(source, out)

    print(f"  ok               : {result.ok}")
    print(f"  changed          : {result.changed}")
    print(f"  removed          : {result.watermarks_removed}")
    print(f"  skipped          : {result.watermarks_skipped}")
    print(f"  fidelity         : {result.fidelity.value}")
    print(f"  seconds          : {result.seconds:.2f}")
    print(f"  searchable       : {result.searchable}")

    cleaned = text_of(result.output)
    gone = WATERMARK not in cleaned
    kept = [line for line in DEED_LINES if line not in cleaned]
    print(f"  watermark gone   : {gone}")
    print(f"  deed intact      : {not kept}{'' if not kept else '  MISSING ' + str(kept)}")
    print(f"  source untouched : {source.read_bytes() == before}")
    failures += (not gone) + bool(kept) + (source.read_bytes() != before)

    print("\n  what Surya would read (rendered page, not the text layer):")
    with pymupdf.open(source) as a, pymupdf.open(result.output) as b:
        raw = a[0].get_pixmap(dpi=110)
        clean = b[0].get_pixmap(dpi=110)
        # A watermark adds mid-grey ink across the page. Fewer non-white pixels
        # after cleaning means less for the recogniser to confuse with glyphs.
        def ink(pix):
            data = pix.samples
            return sum(1 for i in range(0, len(data), pix.n) if data[i] < 240)
        before_ink, after_ink = ink(raw), ink(clean)
    print(f"    ink pixels before: {before_ink:,}")
    print(f"    ink pixels after : {after_ink:,}")
    print(f"    reduction        : {100 * (before_ink - after_ink) / max(1, before_ink):.1f}%")
    if after_ink >= before_ink:
        print("    !! cleaning did not reduce ink - OCR would not improve")
        failures += 1

    print("\n=== text selection ===")
    empty = pages_without_text(result.output)
    print(f"  pages without a text layer: {empty or 'none - every page is selectable'}")
    failures += bool(empty)

    print("\n=== real deeds (must be unchanged in substance) ===")
    for path in sorted((paths.TESTS / "corpus" / "saledeeds").glob("*.pdf"))[:3]:
        original = text_of(path)
        res = prepare(path, out)
        after = text_of(res.output)
        same = original == after
        print(f"  {'ok ' if same else '!! '}{path.name:26} "
              f"changed={res.changed} text identical={same}")
        if not same:
            failures += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
