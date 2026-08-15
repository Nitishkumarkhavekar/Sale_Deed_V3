"""Measure Transaction Identity extraction against the real corpus.

    py -3.13 tools/identity_check.py

Ground truth comes from the filenames, which encode the serial and often the
financial year (`117`, `1451 HSG`, `2025-26-1463`). That is imperfect - some
files are named only by serial - so a mismatch is reported for inspection
rather than counted as a definite error.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.transaction_id import extract, find_candidates
from core import paths

CORPUS = paths.TESTS / "corpus" / "OCR saledeeds"
SERIAL = re.compile(r"(\d{2,6})")


#: A financial year at the start or end of a filename: "2025-26-1463",
#: "00082-2022-23", "1021 25.26", "2797-25-26 TRK". Anchored rather than
#: searched, because an unanchored pattern also eats the serial.
LEADING_YEAR = re.compile(r"^(?:20)?\d{2}\s*[-. ]\s*(?:20)?\d{2}\s*[-. ]?\s*")
TRAILING_YEAR = re.compile(r"\s*[-. ]\s*(?:20)?\d{2}\s*[-. ]\s*(?:20)?\d{2}\b.*$")


def expected_serial(stem: str) -> str | None:
    """The serial the filename implies, zero-padded, or None.

    Filenames are inconsistent - `117`, `1451 HSG`, `00082-2022-23`,
    `2025-26-1463`, `1021   25.26`. The year is stripped from whichever end it
    sits on, then the first remaining run of digits is the serial.

    Worth stating: an earlier version of this took the *last* digit group and
    scored the extractor at 78%. The extractor was right every time; the
    measurement was wrong. A ground truth is as capable of being buggy as the
    thing it judges.
    """
    trimmed = LEADING_YEAR.sub("", stem)
    trimmed = TRAILING_YEAR.sub("", trimmed)
    groups = SERIAL.findall(trimmed)
    if not groups:
        return None
    return groups[0].lstrip("0").rjust(5, "0") or None


def main() -> int:
    if not CORPUS.is_dir():
        print(f"SKIP: no corpus at {CORPUS}")
        return 0

    files = sorted(CORPUS.glob("*.txt"))
    found = agree = disagree = blank = unknown = 0
    ambiguous: list[str] = []
    mismatches: list[tuple[str, str, str]] = []

    print(f"{'file':22} {'extracted':22} {'conf':>5}  verdict")
    print("-" * 74)

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        result = extract(text, source=path.name, ocr_used=True)
        want = expected_serial(path.stem)

        if not result.found:
            blank += 1
            ambiguous.append(path.stem)
            print(f"  {path.stem[:20]:22} {'(blank)':22} {'-':>5}  "
                  f"{result.reason[:28]}")
            continue

        found += 1
        got_serial = result.value.split("-")[2]
        if want is None:
            unknown += 1
            verdict = "no truth in filename"
        elif got_serial == want:
            agree += 1
            verdict = "ok"
        else:
            disagree += 1
            mismatches.append((path.stem, result.value, want))
            verdict = f"MISMATCH want serial {want}"

        print(f"  {path.stem[:20]:22} {result.value:22} "
              f"{result.confidence:5.2f}  {verdict}")

    total = len(files)
    print()
    print(f"  files                : {total}")
    print(f"  extracted            : {found}")
    print(f"  left blank           : {blank}")
    print(f"  agree with filename  : {agree}")
    print(f"  disagree             : {disagree}")
    print(f"  filename gave no truth: {unknown}")

    checkable = agree + disagree
    if checkable:
        print(f"\n  accuracy where checkable: {agree}/{checkable} "
              f"= {100 * agree / checkable:.1f}%")

    if mismatches:
        print("\n  mismatches worth reading:")
        for stem, got, want in mismatches[:8]:
            print(f"    {stem:22} got {got}, filename implies serial {want}")

    if ambiguous:
        print(f"\n  left blank: {', '.join(ambiguous[:8])}")

    # Multi-candidate files are where the scoring earns its keep.
    multi = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len({c.value for c in find_candidates(text)}) > 1:
            multi += 1
    print(f"\n  files citing prior documents: {multi}")

    ok = disagree == 0 and found >= total * 0.9
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
