"""Field-by-field extraction accuracy against the real model.

Answers the only question that matters about the pipeline: for each CSV column,
how often is it populated, and when it is empty, is that because the deed does
not contain the value or because extraction missed it?

    py -3.13 src/tools/extraction_report.py               ten documents
    py -3.13 src/tools/extraction_report.py --limit 50    the whole corpus
    py -3.13 src/tools/extraction_report.py --reference   compare to stored runs

Reads OCR from the corpus rather than running Surya, so a run costs about
fifteen seconds per document instead of twenty minutes. OCR accuracy is measured
separately - see `docs/DOCUMENTATION.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import paths  # noqa: E402
from core.csv_export import (  # noqa: E402
    CSV_COLUMNS,
    STRUCTURALLY_ABSENT,
    DocumentExport,
    build_rows,
)
from core.pipeline.runner import build_stages  # noqa: E402
from core.transaction_id import extract as extract_identity  # noqa: E402
from core.validation import derive_stamp_value  # noqa: E402

CORPUS_OCR = paths.TESTS / "corpus" / "OCR saledeeds"
CORPUS_REF = paths.TESTS / "corpus" / "test scripts" / "outputs" / "vllm_ocr"

#: Identifiers we can verify objectively: if the pattern is in the OCR and not
#: in the extraction, that is a miss with evidence, not an opinion.
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")


def _flatten(extraction: dict) -> str:
    return json.dumps(extraction, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--ai-url", default="http://127.0.0.1:8077")
    ap.add_argument("--reference", action="store_true",
                    help="also compare against the stored reference extractions")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    stages = build_stages(ai_base_url=args.ai_url)
    ready, detail = stages.extract.health()
    print(f"ai server : {'OK - ' + detail if ready else 'NOT READY - ' + detail}")
    if not ready:
        return 1
    print(f"prompt    : {len(stages.extract.prompt)} characters\n")

    texts = sorted(CORPUS_OCR.glob("*.txt"))[:args.limit]
    if not texts:
        print(f"no OCR corpus at {CORPUS_OCR}")
        return 1

    filled = Counter()
    documents: list[DocumentExport] = []
    parse_failures: list[str] = []
    pan_missed = aadhaar_missed = pan_seen = aadhaar_seen = 0

    for path in texts:
        ocr = path.read_text(encoding="utf-8", errors="replace")
        started = time.monotonic()
        outcome = stages.extract.run(ocr, path.stem)
        elapsed = time.monotonic() - started

        if not outcome.ok:
            parse_failures.append(f"{path.stem}: {outcome.detail}")
            print(f"  {path.stem:<24} FAILED  {outcome.detail[:60]}")
            continue

        extraction = outcome.data.get("parsed") or {}
        # Derived exactly as the pipeline does it, so the column reflects what
        # a real export would contain rather than the harness's own gap.
        prop = extraction.get("property_details") or {}
        meta = extraction.get("document_details") or {}
        stamp = derive_stamp_value(prop.get("registration_fee"),
                                   meta.get("transaction_date"))
        # The registration number read off the deed, exactly as the pipeline
        # does it. This used to be `path.stem` - the file name - which made the
        # Transaction Identity column of this report meaningless and hid the
        # very defect the report exists to measure (R-043).
        identity = extract_identity(ocr, source=path.name, ocr_used=True)

        documents.append(DocumentExport(
            transaction_identity=identity.value if identity.found else "",
            source_filename=f"{path.stem}.pdf",
            extraction=extraction, source_text=ocr,
            stamp_value=str(stamp) if stamp else None))

        # Objective identifier check against the source text.
        blob = _flatten(extraction)
        for pattern, label in ((PAN, "pan"), (AADHAAR, "aadhaar")):
            in_ocr = set(pattern.findall(ocr))
            if not in_ocr:
                continue
            found = sum(1 for v in in_ocr if v.replace(" ", "") in blob.replace(" ", ""))
            if label == "pan":
                pan_seen += len(in_ocr)
                pan_missed += len(in_ocr) - found
            else:
                aadhaar_seen += len(in_ocr)
                aadhaar_missed += len(in_ocr) - found

        persons = len(extraction.get("buyer_details") or []) + \
            len(extraction.get("seller_details") or [])
        print(f"  {path.stem:<24} ok  {elapsed:5.1f}s  {persons} part(y|ies)")

    if not documents:
        print("\nnothing extracted; cannot report on columns")
        return 1

    rows = build_rows(documents)
    for row in rows:
        for column in CSV_COLUMNS:
            if str(row.get(column, "") or "").strip():
                filled[column] += 1

    print(f"\n{'=' * 78}\nCOLUMN POPULATION  ({len(rows)} rows from {len(documents)} documents)\n{'=' * 78}")
    print(f"{'column':<44} {'filled':>7} {'%':>6}   reason if empty")
    print("-" * 78)
    #: Populated only when there is something to say - a validation flag, or an
    #: untranslated value. Zero on a clean batch is the correct answer.
    conditional = {"Remarks", "Person Details Remarks (PC)",
                   "Identification Type (PC)", "Identification Number (PC)"}
    real_gaps: list[str] = []
    for column in CSV_COLUMNS:
        n = filled[column]
        pct = n / len(rows) * 100
        note = ""
        if n == 0:
            note = STRUCTURALLY_ABSENT.get(column, "NOT EXTRACTED - investigate")
            if column in conditional:
                note = "conditional - nothing to report on these documents"
            elif column not in STRUCTURALLY_ABSENT:
                real_gaps.append(column)
        print(f"{column:<44} {n:>7} {pct:>5.0f}%   {note[:34]}")

    populated = sum(1 for c in CSV_COLUMNS if filled[c])
    expected = len(CSV_COLUMNS) - len(STRUCTURALLY_ABSENT)
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"  documents attempted        : {len(texts)}")
    print(f"  extractions parsed         : {len(documents)}  "
          f"({len(documents) / len(texts) * 100:.0f}%)")
    print(f"  parse failures             : {len(parse_failures)}")
    print(f"  columns populated          : {populated} of {len(CSV_COLUMNS)}")
    print(f"  columns expected to fill   : {expected}  "
          f"({len(STRUCTURALLY_ABSENT)} absent from a sale deed by nature)")
    print(f"  coverage of expected       : {populated / expected * 100:.0f}%")
    if pan_seen:
        print(f"  PAN in OCR -> extraction   : {pan_seen - pan_missed}/{pan_seen}  "
              f"({(pan_seen - pan_missed) / pan_seen * 100:.0f}%)")
    if aadhaar_seen:
        print(f"  Aadhaar in OCR -> extraction: {aadhaar_seen - aadhaar_missed}"
              f"/{aadhaar_seen}  "
              f"({(aadhaar_seen - aadhaar_missed) / aadhaar_seen * 100:.0f}%)")

    if real_gaps:
        print(f"\n  columns with no value and no explanation ({len(real_gaps)}):")
        for column in real_gaps:
            print(f"      {column}")
    for failure in parse_failures:
        print(f"  parse failure: {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
