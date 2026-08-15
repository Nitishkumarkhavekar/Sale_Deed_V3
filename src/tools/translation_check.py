"""End-to-end translation against the real model.

    py -3.13 tools/translation_check.py

Runs genuine deed phrasing in every required language through detection and the
model, then a full extraction through the pipeline stage and into a CSV, and
reports whether any non-English text survives.

Kept as a tool rather than a test because it loads a 2.5 GB model and takes
minutes on CPU - a suite that runs on every change cannot afford that.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.csv_export import DocumentExport, untranslated_cells, write_csv
from core.pipeline.stages import TranslateStage
from core.translation import TranslationItem, TranslationService, build_config, detect

NON_LATIN = re.compile(r"[^\x00-\x7F]")

#: Realistic deed phrasing, not dictionary words: a name, a road, a city.
SAMPLES: list[tuple[str, str, str]] = [
    ("Kannada",   "ರಮೇಶ್ ಕುಮಾರ್", "transliterate"),
    ("Kannada",   "ಮುಖ್ಯ ರಸ್ತೆ, ಬೆಂಗಳೂರು", "translate"),
    ("Hindi",     "रमेश कुमार", "transliterate"),
    ("Hindi",     "मुख्य सड़क, बेंगलुरु", "translate"),
    ("Telugu",    "రమేష్ కుమార్", "transliterate"),
    ("Telugu",    "ప్రధాన రహదారి", "translate"),
    ("Tamil",     "ரமேஷ் குமார்", "transliterate"),
    ("Tamil",     "பிரதான சாலை", "translate"),
    ("Malayalam", "രമേഷ് കുമാർ", "transliterate"),
    ("Gujarati",  "મુખ્ય રોડ", "translate"),
    ("Bengali",   "প্রধান রাস্তা", "translate"),
    ("Punjabi",   "ਮੁੱਖ ਸੜਕ", "translate"),
    ("Odia",      "ମୁଖ୍ୟ ରାସ୍ତା", "translate"),
    ("Urdu",      "مین روڈ", "translate"),
    ("Marathi",   "मुख्य रस्ता", "translate"),
]


def main() -> int:
    config = build_config()
    service = TranslationService(config)

    ok, detail = service.available()
    print(f"model     : {config.model_dir.name if config.model_dir else '-'}")
    print(f"available : {ok} - {detail}")
    if not ok:
        print("\nSKIP: install the model first "
              "(py -3.13 tools/setup.py --install-translation)")
        return 0
    print(f"probe     : {json.dumps(service.probe())[:120]}")

    print("\n=== every required language ===")
    items = [TranslationItem(key=f"{i}", text=text, kind=kind)
             for i, (_, text, kind) in enumerate(SAMPLES)]
    started = time.time()
    result = service.translate(items)
    elapsed = time.time() - started

    print(f"{'language':11} {'op':14} {'source':26} -> english")
    print("-" * 78)
    failures = 0
    for (language, _, kind), item in zip(SAMPLES, items):
        english = item.output
        clean = not NON_LATIN.search(english)
        if not clean:
            failures += 1
        print(f"  {language:9} {kind:14} {item.text[:24]:26} -> "
              f"{english[:32]}{'' if clean else '   <-- NOT ENGLISH'}")

    print(f"\n  {result.translated}/{len(items)} translated in {elapsed:.1f}s "
          f"on {result.device} ({result.engine})")
    print(f"  cache: {service.cache_stats()}")

    print("\n=== cache: a repeat must not hit the model ===")
    started = time.time()
    again = service.translate([TranslationItem(key="r", text=SAMPLES[0][1],
                                               kind="transliterate")])
    print(f"  repeat took {time.time() - started:.3f}s, "
          f"from_cache={again.items[0].from_cache}")

    print("\n=== a full deed through the stage and into CSV ===")
    extraction = {
        "document_details": {"transaction_date": "2024-06-15",
                             "registration_office": "ಬೆಂಗಳೂರು ಕಚೇರಿ"},
        "property_details": {"schedule_c_property_address": "ಮುಖ್ಯ ರಸ್ತೆ",
                             "village": "ಬೆಂಗಳೂರು", "district": "ಬೆಂಗಳೂರು"},
        "buyer_details": [{"name": "ರಮೇಶ್ ಕುಮಾರ್", "father_name": "ಸುರೇಶ್",
                           "address": "ಮುಖ್ಯ ರಸ್ತೆ, ಬೆಂಗಳೂರು",
                           "pan_card_number": "ABCDE1234F",
                           "aadhaar_number": "123456789012"}],
        "seller_details": [{"name": "John Smith", "address": "12 Church Street"}],
    }
    outcome = TranslateStage().run(extraction)
    print(f"  stage: translated={outcome.data['translated']} "
          f"pending={outcome.data['pending']} "
          f"languages={outcome.data.get('languages')} "
          f"in {outcome.data['duration_s']}s")

    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "out.csv"
        write_csv(target, [DocumentExport(transaction_identity="275/2024-25",
                                          extraction=extraction,
                                          source_filename="275.pdf")])
        with open(target, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

    remaining = untranslated_cells(rows)
    print(f"  CSV columns still non-English: {len(remaining)}")
    for column, value in remaining.items():
        print(f"      {column:34} = {value[:28]}")

    print("\n  identifiers must be untouched:")
    # The buyer's row, selected by relation rather than by position. Rows are
    # ordered seller-first, and the identifiers above belong to the buyer - so
    # `rows[0]` compared the seller's (correctly empty) PAN against the buyer's
    # expected value and reported the translation stage as corrupting
    # identifiers it never touches. A check that cries wolf about Aadhaar
    # corruption is worse than no check at all.
    buyer = next((r for r in rows if r.get("Transaction Relation (PC)") == "B"),
                 rows[0] if rows else {})
    for column, expected in (("PAN (PC)", "ABCDE1234F"),
                             ("Aadhaar Number (PC)", "123456789012")):
        actual = buyer.get(column, "")
        print(f"      {column:22} {actual:16} {'ok' if actual == expected else 'CHANGED'}")
        failures += 0 if actual == expected else 1

    print()
    passed = failures == 0 and not remaining
    print("RESULT:", "PASS" if passed else f"FAIL ({failures} bad, "
          f"{len(remaining)} column(s) not English)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
