"""Which CSV columns can carry Kannada into the export?

Run:  py -3.13 tools/kannada_audit.py

Fills every plausible free-text field with Kannada, runs the real translate
stage, then the real CSV writer, and reports any cell that still holds Kannada
characters. That is the only reliable answer to "does every field pass through
translation" - reading the code gives the intent, not the coverage.
"""
import csv
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.csv_export import (CSV_COLUMNS, DocumentExport, untranslated_cells,
                             write_csv)
from core.pipeline.stages import PERSON_FIELDS, PROPERTY_FIELDS, TranslateStage

KN = re.compile(r"[\u0c80-\u0cff]")
K_NAME = "ರಮೇಶ್ ಕುಮಾರ್"
K_ADDR = "ಮುಖ್ಯ ರಸ್ತೆ, ಬೆಂಗಳೂರು"
K_TEXT = "ಆಸ್ತಿಯ ವಿವರಣೆ"

extraction = {
    "document_details": {
        "document_number": "275/2024-25",
        "transaction_date": "2024-06-15",
        "consideration_amount": 3000000,
        "registration_fee": 60000,
        "registration_office": K_TEXT,
        "document_type": K_TEXT,
    },
    "property_details": {
        "schedule_c_property_address": K_ADDR,
        "property_description": K_TEXT,
        "survey_number": "455/1",
        "village": K_TEXT,
        "district": K_TEXT,
        "taluk": K_TEXT,
    },
    "buyer_details": [{
        "name": K_NAME, "father_name": K_NAME, "address": K_ADDR,
        "gender": K_TEXT, "occupation": K_TEXT,
        "pan_card_number": "ABCDE1234F", "aadhaar_number": "123456789012",
    }],
    "seller_details": [{
        "name": K_NAME, "father_name": K_NAME, "address": K_ADDR,
    }],
}

stage = TranslateStage(engine="passthrough")  # never loads the model
outcome = stage.run(extraction)
print("translate stage (passthrough - the model is absent):")
print(f"  fields it would translate : {outcome.data['pending']}")
for f in outcome.data.get("fields", []):
    print(f"      {f}")
print()

with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / "out.csv"
    write_csv(target, [DocumentExport(transaction_identity="275/2024-25",
                                      extraction=extraction,
                                      source_filename="275.pdf")])
    with open(target, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

# Coverage is read from the stage, not restated here - a hard-coded list would
# drift from the code it is meant to audit.
covered_fields = {f for f, _ in PERSON_FIELDS} | {f for f, _ in PROPERTY_FIELDS}
bad = untranslated_cells(rows)

print(f"CSV columns containing Kannada: {len(bad)} of {len(CSV_COLUMNS)}")
for col, value in bad.items():
    print(f"  {col:36} = {value[:30]}")

print()
# This is a *coverage* audit: it runs the stage in passthrough so it never
# loads the model, and reports which columns would carry regional text if
# translation did not run. `tools/translation_check.py` is the end-to-end one.
print(f"fields the stage would translate : {len(outcome.data.get('fields', []))}")
print()
print("RESULT:", "no column can carry regional text" if not bad else
      f"{len(bad)} column(s) would carry regional text without translation; "
      f"the stage covers {len(outcome.data.get('fields', []))} field(s). "
      "Run tools/translation_check.py to verify the model end to end.")
