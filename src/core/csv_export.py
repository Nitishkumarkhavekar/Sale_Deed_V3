"""42-column CSV export, matching `saledeed main/refering docs/example.csv`.

Structure that matters: **one row per person, not per document.** Document-level
fields (transaction identity, amount, property address, stamp value) repeat on
every party's row, and `Transaction Relation (PC)` carries `S` or `B`. A deed with
one buyer and three sellers produces four rows sharing a `Report Serial Number`.

Two conventions read off the reference file rather than assumed:

  * `Transaction Date` is **DD-MM-YYYY** (`19-07-2025`). The model emits ISO
    (`2025-07-19`), so it is converted here. Exporting ISO would silently produce
    a file the receiving system misreads.
  * `State Code` holds a state *name* ("Karnataka"), not a code, and
    `Country Code` / `Country` / `Nationality` hold "IN".

Identifier columns are written as text. `example.csv` has Aadhaar values
corrupted to `6.63E+11` because a spreadsheet coerced 12-digit strings to floats;
that loses the number permanently. See `excel_safe` below.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .validation import Disposition, ValidationReport

#: The 42 columns, in order, exactly as they appear in example.csv.
CSV_COLUMNS: tuple[str, ...] = (
    "Report Serial Number",
    "Original Report Serial Number",
    "Transaction Date",
    "Transaction Identity",
    "Transaction Type",
    "Transaction Amount",
    "Property Type",
    "Whether property is within municipal limits",
    "Property Address",
    "City / Town",
    "Postal Code",
    "State Code",
    "Country Code",
    "Stamp Value",
    "Remarks",
    "Transaction Relation (PC)",
    "Transaction Amount related to the person (PC)",
    "Person Name (PC)",
    "Person Type (PC)",
    "Gender (PC)",
    "Father's Name (PC)",
    "PAN (PC)",
    "Aadhaar Number (PC)",
    "Form 60 Acknowledgement (PC)",
    "Identification Type (PC)",
    "Identification Number (PC)",
    "Date of Birth/ Incorporation (PC)",
    "Nationality/Country of Incorporation (PC)",
    "Address Type (PC-L)",
    "Address (PC-L)",
    "City/Town (PC-L)",
    "Pin Code (PC-L)",
    "State (PC-L)",
    "Country (PC-L)",
    "Primary STD Code (PC)",
    "Primary Phone Number (PC)",
    "Primary Mobile Number (PC)",
    "Secondary STD Code (PC)",
    "Secondary Phone Number (PC)",
    "Secondary Mobile Number (PC)",
    "Email (PC)",
    "Person Details Remarks (PC)",
)

FAILED_COLUMNS: tuple[str, ...] = (
    "Transaction Identity",
    "Source Filename",
    "Failed Stage",
    "Processing Status",
    "Reason",
    "Flags",
    "Confidence",
)

_log = logging.getLogger("saledeed.export")

COUNTRY = "IN"

#: Indian PIN codes are six digits, and deeds write them with a space after
#: the third - "BENGALURU-560 015". Requiring six consecutive digits missed
#: those, which is most of why Postal Code was filled on a third of rows.
#:
#: The lookarounds exclude a decimal, not merely a digit. A schedule states
#: measurements - "7.315215 metres", "64.660488 square metres" - and the
#: fractional part of every one of them is six digits. Reading those as
#: postal codes put 660488 in the Postal Code column of a real document.
PINCODE = re.compile(r"(?<![\d.])([1-9]\d{2})[ \u00a0-]?(\d{3})(?!\d)(?!\.\d)")
ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


@dataclass
class DocumentExport:
    """One deed ready for export."""

    transaction_identity: str
    #: Extraction JSON. Persons may carry `name_translated` / `address_translated`
    #: added by the translation stage; those are preferred when present.
    extraction: dict[str, Any]
    report: ValidationReport | None = None
    source_filename: str = ""
    #: Passthrough. The Stamp Value formula is undefined (docs/DECISIONS ADR-010),
    #: so nothing is computed here - whatever the caller supplies is written.
    stamp_value: str | None = None
    #: The document's OCR text, when the caller has it. Used only to classify
    #: the property and its municipal status: the deed states both in prose the
    #: extraction schema has no field for, and reading it from the address alone
    #: found 12% against the 96% that are actually stated somewhere.
    source_text: str = ""
    serial: int | None = None
    extras: dict[str, str] = field(default_factory=dict)


@dataclass
class FailedDocument:
    transaction_identity: str
    source_filename: str = ""
    failed_stage: str = ""
    processing_status: str = ""
    reason: str = ""
    flags: str = ""
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def iso_to_ddmmyyyy(value: str | None) -> str:
    """2025-07-19 -> 19-07-2025. Non-ISO input is passed through untouched."""
    m = ISO_DATE.match(str(value or ""))
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else str(value or "")


def extract_pincode(*texts: str | None) -> str:
    """First plausible Indian PIN code found, searched last-field-first.

    Addresses put the PIN at the end ("... Karnataka - 562123"), so the final
    match in the string is the more likely one.
    """
    for text in texts:
        if not text:
            continue
        matches = PINCODE.findall(str(text))
        if matches:
            # `findall` yields the two halves; rejoin without the separator.
            return "".join(matches[-1])
    return ""


#: Characters Excel, LibreOffice and Google Sheets treat as the start of a
#: formula. A cell beginning with one of these is evaluated on open.
FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _defuse(text: str) -> str:
    """Stop a spreadsheet evaluating a cell as a formula.

    Deed content is third-party data: a party name, an address or a remark comes
    from a document this application did not write. A name recorded as
    `=cmd|'/c calc'!A0` is executed when the clerk opens the export, which turns
    a CSV into code execution on their machine.

    A leading apostrophe is the conventional defence - spreadsheets read it as
    "treat the rest as text" and do not display it. Values that do not begin with
    a trigger are returned untouched, so ordinary text and negative numbers
    written as `-5` in a numeric column are unaffected: only the *leading*
    character matters, and numeric columns are built by this module, not by the
    model.
    """
    return f"'{text}" if text[:1] in FORMULA_TRIGGERS else text



def untranslated_cells(rows: list[dict[str, str]]) -> dict[str, str]:
    """Columns still holding non-English text, with the first offending value.

    Checks **every** script the application supports, not just Kannada: a deed
    from another state fails the same way, and a Kannada-only check would report
    a Telugu export as clean.

    Used only to *report* - never to remove. Dropping a value because it is in
    the wrong script would lose data from a legal record, and a blank cell is
    worse than a Kannada one: a reader can see Kannada and act on it, but cannot
    see an absence.
    """
    from .translation import needs_translation

    found: dict[str, str] = {}
    for row in rows:
        for column, value in row.items():
            if value and column not in found and needs_translation(str(value)):
                found[column] = str(value)
    return found


def _translated(source: dict[str, Any], key: str) -> Any:
    """The translated value if the translate stage produced one, else the
    original.

    Every column carrying free text must go through this. Reading a raw key
    directly is how `Property Address` shipped in Kannada while the identical
    address on the person row shipped in English - the stage had done the work
    and one call site ignored it.
    """
    return source.get(f"{key}_translated") or source.get(key)


def _clean(value: Any) -> str:
    """Render a value for CSV. Collapses newlines - a cell must stay one line."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _defuse(re.sub(r"\s+", " ", str(value)).strip())


#: Smallest unit the split is allowed to leave over: one paisa. Shares are
#: rounded to this and the remainder handed out one unit at a time, so a set of
#: shares always adds back to the deed's own consideration.
_PAISA = Decimal("0.01")


def _to_decimal(value: Any) -> Decimal | None:
    """Read a consideration as a number, or None if it is not one.

    The column is populated from `Property.sale_consideration`, which is already
    a `Decimal`, but it also survives a round trip through the CSV as text - and
    a deed occasionally carries a value the model wrote with separators or a
    currency mark. Anything that cannot be read as a number returns None, and
    the caller leaves the cell exactly as it found it rather than inventing a
    figure for a tax return.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    text = str(value or "").strip()
    if not text:
        return None
    # Currency marks, thousands separators (Indian or Western grouping) and
    # stray spaces. Not a general parser: anything left that is not a plain
    # number is refused below.
    text = re.sub(r"[₹$,\s]", "", text)
    text = text.removeprefix("Rs.").removeprefix("Rs").removeprefix("INR")
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def person_shares(total: Any, count: int) -> list[str]:
    """Split a deed's consideration equally between `count` parties.

    **Each side is split on its own.** A deed of ₹1,000 with four buyers and two
    sellers gives ₹250 to every buyer and ₹500 to every seller: the buyers share
    the whole consideration between themselves and so do the sellers, because
    each side is a complete account of the same transaction seen from one end.
    Dividing by six would report a transaction that never happened.

    The shares always add back to `total`. An amount that does not divide evenly
    leaves a remainder, and dropping it would understate the deed - ₹1,000 over
    three parties as ₹333.33 each reports ₹999.99. The remainder is handed out a
    paisa at a time to the earliest parties, so three parties receive ₹333.34,
    ₹333.33 and ₹333.33. Deterministic, so two exports of the same deed agree.

    Whole-rupee shares are rendered without decimals, which is what the column
    held before this and what every evenly-divided deed still produces.
    """
    if count <= 0:
        return []
    amount = _to_decimal(total)
    if amount is None:
        # Not a number - a blank, or something the model wrote that cannot be
        # read as one. Passed through unchanged: a wrong figure in this column
        # is worse than the original text, which at least shows what was found.
        return [_clean(total)] * count

    divisor = Decimal(count)
    # `quantize` rounds; the remainder is then distributed explicitly, so no
    # rounding mode can silently lose or invent money.
    base = (amount / divisor).quantize(_PAISA, rounding=ROUND_DOWN)
    shares = [base] * count
    remainder = amount - base * divisor
    units = int((remainder / _PAISA).to_integral_value(rounding=ROUND_HALF_UP))
    for index in range(min(units, count)):
        shares[index] += _PAISA

    whole = all(share == share.to_integral_value() for share in shares)
    return [_format_share(share, whole) for share in shares]


def _format_share(share: Decimal, whole: bool) -> str:
    """Whole rupees without a decimal point, part-rupees with exactly two."""
    if whole:
        return str(share.to_integral_value())
    return str(share.quantize(_PAISA))


def _identifier(value: Any, excel_safe: bool) -> str:
    """Render PAN or Aadhaar so it survives the trip.

    `excel_safe` wraps the value as an Excel formula string (`="123..."`), which
    stops Excel coercing a 12-digit Aadhaar into `6.63E+11` on open. It is
    non-standard CSV, so it is off by default: a correct file that Excel displays
    badly beats a corrupted file that looks fine.
    """
    text = _clean(value)
    if not text:
        return ""
    return f'="{text}"' if excel_safe else text


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


#: An alias marker. A deed writes "known also as" with `@`. The Name column
#: carries the primary name only, so everything from the `@` onward is removed.

_ALIAS = re.compile(r"\s*@.*$")


def primary_name(value: Any) -> str:
    """The name without an alias.

    `AAKASH SACHIDANAND MISHRA @ AAKASH MISHRA` becomes
    `AAKASH SACHIDANAND MISHRA`. The alias is what the deed says the party is
    *also* known by; the column holds one name, so the first is kept.

    The alias is not silently lost - `build_rows` logs it against the document,
    so the full form stays recoverable from the record of the run.
    """
    return _ALIAS.sub("", str(value or "")).strip()


def looks_like_a_name(value: Any) -> bool:
    """True when this could be somebody's name.

    The test is the presence of a letter, not the absence of punctuation. There
    is deliberately no list of permitted characters: a Kannada name is mostly
    combining marks - `್`, `ಿ`, `ಾ` - and a cleaner built from an allowed-set
    reads those as symbols and mangles the name it was meant to protect.
    """
    return any(ch.isalpha() for ch in str(value or ""))


def _fold(value: Any) -> str:
    """Comparison form: letters and digits of any script, lowercased."""
    return re.sub(r"[^\wऀ-෿]", "", str(value or "").lower())


def _party_key(person: dict[str, Any], relation: str) -> tuple[str, ...]:
    """What makes two extracted parties the same person.

    An Aadhaar settles it, and a PAN nearly does. Without either, a name and a
    father's name together are the strongest evidence available: two different
    people on one deed sharing both is possible in principle and appears nowhere
    in the corpus, whereas the same party listed twice does.
    """
    aadhaar = re.sub(r"\D", "", str(person.get("aadhaar_number") or ""))
    if len(aadhaar) == 12:
        return (relation, "aadhaar", aadhaar)
    pan = str(person.get("pan_card_number") or "").strip().upper()
    if pan:
        return (relation, "pan", pan)
    return (relation, "name", _fold(_translated(person, "name")),
            _fold(_translated(person, "father_name")))


def _document_fields(doc: DocumentExport, serial: int) -> dict[str, str]:
    prop = doc.extraction.get("property_details") or {}
    meta = doc.extraction.get("document_details") or {}
    # Prefer the translated value, exactly as the person fields do. Reading the
    # raw key here meant the translate stage did its work and the export ignored
    # it: `Property Address` arrived in the source language while
    # `Address (PC-L)` - the same address, on the person - arrived in English.
    address = _translated(prop, "schedule_c_property_address")

    # Remarks stays empty. It used to carry the validation summary, a
    # confidence score and the disposition, which is diagnostic output rather
    # than deed content - none of it is a fact about the transaction, and a
    # report consumed by another system should not carry this application's
    # opinion of its own work. All of it remains in the database and on the
    # Validation screen, where it belongs.
    remarks = ""

    return {
        "Report Serial Number": str(serial),
        "Original Report Serial Number": "",
        "Transaction Date": iso_to_ddmmyyyy(meta.get("transaction_date")),
        "Transaction Identity": _clean(doc.transaction_identity),
        "Transaction Type": "",
        "Transaction Amount": _clean(prop.get("sale_consideration")),
        # Address first - it is the specific description of *this* property -
        # then the body of the deed. Not the other way round: a mention of
        # agricultural income elsewhere must not outrank "residential site" in
        # the schedule.
        "Property Type": property_type(address, doc.source_text),
        "Whether property is within municipal limits":
            within_municipal_limits(address, doc.source_text),
        "Property Address": _clean(address),
        # The property's own address, and nothing else. This read the whole
        # OCR text as a fallback, which is how a registration office and a post
        # office ("post Pavagada") reached a column that must name the city the
        # *property* is in. A deed mentions many places; only one of them is
        # this one.
        "City / Town": city_town(address),
        "Postal Code": extract_pincode(address),
        "State Code": _clean(prop.get("state")),
        "Country Code": COUNTRY,
        # Derived by `validation.derive_stamp_value` and carried here by the
        # caller: the registration fee, halved before the cutoff (ADR-010).
        "Stamp Value": _clean(doc.stamp_value),
        "Remarks": remarks,
    }


def _person_fields(person: dict[str, Any], relation: str, amount: str,
                   person_remarks: str, excel_safe: bool,
                   property_city: str = "") -> dict[str, str]:
    # Prefer translated values when the translation stage has supplied them:
    # names are transliterated, addresses translated.
    # An alias is dropped: the column carries one name. See `primary_name`.
    name = primary_name(_translated(person, "name"))
    father = primary_name(_translated(person, "father_name"))
    address = _translated(person, "address")

    return {
        # Same code as the relation: S when this party is selling, B when
        # buying. Set here rather than in the document fields because a row is
        # a party, and one deed has both.
        "Transaction Type": TRANSACTION_TYPE_FOR_RELATION.get(relation, ""),
        "Transaction Relation (PC)": relation,
        "Transaction Amount related to the person (PC)": amount,
        "Person Name (PC)": _clean(name),
        "Person Type (PC)": person_type(person),
        "Gender (PC)": _clean(person.get("gender")),
        "Father's Name (PC)": _clean(father),
        "PAN (PC)": _identifier(person.get("pan_card_number"), excel_safe),
        "Aadhaar Number (PC)": _identifier(person.get("aadhaar_number"), excel_safe),
        "Form 60 Acknowledgement (PC)": "",
        "Identification Type (PC)": identification_type(person),
        # Always blank, by instruction. The identifiers themselves are carried
        # by the PAN and Aadhaar columns; this one is not a second copy.
        "Identification Number (PC)": "",
        "Date of Birth/ Incorporation (PC)": "",
        "Nationality/Country of Incorporation (PC)": COUNTRY,
        "Address Type (PC-L)": address_type(address, person),
        "Address (PC-L)": _clean(address),
        # The property's city, as requested: both City/Town columns name the
        # same place, because both describe the same transaction's property.
        # Not read from the party's own address - a seller's Aadhaar address is
        # routinely in another town, and that town is not what this column is
        # being asked for.
        "City/Town (PC-L)": property_city,
        "Pin Code (PC-L)": extract_pincode(address),
        "State (PC-L)": _clean(person.get("state")),
        "Country (PC-L)": COUNTRY,
        "Primary STD Code (PC)": "",
        "Primary Phone Number (PC)": "",
        "Primary Mobile Number (PC)": "",
        "Secondary STD Code (PC)": "",
        "Secondary Phone Number (PC)": "",
        "Secondary Mobile Number (PC)": "",
        "Email (PC)": "",
        "Person Details Remarks (PC)": person_remarks,
    }


def _person_remarks(doc: DocumentExport, relation: str, ordinal: int) -> str:
    """Empty, deliberately - see the note on `Remarks` in `_document_fields`.

    Kept as a function rather than deleted because the per-person validation
    results it used to render are still produced, still stored, and still shown
    in the interface. Only the export stopped carrying them.
    """
    return ""

    # Unreachable, retained so the shape of what was dropped is on the record.
    if doc.report is None:                                    # pragma: no cover
        return ""
    for result in doc.report.persons:
        if result.relation == relation and result.index == ordinal:
            parts = [result.remarks]
            if result.confidence:
                parts.append(f"conf={result.confidence:.2f}")
            if result.discarded:
                parts.append("discarded:" + ",".join(sorted(result.discarded)))
            return " ".join(p for p in parts if p).strip()
    return ""



# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
#
# The fine-tuned model emits sixteen fields; the report has forty-two columns.
# The model is fixed and must not be retrained, so the rest are either derived
# from what it does emit, or genuinely absent from a sale deed and left blank on
# purpose. Measured across the fifty-document corpus, the deeds contain:
#
#     PAN 84%   Aadhaar 82%   pin code 100%   property kind 96%
#     municipal wording 94%   e-mail 2%   landline 0%   date of birth 4%
#
# Everything below derives only what the documents actually carry. A column that
# stays blank is reported as such by `unpopulated_reasons()` rather than filled
# with a plausible guess - this is a legal record, and an invented value is
# worse than an empty cell because it cannot be told apart from a real one.

#: Transaction Type is the side of the transaction this row records, using the
#: same single-letter code as Transaction Relation: **S for sale, B for buy**.
#: A row is one party, so the value is per row and not per document - the same
#: deed produces `S` rows for the sellers and `B` rows for the buyers.
TRANSACTION_TYPE_FOR_RELATION = {"S": "S", "B": "B"}

#: The fourth character of a PAN encodes the holder's constitution. This is the
#: definition of the format, not a heuristic.
PAN_HOLDER_TYPE = {
    "P": "Individual", "C": "Company", "H": "HUF", "F": "Firm",
    "A": "AOP", "T": "Trust", "B": "BOI", "L": "Local Authority",
    "J": "Artificial Juridical Person", "G": "Government",
}

#: Property classification, in precedence order. Agricultural and commercial are
#: stated explicitly when they apply; residential is the residual case, so it is
#: tested last. Kannada terms sit alongside the English because a deed uses both.
#: Ordered, and the order is the whole correctness of it. A deed that says
#: "Residential Site ... formed in converted Survey No.06" is residential:
#: the survey number says where the land came from, and "converted" is the
#: word for land that is no longer agricultural. Matching the survey number
#: first called it agricultural - which is how the reference example came
#: out wrong. So: what the deed *calls* the property first, what it is made
#: of second.
#: `Property Type` codes. Single letters, as the receiving format requires -
#: the column used to carry words ("Residential"), which no downstream reader
#: accepts.
PROPERTY_TYPES = {
    "agricultural": "A",
    "non_agricultural": "N",
    "commercial": "C",
    "residential": "R",
    "industrial": "I",
    "other": "Z",
    "not_categorised": "X",
}

#: Ordered, and the order is the rule. Each entry is (code, pattern) and the
#: first match wins, so the list runs from the most specific statement of use to
#: the least:
#:
#:   I  an industrial estate or factory is never anything else
#:   C  a shop or commercial complex likewise
#:   R  a house, flat or residential site - this beats N deliberately, because
#:      "converted residential site" is a residential property; the conversion
#:      says how it stopped being agricultural, not what it is now
#:   N  converted or non-agricultural land with no use stated
#:   A  agricultural land, only when nothing above matched - a deed that says
#:      "agricultural land converted to residential" is residential
#:
#: `Z` and `X` are not in this list: `Z` needs a positive signal that the
#: property is something else, and `X` is the absence of any signal at all.
PROPERTY_KINDS = (
    ("I", re.compile(
        r"(?i)(industrial|\bfactory\b|\bKIADB\b|\bKSSIDC\b|godown|warehouse|"
        r"manufactur|ಕೈಗಾರಿಕ|ಕಾರ್ಖಾನೆ)")),
    ("C", re.compile(
        r"(?i)(commercial|\bshop\b|showroom|business\s+premises|office\s+space|"
        r"\bmall\b|complex\s+bearing|ವಾಣಿಜ್ಯ|ಅಂಗಡಿ)")),
    ("R", re.compile(
        r"(?i)(residential|dwelling|\bhouse\b|apartment|\bflat\b|"
        r"residential\s+site|house\s*site|ವಸತಿ|ನಿವೇಶನ|ಮನೆ)")),
    ("N", re.compile(
        r"(?i)(non[-\s]?agricultur|\bN\.?A\.?\s+(land|site|plot)|"
        r"\bconvert(ed|ion)\b|ಪರಿವರ್ತ)")),
    ("A", re.compile(
        r"(?i)(agricultur|\bkharab\b|\bacres?\b|\bguntas?\b|"
        r"ಕೃಷಿ|ಜಮೀನು|ಗುಂಟೆ|ಎಕರೆ)")),
)

#: A property that is plainly something else - neither land nor a building of
#: the four kinds above. Checked only after all of them fail, so it cannot
#: outrank a stated use.
_OTHER_PROPERTY = re.compile(
    r"(?i)(temple|\bchurch\b|\bmosque\b|burial|graveyard|\bschool\b|"
    r"\bcollege\b|hospital|\bwakf\b|charitable|ದೇವಸ್ಥಾನ|ಶಾಲೆ)")

#: Where a deed starts describing the property it is actually conveying.
#:
#: Anchored to a line start, because the words appear mid-sentence in the
#: operative clause of almost every deed - "conveys the schedule property to
#: the purchaser" is a reference to the schedule, not the schedule itself, and
#: cutting there starts the window in the recitals.
#: The lookbehinds are what separate a heading from a reference to one, and
#: they were read off the corpus rather than imagined. A deed refers to its own
#: schedule constantly - "the Government Market value of the schedule property
#: is Rs.1,20,00,000", "constructed on the Schedule 'A' Property" - and every
#: one of those is preceded by an article or a preposition. A real heading is
#: not:
#:
#:     SCHEDULE Agricultural land bearing Sy.No. ...
#:     SCHEDULE "B" PROPERTY All that piece and parcel of ...
#:     SCHEDULE:- All that piece and parcel of the House Property ...
#:     ಶೆಡ್ಯೂಲ್ 'ಡಿ' ಸ್ಪತ್ತು: ಶಿವಮೊಗ್ಗ ಜಿಲ್ಲೆ ...
#:
#: Anchoring to a line start instead looked more principled and matched 2 deeds
#: in 50, because OCR does not preserve the layout that would make it true.
_SCHEDULE_HEADER = re.compile(
    r"(?i)(?<!\bthe )(?<!\bof )(?<!\bon )(?<!\bin )(?<!\bsaid )"
    r"(schedule\b|ಅನುಸೂಚಿ|ಶೆಡ್ಯೂಲ್|ಪರಿಶಿಷ್ಟ)")

#: How much of the deed after the heading counts as the schedule. A property
#: description is a paragraph or two; the pages after it are signatures,
#: witnesses and attestations.
#:
#: Bounded rather than run to the end of the document, and the bound is what
#: makes the section mean anything: an unbounded cut swept in every later page,
#: so the classifier matched whatever word appeared anywhere in the tail. On
#: deed 1896 that turned a schedule reading "ಜಮೀನುಗಳು ಭೂ-ಪರಿವರ್ತನೆಯಾಗಿ"
#: (converted land, N) into C, because a commercial word appeared several pages
#: further on.
SCHEDULE_WINDOW_CHARS = 2000


def schedule_section(text: str | None) -> str:
    """The Schedule of Property, or empty when the deed has no such heading.

    The **last** heading, not the first. A deed refers to "the schedule
    property" throughout its operative clauses and then sets the schedule out at
    the end, so the last occurrence is the description and the earlier ones are
    references to it.

    This exists because classifying from the whole deed reads the *parties'*
    addresses too. "Residing at his house in Bengaluru" is not a statement about
    the land being sold, and on a corpus where 38 of 50 deeds mention a house
    somewhere it is the difference between reading the deed and guessing.
    """
    body = str(text or "")
    matches = list(_SCHEDULE_HEADER.finditer(body))
    if not matches:
        return ""
    start = matches[-1].start()
    return body[start:start + SCHEDULE_WINDOW_CHARS]

_MUNICIPAL = re.compile(
    r"(?i)(municipal|mahanagara|corporation|city\s+council|town\s+council|"
    r"\bBBMP\b|\bCMC\b|\bTMC\b|ಮಹಾನಗರ|ನಗರಸಭೆ|ಪುರಸಭೆ)")
_PANCHAYAT = re.compile(
    r"(?i)(grama\s*panchayat|gram\s*panchayat|village\s+panchayat|"
    r"ಗ್ರಾಮ\s*ಪಂಚಾಯ|ಪಂಚಾಯತ)")


#: A schedule names places with their administrative qualifier. Preference runs
#: from the most city-like to the least: a City or Town is the answer outright,
#: a Taluk is the seat that names it, and a Village or Hobli is a locality
#: within one - reported only when nothing larger is stated.
#: How strongly a word identifies the place it qualifies. A city or town names
#: itself; a taluk names the town it is administered from; a district is the
#: weakest, but it is still a real place and better than nothing.
_PLACE_KINDS = {
    "city": 4, "town": 4, "nagara": 4, "nagar": 4, "pura": 4, "pete": 4,
    "taluk": 3, "taluka": 3, "taluq": 3, "tehsil": 3,
    "district": 2, "dist": 2, "distirct": 2,
}

#: "Bengaluru South Taluk" - the name comes first, the kind follows.
_PLACE_AFTER = re.compile(
    r"(?i)([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)?)\s*[,.]?\s*"
    r"\b(city|town|nagara|taluka|taluk|taluq|tehsil|district|distirct|dist)\b")

#: "Taluka & Dist: Belgaum" - the kind comes first, separated by a colon or
#: dash. Missed entirely before, and it is how most Belgaum-side deeds are
#: written.
_PLACE_BEFORE = re.compile(
    r"(?i)\b(city|town|taluka|taluk|taluq|tehsil|district|distirct|dist)\b"
    r"\s*(?:&\s*(?:dist|district|taluk|taluka)\s*)?[:\-]\s*"
    r"([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)?)")

#: A bare city immediately before the PIN code: "Bangalore - 560 034". The PIN
#: is what makes this safe to trust - it marks the end of a postal address, so
#: the token in front of it is the post town rather than a passing mention.
_PLACE_BEFORE_PIN = re.compile(
    r"(?i)\b([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)?)\s*[-,.\s]*[1-9]\d{2}[\s-]?\d{3}\b")

#: Words that qualify a place rather than name one. "Bengaluru North Taluk" is
#: Bengaluru; "North" alone is not a town.
_DIRECTIONS = frozenset({"north", "south", "east", "west", "new", "old",
                         "upper", "lower", "greater"})

#: Never a city. Structural words from an address, administrative units below a
#: town, and the states and countries that end one - returning any of these
#: would put a non-city in a city column, which is the failure this whole
#: function exists to avoid.
_NOT_A_CITY = frozenset({
    "village", "hobli", "panchayath", "panchayat", "grama", "gram",
    "layout", "block", "sector", "road", "street", "cross", "main",
    "phase", "stage", "extension", "colony", "nagar", "nagara", "site",
    "plot", "survey", "sy", "no", "number", "khata", "katha", "ward",
    "post", "po", "taluk", "taluka", "taluq", "tehsil", "district", "dist",
    "state", "india", "karnataka", "maharashtra", "tamil", "nadu", "kerala",
    "andhra", "pradesh", "telangana", "goa", "limits", "corporation",
    "municipal", "municipality", "city", "town", "property", "schedule",
    "boundary", "east", "west", "north", "south", "registrar", "office",
    "sub", "registration", "bearing", "measuring", "situated", "within",
})


def _clean_place(name: str) -> str:
    """Strip qualifiers and reject anything that is not a place name."""
    words = [w for w in re.split(r"\s+", str(name or "").strip()) if w]
    kept = [w for w in words if w.lower().strip(".,") not in _DIRECTIONS]
    cleaned = " ".join(kept).strip(" .,;:-")
    if not cleaned:
        return ""
    # Rejected if *any* word is structural, not merely the whole string.
    # "Angol Village, Taluka & Dist: Belgaum" offered "Angol Village" as the
    # name in front of "Taluka"; that names a village, and the town this deed
    # actually sits in is the Belgaum the district clause gives.
    if any(w.lower().strip(".,") in _NOT_A_CITY for w in cleaned.split()):
        return ""
    # A single word that is only a structural term is not a city, whatever
    # case the deed wrote it in.
    # A place name is short, alphabetic and at most a few words. Without this,
    # noisy input walks straight through: fed a page of signature scrawl, the
    # last-resort rule below happily returned
    # "1 H. mosama 2 R. Kavalaky 3. Indbox ..." as a city. A cell like that is
    # worse than a blank one, because it looks like data.
    if not (3 <= len(cleaned) <= 40):
        return ""
    if len(cleaned.split()) > 3:
        return ""
    # Letters, spaces and the hyphen or apostrophe a place name may carry.
    if not all(ch.isalpha() or ch in " -'" for ch in cleaned):
        return ""
    # At least one word of real length: "K. R" is initials, not a town.
    if not any(len(w) >= 3 for w in cleaned.split()):
        return ""
    return cleaned


def city_town(*texts: str | None) -> str:
    """The town or city a property sits in, read from the property's address.

    Deeds write the place in three shapes and the previous version recognised
    only one of them - a name immediately followed by a kind word. It therefore
    returned nothing for "Taluka & Dist: Belgaum" (kind first) and nothing for
    "Koramangala, Bangalore - 560 034" (bare name before the PIN), which are
    both ordinary ways to end an address.

    Ranked rather than first-match: a deed naming both a taluk and a district
    should yield the taluk's town, because that is the closer place. Within a
    rank the earliest match wins, since an address runs from the specific to the
    general.

    Still returns "" when the schedule names only a village and a hobli. That is
    the common case for agricultural land - there is no town, and inventing the
    nearest one would be a guess about jurisdiction.
    """
    blob = " ".join(str(t) for t in texts if t)
    if not blob.strip():
        return ""

    candidates: list[tuple[int, int, str]] = []      # (rank, -position, name)

    for match in _PLACE_AFTER.finditer(blob):
        name = _clean_place(match.group(1))
        rank = _PLACE_KINDS.get(match.group(2).lower(), 0)
        if name and rank:
            candidates.append((rank, -match.start(), name))

    for match in _PLACE_BEFORE.finditer(blob):
        name = _clean_place(match.group(2))
        rank = _PLACE_KINDS.get(match.group(1).lower(), 0)
        if name and rank:
            candidates.append((rank, -match.start(), name))

    # Rank 1: weaker than any named administrative unit, but far better than
    # blank - and it is right whenever the address simply ends with its city.
    for match in _PLACE_BEFORE_PIN.finditer(blob):
        name = _clean_place(match.group(1))
        if name:
            candidates.append((1, -match.start(), name))

    # Last resort, and the weakest: an address that simply ends with its city
    # and says nothing else - "Hosur Sarjapur Road Layout, Bangalore". Safe only
    # because `_clean_place` rejects the structural words, villages, hoblis and
    # state names that otherwise end an address; without that guard this would
    # put "Kasaba Hobli" or "Karnataka" in a city column.
    if not candidates:
        for part in reversed(str(blob).split(",")):
            name = _clean_place(part)
            if name:
                candidates.append((0, 0, name))
                break

    if not candidates:
        return ""
    return max(candidates)[2]


def person_type(person: dict[str, Any]) -> str:
    """Constitution of the party - Individual, Company, Trust and so on.

    Read from the PAN when there is one, because the fourth character *is* the
    holder type by definition. Without a PAN, a party with a father's name or a
    gender recorded is an individual; a deed does not record either for a
    company. Anything else stays blank.
    """
    pan = str(person.get("pan_card_number") or "").strip().upper()
    if len(pan) >= 4 and pan[3] in PAN_HOLDER_TYPE:
        return PAN_HOLDER_TYPE[pan[3]]
    if str(person.get("father_name") or "").strip() or \
            str(person.get("gender") or "").strip():
        return "Individual"
    return ""


#: `Identification Type` is a single-letter code, not a name. The full set the
#: report accepts, so an unrecognised value is a coding error rather than a
#: plausible-looking string that a downstream reader silently rejects.
IDENTIFICATION_CODES = {
    "passport": "A",
    "election": "B",          # Election ID / voter card
    "pan": "C",
    "government": "D",        # ID issued by a government body or PSU
    "driving": "E",
    "aadhaar": "G",           # UIDAI letter or Aadhaar card
    "nrega": "H",             # NREGA job card
    "other": "Z",
}


def identification_type(person: dict[str, Any]) -> str:
    """The code for the identification document this party was identified by.

    PAN outranks Aadhaar because it is the identifier the transaction is
    reported against; a deed carrying both is identified by the PAN. Neither
    present leaves the cell empty rather than guessing `Z` - "Others" asserts
    that some other document was seen, and none was.

    The number itself is deliberately not returned. `Identification Number`
    stays blank in every row - see `_person_fields`.
    """
    if str(person.get("pan_card_number") or "").strip():
        return IDENTIFICATION_CODES["pan"]
    if str(person.get("aadhaar_number") or "").strip():
        return IDENTIFICATION_CODES["aadhaar"]
    return ""


#: `Address Type` is a numeric code. 5 is not a failure value - the format
#: provides it for exactly this case, and a deed records where a party lives
#: without stating what the premises are used for.
ADDRESS_TYPES = {
    "residential_business": "1",
    "residential": "2",
    "business": "3",
    "registered_office": "4",
    "unspecified": "5",
}

_BUSINESS_ADDRESS = re.compile(
    r"(?i)(\boffice\b|\bshop\b|godown|factory|premises|industrial|"
    r"\bunit\s*n|business|showroom)")
#: Genuinely residential wording only. `road`, `street`, `cross` and `main`
#: were in this set and are not evidence of anything - every business
#: address has them too, and a shop on MG Road came out as "residential
#: and business" because of it.
_RESIDENTIAL_ADDRESS = re.compile(
    r"(?i)(\bresiding\b|\bresident\b|\bhouse\b|\bflat\b|\bdwelling\b|\bvillage\b|\bnagar\b|\bh\.?\s?no\b|apartment|layout|colony)")

_REGISTERED_OFFICE = re.compile(
    r"(?i)(registered\s+office|corporate\s+office|\bregd\.?\s*off)")


def address_type(address: str | None, person: dict[str, Any] | None = None) -> str:
    """Classify a party's address: 1 to 5.

    Read from what the address says about itself. A registered office states so
    outright. A company - which the PAN's fourth character identifies - has a
    business address by construction. Otherwise residential wording decides,
    and both kinds of wording together give 1.

    Falls back to 5 rather than assuming 2. A deed states where somebody lives
    without saying what the premises are used for, and "Unspecified" is the
    value the format provides for that.
    """
    text = str(address or "")
    if not text.strip():
        return ADDRESS_TYPES["unspecified"]

    if _REGISTERED_OFFICE.search(text):
        return ADDRESS_TYPES["registered_office"]

    # A non-individual PAN holder is at a business address by definition.
    if person is not None:
        pan = str(person.get("pan_card_number") or "").strip().upper()
        if len(pan) >= 4 and pan[3] in ("C", "F", "T", "A", "B", "L", "J", "G"):
            return ADDRESS_TYPES["business"]

    business = bool(_BUSINESS_ADDRESS.search(text))
    residential = bool(_RESIDENTIAL_ADDRESS.search(text))
    if business and residential:
        return ADDRESS_TYPES["residential_business"]
    if business:
        return ADDRESS_TYPES["business"]
    if residential:
        return ADDRESS_TYPES["residential"]
    return ADDRESS_TYPES["unspecified"]


def property_type(schedule_address: str | None = None,
                  source_text: str | None = None) -> str:
    """The property's type as a single code: A, N, C, R, I, Z or X.

    Read from the deed in order of authority, stopping at the first source that
    answers:

      1. **The Schedule of Property**, when the deed has one. This is the
         section that describes what is being conveyed, so it outranks
         everything else by construction.
      2. **The schedule address** the extraction pulled out, for a deed whose
         schedule heading the OCR did not capture.
      3. **The whole deed**, last. It contains the parties' own addresses, so a
         match here is the weakest kind of evidence - but a deed that describes
         its land only in the recitals still deserves an answer.

    `X` when none of them says anything. Not a guess and not a blank: the format
    provides "Not Categorized" precisely for a deed that does not state it, and
    inventing `R` because somebody's home was mentioned is how a farm gets
    reported as a house.
    """
    for source in (schedule_section(source_text), schedule_address, source_text):
        blob = str(source or "").strip()
        if not blob:
            continue
        for code, pattern in PROPERTY_KINDS:
            if pattern.search(blob):
                return code
        if _OTHER_PROPERTY.search(blob):
            return PROPERTY_TYPES["other"]
    return PROPERTY_TYPES["not_categorised"]


def within_municipal_limits(*texts: str | None) -> str:
    """Yes, No, or blank - blank when the deed does not make it clear.

    Answered only on an explicit authority: a municipal body means yes, a grama
    panchayat means no. Both or neither leaves it empty. This is a legal
    characterisation of the property and a confident wrong answer is worse than
    an acknowledged gap.
    """
    blob = " ".join(str(t) for t in texts if t)
    municipal = bool(_MUNICIPAL.search(blob))
    panchayat = bool(_PANCHAYAT.search(blob))
    if municipal and not panchayat:
        return "Yes"
    if panchayat and not municipal:
        return "No"
    return ""


#: Columns that stay empty because a sale deed does not carry the value.
#: Measured on the corpus rather than assumed - see the note above.
STRUCTURALLY_ABSENT = {
    "Original Report Serial Number":
        "only used for a correction or deletion report; this is an original",
    "Date of Birth/ Incorporation (PC)": "not recorded on a sale deed (4% of corpus)",
    "Form 60 Acknowledgement (PC)": "only when a party has no PAN and files Form 60 (0%)",
    "Primary STD Code (PC)": "landline numbers do not appear on deeds (0%)",
    "Primary Phone Number (PC)": "landline numbers do not appear on deeds (0%)",
    "Secondary STD Code (PC)": "landline numbers do not appear on deeds (0%)",
    "Secondary Phone Number (PC)": "landline numbers do not appear on deeds (0%)",
    "Email (PC)": "not recorded on a sale deed (2% of corpus)",
    "Primary Mobile Number (PC)":
        "present in 40% of deeds but never attributed to a named party",
    "Secondary Mobile Number (PC)": "see Primary Mobile Number",
    # `City / Town` was here until R-042 taught `city_town` to read the place
    # from the address and the deed body. It is populated now, so the claim that
    # it cannot be is false and has been withdrawn.
    # `City/Town (PC-L)` was here. It now carries the property's city, the
    # same value as `City / Town` - so the claim that it cannot be filled is
    # false and has been withdrawn.
    "Identification Number (PC)":
        "held blank by instruction; the identifiers are in the PAN and Aadhaar columns",
}


def unpopulated_reasons(row: dict[str, str]) -> dict[str, str]:
    """Why each empty column in this row is empty.

    Separates "the deed does not contain this" from "the deed may contain it and
    nothing found it" - the second is a defect worth chasing, the first is not,
    and a report that does not distinguish them sends people looking for bugs
    that are not there.
    """
    reasons: dict[str, str] = {}
    for column in CSV_COLUMNS:
        if str(row.get(column, "") or "").strip():
            continue
        reasons[column] = STRUCTURALLY_ABSENT.get(
            column, "expected in the document but not extracted")
    return reasons


def _parties_for_side(doc: DocumentExport, side: str, relation: str,
                      seen_parties: set[tuple[str, ...]]) -> list[tuple[int, dict[str, Any]]]:
    """The parties on one side of a deed that will become rows.

    Split out of `build_rows` so the count is available *before* the rows are
    written: the consideration is divided by the number of parties on this side,
    and that number is only correct after the same filtering the rows go through.

    `seen_parties` is shared across both sides and carries the relation in its
    key, so someone appearing as both seller and buyer keeps a row on each side.
    """
    kept: list[tuple[int, dict[str, Any]]] = []
    for ordinal, person in enumerate(doc.extraction.get(side) or [], start=1):
        if not isinstance(person, dict):
            continue

        # One row per party. The model occasionally lists the same person twice
        # - same name, same Aadhaar - and each copy became a row carrying the
        # whole document with it.
        key = _party_key(person, relation)
        if key in seen_parties:
            _log.info("duplicate party dropped", extra={
                "document": doc.transaction_identity,
                "relation": relation, "ordinal": ordinal,
                # Not "name": LogRecord reserves it, and passing it raises
                # KeyError from inside logging - so the export would crash on
                # the very case this line reports.
                "party": _clean(_translated(person, "name"))})
            continue
        seen_parties.add(key)

        raw_name = _clean(_translated(person, "name"))
        if raw_name != primary_name(raw_name):
            _log.info("alias removed from name", extra={
                "document": doc.transaction_identity,
                "relation": relation, "ordinal": ordinal,
                "full": raw_name, "kept": primary_name(raw_name)})

        if not looks_like_a_name(_translated(person, "name")):
            # Punctuation is not a name. Reported rather than exported: a row
            # naming nobody is worse than a document one row short, and the log
            # says which document to look at.
            _log.warning("party has no usable name", extra={
                "document": doc.transaction_identity,
                "relation": relation, "ordinal": ordinal,
                "value": _clean(_translated(person, "name"))})
            continue

        kept.append((ordinal, person))
    return kept


def build_rows(documents: list[DocumentExport], *,
               excel_safe: bool = False) -> list[dict[str, str]]:
    """Expand documents into per-person rows.

    A document with no extracted parties still yields one row, so it is visible
    in the export rather than silently absent.
    """
    rows: list[dict[str, str]] = []
    serial = 0

    for doc in documents:
        serial = doc.serial if doc.serial is not None else serial + 1
        base = _document_fields(doc, serial)
        amount = base["Transaction Amount"]

        emitted = 0
        seen_parties: set[tuple[str, ...]] = set()
        # Sellers first, then buyers - the order the reference report uses, and
        # the order a deed reads in: the party parting with the property is
        # named before the party acquiring it.
        for side, relation in (("seller_details", "S"), ("buyer_details", "B")):
            # Selected before any row is written, because the consideration is
            # split by *how many parties this side actually has* - and that is
            # known only after duplicates and unusable names are removed. Split
            # by the raw list instead and a deed that dropped a duplicate buyer
            # would hand out three quarters of its own value.
            parties = _parties_for_side(doc, side, relation, seen_parties)
            shares = person_shares(amount, len(parties))

            for (ordinal, person), share in zip(parties, shares):
                row = dict.fromkeys(CSV_COLUMNS, "")
                row.update(base)
                row.update(_person_fields(
                    person, relation, share,
                    _person_remarks(doc, relation, ordinal), excel_safe,
                    property_city=base.get("City / Town", "")))
                row.update({k: v for k, v in doc.extras.items() if k in CSV_COLUMNS})
                rows.append(row)
                emitted += 1

        if emitted == 0:
            row = dict.fromkeys(CSV_COLUMNS, "")
            row.update(base)
            row["Nationality/Country of Incorporation (PC)"] = COUNTRY
            row["Country (PC-L)"] = COUNTRY
            # `Address Type` carries a code in every row, with no exception for
            # a document whose parties could not be read. `5` is "Unspecified",
            # which is precisely the case here: nothing about the address could
            # be determined. Requested explicitly - the column must hold one of
            # 1 to 5 and never a blank.
            row["Address Type (PC-L)"] = ADDRESS_TYPES["unspecified"]
            # The row is still emitted, so a document with no parties is
            # visible in the export rather than silently absent - but the
            # explanation belongs in the log, not in a column that must stay
            # empty. `write_csv` reports the count.
            _log.warning("no parties extracted", extra={
                "document": doc.transaction_identity,
                "file": doc.source_filename})
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


#: Columns the receiving system reads as codes rather than as words, and the
#: complete set each one admits. A blank is allowed: a row with no party carries
#: no person data at all, and inventing a code for a party that does not exist
#: would be a claim about nobody.
CODED_COLUMNS = {
    "Address Type (PC-L)": {"1", "2", "3", "4", "5"},
    "Identification Type (PC)": {"A", "B", "C", "D", "E", "G", "H", "Z"},
    "Transaction Type": {"S", "B"},
    "Transaction Relation (PC)": {"S", "B"},
    # A/N/C/R/I/Z/X. This column held words - "Residential", "Commercial" -
    # until it was specified as coded, and a word here is rejected by the
    # receiving system rather than by anything visible in a spreadsheet.
    "Property Type": set(PROPERTY_TYPES.values()),
}


def coded_column_violations(rows: list[dict[str, str]]) -> list[str]:
    """Cells in a coded column holding something outside its code set.

    Exists because the failure it catches is silent. A description written into
    a coded column looks entirely plausible in a spreadsheet and is only
    rejected much later, by the system that consumes the file - which is exactly
    what `Property Type` did while it carried "Residential" instead of `R`.
    """
    problems: list[str] = []
    for index, row in enumerate(rows, start=1):
        for column, permitted in CODED_COLUMNS.items():
            value = str(row.get(column) or "").strip()
            if value and value not in permitted:
                problems.append(f"row {index}: {column} = {value!r}")
    return problems


def write_csv(path: str | Path, documents: list[DocumentExport], *,
              excel_safe: bool = False, encoding: str = "utf-8-sig") -> int:
    """Write the 42-column export. Returns the row count.

    `utf-8-sig` by default: the BOM is what makes Excel read Kannada text
    correctly instead of showing mojibake. Standard readers ignore it.
    """
    rows = build_rows(documents, excel_safe=excel_safe)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    offenders = coded_column_violations(rows)
    if offenders:
        # Loud, and named. These columns are read by another system as codes; a
        # descriptive word in one of them is not a cosmetic problem, it is a
        # value the receiver cannot parse. Reported per row rather than
        # corrected, because a wrong code is worse than a visible fault.
        _log.error("export contains non-coded values in %d cell(s)",
                   len(offenders),
                   extra={"cells": offenders[:20], "path": str(target)})

    remaining = untranslated_cells(rows)
    if remaining:
        # Written anyway - see `untranslated_cells` on why a blank is worse -
        # but never silently.
        _log.warning(
            "export contains untranslated text in %d column(s)",
            len(remaining),
            extra={"columns": sorted(remaining), "path": str(target)})

    with target.open("w", newline="", encoding=encoding) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS),
                                extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    _log.info("CSV written: %d row(s) -> %s", len(rows), target.name,
              extra={"path": str(target), "columns": len(CSV_COLUMNS),
                     "bytes": target.stat().st_size if target.is_file() else 0,
                     "excel_safe": excel_safe})
    return len(rows)


def write_failed_csv(path: str | Path, failures: list[FailedDocument], *,
                     encoding: str = "utf-8-sig") -> int:
    """Write the failed-documents export for the dashboard's second download."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding=encoding) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FAILED_COLUMNS),
                                extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for item in failures:
            writer.writerow({
                "Transaction Identity": item.transaction_identity,
                "Source Filename": item.source_filename,
                "Failed Stage": item.failed_stage,
                "Processing Status": item.processing_status,
                "Reason": _clean(item.reason),
                "Flags": item.flags,
                "Confidence": ("" if item.confidence is None
                               else f"{item.confidence:.2f}"),
            })
    return len(failures)


def verify_against_reference(reference_csv: str | Path) -> list[str]:
    """Compare our column list against the reference file. Returns problems."""
    problems: list[str] = []
    with Path(reference_csv).open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))

    if len(header) != len(CSV_COLUMNS):
        problems.append(f"column count {len(CSV_COLUMNS)} != reference {len(header)}")
    for i, (ours, theirs) in enumerate(zip(CSV_COLUMNS, header), start=1):
        if ours != theirs:
            problems.append(f"column {i}: {ours!r} != reference {theirs!r}")
    return problems
