"""Extraction validation - INFERENCE_PIPELINE.md layers 2 through 7.

This module is load-bearing, not a quality feature. The served model is 4-bit
quantised, and quantisation damages exact-digit reproduction: measured on deed
`07`, the model returned `sale_consideration: 1500000` where the OCR reads
`Rs.1,50,000=00 (Rupees One Lakh Fifty thousand)` - wrong by a factor of ten.
The BF16 model got it right. Nothing downstream can tell the difference between
those two numbers; only cross-checking against the source can.

So the governing rule here is: **a value that does not appear in the OCR is not
trustworthy, regardless of how well-formed it looks.**

Layers implemented:

    2  JSON extraction and schema shape
    3  Field validators - format, then presence in the OCR source
    4  Within-side leak check - PAN and Aadhaar proximity
    5  Registration-fee regex cross-check, independent of the model
    6  PAN coverage - the retry trigger
    7  Disposition - accept, retry, or route to review

Matching is deliberately tolerant, because OCR is not clean: PANs written with
spaces or dots, Aadhaars wrapped across line breaks, amounts in Indian grouping,
Kannada names whose characters survive but whose clusters get mangled.

Standard library only. No I/O. Fully unit-testable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Flag codes (Help page)
# ---------------------------------------------------------------------------


class Flag(str, Enum):
    """Remark codes written into the two CSV remarks columns."""

    PAN_MATCHED = "PM"          # PAN present in OCR and paired correctly
    WRONG_AADHAAR = "WAN"       # Aadhaar not found in OCR
    WRONG_CONSIDERATION = "WSC" # sale_consideration not found in OCR
    WRONG_STAMP_VALUE = "WSV"   # registration fee disagrees with OCR / regex
    STAMP_OVER_SALE = "SSV"     # registration fee exceeds sale consideration
    WRONG_TXN_DATE = "WTD"      # transaction date missing or not in OCR
    OCR_PASSED = "OCR_P"
    OCR_FAILED = "OCR_F"

    # Not in the original code list; added because they are distinct conditions
    # an operator must be able to see.
    HALLUCINATED_PAN = "HPAN"   # PAN well-formed but absent from OCR - discarded
    EXTRA_OCR_PANS = "XPAN"     # OCR contains PANs not attributed to any party
    PAN_AADHAAR_FAR = "PAF"     # possible within-side field swap
    NAME_NOT_IN_OCR = "WNM"     # name not locatable in OCR
    SCHEMA_INVALID = "SCH"      # missing or malformed top-level keys
    TRUNCATED = "TRC"           # generation hit the token ceiling
    NO_TXN_IDENTITY = "WTI"     # registration number not readable on the deed


@dataclass(frozen=True)
class RuleToggles:
    """Validation Rules page. Each rule independently switchable."""

    pan: bool = True
    aadhaar: bool = True
    registration_fee: bool = True
    sale_consideration: bool = True
    transaction_date: bool = True
    ocr_cross_verify: bool = True
    confidence: bool = True
    #: Stamp Value is derived from the registration fee, halved for transactions
    #: dated before the cutoff below. Enabled now that the formula is defined.
    stamp_value: bool = True

    #: Fraction of OCR PANs that must appear on the buyer/seller side.
    pan_coverage_threshold: float = 0.6
    #: How many OCR PANs must be *unaccounted for* before low coverage is acted
    #: on. The ratio alone is too coarse at small denominators: with 2 PANs in the
    #: document it can only be 0.0, 0.5 or 1.0, so a single witness PAN drops a
    #: perfectly good extraction to 0.5 and fails a 0.6 threshold every time.
    #: Measured on deed 1359, where the OCR holds AZMPA8189K, BUVPB8312G and a
    #: truncated BLRPS9269; the model extracted correctly and still scored 0.5.
    #: Requiring 2+ unmatched keeps the metric's real purpose - catching a model
    #: that stopped early - while ignoring noise. A 30-PAN deed missing 15 still
    #: trips it.
    pan_coverage_min_unmatched: int = 2
    #: Max characters between a person's PAN and Aadhaar in the OCR.
    pan_aadhaar_proximity_chars: int = 250
    #: Party count above which split-prompt extraction is preferred. Sources
    #: disagree (spec 30, architecture&plan 20, INFERENCE.md 25).
    pan_split_threshold: int = 25
    #: Plausible registration-fee range, for the Layer 5 regex sweep.
    reg_fee_min: int = 100
    reg_fee_max: int = 1_000_000

    #: Stamp Value = registration fee, halved when the transaction date falls
    #: BEFORE this cutoff. Deeds on or after it carry the fee unchanged.
    #: A document with no usable transaction date cannot be classified either
    #: way, so no Stamp Value is derived rather than guessing at the rate.
    stamp_value_cutoff: date = date(2025, 8, 31)
    #: Extra multiplier from Settings, applied after the halving rule.
    stamp_value_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

PAN_STRICT = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
PAN_IN_TEXT = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

#: Registration fee, English and Kannada. Kept independent of the model so the
#: cross-check is genuinely a second opinion.
REG_FEE_PATTERNS = (
    re.compile(r"(?i)(?:registration|regn\.?)\s*(?:fee|fees|charges?)\s*[:\-]?\s*"
               r"(?:rs\.?|₹|inr)?\s*([\d,.\-]+)"),
    re.compile(r"ನೋಂದಣಿ\s*ಶುಲ್ಕ\s*[:\-]?\s*(?:ರೂ\.?)?\s*([\d,.\-]+)"),
    re.compile(r"(?i)reg\.?\s*fee\s*(?:rs\.?|₹)?\s*([\d,.\-]+)"),
)

ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

REQUIRED_KEYS = ("buyer_details", "seller_details", "property_details", "document_details")


# ---------------------------------------------------------------------------
# Layer 2 - JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Pull the first balanced top-level JSON object out of model output.

    Handles code fences and trailing commentary. Scans for balanced braces rather
    than using a greedy regex, because a truncated or duplicated object would
    otherwise poison the parse - both observed in the benchmark corpus.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for i, ch in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    start, depth = None, 0
                    continue
                return parsed if isinstance(parsed, dict) else None
    return None


def schema_shape_ok(pred: dict) -> tuple[bool, list[str]]:
    """Check top-level keys and container types. Missing values are allowed."""
    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in pred:
            problems.append(f"missing key: {key}")
    for key in ("buyer_details", "seller_details"):
        if key in pred and not isinstance(pred[key], list):
            problems.append(f"{key} must be a list")
    for key in ("property_details", "document_details"):
        if key in pred and not isinstance(pred[key], dict):
            problems.append(f"{key} must be an object")
    return not problems, problems


# ---------------------------------------------------------------------------
# OCR normalisation and presence helpers
# ---------------------------------------------------------------------------


def normalise_for_match(text: str) -> str:
    """Collapse whitespace and separators so OCR artefacts stop mattering.

    NFKC first: Kannada and Indic digits appear in multiple Unicode forms, and
    without normalisation identical-looking text fails to match.
    """
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[\s\-.,:;/\\|_()\[\]{}]+", "", text).upper()


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def parse_amount(raw: str) -> int | None:
    """Parse a rupee amount, discarding a paise fraction.

    Deeds write amounts as `1,50,000=00`, `12,30,000.00` or plain `200.00`. A
    naive digit-strip turns `200.00` into 20000 - a hundredfold error, and the
    same mistake that made a wrong registration fee look grounded on deed 1316.

    A trailing group of exactly two digits after `.` or `=` is treated as paise
    and dropped; anything else is kept.
    """
    text = (raw or "").strip()
    if not text:
        return None
    m = re.match(r"^([\d,\s]+?)\s*[.=]\s*(\d{1,2})\s*$", text)
    whole = m.group(1) if m else text
    digits = _digits(whole)
    return int(digits) if digits else None


def indian_grouping(digits: str) -> str:
    """1500000 -> '15,00,000'. Last three digits, then pairs."""
    if not digits.isdigit() or len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


# -- PAN --------------------------------------------------------------------


def pan_format_valid(pan: str | None) -> bool:
    return bool(pan) and bool(PAN_STRICT.match(pan.strip().upper()))


def pan_in_ocr(pan: str, ocr_normalised: str) -> bool:
    """Tolerant of separators: ALQPP 8332F, ALQPP-8332F, ALQPP.8332F."""
    return pan.strip().upper() in ocr_normalised


def ocr_pans(ocr: str) -> set[str]:
    return set(PAN_IN_TEXT.findall(unicodedata.normalize("NFKC", ocr or "").upper()))


# -- Aadhaar ----------------------------------------------------------------


def aadhaar_format_valid(value: str | None) -> bool:
    return len(_digits(value or "")) == 12


def aadhaar_in_ocr(value: str, ocr_normalised: str, ocr_raw: str) -> bool:
    """Match a 12-digit Aadhaar across arbitrary separators and line wraps.

    Aadhaars are routinely split across lines in OCR, e.g. '9414 5063\\n  3293'.
    The normalised pass catches most cases; the spaced regex is the fallback
    that stops a correct extraction being reported as a hallucination.
    """
    digits = _digits(value)
    if len(digits) != 12:
        return False
    if digits in ocr_normalised:
        return True
    spaced = r"[\s\-.]*".join(digits)
    return bool(re.search(spaced, ocr_raw))


# -- Amounts ----------------------------------------------------------------


def amount_format_valid(value: str | None) -> bool:
    """Plain digit string only. Commas, decimals or spaces indicate a model bug."""
    return bool(value) and str(value).isdigit()


def amount_in_ocr(value: str, ocr_normalised: str, ocr_raw: str) -> bool:
    """Is this amount actually present in the OCR, as a whole number?

    This is the check that caught the 10x quantisation error on deed 07:
    `1500000` and `15,00,000` are both absent from that OCR, so the value is
    rejected rather than exported.

    Separators are commas and whitespace ONLY - never a period. A period is a
    decimal point, not a grouping separator: allowing it made "200.00" satisfy the
    query "20000", reporting a genuinely wrong registration fee on deed 1316 as
    grounded. That OCR reads "registration fee 200.00" - the real fee is 200.

    Whitespace covers line wraps, so a split amount still matches. Boundaries
    stop the run being part of a longer number; a trailing ".00" or "=00" suffix
    satisfies the lookahead naturally since neither is a digit.

    No normalised-text fallback on purpose: normalisation strips separators,
    concatenating adjacent unrelated numbers ("63,60,000 13,00,000" ->
    "63600001300000"), which would let a query match across the join. For a check
    whose whole purpose is catching wrong digits, being permissive is the worse
    failure.
    """
    digits = _digits(str(value))
    if not digits:
        return False
    # Allow arbitrary separators between digits (Indian grouping, spaces, line
    # wraps) but require the run not to be part of a longer number.
    pattern = r"(?<![\d])" + r"[\s,]*".join(re.escape(d) for d in digits) + r"(?![\d])"
    return bool(re.search(pattern, ocr_raw))


# -- Names ------------------------------------------------------------------


def name_in_ocr(name: str, ocr_normalised: str) -> bool:
    """Token-fuzzy presence.

    Strict equality is useless here: OCR mangles Kannada character clusters, so
    a correct name often appears with its tokens present but non-contiguous. The
    published grounding check under-counted for exactly this reason. Pass if the
    whole name matches, or if at least two tokens of three or more characters do.
    """
    if not name:
        return False
    whole = normalise_for_match(name)
    if whole and whole in ocr_normalised:
        return True
    tokens = [normalise_for_match(t) for t in re.split(r"\s+", name) if len(t) >= 3]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t and t in ocr_normalised)
    return hits >= 2 or (len(tokens) == 1 and hits == 1)


# -- Dates ------------------------------------------------------------------


def date_iso_valid(value: str | None) -> bool:
    m = ISO_DATE.match(str(value or ""))
    if not m:
        return False
    y, mo, d = (int(g) for g in m.groups())
    return 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31


def date_in_ocr(value: str, ocr_normalised: str) -> bool:
    """Accept the Indian written forms the ISO value was derived from."""
    m = ISO_DATE.match(str(value or ""))
    if not m:
        return False
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    candidates = (
        f"{d}/{mo}/{y}", f"{d:02d}/{mo:02d}/{y}",
        f"{d}-{mo}-{y}", f"{d:02d}-{mo:02d}-{y}",
        f"{d:02d}.{mo:02d}.{y}",
    )
    return any(normalise_for_match(c) in ocr_normalised for c in candidates)


# ---------------------------------------------------------------------------
# Layer 4 - within-side leak check
# ---------------------------------------------------------------------------


def pan_aadhaar_proximity(pan: str | None, aadhaar: str | None, ocr: str,
                          window: int = 250) -> bool | None:
    """Do this person's PAN and Aadhaar occur near each other in the source?

    Catches the quiet failure where two people on the same side have their
    identifiers crossed. The government database rejects such a deed on lookup,
    so the end user is protected - but surfacing it in-app avoids a wasted filing.

    Returns None when the pair cannot be evaluated (either value absent).
    """
    if not pan or not aadhaar:
        return None
    digits = _digits(aadhaar)
    if len(digits) != 12:
        return None

    pan_positions = [m.start() for m in re.finditer(re.escape(pan.upper()), ocr.upper())]
    spaced = r"[\s\-.]*".join(digits)
    aadhaar_positions = [m.start() for m in re.finditer(spaced, ocr)]
    if not pan_positions or not aadhaar_positions:
        return None

    return any(abs(p - a) <= window for p in pan_positions for a in aadhaar_positions)


# ---------------------------------------------------------------------------
# Layer 5 - registration fee cross-check
# ---------------------------------------------------------------------------


def reg_fee_candidates(ocr: str, lo: int = 100, hi: int = 1_000_000) -> list[int]:
    """Registration-fee values found by regex, independent of the model."""
    found: list[int] = []
    for pattern in REG_FEE_PATTERNS:
        for match in pattern.finditer(ocr or ""):
            value = parse_amount(match.group(1))
            if value is None:
                continue
            if lo <= value <= hi and value not in found:
                found.append(value)
    return found


# ---------------------------------------------------------------------------
# Layer 6 - PAN coverage
# ---------------------------------------------------------------------------


def pan_coverage(pred: dict, ocr: str) -> float:
    """Fraction of OCR PANs extracted onto the buyer/seller side.

    The retry trigger. Cheap and sharp for under-extraction, but note its blind
    spot: parties with no PAN are invisible to it. A run that dropped two
    PAN-less sellers still scored 1.0. Completeness needs other signals too.
    """
    source = ocr_pans(ocr)
    if not source:
        return 1.0
    extracted = {
        (p.get("pan_card_number") or "").strip().upper()
        for side in ("buyer_details", "seller_details")
        for p in (pred.get(side) or [])
        if isinstance(p, dict) and p.get("pan_card_number")
    }
    return len(extracted & source) / len(source)


# ---------------------------------------------------------------------------
# Stamp Value
# ---------------------------------------------------------------------------


def parse_iso_date(value: str | None) -> date | None:
    m = ISO_DATE.match(str(value or ""))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def derive_stamp_value(registration_fee: str | int | None,
                       transaction_date: str | None,
                       rules: "RuleToggles | None" = None) -> int | None:
    """Stamp Value from the registration fee.

    Rule: the fee is halved for transactions dated **before** the cutoff
    (31 August 2025); on or after it the fee carries through unchanged. Any
    Settings multiplier is applied afterwards.

    Returns None when the fee is unusable, or when the date is missing - the
    halving depends entirely on which side of the cutoff the deed falls, so an
    undated document cannot be classified and guessing would silently produce a
    figure that is wrong by a factor of two.
    """
    rules = rules or RuleToggles()
    digits = _digits(str(registration_fee or ""))
    if not digits:
        return None

    when = parse_iso_date(transaction_date)
    if when is None:
        return None

    fee = int(digits)
    value = fee / 2 if when < rules.stamp_value_cutoff else float(fee)
    value *= rules.stamp_value_multiplier
    # Round half up: a stamp value is a rupee amount, never a fraction.
    return int(value + 0.5)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Disposition(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    REVIEW = "review"


@dataclass
class FieldCheck:
    field: str
    value: object
    format_ok: bool | None = None
    in_ocr: bool | None = None
    confidence: float = 1.0
    detail: str = ""

    @property
    def suspect(self) -> bool:
        return self.format_ok is False or self.in_ocr is False


@dataclass
class PersonResult:
    index: int
    relation: str            # "B" or "S"
    name: str | None
    checks: list[FieldCheck] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    discarded: dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def remarks(self) -> str:
        """Person Details Remarks (PC) - column 42."""
        return " ".join(dict.fromkeys(f.value for f in self.flags))


@dataclass
class ValidationReport:
    parsed: bool
    schema_ok: bool
    persons: list[PersonResult] = field(default_factory=list)
    document_flags: list[Flag] = field(default_factory=list)
    document_checks: list[FieldCheck] = field(default_factory=list)
    pan_coverage: float = 0.0
    #: Derived from the registration fee and the transaction date (see
    #: derive_stamp_value). None when either input is unusable.
    stamp_value: int | None = None
    extra_ocr_pans: list[str] = field(default_factory=list)
    reg_fee_candidates: list[int] = field(default_factory=list)
    confidence: float = 0.0
    disposition: Disposition = Disposition.REVIEW
    notes: list[str] = field(default_factory=list)

    @property
    def remarks(self) -> str:
        """Document-level Remarks - column 15."""
        codes = list(dict.fromkeys(f.value for f in self.document_flags))
        if self.extra_ocr_pans:
            codes.append(f"{Flag.EXTRA_OCR_PANS.value}:{','.join(self.extra_ocr_pans)}")
        return " ".join(codes)

    @property
    def suspect_fields(self) -> list[str]:
        out = [c.field for c in self.document_checks if c.suspect]
        for person in self.persons:
            out.extend(f"{person.relation}{person.index}.{c.field}"
                       for c in person.checks if c.suspect)
        return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _confidence(checks: list[FieldCheck]) -> float:
    """Mean confidence over evaluated fields.

    Derived from validator outcomes rather than model log-probabilities:
    collecting per-token scores for an 8000-token generation needs ~8 GB and
    OOMs even on a 16 GB GPU. See docs/DOCUMENTATION.md.
    """
    scored = [c.confidence for c in checks if c.format_ok is not None or c.in_ocr is not None]
    return round(sum(scored) / len(scored), 3) if scored else 0.0


def _score(format_ok: bool | None, in_ocr: bool | None) -> float:
    if format_ok is False:
        return 0.0
    if in_ocr is False:
        return 0.25  # well-formed but ungrounded - the quantisation failure mode
    if in_ocr is True:
        return 1.0
    return 0.6       # format fine, presence not evaluated


def validate_extraction(
    raw_or_pred: str | dict,
    ocr: str,
    toggles: RuleToggles | None = None,
    *,
    truncated: bool = False,
    ocr_succeeded: bool = True,
) -> ValidationReport:
    """Run layers 2-7 over one extraction. Never raises."""
    rules = toggles or RuleToggles()
    ocr = (ocr or "").replace("\r\n", "\n").replace("\r", "\n")
    ocr_norm = normalise_for_match(ocr)

    pred = extract_json(raw_or_pred) if isinstance(raw_or_pred, str) else raw_or_pred
    report = ValidationReport(parsed=pred is not None, schema_ok=False)

    report.document_flags.append(Flag.OCR_PASSED if ocr_succeeded else Flag.OCR_FAILED)
    if truncated:
        report.document_flags.append(Flag.TRUNCATED)
        report.notes.append(
            "generation hit the token ceiling; on this workload that usually "
            "indicates a repetition loop rather than a long answer"
        )

    if pred is None:
        report.disposition = Disposition.RETRY
        report.notes.append("output did not contain parseable JSON")
        return report

    report.schema_ok, problems = schema_shape_ok(pred)
    if not report.schema_ok:
        report.document_flags.append(Flag.SCHEMA_INVALID)
        report.notes.extend(problems)

    # -- Layer 3/4 per person ------------------------------------------------
    source_pans = ocr_pans(ocr)
    claimed_pans: set[str] = set()

    for side, relation in (("buyer_details", "B"), ("seller_details", "S")):
        for i, person in enumerate(pred.get(side) or [], start=1):
            if not isinstance(person, dict):
                continue
            result = PersonResult(index=i, relation=relation, name=person.get("name"))

            # PAN
            pan = (person.get("pan_card_number") or "").strip().upper() or None
            if rules.pan and pan:
                fmt = pan_format_valid(pan)
                present = pan_in_ocr(pan, ocr_norm) if rules.ocr_cross_verify else None
                result.checks.append(FieldCheck("pan", pan, fmt, present,
                                                _score(fmt, present)))
                if not fmt:
                    result.discarded["pan_card_number"] = pan
                    result.flags.append(Flag.HALLUCINATED_PAN)
                elif present is False:
                    # Well-formed but absent from the source: discard rather than
                    # export an identifier the document does not contain.
                    result.discarded["pan_card_number"] = pan
                    result.flags.append(Flag.HALLUCINATED_PAN)
                else:
                    claimed_pans.add(pan)
                    result.flags.append(Flag.PAN_MATCHED)

            # Aadhaar
            aadhaar = person.get("aadhaar_number")
            if rules.aadhaar and aadhaar:
                fmt = aadhaar_format_valid(aadhaar)
                present = (aadhaar_in_ocr(str(aadhaar), ocr_norm, ocr)
                           if rules.ocr_cross_verify and fmt else None)
                result.checks.append(FieldCheck("aadhaar", aadhaar, fmt, present,
                                                _score(fmt, present)))
                if not fmt or present is False:
                    result.flags.append(Flag.WRONG_AADHAAR)

            # Name
            if person.get("name"):
                present = name_in_ocr(str(person["name"]), ocr_norm) \
                    if rules.ocr_cross_verify else None
                result.checks.append(FieldCheck("name", person["name"], True, present,
                                                _score(True, present)))
                if present is False:
                    result.flags.append(Flag.NAME_NOT_IN_OCR)

            # Layer 4 - proximity
            near = pan_aadhaar_proximity(pan, aadhaar, ocr,
                                         rules.pan_aadhaar_proximity_chars)
            if near is False:
                result.flags.append(Flag.PAN_AADHAAR_FAR)
                result.checks.append(FieldCheck("pan_aadhaar_proximity", None, True,
                                                False, 0.4,
                                                "PAN and Aadhaar far apart in OCR"))

            result.confidence = _confidence(result.checks)
            report.persons.append(result)

    report.extra_ocr_pans = sorted(source_pans - claimed_pans)

    # -- Layer 3 document fields --------------------------------------------
    prop = pred.get("property_details") or {}
    doc = pred.get("document_details") or {}

    consideration = prop.get("sale_consideration")
    if rules.sale_consideration and consideration is not None:
        fmt = amount_format_valid(consideration)
        present = (amount_in_ocr(str(consideration), ocr_norm, ocr)
                   if rules.ocr_cross_verify and fmt else None)
        report.document_checks.append(
            FieldCheck("sale_consideration", consideration, fmt, present,
                       _score(fmt, present)))
        if not fmt or present is False:
            report.document_flags.append(Flag.WRONG_CONSIDERATION)

    reg_fee = prop.get("registration_fee")
    report.reg_fee_candidates = reg_fee_candidates(ocr, rules.reg_fee_min, rules.reg_fee_max)
    if rules.registration_fee and reg_fee is not None:
        fmt = amount_format_valid(reg_fee)
        present = (amount_in_ocr(str(reg_fee), ocr_norm, ocr)
                   if rules.ocr_cross_verify and fmt else None)
        # Layer 5 is ADVISORY, not a flag trigger on its own. The regex sweep is
        # unreliable: on deed 117 it returned 20000 while the model's 52500 is
        # present in the OCR as "52,500" and is exactly 1% of the consideration -
        # the standard fee. INFERENCE_PIPELINE.md itself notes the model may be
        # right and the regex may have caught a cess line. So disagreement lowers
        # confidence; only absence from the OCR raises WSV.
        agrees = (int(_digits(str(reg_fee))) in report.reg_fee_candidates
                  if fmt and report.reg_fee_candidates else None)
        score = _score(fmt, present)
        detail = ""
        if agrees is False:
            score = min(score, 0.7)
            detail = f"regex sweep found {report.reg_fee_candidates[:5]} instead (advisory)"
        report.document_checks.append(
            FieldCheck("registration_fee", reg_fee, fmt, present, score, detail))
        if not fmt or present is False:
            report.document_flags.append(Flag.WRONG_STAMP_VALUE)

        # Stamp Value: registration fee, halved before the cutoff date.
        if rules.stamp_value and fmt:
            report.stamp_value = derive_stamp_value(
                reg_fee, doc.get("transaction_date"), rules)
            if report.stamp_value is None and doc.get("transaction_date") is None:
                report.notes.append(
                    "stamp value not derived: the transaction date is missing, and "
                    "the halving rule depends on which side of the cutoff the deed falls")

            # SSV compares the DERIVED stamp value against the consideration,
            # not the raw fee - halving can move a deed across the threshold.
            if report.stamp_value is not None and amount_format_valid(consideration):
                if report.stamp_value > int(_digits(str(consideration))):
                    report.document_flags.append(Flag.STAMP_OVER_SALE)

    paid = prop.get("paid_in_cash")
    if paid not in ("yes", "no"):
        report.document_checks.append(
            FieldCheck("paid_in_cash", paid, False, None, 0.0,
                       'must be "yes" or "no", never null'))

    txn_date = doc.get("transaction_date")
    if rules.transaction_date:
        if not txn_date:
            report.document_flags.append(Flag.WRONG_TXN_DATE)
            report.document_checks.append(
                FieldCheck("transaction_date", None, False, None, 0.0, "missing"))
        else:
            fmt = date_iso_valid(txn_date)
            present = date_in_ocr(str(txn_date), ocr_norm) \
                if rules.ocr_cross_verify and fmt else None
            report.document_checks.append(
                FieldCheck("transaction_date", txn_date, fmt, present,
                           _score(fmt, present)))
            if not fmt:
                report.document_flags.append(Flag.WRONG_TXN_DATE)

    # -- Layers 6 and 7 ------------------------------------------------------
    report.pan_coverage = round(pan_coverage(pred, ocr), 3)

    all_checks = list(report.document_checks) + [c for p in report.persons for c in p.checks]
    report.confidence = _confidence(all_checks) if rules.confidence else 1.0

    report.disposition = _dispose(report, rules)
    return report


def _dispose(report: ValidationReport, rules: RuleToggles) -> Disposition:
    """Layer 7.

    Retry is reserved for *under-extraction* - a parse failure or missing PANs -
    because that is what a second pass can fix. A grounding failure (a value the
    OCR does not contain) will not improve on retry at temperature 0, so it goes
    straight to a human.
    """
    if not report.parsed:
        return Disposition.RETRY

    # Low coverage only triggers a retry when enough PANs are genuinely
    # unaccounted for. See RuleToggles.pan_coverage_min_unmatched.
    unmatched = len(report.extra_ocr_pans)
    if (report.pan_coverage < rules.pan_coverage_threshold
            and unmatched >= rules.pan_coverage_min_unmatched):
        report.notes.append(
            f"PAN coverage {report.pan_coverage:.2f} below "
            f"{rules.pan_coverage_threshold:.2f} with {unmatched} OCR PANs unmatched")
        return Disposition.RETRY
    if report.pan_coverage < rules.pan_coverage_threshold:
        report.notes.append(
            f"PAN coverage {report.pan_coverage:.2f} is low but only {unmatched} "
            "OCR PAN(s) unmatched - likely a witness or advocate, not under-extraction")

    grounding_failures = {
        Flag.WRONG_CONSIDERATION, Flag.WRONG_STAMP_VALUE, Flag.WRONG_AADHAAR,
        Flag.HALLUCINATED_PAN, Flag.PAN_AADHAAR_FAR, Flag.SCHEMA_INVALID,
        Flag.TRUNCATED,
    }
    present = set(report.document_flags) | {f for p in report.persons for f in p.flags}
    if present & grounding_failures:
        return Disposition.REVIEW
    if not report.schema_ok:
        return Disposition.REVIEW
    return Disposition.ACCEPT
