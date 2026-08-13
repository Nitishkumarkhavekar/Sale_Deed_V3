"""Transaction Identity - the registration number, read off the deed itself.

A Kaveri registration number looks like `BGP-1-00275-2025-26`:

    BGP      sub-registrar office code, 2-4 letters
    1        book/series number
    00275    serial within the financial year, zero-padded
    2025-26  financial year

Finding one is easy - the format is distinctive. Finding the **right** one is
not, and that is what this module is for.

**Why the first match is wrong.** A sale deed recites its chain of title, so it
quotes the registration numbers of *prior* documents. Measured on the 50-deed
corpus, 9 of 50 contain more than one identity, and in at least one the deed's
own number appears at line 330 while a 2018-19 prior deed appears at line 107.
Taking the first match would put a previous owner's document number in the
Transaction Identity column - a wrong value, which is worse than a blank one,
because nothing downstream would question it.

Candidates are therefore scored on evidence:

* **Financial year.** The deed's own registration is the newest event in it;
  every cited document is older. This is the strongest single signal.
* **Label.** The registration block writes `ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ` (document number)
  or `ನಂಬರ್`. Prior titles appear after "Vide Doc No." or "registered as
  document No." in prose - a signal that argues *against* a candidate.
* **Repetition.** The deed's own number is usually stamped more than once.
* **Position.** Weakly preferred toward the front, where the registration block
  is - but never on its own, for the reason above.

Nothing here needs the language model. It is deterministic, instant, and works
identically on a digital text layer and on OCR output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

_log = logging.getLogger("saledeed.transaction_id")

#: The canonical form, and what is written to the CSV.
CANONICAL = re.compile(r"^[A-Z]{2,4}-\d-\d{5}-\d{4}-\d{2}$")

#: Deliberately loose, because this runs on OCR output.
#:
#: * Spaces may appear around the hyphens - OCR splits on glyph boundaries.
#: * The separator may be an en dash, em dash or a stray period.
#: * The serial is 4-6 digits: it is normally 5, but OCR drops or doubles one,
#:   and a real office occasionally pads differently.
#:
#: Not loose enough to match a date or an amount: the leading letter group and
#: the trailing two-part year anchor it.
#: The office group accepts digits so an OCR misread (`0GP` for `OGP`) is still
#: found and can be repaired. A candidate whose office contains no letter at all
#: is discarded in `find_candidates` - that shape is a phone number or a survey
#: reference, not a registration number.
PATTERN = re.compile(
    r"\b([A-Z0-9]{2,4})\s*[-–—.]\s*(\d)\s*[-–—.]\s*(\d{4,6})\s*[-–—.]\s*"
    r"(\d{4})\s*[-–—.]\s*(\d{2})\b"
)

#: Labels that introduce the deed's **own** registration number.
#: Wording that marks a candidate as *this* deed's registration rather than a
#: citation. Taken from the registration certificate Kaveri stamps on page 1,
#: which reads:
#:
#:     1 ನೇ ಪುಸ್ತಕದ ದಸ್ತಾವೇಜು
#:     ನಂಬರ BGP-1-00275-2025-26 ಆಗಿ
#:     ದಿನಾಂಕ 19/04/2025 ರಂದು ನೋಂದಾಯಿಸಿ ...
#:
#: `ನಂಬರ` is listed without the trailing virama. The certificate writes it that
#: way and the corpus agrees - the bare form appears in 49 of 50 documents
#: against 46 for `ನಂಬರ್` - so matching only the virama form missed the label on
#: the very block the number sits in, and the year signal was carrying the
#: decision alone. The bare form subsumes both, since one is a prefix of the other.
LABELS = (
    "ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ",     # Kannada: document number
    "ದಸ್ತಾವೇಜು ಸಂ",
    "ನಂಬರ",                 # Kannada: number - covers ನಂಬರ್ as well
    "ದಸ್ತಾವೇಜು ಕ್ರಮಾಂಕ",
    "ಪುಸ್ತಕದ ದಸ್ತಾವೇಜು",    # "book document" - the certificate's own heading
    "ನೋಂದಾಯಿಸಿ",            # "registered on"
    "ನೋಂದಣಾಧಿಕಾರಿ",         # "sub-registrar", who signs the block
    "document number",
    "document no",
    "doc number",
)

#: Phrases that introduce a **prior** document. A candidate preceded by one of
#: these is being cited, not registered - it belongs to the chain of title.
PRIOR_MARKERS = (
    "vide doc", "vide document", "registered as document",
    "registered as doc", "earlier document", "previous document",
    "vide no", "as per document", "mother deed", "parent document",
    "ಮೂಲ ದಸ್ತಾವೇಜು",       # Kannada: original/parent document
)

#: How far back to read for a label or a prior-document marker.
CONTEXT = 90

#: OCR confusions that are safe to correct **inside the office code**, where
#: only letters are valid. Never applied to the digits: turning a real 0 into an
#: O in a serial number would corrupt the value this module exists to get right.
_LETTER_FIXES = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})


@dataclass
class Candidate:
    """One identity found in the text, with the evidence for it."""

    value: str
    office: str
    series: str
    serial: str
    year: str
    position: int
    line: int
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def financial_year(self) -> int:
        """Starting year, for comparison. `2025-26` -> 2025."""
        try:
            return int(self.year.split("-")[0])
        except (ValueError, IndexError):
            return 0


@dataclass
class Extraction:
    """The result, and enough context to explain it in a log."""

    value: str = ""
    confidence: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    page: int | None = None
    reason: str = ""

    @property
    def found(self) -> bool:
        return bool(self.value)


def _normalise(office: str, series: str, serial: str,
               year_a: str, year_b: str) -> str:
    """Assemble the canonical form, repairing what is safe to repair."""
    office = office.upper().translate(_LETTER_FIXES)
    # Pad or trim the serial to five digits. Padding is safe; trimming only
    # removes a leading zero OCR duplicated, never a significant digit.
    serial = serial.lstrip("0") or "0"
    serial = serial[-5:].rjust(5, "0")
    return f"{office}-{series}-{serial}-{year_a}-{year_b}"


def _page_of(text: str, position: int) -> int | None:
    """Which page marker precedes this position, if the text carries them."""
    markers = [m for m in re.finditer(r"=+\s*PAGE\s+(\d+)\s*=+", text[:position],
                                      re.IGNORECASE)]
    return int(markers[-1].group(1)) if markers else None


def find_candidates(text: str) -> list[Candidate]:
    """Every identity in the text, unscored."""
    found: list[Candidate] = []
    for match in PATTERN.finditer(text or ""):
        office, series, serial, year_a, year_b = match.groups()
        # An office code with no letter is not an office code. Without this, a
        # string of digits separated by hyphens - a phone number, a survey
        # reference - would be accepted as a registration number.
        if not any(c.isalpha() for c in office):
            continue
        value = _normalise(office, series, serial, year_a, year_b)
        found.append(Candidate(
            value=value, office=value.split("-")[0], series=series,
            serial=value.split("-")[2], year=f"{year_a}-{year_b}",
            position=match.start(),
            line=text[:match.start()].count("\n") + 1))
    return found


def _score(candidates: list[Candidate], text: str) -> None:
    """Weigh the evidence for each candidate. Highest score wins."""
    if not candidates:
        return

    newest = max(c.financial_year for c in candidates)
    occurrences: dict[str, int] = {}
    for candidate in candidates:
        occurrences[candidate.value] = occurrences.get(candidate.value, 0) + 1
    last_position = max(c.position for c in candidates) or 1

    for candidate in candidates:
        before = text[max(0, candidate.position - CONTEXT):candidate.position].lower()

        # The deed's own registration is the newest event in it. Every document
        # it cites was registered earlier.
        if candidate.financial_year == newest:
            candidate.score += 5.0
            candidate.reasons.append("newest financial year")
        else:
            candidate.score -= 3.0
            candidate.reasons.append(f"older year {candidate.year}")

        # A label introduces the deed's own number.
        if any(label.lower() in before for label in LABELS):
            candidate.score += 4.0
            candidate.reasons.append("registration label")

        # Prose citing a prior document argues against.
        if any(marker in before for marker in PRIOR_MARKERS):
            candidate.score -= 5.0
            candidate.reasons.append("cited as a prior document")

        # Repeated stamping is characteristic of the deed's own number.
        repeats = occurrences[candidate.value]
        if repeats > 1:
            candidate.score += min(2.0, 0.5 * repeats)
            candidate.reasons.append(f"appears {repeats} times")

        # Weak positional preference toward the registration block.
        candidate.score += 1.0 * (1.0 - candidate.position / last_position)


def extract(text: str, *, source: str = "", ocr_used: bool | None = None) -> Extraction:
    """Read the Transaction Identity from a deed's text.

    Returns an empty `Extraction` when nothing convincing is found. That is
    deliberate: the requirement is to leave the column blank rather than write a
    value that might belong to a previous owner.
    """
    candidates = find_candidates(text)
    if not candidates:
        _log.info("no transaction identity found", extra={
            "source": source, "ocr_used": ocr_used, "candidates": 0})
        return Extraction(reason="no candidate matched the registration format")

    _score(candidates, text)
    ranked = sorted(candidates, key=lambda c: -c.score)
    best = ranked[0]

    # Distinct alternatives, for confidence and for the log.
    distinct = {c.value for c in candidates}
    runner_up = next((c for c in ranked[1:] if c.value != best.value), None)

    if len(distinct) == 1:
        confidence = 0.95
    elif runner_up is not None and best.score - runner_up.score >= 5.0:
        confidence = 0.85
    elif runner_up is not None and best.score - runner_up.score >= 2.0:
        confidence = 0.7
    else:
        # Two candidates the evidence cannot separate. A coin flip here would
        # write a previous owner's document number into a legal export.
        _log.warning("transaction identity is ambiguous - leaving it blank",
                     extra={"source": source, "ocr_used": ocr_used,
                            "candidates": sorted(distinct),
                            "best": best.value,
                            "runner_up": runner_up.value if runner_up else None})
        return Extraction(candidates=ranked, reason="ambiguous candidates")

    if not CANONICAL.match(best.value):
        _log.warning("transaction identity failed format validation",
                     extra={"source": source, "value": best.value})
        return Extraction(candidates=ranked,
                          reason=f"{best.value} is not a valid registration number")

    page = _page_of(text, best.position)
    _log.info("transaction identity %s", best.value, extra={
        "source": source, "page": page, "ocr_used": ocr_used,
        "value": best.value, "confidence": round(confidence, 2),
        "line": best.line, "candidates": len(distinct),
        "why": ", ".join(best.reasons)})

    return Extraction(value=best.value, confidence=confidence,
                      candidates=ranked, page=page,
                      reason=", ".join(best.reasons))
