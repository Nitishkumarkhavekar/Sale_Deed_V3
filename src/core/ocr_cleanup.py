"""OCR cleanup - prepare raw OCR text for the extraction model.

The governing principle is restraint. This model was **finetuned on OCR text**,
padding and all, so every transformation applied here moves the input away from
the distribution it was trained on. That is not hypothetical: feeding CRLF text
produced 6,758 tokens where the training tokenizer produced 6,408 - a 5.5%
divergence - and normalising line endings fixed it exactly. The same mechanism
works in reverse if we "tidy" text the model expects to see untidied.

So transformations fall into two classes:

  * **Corrective** - on by default. These bring the text *closer* to what the
    model saw in training, or are provably lossless.
  * **Aggressive** - off by default. These reduce tokens but change layout the
    model may rely on. Enable only with measurement.

One hard constraint on the caller: the **cleaned** text is what must be passed to
the validator, not the raw text. Grounding checks match extracted values against
the OCR, so cleaning the model's input while validating against the original
would produce phantom mismatches.

Observed corpus characteristics (all 50 files in `test/OCR saledeeds`):
    line endings   CRLF, without exception
    page markers   already present - a `====` banner plus `PAGE n  (id)`
    padding        30,000+ runs of 4 or more spaces, used as column separators
    control chars  none beyond newline and tab
    scripts        mixed Kannada and Latin
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: Canonical page marker the specification asks for.
PAGE_MARKER = "===== PAGE {n} ====="

#: Existing markers in the corpus: a rule of '=' characters, then 'PAGE n (id)',
#: then another rule. Matched as a unit so it can be replaced wholesale.
EXISTING_PAGE_BLOCK = re.compile(
    r"^[=\-]{10,}[ \t]*\n[ \t]*PAGE[ \t]+(\d+)[^\n]*\n[=\-]{10,}[ \t]*$",
    re.MULTILINE,
)
#: Bare marker without surrounding rules.
BARE_PAGE_LINE = re.compile(r"^[ \t]*(?:[=\-]{3,}[ \t]*)?PAGE[ \t]+(\d+)[^\n]*$",
                            re.MULTILINE)
#: Form feed, the other common page delimiter.
FORM_FEED = "\f"

#: Surya writes a rule of "=" after every page. Once the page marker itself is
#: canonicalised the rule is pure noise, and it is long enough to cost real
#: tokens on a 30-page deed.
PAGE_RULE = re.compile(r"^[=\-]{20,}[ \t]*$", re.MULTILINE)

#: Inline markup Surya emits: <b>, </b>, <i>, <u>, <br>, <sub>, <sup>, <math>.
#: The training corpus contains NONE of these - checked across all 50 files - so
#: leaving them in hands the model a token stream it never saw. Worse, they wrap
#: precisely the fields that matter: the sample output has
#: `<b>Rs.30,00,000/-</b>` around an amount and `<math>455/1</math>` around a
#: survey number. Same class of defect as CRLF, with a larger blast radius.
MARKUP_TAG = re.compile(r"</?(?:b|i|u|br|em|strong|sub|sup|span|p|div)\s*/?>",
                        re.IGNORECASE)
#: <math>...</math> keeps its contents - the value inside is real data.
MATH_WRAPPER = re.compile(r"</?math[^>]*>", re.IGNORECASE)
#: LaTeX fragments Surya emits inside <math>, e.g. 42\frac{1}{2}.
#: A whole number in front of the fraction is captured with it. Surya writes an
#: extent of 42 and a half guntas as `42\frac{1}{2}`, and dropping the boundary
#: yields `421/2` - which reads as survey number 421/2, a different value
#: entirely. The space is what keeps the two numbers apart.
LATEX_FRAC = re.compile(
    r"(?:(?P<whole>\d+)\s*)?\\frac\{(?P<num>\d+)\}\{(?P<den>\d+)\}")
LATEX_CMD = re.compile(r"\\(?:text|mathrm|mathbf|left|right|,|;|!)\s*")


@dataclass(frozen=True)
class CleanupOptions:
    """What to do. Defaults are the conservative, evidence-backed set."""

    # -- corrective (default on) ----------------------------------------
    #: CRLF/CR -> LF. Measured requirement, not cosmetic. See ADR-005.
    normalise_line_endings: bool = True
    #: Trailing whitespace carries no information and costs tokens.
    strip_trailing_whitespace: bool = True
    #: Runs of blank lines beyond two convey nothing.
    collapse_blank_lines: bool = True
    #: Rewrite existing page markers to the canonical form; add them only if the
    #: document has none.
    canonical_page_markers: bool = True
    #: Remove control characters other than newline and tab.
    strip_control_chars: bool = True
    #: Strip Surya's inline markup. ON by default: the model was finetuned on
    #: text without it, so leaving it in is a distribution mismatch, not a
    #: cosmetic issue.
    strip_markup: bool = True

    # -- aggressive (default off) ---------------------------------------
    #: Cap runs of spaces. Reduces tokens noticeably, but these runs are column
    #: separators in this corpus - collapsing them makes unrelated fields
    #: adjacent, which can mislead the model about what belongs together.
    collapse_wide_padding: bool = False
    max_space_run: int = 8
    #: Unicode NFC. Sounds harmless; it can change token IDs for Kannada
    #: clusters, so it stays off until measured against the tokenizer.
    unicode_normalise: bool = False
    #: Drop lines that repeat on most pages (letterhead, footers). Saves prefill
    #: on long deeds but risks removing a line that carries a real field.
    drop_repeated_headers: bool = False
    repeated_header_min_pages: int = 3
    #: Blank out fully masked identifiers ("XXXX XXXX 1234"). Off by default:
    #: the model is trained to emit null for these, and removing them denies it
    #: the evidence that a masked value was present at all.
    blank_masked_identifiers: bool = False


@dataclass
class CleanupReport:
    """What actually changed, so the effect is auditable rather than assumed."""

    chars_before: int = 0
    chars_after: int = 0
    lines_before: int = 0
    lines_after: int = 0
    pages_detected: int = 0
    page_markers_rewritten: int = 0
    page_markers_inserted: int = 0
    crs_removed: int = 0
    control_chars_removed: int = 0
    blank_runs_collapsed: int = 0
    padding_runs_collapsed: int = 0
    repeated_headers_dropped: int = 0
    masked_identifiers_blanked: int = 0
    markup_tags_removed: int = 0
    page_rules_removed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def reduction(self) -> float:
        return self.chars_saved / self.chars_before if self.chars_before else 0.0

    def summary(self) -> str:
        return (
            f"{self.chars_before:,} -> {self.chars_after:,} chars "
            f"({self.reduction:+.1%}), {self.pages_detected} pages, "
            f"{self.crs_removed} CR removed, "
            f"{self.page_markers_rewritten} markers rewritten, "
            f"{self.page_markers_inserted} inserted"
            + (f", {self.markup_tags_removed} markup tags stripped"
               if self.markup_tags_removed else "")
        )


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


def _normalise_line_endings(text: str, report: CleanupReport) -> str:
    """CRLF and lone CR -> LF.

    The single most important step. llama.cpp preserves `\\r` as its own token
    while the tokenizer the model trained with drops it and merges `\\r\\n\\r\\n`
    into `\\n\\n`. Leaving CRLF in place hands the model a token stream it never
    saw during training.
    """
    report.crs_removed = text.count("\r")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_control(text: str, report: CleanupReport) -> str:
    keep = {"\n", "\t"}
    out = []
    removed = 0
    for ch in text:
        if ch in keep or ch == FORM_FEED:
            out.append(ch)
        elif unicodedata.category(ch).startswith("C"):
            removed += 1
        else:
            out.append(ch)
    report.control_chars_removed = removed
    return "".join(out)


def _find_pages(text: str) -> list[tuple[int, int, str]]:
    """Locate page markers. Returns (start, end, page_number) spans."""
    spans: list[tuple[int, int, str]] = []
    for m in EXISTING_PAGE_BLOCK.finditer(text):
        spans.append((m.start(), m.end(), m.group(1)))
    if not spans:
        for m in BARE_PAGE_LINE.finditer(text):
            spans.append((m.start(), m.end(), m.group(1)))
    return spans


def _canonical_markers(text: str, report: CleanupReport) -> str:
    """Rewrite existing markers to the canonical form, or insert them.

    The corpus already carries a `====` banner plus `PAGE n  (id)`. Those are
    replaced rather than supplemented - adding a second marker per page would
    both waste tokens and confuse a reader.
    """
    spans = _find_pages(text)
    if spans:
        out = []
        cursor = 0
        for start, end, number in spans:
            out.append(text[cursor:start])
            out.append(PAGE_MARKER.format(n=number))
            cursor = end
        out.append(text[cursor:])
        report.pages_detected = len(spans)
        report.page_markers_rewritten = len(spans)
        return "".join(out)

    if FORM_FEED in text:
        parts = text.split(FORM_FEED)
        rebuilt = []
        for i, part in enumerate(parts, start=1):
            rebuilt.append(PAGE_MARKER.format(n=i))
            rebuilt.append(part if part.startswith("\n") else "\n" + part)
        report.pages_detected = len(parts)
        report.page_markers_inserted = len(parts)
        return "".join(rebuilt)

    # No page structure available. Mark the whole document as page 1 so the
    # format is uniform for downstream consumers.
    report.pages_detected = 1
    report.page_markers_inserted = 1
    report.notes.append("no page delimiters found; treated as a single page")
    return PAGE_MARKER.format(n=1) + "\n" + text


def _strip_markup(text: str, report: CleanupReport) -> str:
    """Remove Surya's inline markup, keeping the text it wrapped.

    `<math>42\\frac{1}{2}</math>` becomes `42 1/2` rather than being deleted -
    the number is real data on a property schedule.
    """
    count = 0

    def latex_frac(m: re.Match[str]) -> str:
        fraction = f"{m.group('num')}/{m.group('den')}"
        whole = m.group("whole")
        return f"{whole} {fraction}" if whole else fraction

    out = LATEX_FRAC.sub(latex_frac, text)
    out, n = MATH_WRAPPER.subn("", out)
    count += n
    out, n = MARKUP_TAG.subn(lambda m: " " if m.group(0).lower().startswith("<br")
                             else "", out)
    count += n
    out = LATEX_CMD.sub("", out)
    report.markup_tags_removed = count
    return out


def _strip_page_rules(text: str, report: CleanupReport) -> str:
    """Drop the long "=" rules Surya writes between pages."""
    out, n = PAGE_RULE.subn("", text)
    report.page_rules_removed = n
    return out


def _strip_trailing_ws(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _collapse_blank_lines(text: str, report: CleanupReport) -> str:
    collapsed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal collapsed
        collapsed += 1
        return "\n\n"

    out = re.sub(r"\n{3,}", repl, text)
    report.blank_runs_collapsed = collapsed
    return out


def _collapse_padding(text: str, report: CleanupReport, max_run: int) -> str:
    collapsed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal collapsed
        collapsed += 1
        return " " * max_run

    out = re.sub(rf"[ ]{{{max_run + 1},}}", repl, text)
    report.padding_runs_collapsed = collapsed
    return out


def _drop_repeated_headers(text: str, report: CleanupReport, min_pages: int) -> str:
    """Remove lines that recur across most pages (letterhead, footers)."""
    pages = re.split(r"^===== PAGE \d+ =====$", text, flags=re.MULTILINE)
    body_pages = [p for p in pages if p.strip()]
    if len(body_pages) < min_pages:
        return text

    counts: dict[str, int] = {}
    for page in body_pages:
        for line in {ln.strip() for ln in page.split("\n") if len(ln.strip()) >= 12}:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(min_pages, int(len(body_pages) * 0.6))
    boilerplate = {ln for ln, c in counts.items() if c >= threshold}
    if not boilerplate:
        return text

    kept, dropped = [], 0
    for line in text.split("\n"):
        if line.strip() in boilerplate:
            dropped += 1
            continue
        kept.append(line)
    report.repeated_headers_dropped = dropped
    return "\n".join(kept)


MASKED_ID = re.compile(r"\b[Xx]{4}[\s\-]*[Xx]{4}[\s\-]*\d{4}\b")


def _blank_masked(text: str, report: CleanupReport) -> str:
    out, n = MASKED_ID.subn("", text)
    report.masked_identifiers_blanked = n
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def clean(text: str, options: CleanupOptions | None = None) -> tuple[str, CleanupReport]:
    """Clean OCR text. Returns (cleaned_text, report).

    The returned text is what must be sent to BOTH the model and the validator.
    """
    opts = options or CleanupOptions()
    report = CleanupReport(chars_before=len(text or ""),
                           lines_before=(text or "").count("\n") + 1)
    out = text or ""

    if opts.normalise_line_endings:
        out = _normalise_line_endings(out, report)
    if opts.strip_control_chars:
        out = _strip_control(out, report)
    if opts.unicode_normalise:
        out = unicodedata.normalize("NFC", out)
    if opts.canonical_page_markers:
        out = _canonical_markers(out, report)
        # Only after the markers are canonical, so the rules cannot be mistaken
        # for part of a page banner.
        out = _strip_page_rules(out, report)
    else:
        report.pages_detected = len(_find_pages(out))
    if opts.strip_markup:
        out = _strip_markup(out, report)
    if opts.blank_masked_identifiers:
        out = _blank_masked(out, report)
    if opts.strip_trailing_whitespace:
        out = _strip_trailing_ws(out)
    if opts.collapse_wide_padding:
        out = _collapse_padding(out, report, opts.max_space_run)
    if opts.drop_repeated_headers:
        out = _drop_repeated_headers(out, report, opts.repeated_header_min_pages)
    if opts.collapse_blank_lines:
        out = _collapse_blank_lines(out, report)

    out = out.strip("\n") + "\n"
    report.chars_after = len(out)
    report.lines_after = out.count("\n") + 1
    return out, report


def page_texts(cleaned: str) -> list[tuple[int, str]]:
    """Split cleaned text into (page_number, body) pairs.

    Useful for per-page storage: the schema keeps OCR per page so the 30-day
    cache expiry can operate at page granularity.
    """
    parts = re.split(r"^===== PAGE (\d+) =====$", cleaned, flags=re.MULTILINE)
    pages: list[tuple[int, str]] = []
    # re.split with one group yields [pre, num, body, num, body, ...]
    for i in range(1, len(parts) - 1, 2):
        try:
            number = int(parts[i])
        except ValueError:
            continue
        pages.append((number, parts[i + 1].strip("\n")))
    return pages
