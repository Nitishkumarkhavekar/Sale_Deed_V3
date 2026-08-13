"""Language detection by script.

**Why script ranges rather than a statistical detector.**

The inputs here are deed *fields* - a name, a village, a two-line address - not
paragraphs. Statistical detectors (langdetect, fastText, langid) are trained on
running prose and are unreliable below roughly twenty words; on a two-word
Kannada name they guess, and they guess differently between runs because several
seed by random state. A wrong guess is not a near miss: it selects the wrong
source language for the translator and produces confident nonsense.

Script identity, by contrast, is a property of the characters themselves. Every
major Indian language in scope has its own Unicode block, so detection is exact,
instant, deterministic, and needs no model, no download and no dependency:

    Kannada    U+0C80-U+0CFF        Telugu     U+0C00-U+0C7F
    Tamil      U+0B80-U+0BFF        Malayalam  U+0D00-U+0D7F
    Gujarati   U+0A80-U+0AFF        Bengali    U+0980-U+09FF
    Gurmukhi   U+0A00-U+0A7F        Odia       U+0B00-U+0B7F
    Devanagari U+0900-U+097F        Arabic     U+0600-U+06FF

**Devanagari carries two languages: Hindi and Marathi.** The script alone does
not separate them, but the *text* usually does, and this module now looks.

Two kinds of evidence, both deterministic:

* **Letters Hindi does not use.** `ळ` (U+0933) is ordinary in Marathi and absent
  from standard Hindi; `ॲ` (U+0972) is Marathi-specific. One of these in a field
  is close to conclusive.
* **Vocabulary that differs on exactly the words a deed repeats.** `जिल्हा` /
  `जिला`, `तालुका` / `तहसील`, `मध्ये` / `में`, `आणि` / `और`, `आहे` / `है`.
  Registered deeds are formulaic, so these appear many times per document.

Where the evidence is absent - a two-word personal name shared by both languages
- the module does not guess. It falls back to whatever the caller configured,
which is Hindi unless the operator says otherwise. Guessing on no evidence is
what a statistical detector does, and the reason this module exists.

Urdu is likewise reported from Arabic script; Arabic and Persian share the block
but do not appear on Indian sale deeds.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class Script(str, Enum):
    """A writing system, which is what characters actually tell us."""

    LATIN = "latin"
    DEVANAGARI = "devanagari"
    KANNADA = "kannada"
    TELUGU = "telugu"
    TAMIL = "tamil"
    MALAYALAM = "malayalam"
    GUJARATI = "gujarati"
    BENGALI = "bengali"
    GURMUKHI = "gurmukhi"
    ODIA = "odia"
    ARABIC = "arabic"
    #: Digits, punctuation and symbols only - no script to speak of.
    NEUTRAL = "neutral"


#: (first, last, script). Ordered by how often they appear in this corpus so the
#: common case exits early.
_RANGES: tuple[tuple[int, int, Script], ...] = (
    (0x0C80, 0x0CFF, Script.KANNADA),
    (0x0900, 0x097F, Script.DEVANAGARI),
    (0x0C00, 0x0C7F, Script.TELUGU),
    (0x0B80, 0x0BFF, Script.TAMIL),
    (0x0D00, 0x0D7F, Script.MALAYALAM),
    (0x0A80, 0x0AFF, Script.GUJARATI),
    (0x0980, 0x09FF, Script.BENGALI),
    (0x0A00, 0x0A7F, Script.GURMUKHI),
    (0x0B00, 0x0B7F, Script.ODIA),
    (0x0600, 0x06FF, Script.ARABIC),
    (0x0750, 0x077F, Script.ARABIC),          # Arabic Supplement
    (0xFB50, 0xFDFF, Script.ARABIC),          # Presentation Forms-A
    (0x0041, 0x005A, Script.LATIN),
    (0x0061, 0x007A, Script.LATIN),
    (0x00C0, 0x024F, Script.LATIN),           # Latin-1 Supplement + Extended
)

#: FLORES-200 codes, which is what NLLB expects. ISO 639-1 produces silent
#: garbage rather than an error, so the mapping is explicit.
LANGUAGE_FOR_SCRIPT: dict[Script, str] = {
    Script.LATIN: "eng_Latn",
    # Resolved per-field by `discriminate_devanagari`; this is the fallback used
    # when a field carries no evidence either way.
    Script.DEVANAGARI: "hin_Deva",
    Script.KANNADA: "kan_Knda",
    Script.TELUGU: "tel_Telu",
    Script.TAMIL: "tam_Taml",
    Script.MALAYALAM: "mal_Mlym",
    Script.GUJARATI: "guj_Gujr",
    Script.BENGALI: "ben_Beng",
    Script.GURMUKHI: "pan_Guru",
    Script.ODIA: "ory_Orya",
    Script.ARABIC: "urd_Arab",
    Script.NEUTRAL: "eng_Latn",
}

#: Human-readable, for logs and the UI.
LANGUAGE_NAMES: dict[str, str] = {
    "eng_Latn": "English", "hin_Deva": "Hindi", "mar_Deva": "Marathi",
    "kan_Knda": "Kannada", "tel_Telu": "Telugu", "tam_Taml": "Tamil",
    "mal_Mlym": "Malayalam", "guj_Gujr": "Gujarati", "ben_Beng": "Bengali",
    "pan_Guru": "Punjabi", "ory_Orya": "Odia", "urd_Arab": "Urdu",
}

#: English target. Named so callers do not spell it themselves.
ENGLISH = "eng_Latn"


def script_of(char: str) -> Script:
    """The script a single character belongs to."""
    # Digits first, because every Indic block contains its own set of them and
    # they sit inside the letter range. Without this a PIN code written as
    # `४११०४८` counts as Devanagari, gets queued as translatable text, and the
    # model is asked to render a number into English - which is exactly the
    # invention this module exists to avoid. `Nd` covers Devanagari, Kannada,
    # Tamil and the rest in one test.
    if unicodedata.category(char) == "Nd":
        return Script.NEUTRAL

    point = ord(char)
    for first, last, script in _RANGES:
        if first <= point <= last:
            return script
    # Anything else - digits, punctuation, symbols, whitespace - carries no
    # language. Treating it as Latin would make "560001" look like English and
    # a Kannada address with many digits look mixed.
    return Script.NEUTRAL


@dataclass(frozen=True)
class Detection:
    """What a piece of text is written in."""

    language: str
    script: Script
    #: Fraction of script-bearing characters belonging to the dominant script.
    #: 1.0 is a single script; lower means genuinely mixed.
    confidence: float
    #: Every script present with at least one character, by share.
    scripts: dict[Script, float] = field(default_factory=dict)
    #: Why this language was chosen, when the script alone did not decide it.
    #: Only Devanagari fills this in - Hindi and Marathi share the script, so
    #: the choice is evidence-based and worth being able to read back.
    reason: str = ""

    @property
    def is_english(self) -> bool:
        return self.language == ENGLISH

    @property
    def is_mixed(self) -> bool:
        """More than one script carries meaningful weight.

        A Kannada address containing "No. 42" is not mixed in any useful sense -
        the digits are neutral. This only fires when two *scripts* are present.
        """
        return len([s for s in self.scripts if s is not Script.NEUTRAL]) > 1

    @property
    def name(self) -> str:
        return LANGUAGE_NAMES.get(self.language, self.language)



# ---------------------------------------------------------------------------
# Hindi or Marathi
# ---------------------------------------------------------------------------

HINDI = "hin_Deva"
MARATHI = "mar_Deva"

#: Letters standard Hindi does not use. `ळ` is an ordinary Marathi consonant;
#: `ॲ` and `ऱ` are Marathi orthography. One of these is close to conclusive.
_MARATHI_LETTERS = frozenset("ळॲऱ")

#: Words a Marathi deed repeats, against their Hindi counterparts. Chosen for the
#: pairs a registered document cannot avoid - district, taluka, "and", "is", "in"
#: - so a page of either language matches many times rather than once.
_MARATHI_WORDS = frozenset({
    "आहे", "आहेत", "नाही", "आणि", "मध्ये",
    "यांनी", "यांचे", "यांची", "यांच्या",
    "त्यांनी", "त्यांचे", "तसेच", "असून", "केले", "केली", "झाले",
    "मौजे", "तालुका", "जिल्हा", "नोंदणी", "दस्त", "दस्तएवज",
    "मिळकत", "क्षेत्रफळ", "खरेदीदार", "विक्रेता", "वहिवाट",
    "सातबारा", "भूमापन", "गाव", "रस्ता", "सदर", "चौरस",
})

_HINDI_WORDS = frozenset({
    "है", "हैं", "और", "में", "का", "की", "के", "को", "से", "यह", "वह",
    "हुआ", "किया", "गया", "तथा", "एवं",
    "तहसील", "जिला", "पंजीकरण", "विक्रय", "क्रय", "संपत्ति",
    "सड़क", "गाँव", "पटवारी", "खसरा", "खतौनी", "राशि",
})

#: Anything that is not a Devanagari letter separates words. Whole tokens, not
#: substrings: `के` is a Hindi postposition and also the opening of `केले`, Marathi.
_WORD_SPLIT = re.compile(r"[^\u0900-\u097F]+")


def discriminate_devanagari(text: str, *, fallback: str = HINDI) -> tuple[str, str]:
    """Decide Hindi or Marathi from the text, or fall back and say so.

    Returns `(language, reason)`. The reason reaches the log, so an operator
    reading a bad translation can see whether the language was determined or
    merely assumed - "no distinguishing evidence" is the useful case.
    """
    letters = _MARATHI_LETTERS & set(text)
    words = [w for w in _WORD_SPLIT.split(text) if w]
    marathi = sum(1 for w in words if w in _MARATHI_WORDS)
    hindi = sum(1 for w in words if w in _HINDI_WORDS)

    # A letter Hindi does not have outweighs a single shared-looking word.
    score = marathi + 2 * len(letters)
    if score > hindi:
        why = f"{marathi} Marathi word(s)"
        if letters:
            why += f", letter(s) {''.join(sorted(letters))}"
        return MARATHI, why
    if hindi > score:
        return HINDI, f"{hindi} Hindi word(s)"
    return fallback, "no distinguishing evidence"


def detect(text: str, *, devanagari_as: str = "auto") -> Detection:
    """Identify the language of one field.

    `devanagari_as` controls the one script that carries two languages.
    `"auto"` reads the text and decides - see `discriminate_devanagari` - and
    falls back to Hindi where a field offers no evidence. Passing `"hin_Deva"`
    or `"mar_Deva"` forces it, which is what an operator working a single
    jurisdiction should do. Everything else is unambiguous from the characters.

    Text is normalised to NFC first: a decomposed Kannada vowel sign is a
    combining mark that would otherwise be counted separately from its base and
    skew the share.
    """
    if not text or not text.strip():
        return Detection(ENGLISH, Script.NEUTRAL, 0.0, {})

    counts: dict[Script, int] = {}
    for char in unicodedata.normalize("NFC", text):
        script = script_of(char)
        counts[script] = counts.get(script, 0) + 1

    meaningful = {s: n for s, n in counts.items() if s is not Script.NEUTRAL}
    if not meaningful:
        # Digits and punctuation only: an amount, a PIN code, a survey number.
        # Nothing to translate, and calling it English is the honest answer.
        return Detection(ENGLISH, Script.NEUTRAL, 1.0,
                         {Script.NEUTRAL: 1.0})

    total = sum(meaningful.values())
    shares = {s: n / total for s, n in meaningful.items()}
    dominant = max(shares, key=shares.get)

    language = LANGUAGE_FOR_SCRIPT[dominant]
    reason = ""
    if dominant is Script.DEVANAGARI:
        if devanagari_as in ("auto", "", None):
            language, reason = discriminate_devanagari(text)
        else:
            language, reason = devanagari_as, "set by configuration"

    return Detection(language, dominant, shares[dominant], shares, reason)


def needs_translation(text: str, *, target: str = ENGLISH,
                      devanagari_as: str = "auto") -> bool:
    """True when this value would reach an English column in another script.

    Deliberately not `detect(...).language != target`: a value of pure digits
    detects as English and needs nothing, and asking the translator to render
    "560001" wastes a model call and risks it inventing words.
    """
    result = detect(text, devanagari_as=devanagari_as)
    if result.script is Script.NEUTRAL:
        return False
    if result.language != target:
        return True

    # The dominant script is Latin, but a mixed value still has work in it.
    # `KRISHNAPPA ರಾಜು` is ten Latin characters against four Kannada, so the
    # dominant-script test called it English and the Kannada half reached the
    # CSV untranslated. Any non-Latin, non-neutral script means something here
    # still needs rendering.
    return any(script not in (Script.LATIN, Script.NEUTRAL)
               for script in result.scripts)


def summarise(texts: list[str], *, devanagari_as: str = "hin_Deva") -> dict[str, int]:
    """Languages present across many fields, for a document-level log line.

    A deed is a mixed document by nature - Kannada prose, Latin PANs, digits -
    so the useful question is which languages appear and how often, not which
    single language "the document" is.
    """
    tally: dict[str, int] = {}
    for text in texts:
        result = detect(text, devanagari_as=devanagari_as)
        if result.script is Script.NEUTRAL:
            continue
        tally[result.language] = tally.get(result.language, 0) + 1
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]))
