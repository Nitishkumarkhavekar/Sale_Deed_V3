"""Proper nouns: rendered by rule, never by the translation model.

**Why names must not go through NLLB.**

NLLB is a sentence translator. Given a two-word fragment it does what it was
trained to do - it translates *meaning* - and on Indian names the meaning is
often a word. Measured on this project's own model:

    ಲಕ್ಷ್ಮಿ ದೇವಿ   ->  "Goddess Lakshmi"      a person became a deity
    ವೆಂಕಟೇಶ್       ->  "What is Venkatesh?"   a name became a question
    ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ -> "Rural areas of Bangalore"

On a legal record identifying parties to a property transfer, that is not a
quality issue - it is a corrupted record. Tightening the beam and capping the
length reduces the rate but cannot remove the failure mode, because the model is
doing exactly what it was built for.

Rule-based transliteration has the one property that matters here: it is
**deterministic and cannot invent**. Every character maps to a sound. The output
is occasionally clumsy, never wrong in kind, and never a different person.

The pipeline is: script -> IAST (scholarly, unambiguous) -> plain English
orthography. IAST is used as the intermediate rather than ITRANS because its
diacritics are regular, so the mapping to ASCII digraphs is a table rather than
a set of special cases.
"""

from __future__ import annotations

import re
import unicodedata

from .detect import Script, detect

#: IAST diacritics to the digraphs an English reader expects. `ś` and `ṣ` both
#: become "sh" deliberately: the distinction is real in Sanskrit and invisible
#: to the registrar reading the export.
_IAST_TO_ASCII = {
    "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o",
    "ṛ": "ri", "ṝ": "ri", "ḷ": "li", "ḹ": "li",
    "ṃ": "n", "ṁ": "n", "ḥ": "h", "m̐": "n",
    "ś": "sh", "ṣ": "sh", "ñ": "n", "ṅ": "n",
    "ṭ": "t", "ḍ": "d", "ṇ": "n", "ḻ": "l", "ḷ̆": "l",
    "è": "e", "ò": "o", "ĕ": "e", "ŏ": "o",
    "ẖ": "h", "ḫ": "h", "ṟ": "r", "ṉ": "n",
}

#: Scripts this module can render, mapped to the library's scheme names.
_SCHEME_FOR_SCRIPT = {
    Script.KANNADA: "kannada",
    Script.DEVANAGARI: "devanagari",
    Script.TELUGU: "telugu",
    Script.TAMIL: "tamil",
    Script.MALAYALAM: "malayalam",
    Script.GUJARATI: "gujarati",
    Script.BENGALI: "bengali",
    Script.GURMUKHI: "gurmukhi",
    Script.ODIA: "oriya",
}

#: Scripts carrying an inherent vowel that transliteration makes explicit and
#: the language does not pronounce. `रमेश` becomes "ramesha"; the deed says
#: Ramesh. Dravidian scripts mark it with a virama, so the library already
#: handles them and they must NOT be in this set - dropping a real final vowel
#: would turn "Rama" into "Ram", a different name.
_SCHWA_SCRIPTS = {
    Script.DEVANAGARI, Script.GUJARATI, Script.BENGALI,
    Script.ODIA, Script.GURMUKHI,
}

#: Malayalam chillu letters - a consonant with no inherent vowel, written as a
#: single codepoint. Several transliteration tables omit them and pass the glyph
#: through untouched, which puts Malayalam script into an English column.
_CHILLU = {
    "ൺ": "ണ്",  # chillu NN -> nn + virama
    "ൻ": "ന്",  # chillu N
    "ർ": "ര്",  # chillu RR
    "ൽ": "ല്",  # chillu L
    "ൾ": "ള്",  # chillu LL
    "ൿ": "ക്",  # chillu K
}

#: Tamil has no voiced or aspirated stops in its script - one letter serves both
#: - so any `gh`, `bh`, `dh` or word-initial voiced stop in the output is an
#: artefact of the table, not a sound in the name. Tamil words do not begin with
#: a voiced stop, so the unvoiced form is the citation form a reader expects.
_TAMIL_UNVOICE = (("gh", "k"), ("jh", "ch"), ("dh", "t"), ("bh", "p"))

#: How Kannada names are actually spelled in English, against how IAST renders
#: them. `c` is IAST for ಚ and is written "ch" everywhere outside a
#: transliteration table; `au` is the ಔ vowel and is written "ow" in the names
#: it appears in - Gowda, Gowri. Neither is a preference: a deed's own English
#: pages spell them this way, and so does every official record.
#: The lookahead excludes a full stop as well as an `h`. `ಸಿ` is the letter C,
#: and `read_initials` has already turned it into `C.` by the time these run -
#: without that guard the convention rewrote the initial itself, and
#: `ನವೀನ್ ಕುಮಾರ್ ಎಂ. ಸಿ.` came out as `Navin Kumar M. Ch.`
_KANNADA_CONVENTIONS = ((r"c(?![h.])", "ch"), (r"C(?![h.])", "Ch"),
                        (r"au", "ow"), (r"Au", "Ow"))

#: Words that are titles or honorifics rather than part of the name. Left in
#: place - dropping them would change what the record says - but not title-cased
#: as if they were surnames.
_PARTICLES = {"bin", "ibn", "al", "van", "von", "de", "da", "di", "of", "the"}


def available() -> bool:
    """Whether rule-based transliteration can run."""
    try:
        import indic_transliteration  # noqa: F401
    except ImportError:
        return False
    return True


def _strip_diacritics(text: str) -> str:
    """IAST to plain ASCII, digraphs first then any residue."""
    for mark, plain in _IAST_TO_ASCII.items():
        text = text.replace(mark, plain).replace(mark.upper(), plain.upper())
    # Anything still carrying a combining mark: decompose and drop the mark
    # rather than emit a character a spreadsheet may not render.
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


#: IAST vowels, including the marked long forms. Needed because the rule below
#: runs on IAST, where `a` and `ā` are still different characters.
_IAST_VOWELS = frozenset("aāiīuūṛṝḷḹeēoōṃḥ")

#: Aspirates are two characters in IAST but one consonant in the script. The
#: cluster test below has to know that or it keeps every schwa after one.
_ASPIRATES = frozenset({"kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh",
                        "ph", "bh"})


def _drop_schwa(word: str) -> str:
    """Remove the inherent final vowel Devanagari carries into transliteration.

    `रमेश` transliterates as "ramesha" because the final consonant carries an
    implicit `a`. Hindi and Marathi do not pronounce it, and "Ramesha Kumara"
    reads as a different name from the one on the deed.

    **Run on IAST, before the diacritics are stripped.** That ordering is the
    whole correctness of the rule. An inherent schwa is a plain `a`; a real
    final vowel is `ā`. Strip first and the two become the same character, so
    `सुनीता` -> `sunītā` -> `sunita` loses its ending and prints "Sunit" - a
    different name, on a legal document. Marathi place names show it too:
    `कोंढवा` became "Kondhav" rather than "Kondhwa".
    """
    if len(word) <= 3 or not word.endswith("a"):
        return word

    stem = word[:-1]
    if stem[-1] in _IAST_VOWELS:
        return word            # a real vowel precedes; nothing inherent here

    # Strip the final consonant to see what it sits on, counting an aspirate
    # digraph as the single letter it represents: `bh` is one consonant, and
    # treating it as two would make `mukha` look like a cluster and keep it.
    coda = stem[:-2] if stem[-2:] in _ASPIRATES else stem[:-1]

    # A schwa after a consonant *cluster* is pronounced - `mahārāṣṭra` is
    # Maharashtra, not "Maharashtr". Only the single-consonant case is silent.
    if coda and coda[-1] not in _IAST_VOWELS:
        return word
    return word[:-1]


def _title_case(text: str) -> str:
    """Capitalise as a name, not as a sentence.

    `str.title()` is wrong here - it capitalises after every apostrophe and
    breaks "D'Souza" into "D'Souza" only by luck. This capitalises word-initial
    letters and leaves particles alone.
    """
    words = []
    for word in text.split():
        if not word:
            continue
        if word.lower() in _PARTICLES and words:
            words.append(word.lower())
        else:
            words.append(word[0].upper() + word[1:])
    return " ".join(words)


def transliterate(text: str, *, script: Script | None = None) -> str:
    """Render a proper noun in English letters.

    Returns the input unchanged when the script is unsupported or the library
    is missing - never blank, and never a guess.
    """
    if not text or not text.strip():
        return text

    if script is None:
        script = detect(text).script
    if script is Script.LATIN or script is Script.NEUTRAL:
        return text

    # Initials first, before anything is sounded out. `ಜಿ.` is the letter G,
    # not the syllable "ji", and once it has been transliterated that is no
    # longer recoverable.
    text = read_initials(text, script=script)

    scheme = _SCHEME_FOR_SCRIPT.get(script)
    if scheme is None:
        # Urdu and anything else without a reliable rule set. The caller falls
        # back to the model, which is imperfect but better than nothing.
        return text

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _tr
    except ImportError:
        return text

    source = text
    if script is Script.MALAYALAM:
        for chillu, expanded in _CHILLU.items():
            source = source.replace(chillu, expanded)

    try:
        iast = _tr(source, scheme, sanscript.IAST)
    except Exception:  # noqa: BLE001 - a bad glyph must not lose the value
        return text

    if script in _SCHWA_SCRIPTS:
        iast = " ".join(_drop_schwa(w) for w in iast.split())

    plain = _strip_diacritics(iast)

    if script is Script.KANNADA:
        for pattern, conventional in _KANNADA_CONVENTIONS:
            plain = re.sub(pattern, conventional, plain)

    if script is Script.TAMIL:
        for artefact, real in _TAMIL_UNVOICE:
            plain = plain.replace(artefact, real)
        # A voiced stop cannot begin a Tamil word.
        plain = " ".join(
            {"g": "k", "j": "ch", "d": "t", "b": "p"}.get(w[:1], w[:1]) + w[1:]
            if w else w
            for w in plain.split())

    # Collapse the doubled letters that IAST conjuncts sometimes produce, and
    # tidy spacing around punctuation the source may carry.
    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s+([,.])", r"\1", plain)
    return _title_case(plain)


#: A run of characters that is *not* Indic. Split on these and the Latin parts
#: can be handed back exactly as they arrived.
_NON_INDIC_RUN = re.compile(r"([^\u0900-\u0DFF]+)")


# ---------------------------------------------------------------------------
# Initials
# ---------------------------------------------------------------------------
#
# An Indian name is very often written with initials, and Kannada spells those
# initials *phonetically* - the sound of the English letter's name, not the
# letter. `ಜಿ` is how you write "G"; `ಕೆ` is "K"; `ಎಂ` is "M".
#
# Transliterating them by sound is therefore wrong in a specific, systematic
# way: `ಜಿ.ಕೆ. ರಾಜು` came out as "Ji.ke. Raju" where the deed means
# "G.K. Raju". It is not a spelling preference - the initials are different
# letters from the ones printed.
#
# This is a property of the writing system, the same kind of fact as the
# Malayalam chillu table below, and it is applied only where it cannot be
# ambiguous: a token that is *exactly* one of these and is followed by a full
# stop. `ಬಿ` alone could begin an ordinary word; `ಬಿ.` is an initial.
KANNADA_INITIALS = {
    "ಎ": "A", "ಏ": "A", "ಬಿ": "B", "ಸಿ": "C", "ಡಿ": "D", "ಇ": "E", "ಈ": "E",
    "ಎಫ್": "F", "ಜಿ": "G", "ಎಚ್": "H", "ಹೆಚ್": "H", "ಐ": "I", "ಜೆ": "J", "ಕೆ": "K",
    "ಎಲ್": "L", "ಎಂ": "M", "ಎಮ್": "M", "ಎನ್": "N", "ಓ": "O", "ಪಿ": "P",
    "ಕ್ಯೂ": "Q", "ಆರ್": "R", "ಎಸ್": "S", "ಟಿ": "T", "ಯು": "U",
    "ವಿ": "V", "ಡಬ್ಲ್ಯೂ": "W", "ಎಕ್ಸ್": "X", "ವೈ": "Y",
    "ಜೆಡ್": "Z", "ಝಡ್": "Z",
}

#: Devanagari does the same thing, and Marathi deeds carry initials too.
DEVANAGARI_INITIALS = {
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F",
    "जी": "G", "एच": "H", "आय": "I", "जे": "J", "के": "K",
    "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P",
    "क्यू": "Q", "आर": "R", "एस": "S", "टी": "T",
    "यू": "U", "वी": "V", "डब्ल्यू": "W", "एक्स": "X",
    "वाय": "Y", "जेड": "Z",
}

#: A token followed by a full stop - the only position an initial is certain.
_INITIAL_TOKEN = re.compile(r"([^\s.]+)\s*\.")


#: `ಜಿ.ಎಲ್.ರವಿ` is written with no spaces at all. Once the initials are letters
#: the result is `G.L.ravi`, which title-casing cannot fix because it is one
#: token. English writes the name separately: `G.L. Ravi`.
_RUN_ON_INITIALS = re.compile(r"((?:[A-Z]\.)+)\s*(?=[^\sA-Z.])")

#: A name immediately followed by its initials, with no space between.
_NAME_THEN_INITIALS = re.compile(r"([^\s.]{2,})\.(?=[A-Z]\.)")


def read_initials(text: str, *, script) -> str:
    """Turn phonetically written initials back into letters.

    A token counts as an initial when it is *exactly* one of the letter names
    **and** it sits where an initial sits: next to a full stop, on either side,
    or straight after another initial. Deeds use all three arrangements —

        ಜಿ.ಕೆ. ರಾಜು           G.K. Raju         stop follows
        ಶಶಿಕುಮಾರ್.ಆರ್         Shashikumar R.    stop precedes, none follows
        ವಿಜಯ್ಕುಮಾರ್ ಕೆ. ಎಂ    Vijaykumar K.M.   runs on from the previous one

    Position is what makes this safe. ಬಿ is the letter B and also the opening of
    ಬಿಂದು (Bindu); requiring the punctuation means an ordinary word is never
    mistaken for an initial. Ordinary short words are absent from the table for
    the same reason — they are names, not letters.
    """
    table = (KANNADA_INITIALS if getattr(script, "value", "") == "kannada"
             else DEVANAGARI_INITIALS if getattr(script, "value", "") == "devanagari"
             else None)
    if table is None:
        return text

    # Split keeping the separators: whether a stop adjoins is the whole decision.
    parts = re.split(r"([\s.]+)", text)
    was_initial = False
    for i, part in enumerate(parts):
        if not part or re.fullmatch(r"[\s.]+", part):
            continue
        letter = table.get(part.strip())
        if letter is None:
            was_initial = False
            continue

        after = parts[i + 1] if i + 1 < len(parts) else ""
        before = parts[i - 1] if i else ""
        if "." in after or "." in before or was_initial:
            # Supply the stop when the source omitted it, as it does on a
            # trailing initial: `ಶಶಿಕುಮಾರ್.ಆರ್` is Shashikumar R.
            parts[i] = letter if "." in after else letter + "."
            was_initial = True
        else:
            was_initial = False

    read = "".join(parts)
    # And a space before a run of initials that follows a name, for the same
    # reason: the source writes ಶಶಿಕುಮಾರ್.ಆರ್ with no space and English does not.
    read = _NAME_THEN_INITIALS.sub(lambda m: m.group(1) + " ", read)
    return _RUN_ON_INITIALS.sub(lambda m: f"{m.group(1)} ", read)


def has_indic(text: str) -> bool:
    """True when any character belongs to an Indic block worth transliterating."""
    return any("ऀ" <= ch <= "෿" for ch in text)


def transliterate_mixed(text: str, *, script) -> str:
    """Transliterate only the Indic runs, leaving everything else identical.

    A deed writes `KRISHNAPPA ರಾಜು` - part English, part Kannada. Passing the
    whole string through the transliterator returns `Krishnappa Raju`, which is
    correct for the Kannada and wrong for the English: the name was already in
    the report's language and its capitalisation is how the document spells it.

    So each Indic run is converted and each non-Indic run is returned untouched.
    An already-English value has no Indic run and comes back byte-identical.

    Measured: the transliteration library already leaves Latin alone, so this
    does not change today's output for any input in the corpus. It is here to
    state the guarantee rather than rely on it holding by accident - the
    title-casing step is one edit away from restyling `KRISHNAPPA`.
    """
    parts = _NON_INDIC_RUN.split(text)
    out = []
    for part in parts:
        if not part:
            continue
        if _NON_INDIC_RUN.fullmatch(part):
            out.append(part)                 # untouched, exactly as written
        else:
            # The script of *this run*, not the value's dominant one. In
            # `KRISHNAPPA ರಾಜು` the dominant script is Latin - ten characters
            # against four - so passing that down left the Kannada untouched.
            from .detect import detect

            out.append(transliterate(part, script=detect(part).script))
    return "".join(out)


def transliterate_supported(script: Script) -> bool:
    """Whether this script has a rule set. Urdu does not."""
    return script in _SCHEME_FOR_SCRIPT
