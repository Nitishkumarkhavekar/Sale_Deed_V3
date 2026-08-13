"""Clean up what a sentence translator does to a fragment.

NLLB is trained on sentences. Given a deed field it produces a sentence, and the
difference shows up as artefacts that are wrong for a spreadsheet cell rather
than wrong as English. Measured on this project's own model:

    ೧೨೩, ೪ನೇ ಅಡ್ಡರಸ್ತೆ, ಜಯನಗರ  ->  "It is located at 123, 4th Avenue, Jayanagar."
    ಪುರುಷ                        ->  "The male"
    ಮಹಿಳೆ                        ->  "The woman"
    பிரதான சாலை                  ->  "main road"        (inconsistent case)

None of these is a mistranslation. All of them are wrong in a column headed
`Address (PC-L)` or `Gender (PC)`, where the reader expects a value, not prose.

Everything here is conservative: it removes framing the model added, and never
removes anything the source might have contained. A value it does not recognise
passes through with only whitespace and capitalisation normalised.
"""

from __future__ import annotations

import re

#: Sentence frames the model prepends to a bare fragment. Anchored at the start
#: and followed by real content, so "It is located at 123 Main St" loses the
#: frame while a genuine value beginning "It" is untouched.
_FRAMES = (
    r"it is located (?:at|in)\s+",
    r"it is (?:a|an|the)\s+",
    r"this is (?:a|an|the)\s+",
    r"the address is\s+",
    r"the name is\s+",
    r"located (?:at|in)\s+",
    r"i am\s+",
    r"he is\s+",
    r"she is\s+",
)
_FRAME_RE = re.compile(r"^(?:" + "|".join(_FRAMES) + r")", re.IGNORECASE)

#: A leading article the model adds to a one-word value. Only stripped when what
#: remains is short - "The Bank of Baroda" must keep its article.
_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

#: Closed vocabularies where a translator is the wrong tool. A registrar writes
#: "Male", not "The male", and the set of possible values is three long.
_CANONICAL: dict[str, str] = {
    "male": "Male", "the male": "Male", "a male": "Male", "man": "Male",
    "the man": "Male", "gentleman": "Male", "boy": "Male",
    "female": "Female", "the female": "Female", "a female": "Female",
    "woman": "Female", "the woman": "Female", "lady": "Female",
    "the lady": "Female", "girl": "Female",
    "other": "Other", "transgender": "Transgender",
}

#: A question mark on a fragment is always an artefact - the model turned a name
#: or a noun phrase into a question. The content before it is usually right.
_QUESTION_RE = re.compile(r"^(?:what|who|where|which) (?:is|are)\s+(.+?)\?$",
                          re.IGNORECASE)


def _sentence_case(text: str) -> str:
    """Capitalise the first letter, leave the rest alone.

    Not `.title()`: that would turn "4th Cross Road" into "4Th Cross Road" and
    mangle an all-caps abbreviation the source deliberately used.
    """
    return text[0].upper() + text[1:] if text else text


def tidy(text: str, *, kind: str = "translate") -> str:
    """Normalise one translated value for a spreadsheet cell.

    Returns the input unchanged when nothing applies. Never returns blank: a
    missing value in a legal record is worse than an awkward one.
    """
    if not text or not text.strip():
        return text

    cleaned = " ".join(text.split()).strip()

    # "What is Venkatesh?" -> "Venkatesh". The model does this to bare nouns.
    question = _QUESTION_RE.match(cleaned)
    if question:
        cleaned = question.group(1).strip()

    canonical = _CANONICAL.get(cleaned.lower())
    if canonical:
        return canonical

    cleaned = _FRAME_RE.sub("", cleaned, count=1).strip()

    # A trailing full stop belongs to a sentence, not to a field. Question and
    # exclamation marks likewise - neither can be part of an address or a name.
    cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()

    # Re-check the closed vocabulary: stripping "The " may have revealed it.
    canonical = _CANONICAL.get(cleaned.lower())
    if canonical:
        return canonical

    # A leading article on a short value is the model padding a fragment.
    # Length-gated so "The Bank of Baroda" keeps its article.
    if len(cleaned.split()) <= 2:
        stripped = _ARTICLE_RE.sub("", cleaned).strip()
        if stripped:
            cleaned = stripped

    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return _sentence_case(cleaned) if cleaned else text
