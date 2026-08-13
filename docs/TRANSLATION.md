# Translation

**Last updated:** 2026-08-03

How a deed written in Kannada — or Hindi, Telugu, Tamil, Malayalam, Gujarati,
Bengali, Punjabi, Odia, Marathi or Urdu — becomes an English CSV.

---

## The model

**`facebook/nllb-200-distilled-600M`** — No Language Left Behind, distilled to
600M parameters. Runs entirely locally. ~2.5 GB on disk.

### Why this one

**Ungated.** IndicTrans2 scores better on Indic→English and was the first
choice, but both its variants sit behind a HuggingFace licence gate. A fresh
install cannot fetch them without the operator creating a third-party account
and generating a token. A translation system that requires an account before it
works is not an offline system in any useful sense, and the installer could not
be made to "download automatically if not already available" — which the
requirement asks for.

**One model, every language in scope.** NLLB-200 covers 200 languages including
all eleven Indian languages required. The alternative was a per-language model
set, which multiplies download size and gives the detector something new to get
wrong on every document.

**It fits.** The distilled 600M is ~2.5 GB against 5.5 GB for the 1.3B and 17 GB
for the 3.3B. On a 4 GB card already holding 3.2 GB of language model, only the
distilled variant has any chance of the GPU — and it runs acceptably on CPU when
it does not.

**Local only.** Deed text is a legal record and must not leave the machine,
which rules out every hosted API regardless of quality.

### Proper nouns never touch the model

This is the most important correction in the design, and it was made after
measuring what NLLB actually did to names:

| Source | NLLB output | |
|---|---|---|
| `ಲಕ್ಷ್ಮಿ ದೇವಿ` | "Goddess Lakshmi" | a person became a deity |
| `ವೆಂಕಟೇಶ್` | "What is Venkatesh?" | a name became a question |
| `ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ` | "Rural areas of Bangalore" | a district was paraphrased |

None of these is a mistranslation — NLLB is a *sentence* translator doing
exactly what it was built for, and on Indian names the meaning is often a word.
On a record identifying parties to a property transfer it is a **corrupted
document**. Tightening the beam reduces the rate; it cannot remove the failure.

Names, villages, districts and taluks are therefore rendered by
`src/core/translation/transliterate.py` — rule-based, via IAST, using
`indic-transliteration`. It is **deterministic and cannot invent**. The output
is occasionally clumsy; it is never a different person.

```
ಲಕ್ಷ್ಮಿ ದೇವಿ -> Lakshmi Devi     ವೆಂಕಟೇಶ್ -> Venkatesh
रमेश कुमार   -> Ramesh Kumar     রমেশ কুমার -> Ramesh Kumar
```

Three script-specific rules were needed and are covered by tests: Devanagari,
Gujarati, Bengali, Odia and Gurmukhi carry an inherent final vowel that must be
dropped (`रमेश` → "ramesha" → **Ramesh**); Malayalam chillu letters must be
expanded first or they pass through as script; and Tamil has no voiced stops in
its script, so `gh`/`bh` and word-initial voiced consonants are table artefacts.

Urdu has no rule set here and falls back to the model — reported honestly rather
than rendered by guesswork.

### Fragment artefacts

A sentence translator given a field produces a sentence.
`src/core/translation/postprocess.py` corrects what that adds:

| Raw | Corrected |
|---|---|
| "It is located at 123, 4th Avenue, Jayanagar." | "123, 4th Avenue, Jayanagar" |
| "The male" / "The woman" | "Male" / "Female" |
| "main road" | "Main road" |

Gender uses a closed vocabulary — three possible values, so a translator is the
wrong tool. Nothing is ever reduced to blank: a missing value in a legal record
is worse than an awkward one.

### What this still costs

IndicTrans2 is the better model for Indic→English prose. NLLB is a general
multilingual model and its Indic quality is good, not best-in-class. If
translation accuracy proves insufficient, IndicTrans2 is still the model to
reach for — but it is no longer a drop-in. Its runner was deleted in R-031, and
it needs its own tokenizer and `IndicProcessor` handling that
`src/tools/translate_runner.py` does not implement. Budget a new runner, not a
config change.


## Hindi or Marathi

Devanagari is the one script in scope that carries two languages, so it is the
one language decision the pipeline has to make rather than read.

**Two kinds of evidence, both deterministic.** Letters Hindi does not use - `ळ`
is an ordinary Marathi consonant, `ॲ` and `ऱ` are Marathi orthography - and
vocabulary that differs on exactly the words a deed repeats:

| Marathi | Hindi | |
|---|---|---|
| `जिल्हा` | `जिला` | district |
| `तालुका` | `तहसील` | taluka |
| `मध्ये` | `में` | in |
| `आणि` | `और` | and |
| `आहे` | `है` | is |
| `रस्ता` | `सड़क` | road |

A registered deed repeats these many times, so a page is usually decided several
times over. Where a field carries no evidence - a personal name shared by both
languages - **the detector does not guess.** It falls back to the configured
default and records `"no distinguishing evidence"`, which reaches the log next to
the field. Guessing on no evidence is what a statistical detector does, and
avoiding it is why this module detects by script.

### Setting it

Settings → Language → **Devanagari documents**

| Choice | Behaviour |
|---|---|
| Detect automatically | per field, from the evidence above (default) |
| Always Hindi | forces `hin_Deva`, ignoring the evidence |
| Always Marathi | forces `mar_Deva` |

An operator processing one jurisdiction should choose outright; it is faster to
reason about and cannot be surprised by an unusual document.

Stored as `translation_devanagari_as`. There is no OCR language setting because
Surya's recognition model is multilingual and takes no language argument.

### Model

NLLB-200-distilled-600M covers `mar_Deva` - the same checkpoint already in use.
**Nothing extra is downloaded for Marathi.**

### Names

Proper nouns are transliterated by rule, never translated, in Marathi as in
every other script. The distinction matters more here than it looks:

```
मौजे कोंढवा बुद्रुक   translated:    "Beautiful calf"
                      transliterated: "Mauje Kondhava Budruk"
हवेली                 translated:    "Mansion"
                      transliterated: "Haveli"
सौ. सुनीता जोशी       translated:    "Hundred. Sunita Joshi"
                      transliterated: "Sau. Sunita Joshi"
```

Final vowels are handled on the IAST form, before diacritics are stripped, so
`सुनीता` stays "Sunita" rather than becoming "Sunit" - a different name. A schwa
after a consonant cluster is kept (`महाराष्ट्र` -> Maharashtra), and an aspirate
digraph counts as one consonant (`मुख` -> Mukh).

### Storage

```
AI server/translator/nllb-200-distilled-600M/
    config.json  tokenizer.json  sentencepiece.bpe.model  pytorch_model.bin
```

One directory per model, so an upgrade sits beside the current one rather than
overwriting it mid-batch.

---

## Language detection

`src/core/translation/detect.py` — **by script, not by a statistical classifier.**

The inputs are deed *fields*: a name, a village, a two-line address. Statistical
detectors (langdetect, fastText, langid) are trained on running prose and are
unreliable below roughly twenty words; on a two-word Kannada name they guess,
and several of them guess differently between runs because they seed by random
state. A wrong guess is not a near miss — it selects the wrong source language
and produces confident nonsense.

Script identity is a property of the characters themselves. Every language in
scope has its own Unicode block:

| Script | Range | Language |
|---|---|---|
| Kannada | U+0C80–U+0CFF | `kan_Knda` |
| Devanagari | U+0900–U+097F | `hin_Deva` / `mar_Deva` |
| Telugu | U+0C00–U+0C7F | `tel_Telu` |
| Tamil | U+0B80–U+0BFF | `tam_Taml` |
| Malayalam | U+0D00–U+0D7F | `mal_Mlym` |
| Gujarati | U+0A80–U+0AFF | `guj_Gujr` |
| Bengali | U+0980–U+09FF | `ben_Beng` |
| Gurmukhi | U+0A00–U+0A7F | `pan_Guru` |
| Odia | U+0B00–U+0B7F | `ory_Orya` |
| Arabic | U+0600–U+06FF | `urd_Arab` |
| Latin | — | `eng_Latn` |

Exact, instant, deterministic, no model, no download, no dependency.

### The one ambiguity

**Hindi and Marathi share Devanagari** and no character inspection separates
them — the difference is vocabulary and grammar. Detection reports
`Script.DEVANAGARI` honestly and the caller chooses. The default is Hindi,
because Hindi is far more common on Indian legal documents outside Maharashtra.
Set `translation_devanagari_as = mar_Deva` for Maharashtra records.

### Codes are FLORES-200, not ISO 639-1

NLLB expects `kan_Knda`. Passing `kn` produces silent garbage rather than an
error, which is why the mapping is explicit and tested.

### Neutral text

Digits, punctuation and symbols carry no language. An amount, a PIN code, a PAN
or a survey number detects as `NEUTRAL` and is **never sent to the translator** —
that would waste a model call per document and risk the model inventing words
where an exact digit string is required.

---

## The pipeline

```
PDF → OCR → cleanup → extraction → validation → TRANSLATION → database → CSV
```

### Why translation runs after validation, not before

The requirement asked for translation before validation. That would break
validation, and the point is worth stating plainly.

Validation cross-checks every extracted value against the **OCR source**, which
is in the original language. Translate the name to "Ramesh Kumar" first and it
can never be found in Kannada OCR — every name would flag as unverified, and the
layer that catches quantisation errors on financial fields would stop working.

Translation runs after validation and **before** database storage and CSV
export, which is what the export actually requires.

### Write-back

Results are written to `<field>_translated`, never over the original. The CSV
writer prefers the translated value and falls back to the source, so a partial
translation degrades to mixed output rather than losing the Kannada a reviewer
may still need to check against the deed.

### Fields covered

| Group | Fields | Operation |
|---|---|---|
| Person | `name`, `father_name` | transliterate |
| Person | `address`, `gender`, `occupation` | translate |
| Property | `schedule_c_property_address`, `property_description` | translate |
| Property | `village`, `district`, `taluk` | transliterate |
| Document | `registration_office`, `document_type`, `sub_registrar_office` | translate |

**The distinction is load-bearing.** A name must come across by *sound* — ರಮೇಶ್
is "Ramesh"; translating a proper noun yields nonsense. An address must come
across by *meaning* — ಮುಖ್ಯ ರಸ್ತೆ is "Main Road", not "Mukhya Raste". Getting
these the wrong way round produces output that reads plausibly and is wrong in
half the columns, which nothing downstream would catch.

---

## Architecture

```
core/translation/
    detect.py     script identification, no model
    config.py     resolved settings + the model rationale
    service.py    TranslationService - the single entry point
tools/
    translate_runner.py    NLLB, in the OCR virtual environment
```

**One service.** Detection, caching, batching, the subprocess and the retry
policy all live in `TranslationService`. Before this, the logic sat inside
`TranslateStage`, which meant any new caller would have reimplemented it
slightly differently.

**The model runs in a subprocess** — in `models/SuryaOCR/venv_new`, shared with Surya
because both need torch and transformers. Loading it in-process would hold
~2.5 GB of VRAM for the lifetime of the window, and duplicating torch for a
600M model would waste ~3 GB of disk.

**Batching is by source language.** NLLB sets the source through the tokenizer,
so a batch must be homogeneous. Items are grouped before generation.

**The cache is content-addressed** on `(text, source, target)`. Deeds repeat:
the same village, district and registration office across a whole batch, so a
500-document run collapses to a few hundred distinct strings.

---

## Configuration

Stored in the `settings` table; every key is overridable by environment
variable for a single run.

| Setting | Default | Environment | Meaning |
|---|---|---|---|
| `translation_enabled` | `true` | `SALEDEED_TRANSLATION` | Master switch |
| `translation_target` | `eng_Latn` | `SALEDEED_TRANSLATION_TARGET` | Output language |
| `translation_source` | `auto` | — | `auto` detects per field |
| `translation_devanagari_as` | `hin_Deva` | — | Hindi or Marathi |
| `translation_model` | `nllb-200-distilled-600M` | `SALEDEED_TRANSLATION_MODEL` | Directory name |
| — | — | `SALEDEED_TRANSLATION_MODEL_DIR` | Full path override |
| `translation_device` | `auto` | `SALEDEED_TRANSLATION_DEVICE` | `auto`/`cuda`/`cpu` |
| `translation_batch_size` | `16` | — | Sentences per pass |
| `translation_timeout_s` | `600` | — | Per batch |
| `translation_max_retries` | `1` | — | On transient failure |

`device = auto` selects CUDA **only when enough VRAM is genuinely free**.
`torch.cuda.is_available()` answers "is there a GPU", not "is there room on it" —
and with llama-server resident on a 4 GB card there usually is not.

---

## Installation

```
py -3.13 src/tools/setup.py --install-translation      # download (~2.5 GB, once)
py -3.13 src/tools/setup.py --verify-translation       # check without downloading
py -3.13 src/tools/setup.py --all                      # part of a full install
```

Idempotent: a present, complete model is skipped. The check is for **weights**,
not for the directory — a partial download leaves the config and tokenizer
behind, and treating that as installed produces a system that reports itself
ready and fails on the first document.

Verification checks the required files exist, the weights exceed 512 MB (a
truncated download is the failure worth catching), and the config parses.

---

## Logging

| Record | Level | Carries |
|---|---|---|
| Batch translated | INFO | count, languages, engine, model, device, seconds, cache hits |
| Per field | DEBUG | field, operation, source language, **original**, **translation** |
| Unavailable | WARNING | reason, languages that will pass through untranslated |
| Retry | WARNING | attempt number, error, field count |
| Failed | ERROR | attempts, error, field count |
| Export incomplete | WARNING | every CSV column still holding non-English text |

DEBUG carries the original and the translation. It is off by default because a
500-document batch would produce thousands of lines; enable with
`SALEDEED_DEBUG=true`.

---

## Error handling

**Translation never loses a deed.** A legal record is worth more than an English
column, so every failure path returns the original text and records why:

- Model missing, disabled or unreachable → values pass through, WARNING logged,
  the export names the affected columns.
- Runner crashes or times out → one retry, then the originals stand.
- Model returns nothing for a field → that field keeps its original.
- Anything unexpected → caught, logged with a traceback, batch continues.

`TranslationItem.output` falls back to the source and is never blank. A blank
cell is worse than a Kannada one: a reader can see Kannada and act on it, but
cannot see an absence.

---

## Verifying

```
py -3.13 src/tools/kannada_audit.py        # which CSV columns still hold non-English
py -3.13 src/tools/translation_check.py    # end to end against the real model
py -3.13 -m pytest tests/test_translation_service.py -q
```

---

## Troubleshooting

**"no model weights in ..."** — the model is not installed, or the download was
interrupted. Run `tools/setup.py --install-translation`. A partial download
leaves `config.json` behind, so the directory existing is not evidence.

**Kannada still in the CSV** — check the export log for
`export contains untranslated Kannada in N column(s)`, which names them. Then
`src/tools/kannada_audit.py` to see whether the field is covered at all.

**Translation is slow** — it is running on CPU. Check the log for `device`. On a
4 GB card the language model holds most of the VRAM, so CPU is expected and
usually correct; the alternative is an out-of-memory failure mid-batch.

**Marathi rendered as Hindi** — expected. Set
`translation_devanagari_as = mar_Deva`.

**A name looks translated rather than transliterated** — a known weakness. NLLB
has no transliteration mode. Report the case; IndicTrans2 handles proper nouns
better and is a drop-in replacement.

**"translation is disabled in settings"** — `translation_enabled` is false, or
`SALEDEED_TRANSLATION` is set to something falsy in the environment.

---

## Known limitations

- **Hindi/Marathi** cannot be separated by script.
- **Urdu names** fall back to the model, which is where the name defects above
  originated. Verify Urdu party names against the source.
- **Place-name spelling varies between paths.** A transliterated village gives
  "Bengaluru"; the same name inside a translated address gives "Bangalore",
  because the model uses its own conventional English form. Both are correct
  English; they are not identical strings.
- **Translation quality is unmeasured.** There is no reference set of
  translated deeds to score against, so accuracy is asserted by the model's
  published benchmarks, not by measurement on this corpus.
- **Only Kannada has been tested on real documents.** The other ten languages
  are covered by unit tests using real text in each script, but no genuine
  Hindi, Telugu or Tamil sale deed has been processed end to end.
