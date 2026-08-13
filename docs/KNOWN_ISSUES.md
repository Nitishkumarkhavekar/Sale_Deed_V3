# Known Issues, Limitations and Future Improvements

Append-only. Resolved items are moved to the Resolved section with a date, not
deleted.

> **This file was recreated on 2026-08-05.** The previous copy — carrying
> B-001 to B-003, L-001 to L-008 and R-001 to R-044 — disappeared from `docs/`
> along with `DECISIONS.md`, `DEVELOPMENT_LOG.md`, `PROJECT_STATUS.md` and
> `CLEANUP_REPORT.md`, between 14:48 on 2026-08-04 and 12:00 on 2026-08-05.
>
> I could not establish what removed them, and I am not going to guess. What I
> did check, because it was the serious possibility:
>
> * `core/backup.py` — every `unlink` targets the archive the same function has
>   just written, as failure cleanup. It cannot reach `docs/`.
> * `logging_setup.purge_old_logs` — globs `*.log*` inside the log directory
>   only, and the scheduler is passed `paths.LOG_DIR`.
> * The test suite — no test writes or deletes outside `tmp_path`, and none
>   references `docs/` in a mutating call.
>
> **The application is not deleting files.** That is verified, not assumed.
>
> Most of the lost history survives in `CHANGELOG.md`, which carries the same
> defects release by release. `docs/TEST_REPORT.md` and `docs/CODE_MAP.md` are
> intact. If you have a backup of the five files, restoring it is better than
> anything I can reconstruct — I would be rewriting from memory of my own work,
> which is exactly the kind of record that should not be reconstructed from
> memory.

---

## Resolved

### R-051 - Long deeds extracted nothing, and two stage states lied
**Severity:** silent data loss - **Opened:** 2026-08-12 - **Closed:** 2026-08-12

Chasing the last two documents that exported a blank City/Town found three
separate faults. Neither document had a city because neither had a property
address, and neither had a property address because extraction had produced
nothing at all.

**1. No bound on the request size.** The whole OCR text was sent however long it
was. A 59,012-character deed came back
`HTTP 400 ... request (16618 tokens) exceeds the available context size (16384)`
on every attempt, and the document extracted nothing. `fit_to_context` now
brings the text within `MAX_INPUT_CHARS` by dropping from the *middle*: a deed
opens with the parties and closes with the schedule and boundaries, which is
where all sixteen extracted fields live, while the middle is "WHEREAS" recitals
of prior title. The gap is marked rather than silent, and the trim is logged and
recorded on the extraction row.

**2. A stage that produced no answer was recorded DONE.** The rule was "not a
stage failure: the model answered, the answer is not trustworthy" - correct, but
applied even when there was no answer. The deed above ran three attempts,
returned nothing each time, and finished with `extract_state=done`, zero
persons and no property row. Recorded as a success, it appeared in no
failed-document report. DONE now requires that the model actually answered;
otherwise the stage is FAILED, which is both true and actionable.

**3. A failed claim stranded the document permanently.** When
`claim_next` returned a different document - it returns the lowest-id
claimable one, so asking for the second gets the first - the caller gave up and
`_process_one` parked the document in NEEDS_REVIEW with `extract_state`
still PENDING. `claim_next` admits only PROCESSING documents, so that pair can
never be claimed again: OCR done, extraction never attempted, nothing left to
pick it up. One real document (`BMH-1-00045`) was found in exactly that state.
A failed claim now leaves the states alone; retries genuinely exhausted are
marked FAILED rather than requeued forever, which would spin the runner on a
document it cannot take.

Nine mutants, all caught, including the bound not being applied, the trim losing
the schedule, DONE returning for an empty answer, the stranding returning, and
exhausted retries being ignored.

### R-050 - Both City/Town columns name the property's city
**Severity:** wrong and missing values in a submitted report -
**Opened:** 2026-08-12 - **Closed:** 2026-08-12

Two faults, opposite in kind.

**`City / Town` searched the whole deed.** It was called as
`city_town(address, doc.source_text)`, so when the property's own address
yielded nothing it fell back to the entire OCR text and returned whatever place
it found there - a registration office, or the post office in "post Pavagada".
A deed names many places and only one of them is the property's. It now reads
the property address and nothing else.

**`City/Town (PC-L)` was never populated at all**, and was recorded in
`STRUCTURALLY_ABSENT` as a column deeds do not state. It now carries the same
property city as `City / Town`, as requested: one property, one city, both
columns. Deliberately *not* read from the party's own address - a seller's
Aadhaar address is routinely in another town, and that town is not what the
column is being asked for.

**The extractor recognised one address shape out of three.** It required a name
immediately followed by a kind word (`Bengaluru South Taluk`), so it returned
nothing for the two other ordinary ways a deed ends an address:

| address | before | after |
|---|---|---|
| `Angol Village, Taluka & Dist: Belgaum.` | blank | Belgaum |
| `4th 'C' Block, Koramangala, Bangalore - 560 034` | blank | Bangalore |
| `Sector - 7, Hosur Sarjapur Road Layout, Bangalore` | blank | Bangalore |

Three shapes are now read - kind after the name, kind before it
(`Dist: Belgaum`), and a bare city closing the address - ranked so a town
named outright beats the taluk administering it, which beats a district.

Coverage over every distinct property address held: **57% -> 100%, three gained,
none lost, none changed.** Across all four batches re-exported: 90 rows, **0
mismatched pairs** between the two columns, **0 rows blank where a property
address exists**.

**What still returns blank, on purpose.** A schedule naming only a village and a
hobli has no town in it, and inventing the nearest one would be a guess about
jurisdiction. The stop-list is what makes the weakest rule safe: without it the
"bare city closing the address" fallback would happily return "Kasaba Hobli",
"2nd Block" or "Karnataka". Every candidate is rejected if *any* of its words is
structural - which is also what fixed `Angol Village` being offered as the
town in front of "Taluka".

Seven mutants, all caught - including both columns reverting to blank, the PC-L
column reading the party's address, and the whole-deed fallback returning.

### R-048 - Performance: measured, ranked, and fixed where it was safe
**Severity:** throughput and responsiveness - **Opened:** 2026-08-12 -
**Closed:** 2026-08-12

Measured before changing anything. Every number below is from this machine with
the AI server running, and query counts are exact rather than sampled.

**UI and database.** `BatchRepository.progress` claimed in its docstring to use
"one grouped query" and issued seven; the dashboard then called it once per
completed batch in a loop, and `list_paginated` lazily loaded `batch.user` per
row. `list_for_batch` lazily loaded three child collections per document - a
ten-row page cost thirty round trips, and a thousand-document export cost three
thousand. All collapsed with conditional aggregation and `selectinload`.

| | before | after |
|---|---|---|
| dashboard | 25.4 ms, 17 queries | 9.5 ms, 7 queries |
| PDF processing | 10.3 ms, 8 queries | 2.3 ms, 1 query |
| data view | 33.6 ms, 22 queries | 9.6 ms, 8 queries |
| status poll | 9.2 ms, 8 queries | 1.5 ms, 2 queries |
| poll load | 192 queries/min | 48 queries/min |

`progress_many` versus the old loop, benchmarked head-to-head in one process:
18.83 ms -> 6.03 ms.

**Startup.** `start_up()` costs ~412 ms - status probes, runner settings, the
retention scheduler and a crash-recovery scan - and `main()` called it *before*
`window.show()`, despite its own docstring saying it is "called after the window
is painted, so neither step delays first paint". Now deferred to the first idle
turn of the event loop.

**Pipeline.** OCR is 87% of wall time (105-297 s per document, ~11 s/page).
Fixed overhead per document, measured:

* interpreter start + torch import: **8.0 s**
* Surya model load (`seconds_load`): **9.6 s**
* language-model swap out and back in: **~10.5 s** (5.25 s each way)

The swap half is fixed: the runner now drains OCR for a batch before extracting
any of it, so swaps go from two per document to two per batch. It uses
`_claim_downstream`, which already existed for exactly this state, so
crash-safety is unchanged - `ocr_state` is committed DONE with the text before
the runner moves on.

**Two things deliberately NOT changed, with the measurements behind them:**

*Surya batch size.* The ceiling picks by total VRAM, so a 4 GiB card gets the
smallest rung (detector 4 / recognition 16) even though ~3.2 GiB is free once
the language model releases. Measured on one 14-page deed: 4/16 = 134.7 s,
8/32 = 117.7 s (13% faster), 16/64 = **CUDA out of memory**, which would fail
the document. But 8/32 does not produce the same text - 97.83% token similarity
against 4/16. The differences fall in noisy regions where both outputs are
garbage, but that cannot be proven to hold generally, and this is a legal
document. **A 13% speedup that silently changes the recognised text of a deed is
not mine to take.** The measurement is here so it can be decided deliberately.

*Per-document Surya startup (17.6 s).* Eliminating it means processing several
PDFs per invocation or holding a warm worker. A warm worker conflicts directly
with the GPU-sharing design - Surya resident would keep the language model out
of a 4 GiB card. Batching several PDFs per invocation is the right fix and is
the largest remaining win (~13% of OCR wall time, ~29 minutes on a 100-document
batch), but it changes the runner's one-document-at-a-time contract and needs
its own verification pass rather than being appended to this one.

*Translation on CPU.* One field took 30.16 s at `device=cpu`. NLLB is on the CPU
because the card has no room beside the language model. Same trade-off; noted,
not changed.


### R-047 — Failed OCR: a list, a rerun, and the reference design
**Severity:** feature · **Opened:** 2026-08-12 · **Closed:** 2026-08-12

A document whose OCR failed was visible only as a row in the batch's failed
export. There was no way to see *which* files failed OCR specifically, and no
way to retry one without reprocessing the whole batch from the start.

**Listed by stage, not by outcome.** `failed_ocr` filters on
`ocr_state == FAILED`, not `overall_state == FAILED`. A document that failed
extraction or validation is a failed document, but rerunning OCR on it would
redo minutes of GPU work that was never the problem.

**One OCR implementation, not two.** Rerunning does not OCR anything itself. It
returns the document to PENDING and the normal runner claims it through the
same `claim_next("ocr")` every other document uses. There is no second code
path that could drift from the first.

**The subtle part is the retry cap.** `claim_next` admits a document only while
`ocr_attempts <= max_attempts`, and a failed document has spent them. A rerun
that reset the stage but not the counter would report success, requeue nothing,
and leave the operator pressing a button that does nothing. `requeue_ocr`
clears it, and the mutation that removes that line is caught by
`test_a_rerun_makes_the_document_claimable_again`.

Later stages are reset with it: OCR text is the input to extraction,
translation and validation, so a document that reruns OCR while keeping a DONE
extraction would export results derived from text that no longer exists.

Verified end to end against a genuinely unreadable file - not a database
fixture: 20/20 checks, covering failure, rerun-fails-again, rerun-succeeds, and
the pipeline continuing past OCR to `extract=done` with 14 pages of text stored.

**UI.** Updated toward the supplied reference: violet primary colour, gradient
hero banner, Control Panel / Data View tabs, statistic tiles, a wide progress
bar with an explicit `n / n (x%)` readout, and a light/dark toggle. Status
colours were *not* recoloured to match the brand - green still means passed and
saffron still means needs review, and repainting a status would make the table
lie.

Two things worth knowing for the next change here:

* The tabs are a `<div>`, not a second `<nav>`. `nav a.active` is how the
  current page is located; a second nav in the top bar captures that selector
  and reports the wrong page.
* `.content` is now the scroll container and `#content` the element navigation
  replaces. They were the same element, so anything placed beside the page
  content was wiped on the first navigation.

### R-046 — Coded output columns
**Severity:** wrong values in a submitted report · **Opened:** 2026-08-12 ·
**Closed:** 2026-08-12

`Address Type` must be `1`-`5` and `Identification Type` a letter from the
SFT/Form 61A set (`A` passport, `B` elector ID, `C` PAN, `D` government/PSU ID,
`E` driving licence, `G` UIDAI letter, `H` NREGA card, `Z` other). Both were
empty. `Identification Number` is now held blank in every row by instruction -
the identifiers reach the report through the PAN and Aadhaar columns and this
column is not a second copy.

PAN outranks Aadhaar for the code: a deed carrying both is identified by the
PAN. Neither present leaves the cell empty rather than claiming `Z`, which
would assert that some other document was seen when none was.

`Address Type` falls back to `5` (Unspecified) rather than assuming `2`: a deed
records where somebody lives without saying what the premises are used for, and
the format provides `5` for exactly that. A document with no parties at all
leaves the whole person half blank, Address Type included - `5` describes a
party's address, and there is no party.

Verified on 4 real deeds through the live pipeline: only `2` and `5` in Address
Type, only `C` and `G` in Identification Type, `Identification Number` empty in
10 of 10 rows, no duplicates, column order intact.

Two defects were found while validating and fixed:

* `python -m ai_server.server` - the documented start command - **died at
  startup on a correct install.** `main()`'s argparse defaults were
  pre-restructure relative paths that overrode `build_default`'s correct
  absolute ones. Same fallout as R-040.
* `City / Town` was still listed in `STRUCTURALLY_ABSENT` after R-042 taught the
  exporter to populate it. That list is subtracted from the coverage
  denominator in `extraction_report`, so the stale entry was inflating reported
  coverage on every run. `TestTheAbsentColumnListIsHonest` now guards it.

Still open, and outside these columns: one deed of the four exceeded the
16,384-token context and extracted nothing; `Postal Code` is empty because
property addresses are village-style with no PIN; one Property Address exported
in Kannada, so translation coverage there is incomplete.

### R-045 — Names, duplicate rows, and the "@" alias
**Severity:** wrong names on a legal report · **Opened:** 2026-08-04 ·
**Closed:** 2026-08-05

Three reported issues: a large defect in name rendering, a small real one in
row building, and a question of what the Name column should contain.

#### 1. Names: Kannada writes English initials phonetically

The model was **not** the problem. Measured against each document's own OCR,
**206 of 210 names (98.1%) appear verbatim** in the source and **none was
invented**.

The corruption was in transliteration. Kannada spells an English initial by its
*sound*: `ಜಿ` is how you write **G**, `ಕೆ` is **K**, `ಎಂ` is **M**. Sounding
them out gives:

```
ಜಿ.ಕೆ. ರಾಜು        ->  Ji.ke. Raju         should be  G.K. Raju
ಎಂ.ಟಿ. ರಂಗೇಗೌಡ    ->  En.ti. Rangegauda   should be  M.T. Rangegowda
ಡಿ.ಎಂ. ಮುನಿಯಪ್ಪ   ->  Di.en. Muniyappa    should be  D.M. Muniyappa
```

Not a spelling preference — different letters from the ones printed. Indian
names carry initials constantly, which matches the reported accuracy.

`read_initials()` reads them back, in all three positions real deeds use:

| Written | Read as | Position |
|---|---|---|
| `ಜಿ.ಕೆ. ರಾಜು` | G.K. Raju | stop follows |
| `ಶಶಿಕುಮಾರ್.ಆರ್` | Shashikumar R. | stop precedes only |
| `ವಿಜಯ್ಕುಮಾರ್ ಕೆ. ಎಂ` | Vijaykumar K. M. | runs on from the previous |

**Position is the safety.** `ಬಿ` is the letter B *and* the opening of `ಬಿಂದು`,
so an adjoining full stop is required and ordinary words are untouched.

Three further defects fixed alongside:

| | Was | Now |
|---|---|---|
| `ಜಿ.ಎಲ್.ರವಿ` (no spaces at all) | G.L.ravi | **G.L. Ravi** |
| `ಚನ್ನೇಗೌಡ` (IAST `c`, `au`) | Cannegauda | **Channegowda** |
| `ಹೆಚ್` — a second spelling of H | hec | **H** |

**Measured: 0/50 → 45/50 (90%)** of Kannada names with initials now render as
letters. The five remaining are OCR artefacts — a whole sentence captured as a
name, honorifics such as `ಡಾ` (Dr.) — not transliteration faults.

*A regression caught during the work*: the Kannada `c`→`ch` convention rewrote
the letter C itself, turning `ಎಂ. ಸಿ.` into `M. Ch.`. The lookahead now excludes
a full stop.

#### 2. Duplicate rows

Real but rare: **1 document in 48** listed the same party twice — same name,
same Aadhaar — and each copy became a row carrying the whole document with it.
`_party_key()` treats an Aadhaar as decisive, then a PAN, then name plus
father's name. **210 rows → 205; byte-identical duplicates 2 → 0**, with no
distinct party merged.

#### 3. The "@" symbol — the alias is removed

`@` marks an alias on an Indian deed:

```
AAKASH SACHIDANAND MISHRA @ AAKASH MISHRA
AYUSHI YADAV @ AYUSHI UPENDRA YADAV
```

I first argued for keeping it, on the grounds that the alias is a fact the deed
states. **That was overruled: the Name column carries one name.** `primary_name()`
removes the marker and everything after it, on both the name and the father's
name, so `AAKASH SACHIDANAND MISHRA @ AAKASH MISHRA` exports as
`AAKASH SACHIDANAND MISHRA`.

The alias is not silently discarded — `build_rows` logs the full form against
the document, so it stays recoverable from the record of the run.

The pattern is `\s*@.*$` and nothing else. An earlier draft also matched
`urf`/`uruph`/`alias`; that was unrequested, riskier (those can be parts of
a real name), and its word-boundary escape had been corrupted into a literal
backspace character by the editing script, so the branch never ran at all.
Removed rather than repaired.

Separately, a "name" containing **no letter at all** — `@` alone, or punctuation
— is still rejected. `looks_like_a_name()` tests for the presence of a letter
rather than filtering characters, and that choice matters: a Kannada name is
mostly combining marks (`ಿ`, `ಾ`, `್`), and a cleaner built from an
allowed-character set reads those as symbols and mangles the name.

#### A crash these tests found

`extra={"name": ...}` on the duplicate-party log line. `name` is a reserved
`LogRecord` attribute, and `logging.makeRecord` raises
`KeyError: "Attempt to overwrite 'name' in LogRecord"` **from inside logging**.
The export would have crashed on precisely the condition the line reports — and
only once logging was configured at that level, which is why it passed in
isolation and failed in the full suite.

Renamed, and `TestLogExtrasAreSafe` now walks the AST of every `extra={...}` in
`src/` for all 23 reserved names.

**Tests.** 15, mutation-verified:

| Mutation | Caught |
|---|---|
| initials no longer read (the original defect) | yes |
| only trailing-stop initials found | yes |
| the `ch` convention eats the letter C again | yes |
| duplicate parties come back | yes |
| a nameless party is exported again | yes |
