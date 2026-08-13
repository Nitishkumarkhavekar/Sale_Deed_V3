# Changelog

Append-only. Newest release at the top. Dates are IST.

---

## [0.3.5] — 2026-08-04 (later)

The backend printed nothing to the terminal. Three causes, each sufficient alone.

### Fixed

**The console handler sat at WARNING** (R-039), with a comment explaining that
the desktop UI is the status display and the terminal should stay quiet. Right
for the packaged application, wrong for a service run from a terminal - which
prints nothing and reads as a dead process. INFO by default now;
`SALEDEED_LOG_CONSOLE=warning` or `console_level=` restores the old behaviour.

**No request logging existed.** `log_message` was overridden to `return` with
the comment *"the app owns logging"*, and the app never logged a request.

**The AI server has no terminal** - `CREATE_NO_WINDOW`, output redirected to
`runtime/logs/ai_server.out.log`. Correct for the packaged product, and it means
the process with the interesting logs is the one that cannot show them.
`launcher.py --verbose` follows that file into the launcher's terminal.

### Added

- Function name in every line: `[time] LEVEL logger.function() - message`
- Request/response logging from `_send`, the single exit point every response
  passes through - one line per request, with an id and a duration. 4xx/5xx at
  WARNING with the reason; `/health` at DEBUG because the shell polls it and at
  INFO it buries everything else within a minute.
- One line per pipeline stage from `_log_stage` - the only place that knows both
  the outcome and the document. Logging inside each stage too would double every
  step.
- Batch start/finish, CSV writes with row count and size, and `exc_info=True` on
  engine start-up failure.

### Measured

```
[2026-08-04 14:38:50] INFO  runner.start()      batch runner started: 1 worker(s)
[2026-08-04 14:38:52] INFO  pdf_prepare.prepare() prepared 1359-2025-26.pdf  seconds=1.28
[2026-08-04 14:41:11] INFO  runner._log_stage() ocr ok in 139.15s  chars=24512 pages=14
[2026-08-04 14:41:41] WARN  runner._log_stage() extract failed: no parseable JSON
[2026-08-04 14:40:22] INFO  export.write_csv()  CSV written: 2 row(s)  bytes=1627
```

### Unchanged, deliberately

The 238 `print()` calls in `tools/` and 29 in `launcher/` are those programs'
user interface - a setup script's progress table is not a log. `print()` was
removed only where it carried runtime information.

---

## [0.3.4] — 2026-08-04

The AI server could not start. Reported by the user; the full suite was green.

### Fixed

**`No module named 'ai_server'`** (R-033) — the supervisor spawns the inference
service with `python -m ai_server.server` and the project root as the working
directory. `-m` resolves against the *child's* `sys.path`, which starts there,
and since the restructure the packages live in `src/`. Every start died
instantly, the supervisor retried three times and gave up, and the only thing
the operator saw was:

> Some actions are unavailable. Process - AI server offline.

The message was correct. Nothing was listening on 8077.

`launcher.runner.child_env()` now builds `PYTHONPATH` for supervised children,
preserving any existing value, and `Service` gained an `env` field.

**A self-contradicting readiness message** (R-034) —
`AI server not ready: ready (pressure critical)`. `ready` is the AND of two
independent things, the model being loaded and the governor admitting work, and
the message conflated them. It now distinguishes *still loading* from *up but
not admitting work*, and says the second clears by itself.

**`tools/e2e_test.py` gave up on the first probe** (R-034) — loading the model
takes host RAM to ~3% free for about fifty seconds; the governor correctly
refuses work, then it clears. Measured:

```
   t   RAM free       %   pressure   ready
   0      0.27G    3.6%   critical   False
  25      0.19G    2.6%   critical   False
  50      0.87G   11.7%       high    True    <- recovers on its own
 115      0.92G   12.4%       high    True
```

The harness now waits up to three minutes and says what it is waiting for.

**OCR and the language model fought over the card** (R-035) — invisible until
the server started working. With `llama-server` holding ~3.2 GiB of a 4 GiB
card, every document failed:

```
Surya exited 6: device: cuda (3.2 GiB free of 4.0 GiB)
  OOM while trying to allocate 2581594112 bytes (free: 0, total: 4294443008)
```

Surya measured 3.2 GiB free, chose CUDA, and met `free: 0` part-way through.
`torch.cuda.mem_get_info()` cannot be trusted on Windows: a fresh CUDA context
reports 3.2 GiB free while nvidia-smi shows 3062 MiB in use, because WDDM lets
the driver over-promise and fail the allocation later.

`ResourceGovernor.gpu_lease()` exists for this and is correct, but it is an
**in-process lock** and the language model is a different process. Nothing in
the application can make `llama-server` release VRAM, and there is no unload
endpoint.

`OcrStage._device_for_this_run()` and `TranslationService._device()` now ask the
AI server whether the model is *resident* - a question with a definite answer -
and take the CPU deliberately when it is. OCR costs ~2.9 min/page on CPU against
~1.3 on GPU. Finishing slowly beats failing quickly on a legal document.

Also found: `app/services.py` builds `BatchRunner` with **no governor at all**,
so even the in-process arbitration never ran. Recorded rather than changed - the
device fix addresses the actual contention, and the real answer is an
unload/reload endpoint on the AI server, which is a design change.

**The governor refused work because of memory the work would free** (R-037) —
with the server finally running, the UI reported *"Not enough free memory - 0.4
GB of 7.4 GB available"* and never recovered. `admit_new_work` was
`pressure < CRITICAL`, and `pressure` came from free RAM alone. On a 7.42 GiB
machine the language model is the largest single consumer, so loading it drove
RAM under the 8% threshold and the governor stopped admitting work. Nothing else
was going to release that memory:

```
model loaded     RAM 0.58 GiB ( 8%)  pressure=critical  ready=False
model released   RAM 3.23 GiB (44%)  pressure=normal
                 freed by releasing the model: +2.65 GiB
```

A safeguard that cannot be satisfied is a deadlock - and the memory was
reclaimable all along, because OCR releases the model before it starts (R-035).
The governor now distinguishes memory that *is* free from memory that can be
*made* free, and admits on the second.

`pressure` deliberately still reports the machine as it is: the UI reads it, and
worker counts are scaled from it - planning four workers against memory that is
not free yet would thrash the machine the moment they started. Only the
admission decision considers what could be freed. With nothing loaded there is
nothing to reclaim and exhausted memory still holds work back.

**A relative log path** — `logging_setup` defaulted to `"./logs"`, resolved
against the working directory, so logs landed wherever the process was started
from while the application looked under `runtime/`. Now absolute, from `paths`.

### Changed

**The end-to-end harness now runs the pipeline it claims to.** It hard-coded
`ocr_engine="textlayer"` and `translator_engine="passthrough"`, so it had never
once exercised Surya or the translation model — the two slowest and most
failure-prone stages. Production configuration is now the default, with
`--ocr` / `--translate` to override and `--limit N` to bound a run.

### Added

`TestReclaimableMemoryAdmission` (8) and `TestSupervisedChildrenCanActuallyStart` — five tests, mutation-verified. The
decisive one spawns a real subprocess exactly as the supervisor does and imports
the module; it fails at module resolution long before a model loads, so it costs
a second. Another scans the source for `-m <our package>` spawns and checks every
one, so the next is covered the day it is written.

### Why 684 tests missed it

Every check that touched the AI server was static — the model file exists, the
binary exists, **the port is free**. That last one was reported as `[ ok ]`, and
it is the same observation as "nothing is running": the preflight state and the
failure state are indistinguishable by that check. Not one test started a
process.

---

## [0.3.3] — 2026-08-03 (later)

Twenty top-level folders became five. No behaviour changed.

### Changed

```
src/        core  app  ai_server  launcher  tools  migrations
models/     AI server  SuryaOCR  saledeed main        36 GB, static
runtime/    logs uploads exports data temp cache backups config   disposable
tests/      the suite + corpus/
docs/       documentation and notes
```

Grouped by how the contents behave, not by what they are. The line that earns
its keep is `models/` against `runtime/`: one is 36 GB a cleanup script must
never touch, the other can be deleted at any time and rebuilds itself. Under the
old layout `AI server/` and `temp/` were siblings.

**`src/core/paths.py` is new** and owns every filesystem location. About twenty
files each computed the project root with `Path(__file__).resolve().parents[N]`
and appended a directory name; moving anything made `N` wrong in twenty places,
each failing late and somewhere unrelated - a missing model reads as "no GPU".

Also updated: `alembic.ini`, both `.bat` files, `launcher.py`, and 51 path
references across the documentation. `pytest.ini` is new - `tests/corpus/`
contains research scripts named `test_gemma_*.py` that would now be collected
on sight.

### Fixed

**`shutil.move` began deleting 8.95 GB of Surya.** The rename failed on a
read-only `.git` pack file inside the vendor folder, and `shutil.move` silently
falls back to copy-then-delete. It had finished copying and was partway through
removing the original when it hit the same file and stopped. The source survived
only because the deletion failed early. The remaining moves used `os.rename`
alone, with a manifest written before each one - this project has no version
control, and a rename that cannot be reversed is a demolition.

**A test was passing over nothing.** `TestCompatibility._sources()` globs the
source directories to scan them for hard-coded drive letters. After the move it
globbed paths that no longer existed, found zero files, and reported zero
offenders - three tests went green over an empty sequence. It now asserts it
found something. Third vacuous verification found in this project.

### Verified

Not just imports - the subprocesses, which is where a path mistake actually
shows up:

```
Surya venv, relocated       python 3.12.10, transformers 4.57.1, torch cu126
Surya --probe               {"device": "cuda", "3.2 GiB free of 4.0 GiB"}
translation, end to end     "Bangalore South Taluk"   <- real NLLB subprocess
pytest                      684 passed, 8 skipped
launcher.py --check         all checks passed
System Setup --report-only  Surya found, models found, migrations at head
ui_smoke / service_sweep    PASS, 0 JavaScript errors
60 modules under src/       all import
```

---

## [0.3.2] — 2026-08-03

Project cleanup. The point of the exercise was tidying; what it actually found
was two features that did not work. Both had passed every test, because a test
suite checks the code that runs and neither of these did.

### Fixed

**Translation was dead in the pipeline** (R-031) — every batch was writing
untranslated Kannada to the CSV while the translation service, called directly,
worked perfectly. `build_stages()` still carried the pre-NLLB wiring, which
located a checkpoint by globbing `*.safetensors`. NLLB ships `pytorch_model.bin`,
so the glob never matched and the code fell through to a `passthrough` fallback
designed for the gated IndicTrans2 weights. The fallback looked deliberate, and
a test pinned it in place by asserting exactly that behaviour.

```
before   pipeline: (False, 'translation is disabled in settings')
         service : (True,  'nllb-200-distilled-600M via venv_new')
after    pipeline: (True,  'nllb-200-distilled-600M via venv_new')
```

The replacement test asserts the two reach the *same* verdict, so they cannot
silently disagree again.

**Scanned deeds were cleaned but never made searchable** (R-032) —
`add_text_layer()` was referenced nowhere outside its own definition. Surya's
line boxes never crossed the subprocess boundary, so there was nothing to place
and the call site had never been written. "Copy Text" returned nothing on
exactly the pages that needed it. Now wired end to end: the runner emits
normalised line boxes in `--json`, `OcrStage` carries them, and
`BatchRunner._make_searchable()` writes the layer onto image-only pages.

Two further defects surfaced while proving it worked, both inside
`add_text_layer` and both of the kind that reports success:

- the font was sized by box height alone, so `insert_textbox` found the string
  too wide, wrote nothing, returned a negative number - and the code counted the
  page as written anyway. The first verification reported `written=1` over an
  empty page.
- `helv` cannot encode Kannada. It does not fail; it writes replacement bytes,
  which would have become the searchable text of a legal document.

Verified on real scanned pages (`3-2025-26.pdf` 11-12): no text before,
selectable Kannada and English after, other pages byte-identical, geometry
unchanged. Font subsetting took the size cost from 1522 KB to 86 KB.

### Removed

- `tools/indictrans_runner.py` and `find_translator()` — superseded by NLLB, and
  actively harmful while they remained reachable.
- 16 `__pycache__` / `.pytest_cache` directories (1.9 MB).

### Added

- `README.md` at the project root — there was none.
- `TestScannedPagesBecomeSearchable` (7 tests), each mutation-tested: the fix
  was reverted one line at a time and the suite confirmed to fail.

### Unchanged, deliberately

The `backend/` + `frontend/` restructure was **declined**. There is no React, no
TypeScript, no npm and no FastAPI in this project - one JavaScript file exists,
inside the desktop shell. The tree is already separated by responsibility
(`src/core/` `src/app/` `src/ai_server/` `src/launcher/`); renaming it would break every import,
document reference and launcher path for no functional gain. Reasoning in full
in the cleanup report.

### Verification

```
pytest tests/ -q         684 passed, 8 skipped
launcher.py --check      all checks passed
tools/ui_smoke.py        PASS, 0 JavaScript errors
tools/service_sweep.py   PASS
```

---

## [0.3.1] — 2026-07-31 (late)

Four features that were built but unreachable are now connected. Each had the
same shape: a correct, tested backend that nothing called - invisible to unit
tests of either side.

### Added

**Watermark page wired** — `AppService.watermark` raised `NotImplementedError`;
the page rendered but every button was inert. Now browse / scan / remove / open /
clear all work, with `_watermark_page` reporting real per-file state.
`allow_lossy` stays **False**: a raster watermark is burned into the scan, so
"removing" it means inventing content on a legal document.

Verified on a PDF carrying an annotation watermark over real deed text:

```
detected  : 1 found (annotation)      source unmodified : True
result    : lossless                  DUPLICATE COPY    : removed
SALE DEED / ABCDE1234F / 455/1        : all preserved
```

**Capability gating surfaced** — `Capabilities` modelled
`can_browse/can_export/can_upload/can_process` with reasons, and no template
consulted it. Now a banner in `base.mustache` lists what is unavailable and why,
plus `disabled` and `title` on Start, Browse, Add Batch and Download CSV. A
builder that sets a stricter local value still wins.

**Auto batch mode connected** — the Settings page has offered manual/auto since
the UI was built; the runner was constructed `MANUAL` and never read the stored
value, so choosing Auto changed a database row and nothing else. Applied at
startup and on save, so it takes effect without a restart. Cooldown floors at 5 s
- zero would start the next batch before the GPU released memory.

Verified against the live database with a 6 s cooldown:

```
t+0.4s  batch A finished (completed)
t+4.1s  cooldown observed: auto cooldown 2s remaining
t+6.1s  batch B promoted (running)        gap 5.6s
```

**RetentionScheduler started** — written but never launched. Now opt-in via
`SALEDEED_RETENTION`, deferring while a batch runs and stopping on shutdown.
Off by default deliberately: retention *deletes* data, and a destructive job that
starts itself on first launch is the wrong default for a records system.

New settings seeded: `auto_cooldown_seconds`, `retention_interval_hours`.

### Fixed

- **The capability banner never rendered.** `base.mustache` is rendered from the
  *chrome* context, not the page model, so `degraded` was computed correctly and
  never reached the shell - `{{#degraded}}` collapsed to nothing on all 8 pages
  and the feature was invisible. Found by rendering every page rather than by
  asserting the markup exists. Capability keys are now forwarded into the shell.

### Fixed (post-launch)

- **The window opened unstyled.** Reported on the first real launch of
  `Run Sale Deed AI.bat`: `QResource '/assets/theme.css' not found`. Two faults -
  the templates used `qrc:` (the Qt resource scheme, for resources compiled into
  the binary) instead of the registered `app://` scheme, and `AssetHandler` was
  rooted at `ui/` when it strips a leading `assets/` from the path and so needed
  `ui/assets/`. Both fixed; see R-018.
- **The asset test passed vacuously.** It searched for `app://` URLs and asserted
  every match existed - there were none, so it asserted over an empty set. Now
  asserts the match set is non-empty and mirrors the handler's own resolution.

- **Every page navigation failed** with `Could not open dashboard: failed`.
  Bridge work runs on a thread pool, and the worker invoked the JavaScript
  callback directly - a `QJSValue` belongs to the GUI thread, so the call
  silently never arrived and the promise settled empty. Results are now
  marshalled back through a queued signal; the work itself still runs off the UI
  thread. See R-019.

- **"Upload PDF" terminated the application.** `pick_files` was dispatched to
  the thread pool and opens a `QFileDialog`; a Qt widget created off the GUI
  thread aborts the process natively, below the interpreter, so nothing was
  logged. Now routed to the GUI thread, with a self-check in `_pick_files` and
  `BaseException` containment in the worker. See R-020.
- **Watermark buttons drove the wrong selection.** `btn-wm-browse` and
  `btn-wm-clear` called `pick_files` / `clear_selection` - the *upload*
  selection - and scan / remove / open had no handler at all. The backend was
  wired last release; the front end was not.
- **QWebChannel calling convention.** Slots declared `(str, QJSValue)` are never
  matched: the channel strips a trailing JS function as the reply handler and
  invokes the slot with what remains, logging `No candidates found for "render"
  with 1 arguments`. Slots now take `(request_id, payload)` and reply on a
  `completed(QString,QString)` signal, which keeps work off the UI thread.

- **Navigation rebuilt the QWebChannel on every page change.**
  `document.write` tore down the JavaScript context and constructed a second
  channel over the same transport, orphaning every reply already in flight -
  `execCallbacks[message.id] is not a function`, repeating at the poll interval,
  and `pick_files timed out after 120s`. Navigation now replaces `#content`
  only. See R-022.

- **The webview could never load its stylesheet or the channel.** Four silent
  faults: `LocalScheme` blocked navigation entirely, `setHtml` gave the page no
  origin so subresource requests were refused before the handler saw them,
  `qrc:` is cross-origin from `app://ui/` so `QWebChannel` was undefined, and
  the scheme handler was garbage-collected because `installUrlSchemeHandler`
  does not take ownership. See R-023.

- **Opening a batch raised `DetachedInstanceError`.** `_batch_detail` built its
  result after the session closed, and `user` is a relationship that was never
  loaded. Broken since the page was written; it could not surface until the
  webview worked. See R-024.

### Added

**`src/tools/service_sweep.py`** - exercises every service entry point against the
live database. The defect it found needs real rows; with an empty database the
same call returns "Not found" and passes.

**`src/tools/ui_smoke.py`** - drives a real `QWebEngineView` with a real channel and
the real `app.js`, and reports the browser console. Every UI defect this session
lived in the gap between "the renderer produces correct HTML" and "the browser
can use it", where source-level assertions are blind.

### Tests

**374 -> 497 passing**, plus the smoke test. `test_app_wiring.py` (61) covers all four, including a
parametrised render of every page in both degraded and healthy states - the
check that would have caught the banner defect immediately.

---

## [0.3.0] — 2026-07-31 (evening)

Surya OCR working, single-command launcher, full test sweep. Six real defects
found by the new tests and fixed.

### Added

**Surya OCR — the blocker that had been open longest**
- `src/tools/surya_runner.py` — runs inside Surya's own interpreter as a subprocess.
  Isolation is required twice over: Surya pins `transformers==4.57.1` against the
  rest of the project, and its three models hold ~3.2 GB of VRAM that only a
  process exit releases.
- Spatial layout reconstruction ported from the user's `run_ocr.py` at
  `TARGET_COLS = 110`. This is the padding the model was finetuned on; changing
  it changes the input distribution.
- Renders through Surya's own loader at `IMAGE_DPI_HIGHRES` (192), matching the
  corpus, rather than pre-rendering images at some other DPI.
- `--device auto|cuda|cpu` chooses from **free VRAM at start**, not from
  `torch.cuda.is_available()`. On a 4 GB card the language model already holds
  3.21 GiB, so a naive availability check would send Surya to a GPU with nothing
  left and it would die mid-batch.
- `find_surya()` + `ocr_engine="auto"` — resolves to Surya when present and to
  the text layer when not, so a machine without it still processes the
  digitally-generated deeds that make up most of a batch.
- Environment rebuilt: Python 3.12.10, `surya-ocr==0.17.1`, `requests`,
  `torch==2.13.0+cu126`.

**Application launcher**
- `launcher.py`, `Run Sale Deed AI.bat`, and a `src/launcher/` package
  (`config` / `steps` / `supervisor` / `runner`).
- 13 preflight checks, each reporting `ok|warn|fail` with the command that fixes
  it. All of them run even after a failure, so one pass surfaces every problem.
- Starts PostgreSQL if stopped, applies Alembic migrations, verifies the model,
  starts the AI server, waits for health, opens the window.
- Windows **Job Object** with `KILL_ON_JOB_CLOSE`. `terminate()` on the AI server
  leaves `llama-server.exe` orphaned holding VRAM and port 8077; the kernel now
  kills the whole tree even if the launcher itself is killed.
- Health wait is for the *endpoint*, not a warm model — the UI is designed to
  open during the 30-60 s load and gate what needs inference.

**Test suite: 164 -> 347 passing, all 19 categories**
- `test_database.py` (36), `test_security.py` (36), `test_platform.py` (34),
  `test_operations.py` (30), `test_surya.py` (20), `test_integration.py` (19),
  `test_watermark.py` (14).
- Load tests are scale-parameterised by `SALEDEED_LOAD_PDFS` so the same tests
  run at 25 in CI and 1000 on the deployment machine.

### Fixed

- **`statement_timeout` was absent on every reused connection**
  (`src/core/db/engine.py`). `SET` is transaction-scoped and the pool's `ROLLBACK`
  reverted it — measured 30 s on the first connect, `0` on every one after. The
  runaway-query stall it exists to prevent could still happen. Now a libpq
  startup option.
- **CSV formula injection** (`src/core/csv_export.py`). A party name is third-party
  data landing in a spreadsheet cell; `=cmd|'/c calc'!A0` executed on open.
  Defused with a leading apostrophe. The 42-column comparison still passes.
- **Upload accepted any file named `.pdf`** (`src/app/services.py`). Now checks for
  `%PDF-` in the first 1 KB, so a mislabelled file is rejected at selection
  rather than failing later as an apparently broken document.
- **Unescaped exception text in the dashboard** (`src/app/services.py`).
  `{{{message}}}` renders raw HTML and the notice was built from `{exc}`, which
  carries user-chosen filenames.
- **LaTeX fractions corrupted property extents** (`src/core/ocr_cleanup.py`).
  `42\frac{1}{2}` became `421/2` — which reads as survey number 421/2. Now
  `42 1/2`.
- `_postgres_bin()` honours `%ProgramFiles%` instead of a literal `C:\`.

### Measured

| | |
|---|---|
| Surya, 5 pages, CPU | 858.9 s |
| Surya, 5 pages, CUDA | 388.9 s (**2.2x**, not the 4-10x estimated) |
| Launcher to health 200 | 4.5 s |
| Full test suite | 9.6 s |

The GPU gain is bounded by VRAM, not compute: with ~3.2 GiB free the recognition
model runs small batches and stays memory-bandwidth bound. CUDA output was also
*more* accurate — 9 of 9 survey numbers matched the reference against 8 of 9 on
CPU.

---

## [0.2.0] — 2026-07-31

Database live, desktop UI, remaining spec modules, test suite.

### Added

**Database (live)**
- PostgreSQL **17.10** installed; role and database `saledeed` created.
- `0001_initial` applied: 11 tables, 4 native ENUM types, 17 seeded settings.
- `src/tools/db_setup.py` — `--check / --upgrade / --seed / --verify`, plus
  `--sqlite` for logic verification without a server. **15/15 against real
  PostgreSQL**, including `FOR UPDATE SKIP LOCKED`.

**Desktop UI**
- `src/app/main.py` — PySide6 shell, QWebEngineView, custom `app://` scheme.
- `src/app/ui/bridge.py` — QWebChannel bridge; every call async so the UI thread
  never blocks.
- `src/app/services.py` — Qt-free service layer.
- `src/app/ui/renderer.py` + 10 Pystache templates + `theme.css` (UX4G) + `app.js`.
  All 8 screens render; the window launches and the event loop runs clean.

**Remaining spec modules**
- `src/core/logging_setup.py` — DEBUG-gated; handlers are *not installed* when off.
  Database logging behind a queue so it can never stall a stage worker.
  `LogContext` attaches batch/document identifiers to every record.
- `src/core/watermark.py` — lossless removal of OCG, annotation and text-overlay
  watermarks; raster watermarks detected and refused.
- `src/core/backup.py` — `pg_dump` archiving, verify-before-purge, retention
  scheduler, 30-day OCR expiry, log rotation.

**Tests** — 164 passing: 139 unit, 25 integration, 5 regression.

**Stamp Value** — derivation implemented (ADR-015): registration fee, halved
before 31 August 2025.

### Fixed

- **Idempotency was broken in three repository methods.** `save_pages`,
  `replace_persons` and `record_flags` set the foreign key directly instead of
  appending to the parent relationship, leaving the loaded collection stale so
  delete-then-insert deleted nothing. A retried document would have crashed on a
  unique constraint. Found by running the verification suite.
- **Documents were stranded on retry.** A requeued document was also marked
  `NEEDS_REVIEW`, making it permanently unclaimable. Fixed with a `REQUEUED`
  sentinel and proven against live PostgreSQL.
- **Third instance of the decimal-point defect.** `reg_fee_candidates` read
  `200.00` as 20000. Added `parse_amount()`.
- **PAN coverage was unusable at small denominators** (ADR-014). With two PANs the
  ratio can only be 0.0/0.5/1.0, so one witness PAN failed a deed forever. Retry
  now also requires 2+ unmatched PANs.
- **GPU lease was not taken by the OCR stage.** Harmless with the text-layer
  fallback, but Surya is a GPU model and would have raced the LLM for VRAM. The
  lease is now conditional on a `uses_gpu` property.
- **Log context did not attach.** Filters on a logger do not see records
  propagated from child loggers; moved to handler filters.
- **ORM/database double-delete warning** — `passive_deletes=True` on 7
  relationships so PostgreSQL's cascade does the work.
- `WinError 2` launching `llama-server` (Windows rejects relative forward-slash
  paths); `--flash-attn` now requires an explicit value in b10184.

### Changed

- `char_length()` -> `length()` in CHECK constraints, and `BigInteger` primary
  keys given a SQLite variant — both so the schema can be created on SQLite for
  verification. PostgreSQL behaviour is unchanged (`BIGSERIAL` retained).
- `RuleToggles.stamp_value` now defaults to **True** (ADR-015 supersedes ADR-010).
- Logging is configured in `src/app/main.py` and `src/ai_server/server.py`, with
  `shutdown()` in a `finally` so the queue listener flushes.

### Dependencies

`PySide6 6.11.1` (+Addons), `pystache 0.6.8`, `PyMuPDF 1.28.0`, `pytest`.
PostgreSQL 17.10 as a system service.

### Known issues

Surya OCR and IndicTrans remain absent — two of three pipeline stages have no
engine. Q4_K_M accuracy loss is measured and real (deed 07, 10x error on a sale
consideration), caught by validation. Extraction retry on v6.7 is accepted as
unavailable and will not be pursued.

---

## [0.1.0] — 2026-07-30

First development session. Model deployment layer complete; application layer not
yet started.

### Added

**Hardware and resource management**
- `src/ai_server/hardware.py` — CPU, physical/logical cores, RAM, GPU, VRAM and disk
  detection. NVIDIA-only, pinned by UUID rather than index. Non-NVIDIA adapters
  detected purely to exclude them. Standard library only, so it runs on a bare
  interpreter before anything is installed.
- `src/ai_server/profiles.py` — reads model geometry from the checkpoint's own
  `config.json`, computes a VRAM budget honouring Gemma 3 interleaved
  sliding-window attention, and selects the highest-fidelity configuration that
  fits. `ladder_report()` explains why a better quantisation was not chosen.
- `src/ai_server/resources.py` — runtime governor. Four pressure levels with
  one-directional hysteresis, per-stage concurrency bounded by both cores and
  available RAM, exclusive GPU lease for cards too small for model co-residency,
  and memory trimming (`gc.collect` + `SetProcessWorkingSetSize`).

**Inference engines**
- `src/ai_server/engines/base.py` — `InferenceEngine` contract,
  `ExtractionRequest` / `ExtractionResult`, `EngineHealth`,
  `ModelOutOfMemoryError` carrying the shortfall.
- `src/ai_server/engines/llamacpp.py` — `llama-server` supervision: argv built from
  the selected profile, UUID device pinning, forbidden-backend detection
  (refuses Vulkan/OpenCL, which could bind the integrated GPU), OOM diagnosis
  from startup output, optional idle unload with lazy reload, continuous
  batching.
- `src/ai_server/engines/mock.py` — GPU-free stub. Scrapes identifiers from the OCR
  with regexes so the validators see realistic input; enables full pipeline
  testing and CI on hardware that cannot host the model.

**AI server**
- `src/ai_server/server.py` — async job queue with worker threads, and an HTTP API
  (`/health`, `/hardware`, `/profile`, `/extract`, `/extract/batch`,
  `/jobs`, `/jobs/<id>`, `/shutdown`). Standard library HTTP, zero pip
  dependencies. Returns `503` with `retry: true` under critical pressure rather
  than piling work on a struggling machine. CRLF normalised at the boundary.

**Tools**
- `src/tools/repack_checkpoint.py` — lossless byte-level repack: renames
  `model.language_model.model.` -> `model.`, drops the unused vision tower,
  emits a standard `Gemma3ForCausalLM` config, inlines the chat template, and
  verifies contiguous offsets and dtype consistency in the output.
- `src/tools/llamacpp/` — llama.cpp b10184 CUDA 12.4 Windows binaries.

**Documentation**
- `/docs` — ten files: `PROJECT_STATUS`, `DEVELOPMENT_LOG`, `CHANGELOG`, `TODO`,
  `ARCHITECTURE`, `DECISIONS`, `API_DOCUMENTATION`, `DATABASE_SCHEMA`,
  `TEST_REPORT`, `KNOWN_ISSUES`.

### Model artifacts

Produced from the trained weights. The original was never written to and is
verified unmodified at 8,600,283,312 bytes.

| Artifact | Size | Notes |
|---|---|---|
| `models/AI server/gemma4b-text/` | 7.23 GiB | 444 tensors, standard layout, lossless |
| `models/AI server/gguf/deeds-v6_7-f16.gguf` | 7.24 GiB | intermediate |
| `models/AI server/gguf/deeds-v6_7-Q4_K_M.gguf` | 2.33 GiB | production model |

Quantisation: 7401 MiB -> **2368 MiB at 5.12 BPW**. The profile model predicted
2.30 GiB at 5.1 BPW, validating the budget arithmetic against measurement.

### Fixed

- **Non-standard checkpoint key layout** (R-001). The doubled `model.` prefix
  caused transformers 5.8.x to load the checkpoint without error while silently
  random-initialising the language model. Fixed permanently by repacking. **The
  transformers 5.5.0 pin is no longer required** — this was the most dangerous
  constraint in the project, because violating it produced fluent garbage rather
  than an exception.
- **Tokenisation divergence** (R-002). GGUF produced 6,758 tokens against the HF
  reference's 6,408 on one deed. Root cause was CRLF line endings, not the
  tokenizer: llama.cpp preserves `\r` (token 251) while the HF tokenizer drops it
  and merges `\r\n\r\n` into `\n\n`. Identical after normalisation. CRLF->LF is
  now enforced at the AI server boundary.
- **Profile selector chose a 99%-full configuration** (R-003), which would have
  OOMed on any incidental allocation. Added a 95% utilisation ceiling and
  reordered the search to prefer KV precision over weight precision.
- Dataclass field-ordering error in `HardwareInfo` (defaulted field before
  non-defaulted ones).
- `disks` computed but never passed to the `HardwareInfo` constructor — caught by
  an IDE diagnostic.
- Progress output spammed one line per chunk when not attached to a terminal;
  now gated on `sys.stdout.isatty()`.

### Changed

- `generation_config.json` in the **repacked copy** now sets `do_sample: false`
  and drops `top_k` / `top_p`. Extraction must be greedy; the shipped defaults
  would have let any HF call that omitted `do_sample=False` sample silently. The
  original file is unchanged.
- Decode defaults set to `max_tokens=2048`, `repetition_penalty=1.1`, from the
  project's own benchmark: `gemma6.8 score.md:31` records one runaway repetition
  loop consuming 43% of total wall time under an 8000-token cap.

### Dependencies

The AI server itself has **zero pip dependencies**.

Conversion-only, in an isolated scratchpad venv (not required at runtime):
`numpy 2.5.1`, `torch 2.13.0+cpu`, `transformers 4.57.6`, `gguf`,
`sentencepiece`, `protobuf`. Note: llama.cpp pins `numpy~=1.26.4`, which does not
support Python 3.13; numpy 2.x was used instead and works.

### Known issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Summary: CUDA runtime download incomplete
so no real GPU inference has run (B-001); model is v6.7 and the retry stage cannot
function (B-002); Surya OCR and IndicTrans absent (B-003); Q4_K_M accuracy loss
unmeasured (L-001); Stamp Value formula undefined (L-002).

### Notes

- No model weights were downloaded at any point. Only converter source, the
  llama.cpp binaries, and Python packages.
- `pyproject.toml` and `.gitignore` were created early in the session and removed
  at user request; nothing from that attempt remains.
