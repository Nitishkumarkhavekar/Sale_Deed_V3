# Sale Deed AI — Complete Documentation

**The single reference for this application.** Everything previously spread
across fourteen files in `docs/` is here, section by section, describing what the
code actually does today rather than what was once planned.

Figures in this document were read from the source on **19 August 2026**, not
copied from older notes: 71 modules, 25,301 lines, 11 tables, 42 export columns,
13 preflight checks, 1,636 tests.

---

## Table of contents

1. [What this application is](#1-what-this-application-is)
2. [Running it](#2-running-it)
3. [System requirements](#3-system-requirements)
4. [Installation](#4-installation)
5. [Architecture](#5-architecture)
6. [Folder structure](#6-folder-structure)
7. [Code map](#7-code-map)
8. [The processing pipeline](#8-the-processing-pipeline)
9. [PDF handling and watermark removal](#9-pdf-handling-and-watermark-removal)
10. [OCR](#10-ocr)
11. [AI extraction](#11-ai-extraction)
12. [Translation](#12-translation)
13. [Validation and business rules](#13-validation-and-business-rules)
14. [CSV export — the 42 columns](#14-csv-export--the-42-columns)
15. [Database](#15-database)
16. [The desktop application](#16-the-desktop-application)
17. [The AI server API](#17-the-ai-server-api)
18. [Configuration and environment variables](#18-configuration-and-environment-variables)
19. [Logging](#19-logging)
20. [Error handling](#20-error-handling)
21. [Security](#21-security)
22. [Testing](#22-testing)
23. [Troubleshooting](#23-troubleshooting)
24. [Backup and maintenance](#24-backup-and-maintenance)
25. [Performance, measured](#25-performance-measured)
26. [Known limitations](#26-known-limitations)

---

## 1. What this application is

Offline processing of Indian sale deeds. A folder of registration PDFs goes in;
a 42-column CSV comes out, one row per party to each transaction.

The steps are: read the PDF, remove separable watermarks, OCR the Kannada,
extract sixteen fields with a fine-tuned Gemma-3-4B, validate them against the
OCR source, translate what needs translating, and export.

**Nothing leaves the machine.** There is no cloud API anywhere in the pipeline.
Deed text is a legal record identifying parties to a property transfer, and every
model — extraction, OCR, translation — runs locally. This is the single
constraint that shaped most of the technical choices below.

### Features

| | |
|---|---|
| Batch processing | Up to 4 queued batches; run, pause, stop, resume, delete |
| Watermark removal | Lossless — refuses to inpaint a scan |
| OCR | Surya, Kannada and English, with spatial layout reconstruction |
| Field extraction | Fine-tuned Gemma-3-4B, 16 fields, JSON output |
| Validation | Seven layers, cross-checked against the OCR source |
| Translation | 12 languages to English; names transliterated by rule |
| Export | 42-column CSV, Excel-safe, plus a failed-document export |
| Review | Data View, per-document PDF viewer with a selectable text layer |
| Resume | Per-document, per-stage; survives crash, shutdown and power loss |

---

## 2. Running it

```
Run Sale Deed AI.bat
```

That is the whole command. It starts PostgreSQL if stopped, applies migrations,
verifies the model, starts the AI server, waits for it to become healthy, and
opens the window. On shutdown it stops what it started.

On a machine that has never run this before:

```
system_setup.bat
```

| Command | Does |
|---|---|
| `py -3.13 launcher.py --check` | validate everything, change nothing |
| `py -3.13 launcher.py --no-ai` | browsing and export, no inference |
| `py -3.13 launcher.py --headless` | services without the window |
| `py -3.13 launcher.py --verbose` | follow the AI server's log in this terminal |
| `system_setup.bat --report-only` | detect and report, change nothing |
| `system_setup.bat --no-launch` | set up but do not start |
| `system_setup.bat --skip-tests` | skip the test suite during setup |

### The operator's workflow

1. **Upload PDFs** — select files, name a batch, add it to the queue.
2. **Press Start** — the runner claims documents one at a time.
3. **Watch PDF Processing** — per-stage progress, per-document status.
4. **Failed OCR** — see which documents failed OCR specifically, and rerun one
   without reprocessing the batch.
5. **Data View** — review extracted rows, open the source PDF, export the CSV.

---

## 3. System requirements

| | Minimum | This was developed on |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 11 |
| Python | 3.13 **and** 3.12 | 3.13.14 / 3.12.10 |
| RAM | 8 GB | 7.4 GB |
| GPU | NVIDIA, 4 GB VRAM, CUDA | RTX 3050 Laptop, 4 GB |
| Disk | 60 GB free | — |
| Database | PostgreSQL 17 | 17.10 |

**Two Python versions are required and this is intentional.** 3.13 runs the
application; 3.12 runs Surya OCR, which pins `transformers==4.57.1` against
everything else. Interpreters are selected by *capability* (`import PySide6`),
never by version number — installing a second Python changes what a bare
`python` resolves to, and choosing by version silently picks the wrong one.

A non-NVIDIA GPU is not used for inference. CUDA cannot enumerate an integrated
AMD adapter, so a CUDA build is structurally incapable of selecting it.

---

## 4. Installation

`system_setup.bat` is a shim; all logic is in `src/tools/system_setup.py`,
because batch is unreadable and untestable at any size worth writing.
`system_setup.bat` is the same installer under a name without a space, for
runbooks and unattended installs; it forwards every argument and preserves the
exit code.

It detects what is already installed, installs only what is missing, and never
replaces or removes anything already present. Safe to run twice.

Every step is idempotent in the same shape: detect, skip if present, install if
not, verify afterwards, and report which of those three happened.

```
Detect environment
  → bootstrap interpreter (only to build .venv; app packages never go system-wide)
  → create .venv, install requirements.txt
  → PostgreSQL: install if missing, create role and database
  → write .env with a generated 20-character password, restricted to the installing user
  → alembic upgrade head
  → verify the model files
  → install the llama.cpp CUDA runtime
  → install Surya into models/SuryaOCR/venv_new (Python 3.12)
  → install the translation model (~2.5 GB, once)
  → resolve the inference profile, write config/hardware_profile.json
  → check the AI server's port is free
  → run the test suite
  → write docs/INSTALLATION_REPORT.md
  → launch
```

### Ports

The AI server's port is resolved the way the launcher resolves it - from
`SALEDEED_AI_URL`, falling back to `http://127.0.0.1:8077` - and never from a
second copy of the default written into the installer. Two copies of a default
are how the installer and the launcher come to disagree about which port the
application uses, and the symptom of that disagreement is a UI reporting *AI
server offline* against a server that started perfectly well somewhere else.

If the port is occupied the installer names the process holding it and stops
there. It kills nothing: the holder is as likely to be a previous run of this
application as a stranger, and an installer that frees ports by force
eventually takes down something that mattered. It also does not pick another
port silently - an alternative port only takes effect if it is written into
`.env`, and rewriting the configuration of a working installation is worse than
reporting a conflict.

To move the application to another port:

```
SALEDEED_AI_URL=http://127.0.0.1:8078
```

### config/hardware_profile.json

Written at every run, read by nothing. It records what this machine resolved
to - GPU, VRAM, quantisation, context, GPU layers, concurrency, threads, and
the reason the selector gave - so an operator can quote it in a support
message.

It is deliberately not a configuration file. `ai_server/profiles.py` recomputes
the profile from *measured free VRAM* every time the server starts, because
free VRAM is a property of the moment rather than of the machine: another
application holding 2 GB changes the right answer. A file that pinned
concurrency at install time would be wrong the first time anything else used
the card, and wrong in the direction that ends in an out-of-memory failure.
The file is gitignored for the same reason - one machine's answer would only
mislead the next.

### The model is verified, never downloaded

The extraction model is fine-tuned and specific to this project. An installer
that helpfully fetched "a Gemma 3 4B" would replace the weights every accuracy
figure in this document was measured against, and nothing downstream would
notice. Setup checks the files are present and correct; it does not fetch them.

### Deployment to another machine

Copy the whole project directory, including `models/`, then run
`system_setup.bat`. `.env` is **not** copied — setup writes a fresh one with a
generated password matching the database it creates.

---

## 5. Architecture

### Process topology

Four independent processes. The separation is the mechanism by which heavy AI
work cannot affect UI responsiveness — it is structural, not best-effort.

```
                    launcher.py
      starts, health-checks, supervises, and kills
      the whole tree via a Windows Job Object
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌──────────────────────────────────────────────────────────┐
│ Desktop application (PySide6 + Pystache in QWebEngine)   │
│   CPU only. Never imports CUDA. Never loads a model.     │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP (localhost)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ AI server  (ai_server/, zero pip dependencies)           │
│   hardware detection · profile selection · governor      │
│   async job queue · GPU lease · model lifecycle          │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP (localhost, OpenAI-compatible)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ llama-server  (subprocess, CUDA)                         │
│   the only long-lived process that holds the GPU         │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
                    PostgreSQL  →  CSV export

  Surya OCR  (tools/surya_runner.py, own interpreter, per document)
    spawned by OcrStage, exits after each document to return VRAM
```

A crash in the inference runtime cannot take down the UI.

**Why Surya is a transient process rather than a fourth server.** Its three
models hold ~3.2 GB, and Python cannot reliably hand that back to the driver —
only process exit does. A long-lived Surya server would amortise its ~30 s load
across documents, but on a 4 GB card it would mean the language model never gets
the GPU.

**Why the AI server is not FastAPI.** It is stdlib `ThreadingHTTPServer`. It must
start fast and carry no dependency capable of breaking the one thing it exists to
do.

### Startup

```
Project validation → Configuration → PostgreSQL → Migrations
  → Model verification → AI service → Health check → Window → Ready
```

Thirteen preflight checks run, each reporting `ok | warn | fail` with the command
that fixes it. **All of them run even after a failure**, so one pass surfaces
every problem rather than sending the operator round a loop.

The checks, in order: Project files, Python runtime, Dependencies, Folders, Disk
space, Hardware, PostgreSQL service, Database, Migrations, Model, Inference
runtime, OCR, Port.

Two deliberate deviations:

- **The health check waits for the endpoint, not the model.** Loading a 4-bit
  Gemma takes 30–60 s and the interface already opens against a loading server,
  showing LOADING and gating what needs inference.
- **A failed AI service does not abort the launch.** Browsing, review and CSV
  export need no inference. The window opens with those enabled and processing
  greyed out.

### Shutdown

A Windows Job Object with `KILL_ON_JOB_CLOSE`. `terminate()` on the AI server
leaves `llama-server.exe` orphaned holding VRAM and port 8077.

### Hardware adaptation

Two layers answering two questions.

**Startup — `profiles.py`: what configuration fits?** Reads geometry from the
checkpoint's own `config.json` (nothing hardcoded), computes a VRAM budget, walks
a fidelity ladder and returns the highest-precision rung that fits with slack.

Search order encodes accuracy priorities, outermost first:

1. **Context** — a configuration that truncates real deeds is useless regardless
   of precision, so context is a constraint rather than a variable.
2. **KV precision** — quantised KV damages exact-copy fields (Aadhaar, PAN) more
   than quantised weights do.
3. **Weight precision.**

KV cost honours Gemma 3 interleaved attention:

```
kv_bytes(ctx) = (5·ctx + 29·min(ctx, 1024)) × 2 × n_kv_heads × head_dim × elem
```

Only 5 of 34 layers scale with context. At 24k that is 622 MB instead of 3.42 GB
— the property that makes this model viable on 4 GB at all. On a 16 GB GPU the
ladder reaches BF16, where quantisation loss is zero, with no code change.

**Runtime — `resources.py`: how much work should be in flight?** Samples RAM,
VRAM, CPU and disk every 3 s and publishes a `ConcurrencyPlan`.

| Pressure | Trigger (free RAM) | Effect |
|---|---|---|
| normal | > 25% | full concurrency |
| elevated | 15–25% | 50% |
| high | 8–15% | 25%, memory trim |
| critical | < 8% or disk < 5 GiB | stop admitting new work |

Degradation is immediate; recovery requires clearing the threshold by a 5% margin
so pools do not oscillate. At critical, in-flight documents finish but nothing new
is admitted — killing work would lose a document, and pausing is safe because
every stage is resumable. Never drops below 1 worker.

### GPU arbitration

At 4 GB the OCR model, the 4B language model and the translation model **cannot
be co-resident**; whichever loads second fails. The governor detects VRAM below a
12 GiB co-residency threshold and serialises the three GPU stages behind an
exclusive lease. On a 16 GB machine the lease is a no-op. Same code.

### OOM prevention

1. **llama.cpp preallocates.** Context and KV are sized at load and never grow.
   Once the server is up, VRAM is flat regardless of prompt length or request
   volume. A 33k-token deed cannot OOM mid-batch; it is rejected at admission.
2. **8% headroom** held back from measured free VRAM.
3. **95% ceiling** on the remaining budget.
4. **Preflight** before load; stepwise degradation (context → KV precision →
   quantisation) on failure.
5. **OOM diagnosis** — `llamacpp.py` scans startup output for allocation-failure
   markers and raises `ModelOutOfMemoryError` carrying the shortfall.

### Device safety

The real risk is a **Vulkan or OpenCL build**, which would enumerate the Radeon
and may select it silently — correct output at a fraction of the speed, easy to
miss. Mitigations: pin `CUDA_VISIBLE_DEVICES` by **UUID** (indices reorder across
driver updates); clear `GGML_VK_VISIBLE_DEVICES` and `ONEAPI_DEVICE_SELECTOR`;
abort startup if the log reports a forbidden backend.

---

## 6. Folder structure

Five folders, grouped by how the contents behave rather than by what they are:

```
src/          all the code (71 modules, 25,301 lines)
  core/         domain logic — pipeline, extraction, validation, export,
                watermark removal, translation. No UI, no HTTP, no Qt.
  app/          desktop shell: window, web channel bridge, templates, services
  ai_server/    inference HTTP service, hardware detection, profile selection
  launcher/     startup orchestration, preflight checks, process supervision
  tools/        operational scripts — setup, OCR and translation runners,
                verification harnesses
  migrations/   Alembic revisions

models/       ~37 GB of weights and vendor installs. Large, static, never
              generated. Delete nothing here.
  AI server/    the fine-tuned GGUF, the translation model
  SuryaOCR/     the Surya installation and its Python 3.12 environment
  saledeed main/  the finetuning prompt and reference documents

runtime/      everything written while running — logs, uploads, exports,
              cleaned PDFs, caches, backups. Disposable: delete it to reclaim
              space and the application rebuilds what it needs.

tests/        the suite (1,636 tests) and corpus/ — sample deeds, reference OCR
              and reference extraction outputs the suite reads

docs/         this file, plus the generated INSTALLATION_REPORT.md
```

Root holds only entry points: `launcher.py`, the `.bat` files,
`requirements.txt`, `alembic.ini`, `pytest.ini`.

The split that matters is **`models/` against `runtime/`**: one is ~37 GB that
must never be touched by a cleanup script, the other is disposable by design.
Anything unsure which it is belongs in `models/`.

`src/core/paths.py` owns every filesystem location. Modules ask it rather than
computing `parents[N]` for themselves, so moving a folder is one line there
instead of twenty scattered ones.

---

## 7. Code map

| Package | Modules | Lines | Role |
|---|---:|---:|---|
| `core/` | 25 | 10,863 | Domain logic. No UI, no HTTP, no Qt. |
| `tools/` | 15 | 4,375 | Scripts: installers, subprocess runners, verification. |
| `app/` | 7 | 4,528 | Desktop shell — window, bridge, screens. |
| `ai_server/` | 11 | 3,759 | The inference service. Its own process. |
| `launcher/` | 6 | 1,206 | Starts, supervises and stops everything. |
| `migrations/` | 7 | 570 | Alembic revisions. |

### Which way the arrows point

```
launcher/  ──────────────► everything (it starts all of it)
app/       ──► core/          ──HTTP──► ai_server/
tools/     ──► core/, ai_server/
migrations/──► core/db/models
core/      ──► nothing above it
```

`core/` importing from `app/` or `ai_server/` would be a defect: it is the layer
that must stay testable without a window, a GPU or a server. The desktop shell
talks to the inference service over HTTP rather than importing it, so that
process never links CUDA.

### `core/` — the domain

**`pipeline/`** — how a deed becomes rows

| Module | Does |
|---|---|
| `stages.py` | The four stages: OCR → extract → validate → translate. Document logic, **no database** — each returns a `StageOutcome`. |
| `runner.py` | `BatchRunner`: claims work, drives the stages, writes results, holds the GPU lease, makes scanned pages searchable. |

**`db/`** — persistence: `models.py` (declarative models and enums),
`repositories.py` (repository pattern behind a `UnitOfWork` — every query lives
here), `engine.py` (engine, `session_scope`, health probe, statement timeout).

**`translation/`** — `service.py` (the single entry point: cache, batching,
subprocess, retry), `detect.py` (script-range detection), `transliterate.py`
(rule-based proper nouns), `config.py`, `postprocess.py`.

**Single concerns:** `validation.py` (field rules and disposition),
`watermark.py` (lossless removal), `csv_export.py` (the 42 columns),
`pdf_prepare.py` (cleaned document plus invisible text layer),
`ocr_cleanup.py`, `backup.py`, `logging_setup.py`, `transaction_id.py`,
`failure_codes.py`, `pdf_validation.py`, `paths.py`.

### `ai_server/` — inference

`server.py` (HTTP endpoints), `profiles.py` (quantisation, context and KV from
the hardware), `hardware.py` (CPU/RAM/GPU/VRAM/disk), `resources.py`
(`ResourceGovernor`), `deployment.py`, and `engines/` (`base.py` the interface,
`llamacpp.py` production, `mock.py` a deterministic stub for tests, `vllm.py`).

### `app/` — the desktop shell

`services.py` (every screen and every action — the largest file in the project),
`status.py` (background probing and capability gating), `main.py` (window,
custom URL scheme, asset and PDF handler), `ui/bridge.py` (30 QWebChannel slots —
**the only path** between the webview and Python), `ui/renderer.py`,
`ui/templates/` (16 logic-less Mustache templates), `ui/assets/app.js` (the
project's only JavaScript file).

Two rules this package lives by, both learned from defects:

- Widgets touch **only** the GUI thread. A `QFileDialog` on a worker thread
  aborts the process natively, with no traceback. `Bridge._GUI_THREAD` is the
  explicit list.
- Slots are `(QString, QString)` and reply on a signal. A trailing JavaScript
  function is stripped by QWebChannel as the reply handler, so a slot with the
  wrong arity is never found and the error names the *slot*, not the cause.

### `tools/` — three unrelated kinds

**Installers** (run once per machine): `system_setup.py`, `setup.py`,
`db_setup.py`, `repack_checkpoint.py`.

**Runners — the application depends on these.** Executed as subprocesses inside
Surya's Python 3.12 environment: `surya_runner.py` (OCR, emits
layout-reconstructed text and per-line boxes) and `translate_runner.py` (NLLB,
batched by source language).

> Do not move or rename these two without updating `core/pipeline/stages.py` and
> `core/translation/config.py`. They are located by path, not by import.

**Verification harnesses** (run by hand): `ui_smoke.py`, `service_sweep.py`,
`e2e_test.py`, `translation_check.py`, `identity_check.py`, `prepare_check.py`,
`kannada_audit.py`, `extraction_report.py`, `baseline_check.py`.

These exist because the unit tests cannot see integration seams. Every UI defect
found in this project so far lived in one of those seams and passed the full
suite while doing so.

### Where to add something new

| If it… | Put it in |
|---|---|
| decides something about a deed | `core/` |
| is a new pipeline step | `core/pipeline/stages.py` |
| is a database query | `core/db/repositories.py` — nowhere else |
| is a new screen or button | `app/services.py` + a template |
| needs the GPU or the model | `ai_server/` |
| is a path | `core/paths.py` — never `parents[N]` again |
| runs before the window opens | `launcher/steps.py` |

---

## 8. The processing pipeline

```
PDF → prepare → OCR → cleanup → extract → validate → translate → database → CSV
```

Four stages, each returning a `StageOutcome`. Stage state is tracked **per
document**, not per batch, so a restart never re-runs completed work and never
duplicates it.

### Continuous commit

Every processed document updates the database immediately. Nothing is deferred to
batch end — the application must survive crash, shutdown and power failure with
no data loss.

### Claiming

`claim_next(stage)` uses `FOR UPDATE SKIP LOCKED`, so concurrent workers cannot
take the same row. A document is admitted to a stage only while its attempt
counter is under the cap; stage ordering is enforced (extraction cannot be
claimed before OCR is done).

### Batch drain order

The runner drains OCR for a whole batch before extracting any of it. Swapping the
language model out and back in costs ~10.5 s, so doing it per document rather
than per batch was pure waste. `_claim_downstream` already existed for this
state, so crash-safety is unchanged: `ocr_state` is committed DONE with the text
before the runner moves on.

### Batch queue

At most 4 batches in `queued`. Enforced in the repository rather than by a
database constraint, so the error message can be meaningful. Modes are manual or
automatic; automatic promotes the next batch after a cooldown.

### Recovery

On startup a scan resets documents left `RUNNING` by a crash, making them
claimable again. Retries genuinely exhausted are marked FAILED rather than
requeued forever, which would spin the runner on a document it cannot take.

---

## 9. PDF handling and watermark removal

`core/pdf_prepare.py` produces the working copy; `core/watermark.py` does
detection and removal; `core/pdf_validation.py` answers "is this file broken?"
when something has already gone wrong.

### Lossless only

Watermarks are removed by deleting the object that draws them — an annotation, an
OCG layer, a drawing operator. **It refuses to inpaint a scan.** A watermark
burned into a raster page cannot be removed without altering the pixels of a legal
document, and guessing what was underneath is not an acceptable thing to do to a
deed.

The source file is never modified. Output goes to a `Cleaned Watermark` folder
inside the input folder, keeping the original filenames, with failures and their
reasons written to a `Failed` folder beside it.

### The invisible text layer

After OCR, pages that carry no text of their own get an invisible, selectable
text layer built from Surya's per-line boxes. A page that already has a real text
layer is left alone — writing a second copy over it would make every drag return
the text twice.

---

## 10. OCR

**Surya OCR 0.17.1**, in `models/SuryaOCR/venv_new` (Python 3.12,
`transformers==4.57.1`). Run as a subprocess, one invocation per document,
exiting afterwards to return VRAM.

Text is reconstructed spatially at `TARGET_COLS = 110` — the same width the
finetuning corpus used, so the model sees the shape it was trained on. Rendering
happens at Surya's own `IMAGE_DPI_HIGHRES`; rendering at another DPI changes both
the recognised text and the bounding boxes.

The recognition model is multilingual and takes no language argument, which is
why there is no OCR language setting.

### Installing it on another machine

The environment cannot be copied. A virtualenv records the absolute path it was
built at, so `models/SuryaOCR/venv_new` carried from `D:\saledeed v3` to
`E:\saledeed v3` leaves an interpreter that starts and then cannot import
`surya`. Setup reports this as *interpreter present, surya not importable* and
writes the traceback to `runtime/logs/setup/surya-import.log`.

Rebuild it in place on the target machine:

```
rmdir /s /q "models\SuryaOCR\venv_new"
py -3.12 -m venv "models\SuryaOCR\venv_new"
"models\SuryaOCR\venv_new\Scripts\python.exe" -m pip install surya-ocr==0.17.1 transformers==4.57.1
```

Surya downloads its recognition weights on first use, so the machine needs
network access once. Without this environment scanned pages are skipped and
only PDFs with a real text layer are processed — most Kaveri deeds are scans,
so this is not optional in practice.

### Cleanup

`core/ocr_cleanup.py` normalises raw OCR into what the model was finetuned on:
CRLF to LF, page markers, Surya markup, LaTeX fractions. CRLF normalisation is a
measured correctness requirement, not hygiene.

### The text-layer alternative

`textlayer` reads the PDF's embedded text with PyMuPDF. Zero dependencies beyond
PyMuPDF and no GPU, but Kannada quality is poor and it returns nothing for a pure
scan. It exists as a fallback and for development, not as a production
substitute. A document yielding under 40 characters per page is treated as a pure
scan and fails with that explanation rather than proceeding on nothing.

---

## 11. AI extraction

**Fine-tuned Gemma-3-4B**, served as `deeds-v6_7-Q4_K_M.gguf` (2.33 GiB) by
`llama-server.exe`.

### Model pipeline

```
AI server/gemma4b/                     trained weights, READ-ONLY forever
        │  tools/repack_checkpoint.py  (lossless: rename keys, drop vision)
        ▼
AI server/gemma4b-text/                444 tensors, standard Gemma3ForCausalLM
        │  convert_hf_to_gguf.py
        ▼
AI server/gguf/deeds-v6_7-f16.gguf     7.24 GiB
        │  llama-quantize
        ▼
AI server/gguf/deeds-v6_7-Q4_K_M.gguf  2.33 GiB  ← served
```

Every derived artifact is regenerable from the original. The original is never
written to.

### OCR text, not images

The model is multimodal, but it was **finetuned on OCR text**, so text is its
native input and images are out of distribution. Measured on a real deed, vision
mode duplicated the buyer into sellers, truncated Aadhaar digits, garbled
addresses and put stamp duty in the registration fee. OCR-text mode fixed all of
them.

### The prompt

`models/saledeed main/prompt_v6_short.txt`. Without it the fine-tuned model
behaves like an instruction-tuned one looking at an unlabelled wall of OCR and
responds with prose, so every extraction fails with "no parseable JSON" and every
column comes out empty. A missing or empty prompt file is logged as an error at
startup.

Output is one JSON object: `buyer_details[]`, `seller_details[]`,
`property_details{}`, `document_details{}` — sixteen fields.

### Context bounds

`MAX_INPUT_CHARS = 40,000`. The served context is 16,384 tokens and the prompt
takes a share; deed text measures ~3.55 characters per token on this corpus.

When a deed exceeds it, `fit_to_context` drops from the **middle** in a 3:2
head-to-tail split. A deed opens with the parties and closes with the schedule
and boundaries — where all sixteen fields live — while the middle is "WHEREAS"
recitals of prior title. The gap is marked rather than silent, and the trim is
logged and recorded on the extraction row.

### Generation settings

Temperature 0. **`repetition_penalty` is 1.0 — disabled, deliberately.** The
old API document recommended 1.1 to suppress runaway loops, and that is wrong for
this workload: a penalty suppresses the repeated key tokens that JSON array
elements share, so it truncates the party list. Measured at 1.1 the model emitted
3 of 5 persons and nulled `paid_in_cash`. Both `ExtractStage` and the server
default to 1.0. `max_tokens` is 2048 against a measured average of 664 for
legitimate output. `truncated: true`
means the ceiling was hit, which on this workload usually indicates a repetition
loop rather than a long answer — treated as a validation failure and routed to
review.

### A stage with no answer is a failure

The rule "the model answered, the answer is not trustworthy" is correct, but it
was once applied even when there was **no** answer, so a deed that returned
nothing three times finished with `extract_state=done`, zero persons and no
property row — recorded as a success and absent from every failed-document
report. DONE now requires that the model actually answered.

---

## 12. Translation

**`facebook/nllb-200-distilled-600M`** — runs locally, ~2.5 GB on disk, covers
all twelve languages in scope.

Chosen because it is **ungated**: IndicTrans2 scores better on Indic→English but
sits behind a HuggingFace licence gate, and a translation system that requires a
third-party account before it works is not an offline system in any useful sense.

### Proper nouns never touch the model

The most important correction in the design, made after measuring what NLLB
actually did to names:

| Source | NLLB output | |
|---|---|---|
| `ಲಕ್ಷ್ಮಿ ದೇವಿ` | "Goddess Lakshmi" | a person became a deity |
| `ವೆಂಕಟೇಶ್` | "What is Venkatesh?" | a name became a question |
| `ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ` | "Rural areas of Bangalore" | a district was paraphrased |

None of these is a mistranslation — NLLB is a *sentence* translator doing what it
was built for, and on Indian names the meaning is often a word. On a record
identifying parties to a property transfer it is a corrupted document.

Names, villages, districts and taluks are rendered by
`core/translation/transliterate.py` — rule-based, via IAST, using
`indic-transliteration`. **Deterministic; cannot invent.**

Script-specific rules, all tested: Devanagari, Gujarati, Bengali, Odia and
Gurmukhi carry an inherent final vowel that must be dropped (`रमेश` → "ramesha" →
**Ramesh**); Malayalam chillu letters must be expanded first; Tamil has no voiced
stops in its script, so `gh`/`bh` and word-initial voiced consonants are table
artefacts.

### Kannada writes English initials phonetically

`ಜಿ` is how you write **G**, `ಕೆ` is **K**, `ಎಂ` is **M**. Sounding them out gives
`Ji.ke. Raju` for what should be `G.K. Raju`. `read_initials()` reads them back in
all three positions real deeds use. **Position is the safety**: `ಬಿ` is the letter
B *and* the opening of `ಬಿಂದು`, so an adjoining full stop is required and ordinary
words are untouched. Measured: 0/50 → 45/50 (90%) of Kannada names with initials
now render as letters.

### Detection by script, not by classifier

Statistical detectors are unreliable below roughly twenty words; on a two-word
Kannada name they guess, and several guess differently between runs. Script
identity is a property of the characters themselves.

| Script | Range | Code |
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

Codes are FLORES-200, not ISO 639-1. Passing `kn` produces silent garbage.

**Hindi and Marathi share Devanagari** and no character inspection separates them.
The detector uses letters Hindi does not use (`ळ`, `ॲ`, `ऱ`) and vocabulary that
differs on exactly the words a deed repeats (`जिल्हा`/`जिला`, `तालुका`/`तहसील`,
`आणि`/`और`). Where a field carries no evidence it **does not guess** — it falls
back to the configured default and records `"no distinguishing evidence"`.

Digits, punctuation and symbols detect as `NEUTRAL` and are **never sent to the
translator**.

### Which fields get which treatment

| Group | Fields | Operation |
|---|---|---|
| Person | `name`, `father_name` | transliterate |
| Person | `address`, `gender`, `occupation` | translate |
| Property | `schedule_c_property_address`, `property_description` | translate |
| Property | `village`, `district`, `taluk` | transliterate |
| Document | `registration_office`, `document_type`, `sub_registrar_office` | translate |

**The distinction is load-bearing.** A name must come across by *sound*; an
address by *meaning*. Getting these the wrong way round produces output that
reads plausibly and is wrong in half the columns.

### Why translation runs after validation

Validation cross-checks every extracted value against the **OCR source**, which is
in the original language. Translate the name to "Ramesh Kumar" first and it can
never be found in Kannada OCR — every name would flag as unverified, and the layer
that catches quantisation errors on financial fields would stop working.

Results are written to `<field>_translated`, never over the original. The CSV
writer prefers the translated value and falls back to the source, so a partial
translation degrades to mixed output rather than losing Kannada a reviewer may
still need.

### Architecture

The model runs in a **subprocess**, in Surya's environment, because both need
torch and transformers. Loading it in-process would hold ~2.5 GB of VRAM for the
lifetime of the window.

Batching is **by source language** — NLLB sets the source through the tokenizer,
so a batch must be homogeneous. The cache is content-addressed on
`(text, source, target)`; deeds repeat the same village, district and
registration office across a whole batch.

`device = auto` selects CUDA only when enough VRAM is genuinely free.
`torch.cuda.is_available()` answers "is there a GPU", not "is there room on it".

---

## 13. Validation and business rules

`core/validation.py` — seven layers, cross-checking extracted values against the
OCR source. Confidence is rule-derived, not model-reported.

### Field formats

| Field | Rule |
|---|---|
| PAN | `^[A-Z]{5}[0-9]{4}[A-Z]$`, lowercase normalised |
| Aadhaar | 12 digits, stored as fixed-width text |
| Amounts | parsed with the decimal point respected |
| Dates | IST, calendar-aware |

Aadhaar and PAN are stored as text, never numeric — numeric coercion produced
`6.63E+11` in an early reference file.

### Disposition

A document is accepted, or routed to review. Extraction retry is unavailable by
instruction: failures route to review rather than being retried.

### Business rules in the export

**Transaction Amount is split per side.** ₹1,000 with 4 buyers and 2 sellers gives
each buyer ₹250 and each seller ₹500. It is **not** divided across the combined
count of buyers and sellers.

**Transaction Identity is read off the deed, not the filename.** The registration
number (`BGP-1-00275-2025-26`) is extracted from the OCR text. Candidates are
scored on evidence — a deed recites the chain of title, so it quotes the
registration numbers of *prior* documents, and taking the first match would put a
previous owner's number in the column. Scoring: newest financial year (+5.0),
registration label (+4.0), cited as prior (−5.0), older year (−3.0), repetition,
weak positional preference. Where two candidates cannot be separated the column is
left blank and the deed routed to review, because a wrong value is worse than a
missing one. The filename is used **only** when it is itself a valid registration
number, checked by the same pattern a text candidate must pass.

**Names carry one name.** `@` marks an alias on an Indian deed; the marker and
everything after it is removed, and the full form is logged against the document
so it stays recoverable.

**Duplicate parties are removed** by Aadhaar, then PAN, then name plus father's
name.

**Property Type** is one of seven codes read from the Schedule Property:
A/N/C/R/I/Z/X.

**Address Type** is 1–5, falling back to `5` (Unspecified) rather than assuming
`2` — a deed records where somebody lives without saying what the premises are
used for. A document with no parties leaves the whole person half blank.

**Identification Type** is a letter from the SFT/Form 61A set (A passport,
B elector ID, C PAN, D government/PSU ID, E driving licence, G UIDAI letter,
H NREGA card, Z other). PAN outranks Aadhaar. Neither present leaves the cell
empty rather than claiming `Z`, which would assert some other document was seen.
`Identification Number` is held blank by instruction — the identifiers reach the
report through the PAN and Aadhaar columns.

**City / Town** reads the property's own address and nothing else. Falling back to
the whole deed returned registration offices and post offices. Three address
shapes are read — kind after the name (`Bengaluru South Taluk`), kind before it
(`Dist: Belgaum`), and a bare city closing the address — ranked so a town named
outright beats the taluk administering it, which beats a district. A stop-list
rejects any candidate containing a structural word, which is what keeps the
weakest rule safe.

**Stamp Value** is a nullable passthrough; the formula is undefined.

**All structured data is exported in English**, whatever the deed's language.

---

## 14. CSV export — the 42 columns

`core/csv_export.py`. One row per person, with document-level fields repeated
across each party's row. The database stores this normalised; the exporter
denormalises.

Sellers first, then buyers — the order the reference report uses and the order a
deed reads in.

A document with no extracted parties still yields one row, so it is visible in the
export rather than silently absent.

### Excel safety

Values beginning `=`, `+`, `-` or `@` are defused, so a spreadsheet cannot execute
a cell as a formula.

### The failed-document export

A second download, 7 columns: Transaction Identity, Source Filename, Failed Stage,
Processing Status, Reason, Flags, Confidence.

### Untranslated columns are reported

`write_csv` reports any column still holding non-English text, and the export log
names them.

---

## 15. Database

PostgreSQL 17 + SQLAlchemy 2.0 + psycopg v3 + Alembic. **11 tables, 117 columns.**

DSN scheme is always `postgresql+psycopg://`. `postgresql://` alone resolves to
psycopg2, which is not installed, and the resulting ImportError is confusing
enough to be worth preventing outright — `normalise_dsn()` rewrites it.

### Tables

| Table | Holds |
|---|---|
| `users` | username, created_at — supplied at upload |
| `batches` | name, state, queue position, file count, total bytes, timestamps |
| `documents` | the resume unit: one row per PDF, per-stage state, attempt counters, transaction identity |
| `ocr_pages` | cleaned OCR text per page. **Transient** — purged after 30 days, never backed up |
| `extractions` | raw model output, parse result, PAN coverage, token counts, model name, quantisation, duration |
| `properties` | one row per document: address, state, consideration, fees, dates, registration office |
| `persons` | one row per party: relation (B/S), ordinal, name, father, gender, Aadhaar, PAN, address, translations |
| `validation_results` | flag code, field, detail, confidence — document-level or person-level |
| `failure_events` | what failed, where, and why |
| `settings` | key-value, mirrors the Settings page; `.env` overrides |
| `logs` | structured application log, written only when DEBUG is enabled |

**Images are never stored** — only filename, path and metadata.

`quantisation` is recorded per extraction because it affects accuracy: results
must be attributable to a specific precision.

### Indexes that matter

`(batch_id, overall_state)` drives dashboard counts.
`(batch_id, ocr_state, extract_state, translate_state)` drives the resume scan.

### Migrations

Six revisions under `src/migrations/versions/`. Alembic is already initialised —
**never run `alembic init`**, it would orphan the history.

```bash
alembic upgrade head          # apply
alembic upgrade head --sql    # emit SQL without a connection
```

ENUM types are dropped explicitly in `downgrade()`: PostgreSQL does not remove a
native ENUM when the table using it is dropped, and without those statements a
re-run of `upgrade` fails with "type already exists".

### Bring-up and verification

```bash
python src/tools/db_setup.py --check      # probe connection, report driver
python src/tools/db_setup.py --upgrade    # alembic upgrade head
python src/tools/db_setup.py --seed       # default settings rows
python src/tools/db_setup.py --verify     # full round-trip against real tables
python src/tools/db_setup.py              # all four, in order
```

`--verify` runs the repositories, the `FOR UPDATE SKIP LOCKED` claim, stage
ordering, crash recovery, idempotency and continuous commit for real, then deletes
its own data.

### Statement timeout

Passed as a libpq startup option, not as a `SET`. As a `SET` it was reverted by
the connection pool's `ROLLBACK`, so only the first query on each pooled
connection was bounded.

---

## 16. The desktop application

PySide6 window hosting a QWebEngine view of Pystache-rendered HTML. There is no
web server and no JavaScript framework.

### Ten screens

Dashboard, Upload PDFs, PDF Processing, Failed OCR, Data View, OCR Text
Extraction, Watermark Remover, Settings, Validation Rules, Help.

### How the page reaches the browser

Pages are served over a custom `app://` scheme rather than pushed in with
`setHtml`. That is not a preference — it is the only arrangement in which the
assets load at all. `setHtml` does not establish the base URL as a real origin,
so Chromium refuses every subresource request to the custom scheme and the
handler is never consulted: no stylesheet, no `app.js`, no bridge. The failure is
silent, because a request that is never made cannot fail.

Navigation replaces `#content` only, so the QWebChannel carrying the reply stays
alive. `.content` is the scroll container; `#content` is what navigation replaces.
They were once the same element, so anything placed beside the page content was
wiped on the first navigation.

### The bridge

30 slots, one string in and one string out, replying on a signal. Work runs off
the GUI thread except for the explicit `_GUI_THREAD` list, which is the set that
opens a dialog.

### Capability gating

`status.py` probes in the background and publishes what the UI may offer right
now. A degraded system shows a banner and disables the specific actions that
cannot work, with a `title` explaining why — rather than failing when pressed.

### PDF viewer

Chromium's own, which brings text selection, copy, search and zoom for free — and
renders the *searchable* PDF the pipeline prepared, so what the operator selects
is the same text the model extracted from.

---

## 17. The AI server API

**Base URL:** `http://127.0.0.1:8077` · JSON over HTTP/1.1 ·
`ThreadingHTTPServer`

Submission is asynchronous by default: `POST /extract` returns a job id
immediately so a 1000-file batch never blocks the caller. Request bodies are
capped at 8 MiB. Errors never leak a traceback.

```bash
python -m ai_server.server \
  --model "AI server/gguf/deeds-v6_7-Q4_K_M.gguf" \
  --model-dir "AI server/gemma4b-text" \
  --binary tools/llamacpp/llama-server.exe \
  --engine llamacpp --host 127.0.0.1 --port 8077
```

`--engine mock` runs without a GPU or model.

| Endpoint | Returns |
|---|---|
| `GET /health` | aggregate readiness — engine, pressure, workers, queue. Never raises. |
| `GET /hardware` | detected CPU, RAM, GPUs, disks, excluded adapters, warnings |
| `GET /profile` | selected quantisation, context, KV type, VRAM breakdown, the full fidelity ladder |
| `POST /extract` | `202` with a job id; `wait: true` returns `200` and the finished job |
| `POST /extract/batch` | `202` with job ids; per-document admission |
| `GET /jobs/<id>` | state, result, tokens, timings, `truncated` |
| `GET /jobs` | queue depth, workers, state counts |
| `POST /model` | load or release the weights, so OCR can have the GPU |
| `POST /shutdown` | drain, stop the engine, release VRAM |

`ready` is `false` when the engine is not loaded **or** the governor has stopped
admitting work. The dashboard's "AI Server ● Active" indicator is driven from it.

| Status | Meaning |
|---|---|
| 400 | `ocr_text` missing/empty, body not an object, or body > 8 MiB |
| 503 | Governor refused admission — includes `"retry": true`, a backpressure signal rather than a fault |
| 500 | Unexpected fault |

`/health` is logged at DEBUG on purpose: the shell polls it every few seconds, and
at INFO it buries everything else within a minute.

---

## 18. Configuration and environment variables

Settings live in the `settings` table and mirror the Settings page. Every key is
overridable by environment variable for a single run. `.env` at the project root
is read by every entry point.

**`.env` is never committed.** It is in `.gitignore` because it holds the database
credential. `system_setup.bat` writes it with a freshly generated 20-character
password and restricts it to the installing user.

### Database

| Variable | Default |
|---|---|
| `SALEDEED_DB_URL` | the complete DSN — wins outright |
| `SALEDEED_DB_HOST` | `localhost` |
| `SALEDEED_DB_PORT` | `5432` |
| `SALEDEED_DB_NAME` | `saledeed` |
| `SALEDEED_DB_USER` | `saledeed` |
| `SALEDEED_DB_PASSWORD` | — |

The parts exist so a target machine differing in one respect does not require a
full URL. The password is URL-quoted: a generated password may contain characters
that would otherwise terminate the URL early.

### Services and logging

| Variable | Effect |
|---|---|
| `SALEDEED_AI_URL` | AI server base URL (default `http://127.0.0.1:8077`) |
| `SALEDEED_LOG_CONSOLE` | `warning` — terminal shows warnings and errors only; `debug` — everything |
| `SALEDEED_DEBUG` | debug level plus a separate `saledeed.debug.log` and the database log handler |
| `SALEDEED_RETENTION` | enable the retention scheduler |
| `SALEDEED_LOAD_PDFS` | scale parameter for the load harness |

### Translation

| Setting | Default | Environment |
|---|---|---|
| `translation_enabled` | `true` | `SALEDEED_TRANSLATION` |
| `translation_target` | `eng_Latn` | `SALEDEED_TRANSLATION_TARGET` |
| `translation_source` | `auto` | — |
| `translation_devanagari_as` | `hin_Deva` | — |
| `translation_model` | `nllb-200-distilled-600M` | `SALEDEED_TRANSLATION_MODEL` |
| — | — | `SALEDEED_TRANSLATION_MODEL_DIR` |
| `translation_device` | `auto` | `SALEDEED_TRANSLATION_DEVICE` |
| `translation_batch_size` | `16` | — |
| `translation_timeout_s` | `600` | — |
| `translation_max_retries` | `1` | — |

---

## 19. Logging

Every line carries the time, the level, the module **and the function**, plus any
structured context the call attached:

```
[2026-08-04 14:41:11] INFO     saledeed.pipeline.runner._log_stage()   ocr ok in 139.15s  chars=24512 pages=14
[2026-08-04 14:41:41] WARNING  saledeed.pipeline.runner._log_stage()   extract failed: response contained no parseable JSON
[2026-08-04 14:40:22] INFO     saledeed.export.write_csv()             CSV written: 2 row(s)  bytes=1627 columns=42
```

Both to the terminal and to rotating files under `runtime/logs/` —
`saledeed.log`, `saledeed.ai.log` for the inference server, `ai_server.out.log`
for that process's raw output. 8 MB per file, five kept.

The AI server normally runs windowless, so its output goes to a file.
`launcher.py --verbose` follows it in the terminal; running
`python -m ai_server.server` directly logs to its own terminal.

**`extra={...}` keys must avoid the 23 reserved `LogRecord` attribute names.**
`extra={"name": ...}` raises `KeyError` from inside `logging.makeRecord` — an
export once crashed on precisely the condition the log line reported, and only
once logging was configured at that level, so it passed in isolation and failed in
the full suite. `TestLogExtrasAreSafe` walks the AST of every `extra={...}` in
`src/`.

---

## 20. Error handling

**An operator is never shown a Python exception type.** `core/failure_codes.py`
maps what actually went wrong to a sentence someone can act on, and strips
filesystem paths out of messages that reach the screen.

**Failure is per document, never per batch.** One bad PDF cannot kill a run.
Stages catch broadly and record the reason on the document.

**Retryable and terminal are distinguished.** A missing engine is an environment
problem — retrying that document will not help, so it is not retried. A timeout
is retryable, once.

**A file is examined only when something has already gone wrong.**
`pdf_validation` runs in the failure path, where "is the file itself broken?"
becomes worth answering, and it is never allowed to raise — an exception there
would turn one bad document into a dead batch.

**Translation never loses a deed.** Every failure path returns the original text
and records why. A blank cell is worse than a Kannada one: a reader can see
Kannada and act on it, but cannot see an absence.

---

## 21. Security

| | |
|---|---|
| Credentials | `.env` only, gitignored, generated per install, ACL-restricted to the installing user |
| Database password | never in source, never in `alembic.ini` (which ships a placeholder) |
| CSV injection | values beginning `=`, `+`, `-`, `@` are defused |
| Upload validation | type, size and count checked before anything is written |
| Exception text | escaped before it reaches the page; never rendered raw |
| Paths | stripped from operator-facing messages |
| Logs | never carry a password or a credential |
| Network | nothing outbound. Every model is local. |
| PDF viewer | Chromium's own, sandboxed; remote URL access and JS window-opening are disabled |

---

## 22. Testing

```
py -3.13 -m pytest -q                    1,636 tests
py -3.13 src/tools/ui_smoke.py           real webview, real channel, real app.js
py -3.13 src/tools/service_sweep.py      every service entry point against live data
py -3.13 launcher.py --check             13 preflight checks
```

**38 test files** under `tests/`. `pytest.ini` sets `testpaths = tests` and
excludes `corpus`, which holds sample deeds and reference outputs rather than
tests.

Markers: `unit` (nothing required), `integration` (live PostgreSQL), `gpu` (a
running AI server — skipped without one).

### What the corpus holds

| Path | Contents |
|---|---|
| `tests/corpus/OCR saledeeds/` | 50 real deeds as OCR text — the regression corpus |
| `tests/corpus/saledeeds/` | sample PDFs |
| `tests/corpus/test scripts/outputs/` | reference extraction outputs, read by `conftest.py`, `test_ocr_cleanup.py` and `tools/extraction_report.py` |
| `tests/corpus/test scripts/prompt.txt` | the prompt those reference outputs were produced with |

### The harnesses exist for a reason

Every UI defect found in this project so far lived in an integration seam and
passed the full unit suite while doing so. `ui_smoke.py` drives a real webview,
a real channel and the real `app.js`.

### Verification tools

```
py -3.13 src/tools/kannada_audit.py        which CSV columns still hold non-English
py -3.13 src/tools/translation_check.py    end to end against the real model
py -3.13 src/tools/identity_check.py       Transaction Identity accuracy over the corpus
py -3.13 src/tools/extraction_report.py    field coverage against reference outputs
py -3.13 src/tools/e2e_test.py             real PDFs through the whole pipeline
```

---

## 23. Troubleshooting

**The window opens but processing is greyed out.** The AI server is not healthy.
Check `runtime/logs/saledeed.ai.log`. Browsing and export still work — this is by
design.

**"only N characters across M pages — the PDF has no usable text layer."** The
document is a pure scan and the OCR engine resolved to `textlayer`. A real OCR
engine is required; check that Surya installed correctly (`launcher.py --check`
reports it).

**OCR is very slow.** Expected on a 4 GB card while the language model is
resident — Surya falls back to CPU, at roughly 2.9 min/page against 1.3 on GPU.
The runner now drains OCR per batch rather than per document to limit model
swaps.

**Extraction returns nothing on a long deed.** It exceeded the context. Since
R-051 the text is trimmed from the middle and the trim is recorded; if it still
returns nothing the stage is marked FAILED rather than DONE.

**Kannada still in the CSV.** Check the export log for
`export contains untranslated Kannada in N column(s)`, which names them, then run
`src/tools/kannada_audit.py`.

**Translation is slow.** It is on CPU. On a 4 GB card the language model holds
most of the VRAM, so this is expected and usually correct; the alternative is an
out-of-memory failure mid-batch.

**Marathi rendered as Hindi.** Expected. Set
`translation_devanagari_as = mar_Deva`.

**"no model weights in …".** The translation model is not installed, or the
download was interrupted. Run `tools/setup.py --install-translation`. A partial
download leaves `config.json` behind, so the directory existing is not evidence.

**`pick_files` timed out.** A dialog was opened from a worker thread. Add the slot
to `Bridge._GUI_THREAD`.

**The database is unreachable.** `launcher.py --check` names the cause —
service stopped, wrong credentials, or database absent — with the command that
fixes it.

**A batch will not start.** Four batches are already queued, or the licence
of the queue cap applies. The error message says which.

---

## 24. Backup and maintenance

`core/backup.py` — `pg_dump`, verify-before-purge, retention. The password goes
through the environment, never the command line, because argv is visible to other
processes.

`RetentionScheduler` is opt-in via `SALEDEED_RETENTION`, defers while a batch is
running, and stops on shutdown.

| Data | Policy |
|---|---|
| `ocr_pages` | purged after 30 days, never backed up |
| Logs | 8 MB per file, five kept |
| `runtime/` | disposable in full — delete it to reclaim space |
| Prior year | archived to backup, then purged |

`models/` is never touched by any cleanup path.

---

## 25. Performance, measured

On the development machine (RTX 3050, 4 GB VRAM, 7.4 GB RAM):

| Stage | Cost |
|---|---|
| OCR | 1.3 min/page on GPU, 2.9 min/page on CPU — **87% of wall time** |
| Extraction | ~14 s per deed |
| Interpreter start + torch import | 8.0 s per document |
| Surya model load | 9.6 s per document |
| Language-model swap out and back | ~10.5 s, now per batch rather than per document |

OCR dominates by two orders of magnitude. A 1000-PDF batch is days on this
hardware.

### UI and database, after optimisation

| | before | after |
|---|---|---|
| dashboard | 25.4 ms, 17 queries | 9.5 ms, 7 queries |
| PDF processing | 10.3 ms, 8 queries | 2.3 ms, 1 query |
| data view | 33.6 ms, 22 queries | 9.6 ms, 8 queries |
| status poll | 9.2 ms, 8 queries | 1.5 ms, 2 queries |
| poll load | 192 queries/min | 48 queries/min |

`start_up()` costs ~412 ms and is deferred to the first idle turn of the event
loop, after the window is painted.

### Two things deliberately not changed

**Surya batch size.** Measured on one 14-page deed: 4/16 = 134.7 s, 8/32 =
117.7 s (13% faster), 16/64 = CUDA out of memory. But 8/32 does not produce the
same text — 97.83% token similarity against 4/16. **A 13% speedup that silently
changes the recognised text of a deed is not a trade to make quietly.** The
measurement is recorded so it can be decided deliberately.

**Per-document Surya startup (17.6 s).** Eliminating it means batching several
PDFs per invocation, which is the largest remaining win (~13% of OCR wall time)
but changes the runner's one-document-at-a-time contract.

---

## 26. Known limitations

- **Hindi and Marathi** cannot be separated by script; the detector uses
  vocabulary evidence and falls back to a configured default rather than guessing.
- **Urdu names** fall back to the model, which is where the name defects
  originated. Verify Urdu party names against the source.
- **Place-name spelling varies between paths.** A transliterated village gives
  "Bengaluru"; the same name inside a translated address gives "Bangalore".
  Both are correct English; they are not identical strings.
- **Translation quality is unmeasured.** There is no reference set of translated
  deeds to score against.
- **Only Kannada has been tested on real documents.** The other eleven languages
  are covered by unit tests using real text in each script, but no genuine Hindi,
  Telugu or Tamil sale deed has been processed end to end.
- **`FOR UPDATE SKIP LOCKED` is PostgreSQL-only** and is not exercised by the
  SQLite verification path.
- **No full 100-PDF end-to-end run.** At the measured rate that is ~22 hours on
  this hardware; it belongs on a larger machine.
- **Windows 10 compatibility** has portability properties tested but no
  observation from a second machine.
- **Stamp Value semantics are undefined**, so the column is a nullable
  passthrough rather than a computed value.
- **`ResourceGovernor` is not passed to `BatchRunner`** by `app/services.py`, so
  `_lease()` returns a null lease and in-process arbitration does not happen. Each
  subprocess defers to the resident model instead, so the effect is limited, but
  the lease is currently dead wiring.
- **A plaintext credential file** exists at `models/AI server/bucket_cred.txt`
  from an earlier cloud-storage phase. `models/` is gitignored so it never
  entered version control, and nothing in the application reads it. It should be
  deleted and the credentials rotated.
