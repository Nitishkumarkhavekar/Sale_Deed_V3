# Code Map — what lives in `src/`, and why

62 modules, 18,462 lines, six packages. This document exists so a reader can find
the right file without opening ten wrong ones first.

**Rule of thumb:** a flat `.py` file is one concern. A folder is a subsystem that
grew past one file. If you are unsure where something belongs, ask which package
is allowed to import which — that answers it.

---

## The six packages

| Package | Modules | Lines | Role |
|---|---:|---:|---|
| [`core/`](../src/core) | 23 | 7,597 | Domain logic. No UI, no HTTP, no Qt. |
| [`ai_server/`](../src/ai_server) | 10 | 3,154 | The inference service. Runs as its own process. |
| [`app/`](../src/app) | 7 | 2,757 | Desktop shell — window, bridge, screens. |
| [`tools/`](../src/tools) | 13 | 3,488 | Scripts: installers, subprocess runners, verification. |
| [`launcher/`](../src/launcher) | 6 | 1,121 | Starts, supervises and stops everything. |
| [`migrations/`](../src/migrations) | 3 | 345 | Alembic revisions. |

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
talks to the inference service over HTTP rather than importing it, so this
process never links CUDA.

---

## `core/` — the domain

Everything the application *means*, independent of how it is displayed or served.

### Subsystems

**[`pipeline/`](../src/core/pipeline)** — how a deed becomes rows

| Module | Lines | Does |
|---|---:|---|
| `stages.py` | 763 | The four stages: OCR → extract → validate → translate. Document logic, **no database** — each returns a `StageOutcome`. |
| `runner.py` | 724 | `BatchRunner`: claims work, drives the stages, writes results, holds the GPU lease, makes scanned pages searchable. |

**[`db/`](../src/core/db)** — persistence

| Module | Lines | Does |
|---|---:|---|
| `models.py` | 433 | SQLAlchemy 2.0 declarative models and enums. |
| `repositories.py` | 692 | Repository pattern and `UnitOfWork`. Every query lives here. |
| `engine.py` | 224 | Engine, `session_scope`, connection checks, statement timeout. |

**[`translation/`](../src/core/translation)** — twelve languages, offline

| Module | Lines | Does |
|---|---:|---|
| `service.py` | 357 | The one place translation happens: cache, batching, subprocess, retry. |
| `detect.py` | 214 | Language detection **by script range**, not by a statistical classifier. |
| `transliterate.py` | 205 | Proper nouns rendered **by rule** — a sentence translator turns `ಲಕ್ಷ್ಮಿ ದೇವಿ` into "Goddess Lakshmi". |
| `config.py` | 186 | Model, device and interpreter selection. |
| `postprocess.py` | 109 | Repairs what a sentence translator does to a two-word fragment. |

### Single concerns

| Module | Lines | Does |
|---|---:|---|
| `validation.py` | 809 | Field rules — PAN, Aadhaar, amounts, dates, Stamp Value, and the pass/review disposition. |
| `watermark.py` | 500 | Detection and **lossless** removal. Refuses to inpaint a scan. |
| `csv_export.py` | 438 | The 42-column export, including formula-injection defusing. |
| `pdf_prepare.py` | 390 | The cleaned document, plus the invisible text layer that makes a scan selectable. |
| `ocr_cleanup.py` | 397 | Normalises raw OCR into what the model was finetuned on. |
| `backup.py` | 364 | Backup, archival and retention. |
| `logging_setup.py` | 331 | Centralised structured logging. |
| `transaction_id.py` | 264 | Reads `BGP-1-00275-2025-26` off page 1 — the registration number, not the filename. |
| `paths.py` | 75 | **Every filesystem location.** Ask this module; never compute `parents[N]`. |

---

## `ai_server/` — inference

A separate process on purpose: it links CUDA and holds ~3.2 GiB of weights, and
the desktop shell must be able to start, run and exit without either.

| Module | Lines | Does |
|---|---:|---|
| `server.py` | 542 | The HTTP endpoints. Standard-library `ThreadingHTTPServer` — no FastAPI. |
| `profiles.py` | 504 | Picks quantisation, context length and KV cache from the hardware. |
| `hardware.py` | 462 | CPU, RAM, GPU, VRAM, disk. Ignores the integrated AMD GPU by design. |
| `resources.py` | 453 | `ResourceGovernor` — enforces exclusive GPU access below 12 GiB, so OCR and the LLM never co-reside. |
| `deployment.py` | 358 | What this machine can actually run, and what it must refuse. |
| [`engines/`](../src/ai_server/engines) | 824 | `base.py` the interface · `llamacpp.py` production · `mock.py` deterministic stub for tests. |

---

## `app/` — the desktop shell

PySide6 window hosting a QWebEngine view of Pystache-rendered HTML. There is no
web server and no JavaScript framework.

| Module | Lines | Does |
|---|---:|---|
| `services.py` | 1,319 | Every screen and every action. The largest file in the project. |
| `status.py` | 438 | Background probing and capability gating — what the UI may offer right now. |
| `main.py` | 344 | Window, custom URL scheme, the handler that serves assets and PDFs. |
| `ui/bridge.py` | 328 | QWebChannel slots. **The only path** between the webview and Python. |
| `ui/renderer.py` | 328 | Pystache rendering and view models. |
| `ui/templates/` | — | Logic-less Mustache templates, one per screen. |
| `ui/assets/app.js` | — | The channel client. The project's only JavaScript file. |

Two rules this package lives by, both learned from defects:

- Widgets touch **only** the GUI thread. A `QFileDialog` on a worker thread
  aborts the process natively, with no traceback.
- Slots are `(QString, QString)` and reply on a signal. A trailing JavaScript
  function is stripped by QWebChannel as the reply handler, so a slot with the
  wrong arity is never found and the error names the *slot*, not the cause.

---

## `tools/` — scripts

Three unrelated kinds share this folder. Knowing which kind you are looking at
matters, because one of them is a runtime dependency and the others are not.

### Installers — run once per machine

| Module | Lines | Does |
|---|---:|---|
| `system_setup.py` | 815 | Detect, install what is missing, verify, launch. Behind `System Setup.bat`. |
| `setup.py` | 539 | Dependencies, llama.cpp runtime, PostgreSQL, GGUF build, translation model. |
| `db_setup.py` | 370 | `--check / --upgrade / --seed / --verify`. |
| `repack_checkpoint.py` | 336 | Losslessly repacks the trained checkpoint into a text-only model. |

### Runners — **the application depends on these**

Executed as subprocesses, inside Surya's Python 3.12 environment, because they
pin `transformers==4.57.1` against the rest of the project.

| Module | Lines | Does |
|---|---:|---|
| `surya_runner.py` | 274 | OCR. Emits layout-reconstructed text and per-line boxes. |
| `translate_runner.py` | 222 | NLLB-200 translation, batched by source language. |

> Do not move or rename these two without updating `core/pipeline/stages.py` and
> `core/translation/config.py`. They are located by path, not by import.

### Verification harnesses — run by hand

These exist because the unit tests cannot see integration seams. Every UI defect
found in this project so far passed the full suite.

| Module | Lines | Does |
|---|---:|---|
| `ui_smoke.py` | 155 | Real webview, real channel, real `app.js`. Catches what source-level tests cannot. |
| `service_sweep.py` | 84 | Every service entry point against the live database. |
| `e2e_test.py` | 214 | Real PDFs → OCR → GPU extraction → validation → PostgreSQL. |
| `translation_check.py` | 140 | End-to-end against the real model. |
| `identity_check.py` | 132 | Transaction Identity accuracy against the corpus. |
| `prepare_check.py` | 118 | Document preparation, end to end. |
| `kannada_audit.py` | 89 | Which CSV columns can carry Kannada into the export. |

---

## `launcher/` — startup

| Module | Lines | Does |
|---|---:|---|
| `steps.py` | 376 | The 13 preflight checks, each independently testable. |
| `supervisor.py` | 300 | Child processes: start, health-watch, restart, clean shutdown via a Windows Job Object. |
| `runner.py` | 265 | The sequence, the console output, the log. |
| `config.py` | 150 | Project-root discovery and `.env` loading. |
| `__main__.py` | 10 | Allows `python -m launcher`. |

`launcher.py` at the project root is a four-line shim: it puts `src/` on the
path and calls `launcher.runner.main`.

---

## `migrations/` — Alembic

`env.py` plus one file per revision. Alembic is already initialised and has
applied revisions — **never run `alembic init`**, it would orphan the history.
`alembic.ini` at the project root points here.

---

## Where to add something new

| If it… | Put it in |
|---|---|
| decides something about a deed | `core/` |
| is a new pipeline step | `core/pipeline/stages.py` |
| is a database query | `core/db/repositories.py` — nowhere else |
| is a new screen or button | `app/services.py` + a template |
| needs the GPU or the model | `ai_server/` |
| is a path | `core/paths.py` — never `parents[N]` again |
| is a one-off script | `tools/`, and say which of the three kinds it is |
| runs before the window opens | `launcher/steps.py` |
