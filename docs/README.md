# Sale Deed AI

Offline processing of Indian sale deeds: read the PDF, remove separable
watermarks, OCR the Kannada, extract the fields with a fine-tuned Gemma-3-4B,
validate them, translate what needs translating, and export a CSV.

Nothing leaves the machine. There is no cloud API in the pipeline.

---

## Run it

```
Run Sale Deed AI.bat
```

That is the whole command. It starts PostgreSQL if stopped, applies migrations,
verifies the model, starts the AI server, waits for it to become healthy, and
opens the window. On shutdown it stops what it started.

On a machine that has never run this before:

```
System Setup.bat
```

Detects what is already installed, installs only what is missing, and never
replaces or removes anything that was already there. Safe to run twice.

Other entry points:

| Command | Does |
|---|---|
| `py -3.13 launcher.py --check` | validate everything, change nothing |
| `py -3.13 launcher.py --no-ai` | browsing and export, no inference |
| `py -3.13 launcher.py --headless` | services without the window |
| `System Setup.bat --report-only` | detect and report, change nothing |

---

## The stack

| Layer | What is used |
|---|---|
| Desktop shell | PySide6 + QWebEngine + Pystache |
| HTTP | stdlib `ThreadingHTTPServer` |
| Database | PostgreSQL 17 + SQLAlchemy 2 + psycopg 3 + Alembic |
| Inference | `llama-server.exe` + fine-tuned Gemma-3-4B GGUF |
| OCR | Surya OCR, in its own Python 3.12 environment |
| PDF | PyMuPDF |
| Translation | NLLB-200-distilled-600M + rule-based transliteration |

Deliberately **not** used: FastAPI, Pydantic, NodeJS, React, Tesseract, Poppler,
Ghostscript, vLLM. Each was considered and rejected for a recorded reason - see
`docs/DECISIONS.md` before adding any of them.

The extraction model is fine-tuned and specific to this project. It is
**verified, never downloaded**: an installer that helpfully fetched "a Gemma 3
4B" would replace the weights every accuracy figure here was measured against,
and nothing downstream would notice.

---

## Layout

Five folders, grouped by how the contents behave rather than by what they are:

```
src/          all the code
  core/         domain logic - pipeline, extraction, validation, export,
                watermark removal, translation. No UI, no HTTP.
  app/          desktop shell: window, web channel bridge, templates, services
  ai_server/    inference HTTP service, hardware detection, profile selection
  launcher/     startup orchestration, preflight checks, process supervision
  tools/        operational scripts - setup, OCR and translation runners,
                verification harnesses
  migrations/   Alembic revisions

models/       36 GB of weights and vendor installs. Large, static, never
              generated. Delete nothing here.
  AI server/    the fine-tuned GGUF, the translation model
  SuryaOCR/     the Surya installation and its Python 3.12 environment
  saledeed main/  the finetuning prompt and reference documents

runtime/      everything written while running - logs, uploads, exports,
              cleaned PDFs, caches, backups. Disposable: delete it to reclaim
              space and the application rebuilds what it needs.

tests/        the suite (742 tests) and corpus/ - sample deeds, reference OCR,
              and the research scripts that produced the finetuning corpus

docs/         architecture, decisions, known issues, test report, changelog
```

Root holds only entry points: `launcher.py`, the two `.bat` files,
`requirements.txt`, `alembic.ini`, `pytest.ini`, this file.

The split that matters is **`models/` against `runtime/`**: one is 36 GB that
must never be touched by a cleanup script, the other is disposable by design.
Anything unsure which it is belongs in `models/`.

[src/core/paths.py](src/core/paths.py) owns every filesystem location. Modules
ask it rather than computing `parents[N]` for themselves, so moving a folder is
one line there instead of twenty scattered ones - which is exactly how the
previous layout made itself expensive to change.

Two Python versions are required and this is intentional: **3.13** runs the
application, **3.12** runs Surya, which pins `transformers==4.57.1` against
everything else. Interpreters are selected by *capability* (`import PySide6`),
never by version number - installing the second Python changes what a bare
`python` resolves to, and choosing by version silently picks the wrong one.

---

## Documentation

| Document | Read it for |
|---|---|
| [docs/CODE_MAP.md](docs/CODE_MAP.md) | what every package and module in `src/` does |
| `docs/ARCHITECTURE.md` | how the pieces fit and why |
| `docs/DECISIONS.md` | choices already made and reversed - read before changing direction |
| `docs/KNOWN_ISSUES.md` | open blockers, limitations, and every resolved defect with its cause |
| `docs/DATABASE_SCHEMA.md` | tables, enums, migrations |
| `docs/API_DOCUMENTATION.md` | the AI server's endpoints |
| `docs/TRANSLATION.md` | languages, detection, transliteration, caching |
| `docs/TEST_REPORT.md` | what is tested, what is measured, what is not |
| `docs/TODO.md` | what is left |
| `docs/CLEANUP_REPORT.md` | the 2026-08-03 audit and what it found |

---

## Verifying a change

```
py -3.13 -m pytest -q            742 tests
py -3.13 src/tools/ui_smoke.py        real webview, real channel, real app.js
py -3.13 src/tools/service_sweep.py   every service entry point against live data
py -3.13 launcher.py --check      13 preflight checks
```

The last three exist because the unit tests cannot see integration seams. Every
UI defect found in this project so far lived in one of those seams and passed
the full suite while doing so.

---


## Logs

Every line carries the time, the level, the module **and the function**, plus
any structured context the call attached:

```
[2026-08-04 14:41:11] INFO     saledeed.pipeline.runner._log_stage()   ocr ok in 139.15s  chars=24512 pages=14
[2026-08-04 14:41:41] WARNING  saledeed.pipeline.runner._log_stage()   extract failed: response contained no parseable JSON
[2026-08-04 14:40:22] INFO     saledeed.export.write_csv()             CSV written: 2 row(s)  bytes=1627 columns=42
```

**Where they go.** Both to the terminal and to rotating files under
`runtime/logs/` - `saledeed.log` for the application, `saledeed.ai.log` for the
inference server, `ai_server.out.log` for that process's raw output. 8 MB per
file, five kept.

**Seeing the AI server's logs.** It normally runs windowless, so its output goes
to a file rather than a console:

```
py -3.13 launcher.py --verbose      follow the AI server's log in this terminal
py -3.13 -m ai_server.server ...    run it directly; it logs to its own terminal
```

**Turning the volume up or down.**

| | Effect |
|---|---|
| `SALEDEED_LOG_CONSOLE=warning` | terminal shows warnings and errors only |
| `SALEDEED_LOG_CONSOLE=debug` | everything, including per-request detail |
| `SALEDEED_DEBUG=true` | debug level plus a separate `saledeed.debug.log` |

`/health` is logged at DEBUG on purpose: the desktop shell polls it every few
seconds, and at INFO it buries everything else within a minute.

---

## Performance, measured

On the development machine (RTX 3050, 4 GB VRAM, 7.4 GB RAM):

| Stage | Cost |
|---|---|
| OCR | 1.3 min/page on GPU, 2.9 min/page on CPU |
| Extraction | ~14 s per deed |

OCR dominates by two orders of magnitude. A 1000-PDF batch is days on this
hardware. No throughput figure is published here that has not been measured.
