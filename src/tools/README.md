# `tools/` — scripts

Three unrelated kinds share this folder. Which kind you are looking at matters,
because one of them is a runtime dependency and the others are not.

## 1. Installers — run once per machine

| Module | Does |
|---|---|
| `system_setup.py` | detect, install what is missing, verify, launch. Behind `System Setup.bat`. |
| `setup.py` | dependencies, llama.cpp runtime, PostgreSQL, GGUF build, translation model |
| `db_setup.py` | `--check` · `--upgrade` · `--seed` · `--verify` |
| `repack_checkpoint.py` | losslessly repacks the trained checkpoint into a text-only model |

Every step is idempotent: detect, skip if present, install if not, verify
afterwards, and report which of those three happened. Nothing already on the
machine is replaced or removed.

## 2. Runners — **the application depends on these**

| Module | Does |
|---|---|
| `surya_runner.py` | OCR. Emits layout-reconstructed text plus per-line boxes. |
| `translate_runner.py` | NLLB-200 translation, batched by source language. |

Both are executed as subprocesses inside Surya's Python 3.12 environment, which
pins `transformers==4.57.1` against the rest of the project. They are never
imported.

> Located **by path, not by import**. Moving or renaming either means updating
> `core/pipeline/stages.py` and `core/translation/config.py`.

## 3. Verification harnesses — run by hand

| Module | Does |
|---|---|
| `ui_smoke.py` | real webview, real channel, real `app.js` |
| `service_sweep.py` | every service entry point against the live database |
| `e2e_test.py` | real PDFs → OCR → GPU extraction → validation → PostgreSQL |
| `translation_check.py` | end to end against the real model |
| `identity_check.py` | Transaction Identity accuracy against the corpus |
| `prepare_check.py` | document preparation, end to end |
| `kannada_audit.py` | which CSV columns can carry Kannada into the export |

These exist because unit tests cannot see integration seams. Every UI defect
found in this project so far passed the full suite while broken.

```
py -3.13 src/tools/ui_smoke.py
py -3.13 src/tools/service_sweep.py
```
