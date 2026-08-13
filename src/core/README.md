# `core/` — the domain

Everything the application *means*, independent of how it is displayed or
served. Nothing here imports Qt, HTTP, or the inference server — which is what
lets the whole domain be tested without a window, a GPU or a running service.

## Subsystems

A folder means the concern grew past one file.

| Folder | Contains |
|---|---|
| `pipeline/` | `stages.py` — document logic with **no database**, each stage returning a `StageOutcome`. `runner.py` — the orchestration that does have one: claims work, drives the stages, writes results, holds the GPU lease. |
| `db/` | `models.py` tables and enums · `engine.py` connections and `session_scope` · `repositories.py` the `UnitOfWork` and **every query**. |
| `translation/` | `detect.py` language by script range · `transliterate.py` proper nouns by rule · `service.py` the one place translation happens · `config.py` · `postprocess.py`. |

## Single concerns

`validation.py` · `watermark.py` · `csv_export.py` · `pdf_prepare.py` ·
`ocr_cleanup.py` · `backup.py` · `logging_setup.py` · `transaction_id.py` ·
`paths.py`

## Standing rules

Each of these is here because breaking it caused a real defect.

- **Every query lives in `repositories.py`.** SQL anywhere else is a defect.
- **Never drop a value because it is not English.** Report it instead — that is
  what `csv_export.untranslated_cells()` is for.
- **Never invent document content.** `watermark.remove` refuses to inpaint a
  scan. A plausible guess on a legal instrument is worse than a visible mark,
  because the mark is obviously a mark and the guess is not.
- **Proper nouns are transliterated, never translated.** A sentence translator
  turns `ಲಕ್ಷ್ಮಿ ದೇವಿ` into "Goddess Lakshmi" and a name into a question.
- **Ask `paths.py` for locations.** Never recompute the project root.
