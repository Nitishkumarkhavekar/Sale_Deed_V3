# Test Report

**Last updated:** 2026-07-30 13:17 IST

Append-only. Each run records what was executed, the result, and what it does
**not** establish — the second matters as much as the first.

There is no automated test suite yet (`tests/` is not created). Everything below
was verified by direct execution and is recorded so the results are reproducible.

---

## Automated suite — 164 passing (2026-07-31)

```bash
python -m pytest tests/ -q
```

| Marker | Tests | Requires |
|---|---:|---|
| unit | 139 | nothing |
| **integration** | **25** | live PostgreSQL — **ran, not skipped** |
| regression | 5 | the 50-deed corpus |

| File | Tests | Covers |
|---|---:|---|
| `test_validation.py` | 86 | layers 2-7, flags, stamp value, disposition |
| `test_csv_export.py` | 30 | 42-column schema, row expansion, identifier safety |
| `test_repositories.py` | 25 | claiming, ordering, idempotency, cascades |
| `test_ocr_cleanup.py` | 23 | line endings, page markers, padding policy |

### What the regression tests encode

Each is a defect found by running real data, not an invented case:

- deed **07** — Q4_K_M returned `1500000` where the OCR reads `1,50,000`
- deed **1316** — a decimal point read as a thousands separator made a wrong
  registration fee report as grounded
- deed **1359** — a two-PAN document failed coverage forever on a metric artefact
- both genuine errors recorded in `gemma6.8 score.md` are asserted to still be
  caught

### A bug the tests found

`reg_fee_candidates("ನೋಂದಣಿ ಶುಲ್ಕ 200.00")` returned **20000**, not 200 — the
same decimal defect surviving in a second place, because `_digits()` strips the
point. Fixed with `parse_amount()`.

Three further failures were **incorrect test expectations**, corrected rather than
worked around: lowercase PAN is normalised per spec; one corpus file has a
duplicated object requiring `extract_json`; and cleanup does not always shrink a
document, since a page marker may be added.

### Not covered by any test file

`logging_setup`, `watermark` and `backup` were each verified by direct execution
but have no test file. That is a real gap, listed in PROJECT_STATUS.

---

## Coverage against the specified 16 categories

| Category | Status |
|---|---|
| Unit | not started |
| Integration | partial — manual, AI server end-to-end |
| End-to-end | partial — mock engine only |
| Regression | not started (corpus exists, harness not written) |
| Load | not started |
| Stress | partial — simulated pressure sweep |
| Performance | partial — conversion and quantisation timings |
| GPU memory | **not started — blocked, no GPU inference yet** |
| Database recovery | not started |
| Failure recovery | not started |
| Translation accuracy | not started |
| OCR accuracy | not started |
| LLM accuracy | **not started — the largest open risk** |
| Watermark removal | not started |
| UI | not started |
| Security | not started |

---

## Run 1 — Hardware detection · PASS

`python -m ai_server.hardware`

Correctly identified 8 physical / 16 logical cores (Ryzen 7 4800H), 7.4 GiB RAM,
170.1 GiB free on D:, RTX 3050 at 3.87 GiB free of 4.00, driver 566.07 / CUDA
12.7, compute capability sm_86, and the AMD Radeon listed as excluded.

Notable: `nvidia-smi` reports 0 MiB used and no processes, confirming the
integrated Radeon drives the display and the RTX 3050 is a pure compute device —
worth roughly 0.5–1.0 GiB of usable VRAM versus a display GPU.

**Does not establish:** that CUDA compute actually works. Detection reads
`nvidia-smi`; it does not run a kernel.

---

## Run 2 — Profile selection · PASS (after a defect was found and fixed)

`python -m ai_server.profiles "AI server/gemma4b-text"`

Correctly parsed geometry from the checkpoint's own config: 34 layers (5 full
attention, 29 sliding at 1024), 3.88 B text parameters, 262,208 vocab.

**Defect found:** the first implementation selected Q5_K_M at **99% of budget** —
within the 8% headroom rule but certain to OOM on any incidental allocation.
Fixed via a 95% ceiling and by reordering the search to prefer KV precision over
weight precision (ADR-004). Re-run selected Q4_K_M / q8_0 at **90%**.

Final selection: Q4_K_M, 24,576 context, q8_0 KV, 35 GPU layers, 8 threads,
3.21 of 3.56 GiB.

---

## Run 3 — Hysteresis and degradation · PASS

Simulated RAM sweep 40% -> 22% -> 12% -> 6% -> 12% -> 22% -> 28% -> 40% free.

| Free RAM | Pressure | Admitting | `pdf_render` workers |
|---:|---|---|---:|
| 40% | normal | yes | 7 |
| 22% | elevated | yes | 2 |
| 12% | high | yes | 1 |
| 6% | critical | **no** | 1 |
| 12% | high | yes | 1 |
| 22% | elevated | yes | 2 |
| 28% | **elevated** | yes | 2 |
| 40% | normal | yes | 7 |

The 28% row is the important one: it correctly **stayed** elevated rather than
returning to normal, because recovery requires clearing the 25% threshold by a 5%
margin. Confirms hysteresis prevents pool oscillation. Worker count never reached
zero.

---

## Run 4 — Checkpoint repack · PASS

`python src/tools/repack_checkpoint.py --dry-run` then without.

Dry run classified all 883 source keys: 444 text (kept and renamed), 439 vision
(dropped), **0 unexpected** — the layout was exactly as the tool expected.

Output verified independently: 444 tensors, 7.23 GiB, offsets contiguous with no
gaps, dtype byte-lengths consistent with declared shapes, and the expected keys
present (`model.embed_tokens.weight` `[262208, 2560]`, `model.norm.weight`,
`model.layers.0.mlp.up_proj.weight`). No legacy prefixes remain.

**Original integrity confirmed:** 8,600,283,312 bytes, 883 tensors, vision tower
present, original key naming, June 19 timestamps.

---

## Run 5 — GGUF conversion and quantisation · PASS

Conversion: 444 tensors -> 7.76 GB f16, 50 s at ~153 MB/s.
Quantisation: 7401.72 MiB -> **2368.31 MiB at 5.12 BPW**, 71 s.

Tensor type mix in the Q4_K_M output: 204 Q4_K, 35 Q6_K, 205 F32 — the standard
Q4_K_M recipe.

**Prediction accuracy:** the profile model estimated 2.30 GiB at 5.1 BPW. Actual
2.31 GiB at 5.12 BPW. The VRAM budget arithmetic is validated against
measurement, not merely plausible.

Metadata verified via `GGUFReader`: `general.architecture: gemma3`, 34 blocks,
2560 embedding, 8 heads / 4 KV heads, **`sliding_window: 1024`** (the interleaved
attention survived conversion — this is what keeps 24k context affordable),
262,208 vocab tokens, 444 tensors, chat template embedded.

---

## Run 6 — Tokenizer equivalence · PASS after a real defect was found

The most valuable test of the session.

**Short strings — 5/5 identical**, including Kannada script, PAN and Aadhaar:

| Input | Result |
|---|---|
| `Registration Fee Rs. 33,000/-` | identical |
| `PAN ALQPP8332F` | identical |
| `ನೋಂದಣಿ ಶುಲ್ಕ` | identical |
| `Aadhaar 2413 9130 5374` | identical |
| `SATISH V PATHAK` | identical |

**Full deed — initially FAILED.** `117.txt` (25,493 chars): HF 6,408 tokens,
GGUF 6,758. A 5.5% divergence that would have silently degraded extraction
accuracy.

Diagnosed exactly: the file contains 339 CR characters. GGUF over-produced 339 ×
`\r` (token 251, `<0x0D>`) and 22 × `\n`, and under-produced 11 × `\n\n` —
totalling the 350 difference. The HF tokenizer drops `\r` and merges `\r\n\r\n`.

After CRLF -> LF normalisation: **6,408 tokens on both sides, byte-identical.**

Also cleared a false alarm: `tokenizer.ggml.pre = granite-embed-multi-311m` in the
GGUF looked wrong but is benign — it is llama.cpp's identifier for a matching
pre-tokeniser regex signature. The token IDs prove equivalence.

**Consequence:** CRLF normalisation is a measured correctness requirement, now
enforced at the AI server boundary and mandated for the OCR cleanup module
(ADR-005).

---

## Run 7 — AI server end-to-end · PASS (mock engine)

`python -m ai_server.server --engine mock --port 8078`

| Check | Result |
|---|---|
| `GET /health` | ready, pressure normal, per-stage worker plan present |
| `POST /extract` (async) | `202` with job id |
| `GET /jobs/<id>` | reached `done`; 6,373 -> 244 tokens |
| Output validity | parsed as JSON; 1 buyer, 2 sellers |
| Identifier extraction | 3 PANs, all present in the source OCR |
| `POST /extract/batch` | 3 jobs accepted |
| `GET /jobs` | `{"queued_depth": 0, "states": {"done": 4}}` |
| `GET /profile` | Q4_K_M / 24576 / q8_0 / 22528 prompt capacity |
| `POST /shutdown` | graceful |

**Does not establish:** any extraction accuracy. The mock scrapes identifiers with
regexes rather than inferring them. It proves the plumbing — queue, polling,
batching, backpressure, GPU lease — not model quality.

---

## Run 8 — First real GPU inference · PASS

Blocker B-001 cleared: the CUDA runtime completed and
`llama-cli --list-devices` reports
`CUDA0: NVIDIA GeForce RTX 3050 Laptop GPU (4095 MiB, 3303 MiB free)`.

Two defects fixed to get here:

1. **`WinError 2` launching the subprocess.** Windows `CreateProcess` rejects a
   relative path containing forward slashes even when the file exists;
   `Path.is_file()` returned True and `Popen` still failed. Fixed by resolving the
   binary to an absolute path (`_resolve_binary`).
2. **`--flash-attn` requires a value** in b10184. Bare `--flash-attn` is rejected;
   corrected to `--flash-attn on`.

First extraction on `117.txt`: **6,805 prompt tokens -> 303 completion in
10.2 s**, valid JSON, Kannada names preserved, three PANs matching the source.
Materially faster than the 20-40 s/deed estimate.

---

## Run 9 — Repetition penalty defect · FIXED (ADR-011)

The first GPU output extracted only 3 of 5 persons and nulled `paid_in_cash` and
`registration_office`. Initially read as quantisation damage. It was not.

| | penalty 1.1 | penalty 1.0 | BF16 reference |
|---|---|---|---|
| Persons | **3** | **5** | 5 |
| `paid_in_cash` | `null` | `"no"` | `no` |
| `registration_office` | `null` | `YELBURGA` | ಯಲಬುರ್ಗಾ |
| Output tokens | 303 | 487 | — |

Cause: JSON array elements repeat the same key tokens, so a repetition penalty
suppresses list continuation. Default changed to 1.0. See ADR-011.

**This was a dangerous class of failure** — it produced well-formed but
*incomplete* JSON that every structural validator would pass. PAN coverage would
not have caught it either: the dropped parties had no PAN, so coverage was
3/3 = 1.0 and no retry would have fired.

---

## Run 10 — Q4_K_M versus BF16 on real deeds · PARTIAL (5 of 10)

Same 10-deed sample used by `verify_extraction.py`, comparing against the BF16
references in `tests/corpus/test scripts/outputs/vllm_ocr/`.

| Deed | Tokens | Time | Persons | vs BF16 |
|---|---:|---:|---:|---|
| 07 | 528 | 14.0 s | 6 | **consideration 1500000 != 150000** |
| 117 | 487 | 13.6 s | 5 | match |
| 1316 | 286 | 9.4 s | 2 | match |
| 1421HDG | 391 | 14.3 s | 3 | match |
| 510-22-23 | 626 | 21.6 s | 7 | match |
| 303, 760, 2231, 2717, 2785 | — | — | — | not run (503 backpressure) |

**Field-level agreement: 24/25 = 96%.** Average 14.6 s/deed.

### The one disagreement is a real quantisation error

Deed 07: Q4_K_M returned `sale_consideration: 1500000`; BF16 returned `150000`.

Checked against the source. The OCR contains
`Rs.1,50,000=00 (Rupees One Lakh Fifty thousand)`. Neither `1500000` nor
`15,00,000` occurs anywhere in the file. **BF16 is correct; Q4_K_M is wrong by a
factor of ten** on a financial field.

This is precisely the failure mode predicted for 4-bit weights — exact-digit
fields degrade more than prose. One 10x error on sale consideration in a
five-deed sample is not acceptable for production without a guard.

**The guard already exists in the design.** Layer 3 OCR-presence checking and
Layer 5 amount cross-checking would both catch this, because the extracted value
is absent from the source. It would be flagged `WSC` and routed to review rather
than exported. This raises the priority of `src/core/validation.py` from important to
load-bearing: on quantised weights it is the mechanism that makes the output
trustworthy.

### Why 5 deeds did not run

The governor returned `503` with `retry: true` — correct backpressure under `high`
to `critical` RAM pressure (the host has been running at 0.6-1.4 GiB free of
7.4 GiB). The mechanism worked as designed; it simply prevents a full sweep on
this machine while other applications are open. See limitation L-003.

---

## Run 11 — Validation engine · PASS after two self-inflicted bugs were found

`src/core/validation.py` implementing layers 2-7. Tested against ground truth rather
than against itself.

### Does it catch the quantisation error?

| Input | `in_ocr` | Flags | Disposition |
|---|---|---|---|
| Q4_K_M `sale_consideration: 1500000` (wrong) | **False** | `OCR_P WSC` | **review** |
| BF16 `sale_consideration: 150000` (correct) | True | `OCR_P` | **accept** |

Yes. The 10x error is caught and routed to review instead of exported.

### Two bugs found in my own validator

**Bug 1 - `amount_in_ocr` matched substrings.** Deed 1316's `reg_fee: 20000`
reported as grounded. Root cause: I included `.` in the separator class, so the
OCR text `ನೋಂದಣಿ ಶುಲ್ಕ 200.00` — where the real fee is **200** — satisfied the
query `20000` by treating the decimal point as a thousands separator. A second
normalised-text pass compounded it: stripping separators concatenates adjacent
numbers (`63,60,000 13,00,000` -> `63600001300000`), allowing matches across the
join.

This mattered because `gemma6.8 score.md` records 1316's reg_fee as a **genuine
error**, and my check was reporting it clean. For a validator whose entire purpose
is catching wrong digits, permissiveness is the worse failure.

Fixed: separators are commas and whitespace only; normalised fallback removed.

**Bug 2 - Layer 5 fired WSV too eagerly.** Deed 117's `52500` *is* present as
`52,500` and is exactly 1% of the 5,250,000 consideration — the standard fee — but
the regex sweep returned `20000` from elsewhere and the mismatch raised a flag.
`INFERENCE_PIPELINE.md` §4 itself notes the model may be right and the regex may
have caught a cess line. Layer 5 is now **advisory**: disagreement lowers
confidence; only absence from the OCR raises WSV.

### Ground-truth verification after the fixes

| Deed | Value | Result | Expected |
|---|---|---|---|
| 1316 | `20000` | not grounded | correct — real error |
| 1316 | `200` | grounded | correct — actual fee |
| 07 | `1500000` | not grounded | correct — Q4 error |
| 07 | `150000` | grounded | correct — BF16 value |
| 117 | `52500` | grounded | correct — comma grouping |

**Both genuine errors independently recorded in `gemma6.8 score.md` are now
caught**: deed 1316's registration fee (`WSV`) and deed 2231's seller Aadhaar
(`WAN`). The validator found them without being told they existed.

### Behaviour across all 50 BF16 reference outputs

| | |
|---|---|
| accept | 35 |
| review | 10 |
| retry | 5 |
| mean confidence | 0.940 |

Flag frequency: `OCR_P` 50, `PM` 41, `WAN` 6, `WSV` 2, `HPAN` 2, `PAF` 2,
`SCH` 1, `WTD` 1.

A 30% non-accept rate on BF16 output is higher than the ~1.5% genuine-error rate
the published grounding check reported, so some of these are conservative rather
than real. Two known contributors, both acceptable for now because they err
toward review rather than silent export:

- `date_in_ocr` only recognises numeric date forms. `INFERENCE.md` notes deeds
  also write dates narratively ("Twenty Second day of March 2022"), which will
  not match.
- Kannada name matching is token-fuzzy; the published check documented the same
  under-counting on non-contiguous clusters.

**Not yet measured:** whether the residual 10 reviews are genuine data problems
or validator conservatism. Requires manual inspection of those documents.

---

## Run 12 — CSV export · PASS

`src/core/csv_export.py`, verified against `example.csv` rather than against
assumptions.

| Check | Result |
|---|---|
| Column count and names vs reference | **all 42 identical, in order** |
| Per-person expansion | deed 117 -> 5 rows (1 buyer + 4 sellers) |
| Serial sharing | all rows of a deed share one `Report Serial Number` |
| Multi-document serials | 3 deeds -> 13 rows, serials 1/2/3 with 5/4/4 rows |
| Date conversion | `2025-04-01` -> `01-04-2025` |
| PIN extraction | `... Karnataka - 562123` -> `562123` |
| Kannada round-trip | `ಅಚಲ ಎಲ್ ನರಗುಂದ` preserved through write and re-read |
| Remarks col 15 | `OCR_P conf=0.97` |
| Remarks col 42 | `PM conf=1.00` |
| Failed export | written with stage, status, reason, flags |

### Two conventions taken from the reference, not assumed

**Dates are `DD-MM-YYYY`.** `example.csv` shows `19-07-2025`; the model emits ISO
`2025-07-19`. Exporting ISO would have produced a file the receiving system
misreads on every row — a silent, total failure. Converted at the boundary.

**`State Code` holds a state name** ("Karnataka"), not a code, and
`Country Code` / `Country` / `Nationality` hold `IN`.

### Aadhaar corruption

`example.csv` contains Aadhaar values as `6.63E+11` — a spreadsheet coerced the
12-digit strings to floats and the numbers are unrecoverable. The exporter writes
them as text:

| Mode | Aadhaar written |
|---|---|
| default | `241391305374` |
| `excel_safe=True` | `="241391305374"` |

Default is correct standard CSV. `excel_safe` wraps identifiers as Excel formula
strings so a double-click open cannot coerce them; it is off by default because a
correct file that Excel displays awkwardly beats a corrupted file that looks fine.

Encoding is `utf-8-sig`: the BOM is what makes Excel render Kannada rather than
mojibake, and standard readers ignore it.

---

## Not yet run — and why it matters

### Q4_K_M accuracy versus BF16 · BLOCKED on B-001

**The largest unquantified risk in the project.** Q4_K_M is lossy and the
published 96% / 98% figures were measured at BF16 on different hardware.
Exact-digit fields (12-digit Aadhaar, `AAAAA0000A` PAN) are the most exposed,
because quantisation noise damages exact-copy tasks more than prose.

The harness already exists in the repository: 50 OCR inputs in
`tests/corpus/OCR saledeeds/`, 50 BF16 reference outputs in
`tests/corpus/test scripts/outputs/vllm_ocr/`, and grounding logic in
`verify_extraction.py`. This is roughly an hour of compute once the CUDA runtime
lands, and it converts a guess into a number.

### First real GPU inference · BLOCKED on B-001

`llama-cli --list-devices` currently returns `Available devices: (none)` because
`cudart`/`cublas` DLLs are missing (download at 212/391 MB). **No inference has
run on the GPU at any point in this session.**

---

# Run 2026-07-31 (evening) — full 19-category sweep

**Result:** 347 passed, 8 skipped, 0 failed (355 collected)

*(Updated after R-015: two tests added for interpreter selection.)*
**Command:** `python -m pytest tests/ -q`

Suite grew 164 → 353 tests. Skips are environmental and each prints its reason:
6 need a running AI server (`gpu`), 2 need optional tooling.

## Coverage against the 19 requested categories

| # | Category | Status | Where |
|---|----------|--------|-------|
| 1 | Unit | **Done** | all files — 280 marked `unit` |
| 2 | Integration | **Done** | `test_integration.py` |
| 3 | End-to-End | **Partial** — stub extractor, model not in the loop | `test_integration.py` |
| 4 | Functional | **Done** | `test_validation.py`, `test_csv_export.py` |
| 5 | Performance | **Done** | `test_platform.py` |
| 6 | Load (100–1000) | **Harness only — not run at scale** | `test_platform.py` |
| 7 | Stress | **Done** | `test_platform.py` |
| 8 | Scalability | **Partial** — export only | `test_platform.py` |
| 9 | Database | **Done** | `test_database.py` |
| 10 | AI model validation | **Not in this suite** | measured separately |
| 11 | OCR | **Done** | `test_surya.py`, `test_watermark.py` |
| 12 | UI/UX | **Partial** — no Qt widget tests | `test_platform.py` |
| 13 | API | **Done** | `test_operations.py` |
| 14 | Security | **Done** | `test_security.py` |
| 15 | Installation | **Done** | `test_platform.py` |
| 16 | Compatibility | **Partial** — one machine only | `test_platform.py` |
| 17 | Regression | **Done** | every fixed defect has a test |
| 18 | Backup & recovery | **Done** | `test_operations.py` |
| 19 | Logging & monitoring | **Done** | `test_operations.py` |

| File | Tests |
|------|------:|
| `test_validation.py` | 86 |
| `test_database.py` | 36 |
| `test_security.py` | 36 |
| `test_platform.py` | 34 |
| `test_csv_export.py` | 30 |
| `test_operations.py` | 30 |
| `test_repositories.py` | 25 |
| `test_ocr_cleanup.py` | 23 |
| `test_surya.py` | 20 |
| `test_integration.py` | 19 |
| `test_watermark.py` | 14 |

Markers: `-m unit` (280), `-m integration` (59), `-m slow` (5), `-m gpu` (6).

## Defects found — five, all fixed, each with a regression test

### 1. Statement timeout absent on every pooled connection · `src/core/db/engine.py`

`SET statement_timeout` is transaction-scoped, and the pool issues `ROLLBACK` on
return (`reset_on_return` defaults to `rollback`), reverting it. Measured:

```
connect 1: 30s
connect 2: 0      <- unbounded
connect 3: 0
```

Only the first query on a connection was bounded, so the runaway query this
exists to prevent could still stall a batch indefinitely. Now passed as a libpq
startup option, applied by the server and unaffected by rollback. All three
connections now read `30s`.

### 2. CSV formula injection · `src/core/csv_export.py`

A party name is third-party data landing in a spreadsheet cell. These were
written unescaped: `=cmd|'/c calc'!A0`, `=HYPERLINK(...)`, `=1+1`, `+1+1`,
`@SUM(1:1)`. Excel evaluates a leading `=`, `+`, `-` or `@`, making a hostile
deed code execution on the clerk's machine. Defused with a leading apostrophe;
the 42-column comparison against `example.csv` still passes.

### 3. Upload accepted any file named `.pdf` · `src/app/services.py`

Extension-only check. A renamed executable passed and failed later in OCR, where
the error reads as a broken document rather than a rejected file. Now verifies
`%PDF-` in the first 1 KB.

### 4. Unescaped exception text in the dashboard · `src/app/services.py`

`{{{message}}}` renders raw HTML and the notice was built from `{exc}` —
exception text routinely contains a filename the user chose. A PDF named
`<img src=x onerror=...>.pdf` would have rendered as markup. Escaped at source.
The template check is now a reviewed allowlist of six fields, so a seventh fails
until justified.

### 5. LaTeX fraction corrupted property extents · `src/core/ocr_cleanup.py`

Surya writes 42½ guntas as `42\frac{1}{2}`; the substitution produced `421/2`,
which reads as **survey number 421/2** — a different value, silently. Now
`42 1/2`. A follow-on fix stopped the whitespace class eating the preceding
space (`bare\frac{3}{4}` → `bare3/4`).

## Not executed — stated so a green suite does not imply more than it proved

**Load at 100–1000 PDFs.** Defined, not run at scale. OCR measured 2.9 min/page
CPU and 1.3 min/page GPU here; a thousand ten-page deeds is days to weeks. The
harness reads `SALEDEED_LOAD_PDFS`, so the same tests run at any scale
(`SALEDEED_LOAD_PDFS=1000 pytest -m slow` — verified at 1000 for export, 5
passed). The **full pipeline** at that scale has not been run anywhere.

**AI model validation.** Excluded deliberately: minutes per document would make
the suite unrunnable on every change. Measured separately — Q4_K_M scored 24/25
fields against BF16, one 10× error on deed 07 caught by validation. Ten
documents remain untriaged.

**Compatibility across Windows versions and hardware.** Cannot be done from one
machine. What is tested is that no assumption would break elsewhere: no
unguarded absolute paths, no fixed drive letters, no assumption a GPU exists,
CRLF normalised. Strictly weaker than running on Windows 10.

**Qt widget interaction.** Templates, rendering and gating are covered; widget
tests need a display and an event loop. Window verified manually.

## Measured

| Measurement | Value |
|---|---|
| Full suite | 9.6 s |
| Launcher preflight | 13/13 pass |
| Launcher → AI server health | 4.5 s |
| Orphaned processes after kill | 0 |
| Surya OCR, 5 pages, CPU | 858.9 s |
| Surya OCR, 5 pages, GPU | 388.9 s (2.2×) |
| Cleanup, 50-page deed | < 2 s |
| CSV export, 1000 documents | < 10 s |
| Status snapshot (polled) | < 10 ms |

The 2.2× GPU speedup is below the 4–10× estimated beforehand. Cause is VRAM, not
the GPU: with ~3.2 GiB free the recognition model runs small batches and the work
is memory-bandwidth bound. On an 8 GB card the gap should widen — a prediction,
not a measurement. GPU output was also more accurate: 9 of 9 survey numbers
matched the reference against 8 of 9 on CPU.

## Gaps unchanged by this round

- ~~**IndicTrans2 translation absent**~~ — closed. Translation runs on NLLB-200
  and is live in the pipeline; the wiring that had silently disabled it was
  fixed in R-031.
- **No completed 100-PDF end-to-end run.**
- **10 documents from the BF16 run untriaged.**
