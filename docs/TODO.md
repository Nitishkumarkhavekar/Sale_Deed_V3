# TODO

**Last updated:** 2026-07-31 17:10 IST

Three states. Completed work moves to [CHANGELOG.md](CHANGELOG.md); only the
current session is listed here.

---


## Design change worth making

- [ ] **Let the AI server release VRAM on request.** On a 4 GiB card the
      language model and Surya cannot co-reside, so OCR now runs on the CPU
      whenever the model is loaded (R-035). That costs ~1.6 min per page:
      ~2.9 on CPU against ~1.3 on GPU, or roughly **22 extra minutes per
      14-page deed**.

      An unload/reload endpoint on the AI server would recover it. Reloading
      the model measures ~50 s, so the trade is 50 s against ~22 min - worth
      making, but it is a design change to the server's lifecycle, not a bug
      fix, and it needs care around a reload arriving while a job is in flight.

      Note that the free-VRAM heuristic both runners use cannot substitute:
      `torch.cuda.mem_get_info()` reports 3.2 GiB free on Windows while
      nvidia-smi shows 3062 MiB in use. WDDM over-promises and the allocation
      fails later. "Is the model loaded" is the only question with a reliable
      answer.

- [ ] **Pass a `ResourceGovernor` to `BatchRunner`.** `app/services.py` builds
      it without one, so `_lease()` returns `_NullLease()` and the in-process
      arbitration between OCR, extraction and translation never happens. Less
      urgent now that each subprocess defers to the resident model, but the
      lease is the right mechanism and it is currently dead wiring.

## Blocked — needs something I cannot supply

- [x] ~~**IndicTrans2 weights**~~ — **no longer needed.** Translation ships on
      NLLB-200-distilled-600M, which is ungated and covers the same FLORES-200
      codes (R-026). The IndicTrans2 runner was deleted in R-031 after its
      leftover discovery code was found to be disabling translation in the
      pipeline. The steps below are kept only if the model is ever revisited -
      note that doing so now means writing a new runner, not restoring one.
      Both variants are gated on HuggingFace (`gated=auto`), needing an account,
      a token, and licence acceptance. Steps:
      1. Create a token at huggingface.co → Settings → Access Tokens (read scope)
      2. Accept the licence at
         `huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M`
      3. `SuryaOCR/venv_new/Scripts/python.exe -m huggingface_hub.commands.huggingface_cli login`
      4. `snapshot_download` into `models/AI server/translator/`
      Discovery picks it up automatically - no code change needed.
- [ ] **Full 100-PDF end-to-end run** — not a code problem. At the measured
      1.3 min/page on this GPU (L-008), 100 ten-page deeds is ~22 hours. Belongs
      on the 8 GB deployment machine.
- [ ] **Windows 10 compatibility run** — only portability *properties* can be
      tested from one machine. Converting that to an observation needs a second
      machine.

## Accepted — not to be pursued

- [x] ~~Extraction retry on v6.7~~ — user instruction, 2026-07-31. Retry stays
      unavailable; failures route to review. Surfaced on the Validation page.

---

## In progress

Nothing.

---

## Pending — quality

- [ ] **Triage the 10 review documents** from the 50-deed BF16 run. Some is
      validator conservatism (narrative dates, Kannada fuzzy matching); the
      residue is unknown and should be separated.
- [ ] **`3-2025-26` extracted 5 of 7 parties.** Financials correct. PAN coverage
      cannot detect this, because the missing parties have no PAN — so a
      completeness signal is missing from the design, not just the data.
- [ ] Full 50-deed Q4 vs BF16 sweep (5 of 10 completed).
- [ ] **Re-measure Surya on 8 GB VRAM.** The 2.2x CPU→GPU gain is bounded by
      VRAM, not compute. The prediction that an 8 GB card widens it is untested.

## Pending — features

- [ ] Income Tax Department logo — omitted pending licence confirmation.
- [ ] Consider a phase-separated pipeline (all OCR, then all extraction) so a
      4 GB card can give Surya the GPU. Today `llama-server` holds 3.21 GiB for
      its whole lifetime, so Surya falls back to CPU whenever it is running.

## Pending — testing

All 19 categories are now addressed (347 tests). What remains is not coverage
but *strength* — three categories pass on weaker evidence than the label implies,
recorded in [TEST_REPORT.md](TEST_REPORT.md):

- [ ] **Load at 100–1000 PDFs** — harness is scale-parameterised
      (`SALEDEED_LOAD_PDFS`) and verified at 1000 for export. The full pipeline
      has not been run at that scale anywhere.
- [ ] **Compatibility** — portability properties only, one machine.
- [ ] **AI model validation** — deliberately outside the suite; measured
      separately and still only over 5 of 50 deeds.
- [ ] **Qt widget interaction** — templates and gating are covered; widget tests
      need a display and an event loop.

---

## Decisions awaiting user input

- [ ] **PAN split threshold** — spec says 30, `INFERENCE.md` says 25,
      `architecture&plan.txt` says 20. Currently configurable, defaulting to 25.
- [ ] **Remove `mock.py`?** Keeping it allows GPU-free CI; removing it guarantees
      no code path can produce non-model output. It is never selected
      automatically.
- [ ] Rotate the R2 credentials in `models/AI server/bucket_cred.txt` (still plaintext).
- [ ] **Delete the stale `models/SuryaOCR/venv/`?** It was built on another machine and
      its interpreter paths are dead. `venv_new/` replaces it. Left in place
      pending confirmation — it is several GB.

---

## Completed this session

- [x] **Surya OCR working** (R-009) — runner script, spatial layout
      reconstruction at the corpus's `TARGET_COLS = 110`, venv rebuilt on Python
      3.12.10, `transformers` pinned to 4.57.1, CUDA torch installed. Verified
      against the user's own `275_ocr.txt`.
- [x] **Application launcher** — `launcher.py`, `Run Sale Deed AI.bat`, and a
      four-module package. 13/13 preflight, AI server healthy in 4.5 s, zero
      orphaned processes after kill (Windows Job Object).
- [x] **Test suite 164 → 347** across all 19 requested categories.
- [x] **Six real defects found and fixed**, each with a regression test:
      R-010 statement_timeout, R-011 CSV formula injection, R-012 upload
      validation, R-013 unescaped exception text, R-014 LaTeX fractions,
      R-015 launcher interpreter selection.
- [x] **IndicTrans2 integration** — runner script sharing Surya's interpreter,
      transliterate/translate routing, NFC normalisation replacing the
      unbuildable IndicTransToolkit, auto-discovery, 27 tests. Waiting only on
      gated weights.
- [x] **Watermark page wired** — `AppService.watermark` no longer raises;
      browse/scan/remove/open/clear all work. Verified on a PDF carrying an
      annotation watermark: source byte-identical afterwards, `DUPLICATE COPY`
      gone, `SALE DEED` / PAN / survey number all preserved, reported `lossless`.
- [x] **Capability gating surfaced** — a banner in `base.mustache` plus per-button
      `disabled` and `title` on Start, Browse, Add Batch and Download CSV.
      Renders on all 8 pages when degraded, on none when healthy.
- [x] **Auto batch mode connected** — the Settings dropdown existed but the
      runner was hard-coded MANUAL and never read it. Verified against the live
      database: A completed, `auto cooldown 2s remaining` observed, B promoted
      after 5.6 s of a 6 s window.
- [x] **RetentionScheduler started** — opt-in via `SALEDEED_RETENTION`, defers
      while a batch runs, stopped on shutdown.
- [x] `/docs` synchronised.
