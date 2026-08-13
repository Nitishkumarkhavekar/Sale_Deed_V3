# Architecture

**Last updated:** 2026-07-31 17:15 IST

---

## Process topology

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

A crash in the inference runtime cannot take down the UI. The same HTTP contract
is served by vLLM on larger hardware, so the deploy target is a configuration
change.

**Why Surya is a transient process rather than a fourth server.** Its three
models hold ~3.2 GB, and Python cannot reliably hand that back to the driver —
only process exit does. A long-lived Surya server would amortise its ~30 s load
across documents, but on a 4 GB card it would also mean the language model never
gets the GPU. Worth revisiting on 8 GB hardware. See ADR-016.

**Why not FastAPI.** The AI server is stdlib `ThreadingHTTPServer`. It must start
fast and add no dependency capable of breaking the one thing it exists to do. The
launcher starts this server; the health contract is unchanged.

---

## Startup

`launcher.py` (or `Run Sale Deed AI.bat`) is the single entry point:

```
Project validation → Configuration → PostgreSQL → Migrations
  → Model verification → AI service → Health check → Window → Ready
```

Thirteen preflight checks run, each reporting `ok | warn | fail` with the command
that fixes it. **All of them run even after a failure**, so one pass surfaces
every problem rather than sending the user round a loop.

Two deviations from a naive reading of that order, both deliberate:

- **The health check waits for the endpoint, not the model.** Loading a 4-bit
  Gemma takes 30–60 s and the interface already opens against a loading server,
  showing LOADING and gating what needs inference.
- **A failed AI service does not abort the launch.** Browsing, review and CSV
  export need no inference. The window opens with those enabled and processing
  greyed out.

Shutdown uses a Windows Job Object with `KILL_ON_JOB_CLOSE`, because
`terminate()` on the AI server leaves `llama-server.exe` orphaned holding VRAM
and port 8077. See ADR-017.

---

## Module map

```
ai_server/
  hardware.py      CPU/RAM/GPU/VRAM/disk detection. NVIDIA-only, pinned by UUID.
  profiles.py      Model geometry -> VRAM budget -> fidelity ladder selection.
  resources.py     Runtime governor: pressure levels, dynamic concurrency,
                   GPU lease, memory trimming.
  server.py        Async job queue + HTTP API.
  engines/
    base.py        InferenceEngine contract, request/result types.
    llamacpp.py    llama-server supervision, OOM diagnosis, idle unload.
    mock.py        GPU-free stub for pipeline tests and CI.

core/
  validation.py    INFERENCE_PIPELINE layers 2-7, flag codes, confidence.
  ocr_cleanup.py   CRLF, page markers, Surya markup, LaTeX fractions.
  csv_export.py    42-column export; formula-injection defusing.
  watermark.py     Detection and lossless removal.
  backup.py        pg_dump, verify-before-purge, retention.
  logging_setup.py Structured logging, queue-backed database handler.
  db/              SQLAlchemy 2.0 models, repositories, engine.
  pipeline/        Stages and the resumable batch runner.

app/
  main.py          PySide6 shell, QWebEngineView, app:// scheme.
  services.py      Qt-free service layer, cache-backed rendering.
  status.py        Background probes, circuit breaker, capabilities.
  ui/              Bridge, renderer, 9 screens + 2 partials.

launcher/
  config.py        Root discovery, .env, resolved paths.
  steps.py         One function per requirement, independently testable.
  supervisor.py    Child processes, Job Object, health watch, restart.
  runner.py        The startup sequence and console output.

tools/
  repack_checkpoint.py   Lossless key flatten + vision strip.
  surya_runner.py        Surya OCR in its own interpreter (ADR-016).
  setup.py               One-command install for a new machine.
  db_setup.py            Migration, seeding, verification.
  llamacpp/              llama.cpp CUDA 12.4 binaries.
```

`src/ai_server/` holds serving **code**. `models/AI server/` (with a space) holds model
**weights**. Distinct, and unfortunately similar — noted to prevent confusion.

---

## Model pipeline

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

Every derived artifact is regenerable from the original. Deleting them loses
nothing but time. The original is never written to.

---

## Automatic hardware adaptation

Two layers answering two different questions.

**Startup — `profiles.py`: what configuration fits?**

Reads geometry from the checkpoint's own `config.json` (nothing hardcoded),
computes a VRAM budget, walks a fidelity ladder and returns the highest-precision
rung that fits with slack.

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
— the property that makes this model viable on 4 GB at all.

Scaling is automatic: on a 16 GB GPU the ladder reaches BF16, where quantisation
loss is exactly zero, with no code change.

**Runtime — `resources.py`: how much work should be in flight?**

Samples RAM, VRAM, CPU and disk every 3 s and publishes a `ConcurrencyPlan`.
Four pressure levels. Degradation is immediate; recovery requires clearing the
threshold by a 5% margin so pools do not oscillate.

| Pressure | Trigger (free RAM) | Effect |
|---|---|---|
| normal | > 25% | full concurrency |
| elevated | 15–25% | 50% |
| high | 8–15% | 25%, memory trim |
| critical | < 8% or disk < 5 GiB | stop admitting new work |

At critical, in-flight documents finish but nothing new is admitted — killing
work would lose a document, and pausing is safe because every stage is resumable.
Never drops below 1 worker.

---

## GPU arbitration

At 4 GB the OCR model, the 4B language model and the translation model **cannot
be co-resident**; whichever loads second fails. The governor detects VRAM below a
12 GiB co-residency threshold and serialises the three GPU stages behind an
exclusive lease.

On a 16 GB machine the lease is a no-op and the stages overlap freely. Same code.

---

## OOM prevention

Layered, with the strongest guarantee being architectural.

1. **llama.cpp preallocates.** Context and KV are sized at load and never grow.
   Once the server is up, VRAM is flat regardless of prompt length or request
   volume. A 33k-token deed cannot OOM mid-batch; it is rejected at admission.
   This is categorically better than HF `generate`, where cache and activation
   memory scale per request.
2. **8% headroom** held back from measured free VRAM.
3. **95% ceiling** on the remaining budget (ADR-004).
4. **Preflight** before load; stepwise degradation (context -> KV precision ->
   quantisation) on failure.
5. **OOM diagnosis** — `llamacpp.py` scans startup output for allocation-failure
   markers and raises `ModelOutOfMemoryError` carrying the shortfall, rather than
   surfacing an exit code.

---

## Device safety

CUDA cannot enumerate the AMD integrated GPU, so a CUDA build is structurally
incapable of using it. The real risk is a **Vulkan or OpenCL build**, which would
enumerate the Radeon and may select it silently — correct output at a fraction of
the speed, easy to miss.

Mitigations: pin `CUDA_VISIBLE_DEVICES` by **UUID** (indices reorder across driver
updates); clear `GGML_VK_VISIBLE_DEVICES` and `ONEAPI_DEVICE_SELECTOR`; and abort
startup if the log reports a forbidden backend.

---

## Text normalisation boundary

CRLF is normalised to LF in `AiServer.build_request` before any text reaches the
model. This is a measured correctness requirement, not hygiene — see ADR-005.
The OCR cleanup module must do the same at its own boundary.
