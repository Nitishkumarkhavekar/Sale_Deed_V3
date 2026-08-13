# AI Server API

**Last updated:** 2026-07-30 13:17 IST
**Base URL:** `http://127.0.0.1:8077`
**Transport:** JSON over HTTP/1.1, `http.server.ThreadingHTTPServer` (ADR-006)

Maintained by hand — there is no OpenAPI generator, by design.

Submission is **asynchronous by default**: `POST /extract` returns a job id
immediately so a 1000-file batch never blocks the caller. The UI polls
`GET /jobs/<id>`.

Request bodies are capped at 8 MiB. All responses are JSON; errors never leak a
traceback.

---

## Starting the server

```bash
python -m ai_server.server \
  --model "AI server/gguf/deeds-v6_7-Q4_K_M.gguf" \
  --model-dir "AI server/gemma4b-text" \
  --binary tools/llamacpp/llama-server.exe \
  --engine llamacpp --host 127.0.0.1 --port 8077
```

`--engine mock` runs without a GPU or model, for pipeline tests and CI.

---

## GET /health

Aggregate readiness. Never raises — it is the diagnostic of last resort.

```json
{
  "ready": true,
  "uptime_s": 4.1,
  "engine": {
    "ready": true, "engine": "llamacpp", "model": "deeds-v6_7-Q4_K_M.gguf",
    "detail": "ready", "device": "cuda", "loaded": true,
    "vram_used_bytes": 3448143872, "vram_total_bytes": 3822059520,
    "requests_served": 12, "idle_seconds": 3.2
  },
  "pressure": "normal",
  "admitting_work": true,
  "gpu_exclusive": true,
  "gpu_holder": null,
  "workers": {"pdf_render": 6, "extract": 1, "ocr": 1, "translate": 1,
              "validate": 7, "ocr_postprocess": 7, "translate_post": 7, "export": 7},
  "resources": {"ram_available_bytes": 0, "ram_total_bytes": 0, "cpu_busy": 0.07,
                "vram_free_bytes": 0, "vram_total_bytes": 0, "disk_free_bytes": 0},
  "queue": {"queued_depth": 0, "workers": 1, "states": {"done": 12}}
}
```

`ready` is `false` when the engine is not loaded **or** the governor has stopped
admitting work. Drive the dashboard's "AI Server ● Active" indicator from this.

---

## GET /hardware

Detected hardware. `excluded_adapters` lists non-NVIDIA GPUs that are never used
for inference.

```json
{
  "os": "Windows 11",
  "cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD",
  "physical_cores": 8, "logical_cores": 16,
  "ram_total_bytes": 7943000064,
  "cuda_available": true, "driver_version": "566.07", "cuda_version": "12.7",
  "gpus": [{"index": 0, "uuid": "GPU-7f5cf8bf-...", "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "total_bytes": 4294967296, "free_bytes": 4155703296, "compute_capability": "8.6"}],
  "excluded_adapters": ["AMD Radeon(TM) Graphics"],
  "disks": [{"path": "D:", "free_bytes": 182661734400, "total_bytes": 209715200000}],
  "warnings": ["Non-NVIDIA adapter present. Use a CUDA build of the runtime: ..."]
}
```

---

## GET /profile

Selected inference configuration, the VRAM breakdown, and the full fidelity
ladder — the "why not a better quantisation?" answer.

```json
{
  "model_dir": "AI server/gemma4b-text",
  "device": "cuda",
  "quantisation": "Q4_K_M",
  "lossless": false,
  "quant_note": "measurable loss; verify on the deed corpus",
  "n_ctx": 24576,
  "prompt_capacity_tokens": 22528,
  "kv_type": "q8_0",
  "n_gpu_layers": 35,
  "n_parallel": 1,
  "n_threads": 8,
  "vram": {"weights_bytes": 2469606195, "kv_bytes": 341787340,
           "overhead_bytes": 629145600, "total_bytes": 3440539135,
           "budget_bytes": 3822059520},
  "reason": "highest-fidelity rung fitting 3.56 GiB (8% headroom held back) at 24,576-token context",
  "warnings": ["Q4_K_M is lossy relative to your trained weights - measure against the BF16 baseline before production use"],
  "ladder": "Fidelity ladder at 24,576 context ..."
}
```

Surface `warnings` in the Settings page. `lossless: false` means outputs are not
guaranteed identical to the trained model.

---

## POST /extract

Submit one deed.

**Request**

| Field | Type | Default | Notes |
|---|---|---|---|
| `ocr_text` | string | **required** | CRLF is normalised to LF server-side |
| `prompt` | string | `prompt_v6_short.txt` | instruction block |
| `document_id` | string | `""` | echoed back for correlation |
| `max_tokens` | int | 2048 | legitimate outputs average ~664 |
| `temperature` | float | 0.0 | must stay 0 for extraction |
| `repetition_penalty` | float | 1.1 | suppresses runaway loops |
| `grammar` | string | null | GBNF; forces schema-valid JSON |
| `stop` | string[] | `[]` | |
| `wait` | bool | false | block until finished |
| `timeout_s` | float | 600 | only with `wait` |

```json
{"ocr_text": "...", "document_id": "117"}
```

**202 Accepted**

```json
{"job_id": "3a1c017f931a42a7", "state": "queued"}
```

With `"wait": true`, returns **200** and the completed job object instead.

**Errors**

| Status | Meaning |
|---|---|
| 400 | `ocr_text` missing/empty, body not an object, or body > 8 MiB |
| 503 | Governor refused admission — includes `"retry": true` |
| 500 | Unexpected fault; message carries the exception type |

The 503 is a retryable backpressure signal, not a fault. It occurs under
critical resource pressure; in-flight work continues.

---

## POST /extract/batch

Submit many. Top-level fields act as defaults; each entry may override them.

```json
{
  "prompt": "...",
  "max_tokens": 2048,
  "documents": [
    {"ocr_text": "...", "document_id": "117"},
    {"ocr_text": "...", "document_id": "303"}
  ]
}
```

**202 Accepted** — `{"job_ids": ["...", "..."], "count": 2}`

Submission is atomic per document, not per batch: if admission is refused
part-way, earlier jobs remain queued.

---

## GET /jobs/<job_id>

```json
{
  "job_id": "3a1c017f931a42a7",
  "document_id": "117",
  "state": "done",
  "result": "{ \"buyer_details\": [ ... ] }",
  "error": null,
  "truncated": false,
  "prompt_tokens": 6373,
  "completion_tokens": 244,
  "queued_s": 0.0,
  "duration_s": 1.87
}
```

`state` ∈ `queued` | `running` | `done` | `failed`.

**`truncated: true` deserves attention.** It means generation hit the token
ceiling, which on this workload usually indicates a repetition loop rather than a
genuinely long answer — legitimate outputs average 664 tokens against a 2048 cap.
Treat as a validation failure and route to review.

`404` if the id is unknown.

---

## GET /jobs

`{"queued_depth": 0, "workers": 1, "states": {"done": 12, "failed": 1}}`

---

## POST /shutdown

Graceful stop: drains workers, stops the engine (releasing VRAM), stops the
governor. Returns `{"stopping": true}` before shutting down.

---

## Change history

| Date | Change |
|---|---|
| 2026-07-30 | Initial API: health, hardware, profile, extract, extract/batch, jobs, shutdown |
| 2026-07-31 | No API change. Contract now covered by tests: routes implemented, `HTTPStatus.NOT_FOUND` for unknown paths, `INTERNAL_SERVER_ERROR` instead of a leaked traceback, threaded handler, malformed JSON rejected without a 5xx, `/health` under 1 s for the 2 s UI poll. Live tests are marked `gpu` and skip without a running server. |

---

## A note on the framework

This is stdlib `ThreadingHTTPServer`, not FastAPI. The server must start fast and
carry no dependency capable of breaking the one thing it exists to do. The
launcher (`launcher.py`) starts this server and waits for `/health` to answer —
deliberately for the *endpoint*, not for `ready: true`, since the model takes
30-60 s to load and the interface is built to open during that window.
