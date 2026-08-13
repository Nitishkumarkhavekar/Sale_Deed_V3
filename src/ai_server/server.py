"""AI server - the orchestration layer the desktop application talks to.

Sits above the inference runtime and owns everything the UI must never do:
model lifecycle, GPU arbitration, resource governance and job queueing. The
desktop process speaks HTTP to this and never imports CUDA or loads a model.

Why stdlib HTTP and not FastAPI: the requirement is a lightweight application
that minimises RAM on an 8 GB machine, the API surface is four endpoints, and
this keeps the whole server at zero pip dependencies - which matters more for
enterprise deployment than framework conveniences would.

Submission is asynchronous by default. `POST /extract` returns a job id
immediately so a 1000-file batch never blocks the caller; the UI polls
`GET /jobs/<id>`. A synchronous mode exists for scripts and tests.

Endpoints
    GET  /health          aggregate readiness - engine, governor, pressure
    GET  /hardware        detected CPU/RAM/GPU/disk
    GET  /profile         selected inference profile and the fidelity ladder
    POST /model           {"loaded": false} releases the weights, true reloads
    POST /extract         submit one deed          -> {"job_id": ...}
    POST /extract/batch   submit many deeds        -> {"job_ids": [...]}
    GET  /jobs/<id>       poll one job
    GET  /jobs            queue summary
    POST /shutdown        graceful stop
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: Module logger. `logging_setup.configure` attaches the handlers; until it
#: runs these records go nowhere, which is the correct behaviour for a library
#: and the reason the name is fixed here rather than passed in.
from core import paths

log = logging.getLogger("saledeed.ai.server")

from .engines.base import EngineError, ExtractionRequest, InferenceEngine, ModelOutOfMemoryError
from .profiles import Profile, ladder_report, select_profile
from .resources import Pressure, ResourceGovernor

#: Refuse bodies larger than this. A deed's OCR text is at most a few hundred KB.
MAX_BODY_BYTES = 8 * 1024 * 1024


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """One unit of work, tracked from submission to completion."""

    job_id: str
    request: ExtractionRequest
    state: JobState = JobState.QUEUED
    result_text: str | None = None
    error: str | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "document_id": self.request.document_id,
            "state": self.state.value,
            "result": self.result_text,
            "error": self.error,
            "truncated": self.truncated,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "queued_s": round((self.started_at or time.time()) - self.submitted_at, 2),
            "duration_s": (
                round(self.finished_at - self.started_at, 2)
                if self.started_at and self.finished_at
                else None
            ),
        }


class JobQueue:
    """Bounded work queue with worker threads that respect the governor.

    Workers acquire the GPU lease before inference, so on a card too small for
    co-residency the OCR, extraction and translation models never collide.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        governor: ResourceGovernor,
        *,
        workers: int = 1,
        maxsize: int = 4096,
    ) -> None:
        self.engine = engine
        self.governor = governor
        self._q: queue.Queue[str] = queue.Queue(maxsize=maxsize)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._worker_count = max(1, workers)

    def start(self) -> None:
        for i in range(self._worker_count):
            t = threading.Thread(target=self._work, name=f"extract-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
        self._threads.clear()

    def submit(self, request: ExtractionRequest) -> Job:
        """Queue a job. Rejected under critical pressure rather than piled on."""
        plan = self.governor.plan()
        if not plan.admit_new_work:
            raise RuntimeError(
                f"refusing new work: system pressure is {plan.pressure.label} ({plan.reason}). "
                "In-flight documents will finish; retry shortly."
            )
        job = Job(job_id=uuid.uuid4().hex[:16], request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self._q.put(job.job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            states: dict[str, int] = {}
            for job in self._jobs.values():
                states[job.state.value] = states.get(job.state.value, 0) + 1
        return {"queued_depth": self._q.qsize(), "workers": self._worker_count, "states": states}

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            job = self.get(job_id)
            if job is None:
                continue

            job.state = JobState.RUNNING
            job.started_at = time.time()
            try:
                # Serialise GPU work when VRAM cannot host two models at once.
                with self.governor.gpu_lease("extract"):
                    result = self.engine.generate(job.request)
                job.result_text = result.text
                job.prompt_tokens = result.prompt_tokens
                job.completion_tokens = result.completion_tokens
                job.truncated = result.truncated
                job.state = JobState.DONE
            except ModelOutOfMemoryError as exc:
                job.error = f"out of VRAM: {exc}"
                job.state = JobState.FAILED
            except (EngineError, Exception) as exc:  # noqa: BLE001 - a job must not kill a worker
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = JobState.FAILED
            finally:
                job.finished_at = time.time()
                self._q.task_done()


class AiServer:
    """Owns the engine, the governor and the queue for the process lifetime."""

    def __init__(
        self,
        engine: InferenceEngine,
        profile: Profile,
        model_dir: str | Path,
        *,
        governor: ResourceGovernor | None = None,
        prompt_file: str | Path | None = None,
        workers: int = 1,
    ) -> None:
        self.engine = engine
        self.profile = profile
        self.model_dir = str(model_dir)
        self.governor = governor or ResourceGovernor()
        # Tell the governor how much host RAM this engine is holding that it
        # could hand back. Without it the governor sees only free memory, calls
        # the machine critical because the model is loaded, and refuses to admit
        # the very work whose first act is to unload it (R-037).
        if self.governor.reclaimable_provider is None:
            self.governor.reclaimable_provider = (
                lambda: self.engine.health().reclaimable_bytes)
        self.queue = JobQueue(engine, self.governor, workers=workers)
        self.started_at = time.time()
        self._prompt = ""
        if prompt_file and Path(prompt_file).is_file():
            self._prompt = Path(prompt_file).read_text(encoding="utf-8").strip()

    @property
    def default_prompt(self) -> str:
        return self._prompt

    def start(self) -> None:
        self.governor.start()
        self.engine.start()
        self.queue.start()

    def stop(self) -> None:
        self.queue.stop()
        self.engine.stop()
        self.governor.stop()

    # -- payload handling -------------------------------------------------

    def build_request(self, payload: dict[str, Any]) -> ExtractionRequest:
        """Turn a JSON body into a request, normalising line endings.

        CRLF normalisation is not cosmetic. The OCR corpus ships CRLF, and
        llama.cpp preserves the carriage returns while the tokenizer the model
        was trained with drops them - measured at 6758 vs 6408 tokens on one
        deed. Feeding raw CRLF gives the model a token stream it never saw in
        training, so it is normalised at the boundary.
        """
        ocr = payload.get("ocr_text") or ""
        if not ocr.strip():
            raise ValueError("ocr_text is required and must not be empty")
        ocr = ocr.replace("\r\n", "\n").replace("\r", "\n")

        return ExtractionRequest(
            ocr_text=ocr,
            prompt=(payload.get("prompt") or self._prompt).strip(),
            document_id=str(payload.get("document_id") or ""),
            max_tokens=int(payload.get("max_tokens") or 2048),
            temperature=float(payload.get("temperature") or 0.0),
            # Default 1.0 = disabled. See ExtractionRequest.repetition_penalty:
            # any penalty truncates the JSON party list.
            repetition_penalty=float(payload.get("repetition_penalty") or 1.0),
            grammar=payload.get("grammar"),
            stop=tuple(payload.get("stop") or ()),
        )

    def health(self) -> dict[str, Any]:
        engine_health = self.engine.health()
        plan = self.governor.plan()
        snap = self.governor.snapshot()
        return {
            "ready": engine_health.ready and plan.admit_new_work,
            "uptime_s": round(time.time() - self.started_at, 1),
            "engine": engine_health.as_dict(),
            "pressure": plan.pressure.label,
            "admitting_work": plan.admit_new_work,
            "gpu_exclusive": plan.gpu_exclusive,
            "gpu_holder": self.governor.gpu_holder,
            "workers": plan.workers,
            "resources": {
                "ram_available_bytes": snap.ram_available_bytes,
                "ram_total_bytes": snap.ram_total_bytes,
                "cpu_busy": round(snap.cpu_busy, 3),
                "vram_free_bytes": snap.vram_free_bytes,
                "vram_total_bytes": snap.vram_total_bytes,
                "disk_free_bytes": snap.disk_free_bytes,
            },
            "queue": self.queue.summary(),
        }

    def profile_info(self) -> dict[str, Any]:
        p = self.profile
        return {
            "model_dir": self.model_dir,
            "device": p.device,
            "quantisation": p.quant.name,
            "lossless": p.quant.lossless,
            "quant_note": p.quant.note,
            "n_ctx": p.n_ctx,
            "prompt_capacity_tokens": p.prompt_capacity,
            "kv_type": p.kv_type,
            "n_gpu_layers": p.n_gpu_layers,
            "n_parallel": p.n_parallel,
            "n_threads": p.n_threads,
            "vram": {
                "weights_bytes": p.weight_bytes,
                "kv_bytes": p.kv_bytes,
                "overhead_bytes": p.overhead_bytes,
                "total_bytes": p.total_bytes,
                "budget_bytes": p.budget_bytes,
            },
            "reason": p.reason,
            "warnings": list(p.warnings),
        }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "SaleDeedAI/3.0"
    protocol_version = "HTTP/1.1"
    app: AiServer  # injected by make_http_server
    shutdown_hook: Callable[[], None]

    #: Polled by the desktop shell every few seconds. Logging these at INFO
    #: buries everything else within a minute, so they go to DEBUG - present
    #: when someone is looking for them, invisible when they are not.
    QUIET_PATHS = frozenset({"/health"})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        # BaseHTTPRequestHandler writes its own unformatted line straight to
        # stderr. Silenced here because `_send` logs the same request through
        # the application's handlers, with a duration and an id attached -
        # leaving both would double every entry.
        return

    def _begin(self) -> None:
        """Start the clock and give this request an id."""
        self._started = time.monotonic()
        self._request_id = uuid.uuid4().hex[:8]
        log.debug("request %s %s", self.command, self.path,
                  extra={"request_id": self._request_id,
                         "client": self.client_address[0]})

    # -- helpers ----------------------------------------------------------

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        # Every response leaves through here, which makes it the one place that
        # can report a status and a duration without being repeated per route.
        path = urlparse(self.path).path.rstrip("/") or "/"
        elapsed = (time.monotonic() - getattr(self, "_started", time.monotonic()))
        level = logging.DEBUG if path in self.QUIET_PATHS else logging.INFO
        if status >= HTTPStatus.BAD_REQUEST:
            level = logging.WARNING
        # Status and duration are already in the message; repeating them as
        # structured fields doubles the width of every line for nothing. The
        # id is what a reader needs to correlate, and the error only exists
        # when something went wrong.
        context = {"request_id": getattr(self, "_request_id", "-")}
        if payload.get("error"):
            context["error"] = payload["error"]
        log.log(level, "%s %s -> %d in %.3fs", self.command, path,
                int(status), elapsed, extra=context)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"body exceeds {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._begin()
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/health":
                self._send(HTTPStatus.OK, self.app.health())
            elif path == "/hardware":
                hw = self.app.governor.hw
                self._send(HTTPStatus.OK, {
                    "os": hw.os_name,
                    "cpu": hw.cpu_name,
                    "physical_cores": hw.physical_cores,
                    "logical_cores": hw.logical_cores,
                    "ram_total_bytes": hw.ram_total_bytes,
                    "cuda_available": hw.cuda_available,
                    "driver_version": hw.driver_version,
                    "cuda_version": hw.cuda_version,
                    "gpus": [
                        {"index": g.index, "uuid": g.uuid, "name": g.name,
                         "total_bytes": g.total_bytes, "free_bytes": g.free_bytes,
                         "compute_capability": g.compute_capability}
                        for g in hw.gpus
                    ],
                    "excluded_adapters": hw.other_adapters,
                    "disks": [{"path": d.path, "free_bytes": d.free_bytes,
                               "total_bytes": d.total_bytes} for d in hw.disks],
                    "warnings": hw.warnings,
                })
            elif path == "/profile":
                info = self.app.profile_info()
                info["ladder"] = ladder_report(self.app.model_dir, self.app.governor.hw)
                self._send(HTTPStatus.OK, info)
            elif path == "/jobs":
                self._send(HTTPStatus.OK, self.app.queue.summary())
            elif path.startswith("/jobs/"):
                job = self.app.queue.get(path.split("/")[-1])
                if job is None:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "unknown job_id"})
                else:
                    self._send(HTTPStatus.OK, job.as_dict())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        self._begin()
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/shutdown":
                self._send(HTTPStatus.OK, {"stopping": True})
                threading.Thread(target=self.shutdown_hook, daemon=True).start()
                return

            if path == "/model":
                # Release or reload the weights without stopping the service.
                #
                # On a 4 GiB card the language model and Surya cannot co-reside,
                # and on a 7.4 GiB machine they cannot share host RAM either.
                # Nothing in the application could make llama.cpp let go, so OCR
                # either OOMed or was pushed onto a CPU that had no room for it
                # (R-035). The pipeline now asks.
                #
                # The service stays up throughout: /health keeps answering and
                # reports `loaded: false`, which is the truth and is what the UI
                # already knows how to display.
                payload = self._read_json()
                want = bool(payload.get("loaded", True))
                before = self.app.engine.health().loaded
                if want and not before:
                    self.app.engine.start()
                elif before and not want:
                    self.app.engine.stop()
                after = self.app.engine.health().loaded
                self._send(HTTPStatus.OK, {
                    "loaded": after, "changed": after != before,
                    "detail": ("loaded" if after else "released - "
                               "reload costs about a minute")})
                return

            if path == "/extract":
                payload = self._read_json()
                request = self.app.build_request(payload)
                job = self.app.queue.submit(request)
                if payload.get("wait"):
                    timeout = float(payload.get("timeout_s") or 600)
                    deadline = time.monotonic() + timeout
                    while (time.monotonic() < deadline
                           and job.state in (JobState.QUEUED, JobState.RUNNING)):
                        time.sleep(0.1)
                    self._send(HTTPStatus.OK, job.as_dict())
                else:
                    self._send(HTTPStatus.ACCEPTED,
                               {"job_id": job.job_id, "state": job.state.value})
                return

            if path == "/extract/batch":
                payload = self._read_json()
                documents = payload.get("documents")
                if not isinstance(documents, list) or not documents:
                    raise ValueError("documents must be a non-empty list")
                ids = []
                for entry in documents:
                    merged = {**payload, **entry}
                    merged.pop("documents", None)
                    ids.append(self.app.queue.submit(self.app.build_request(merged)).job_id)
                self._send(HTTPStatus.ACCEPTED, {"job_ids": ids, "count": len(ids)})
                return

            self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})

        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            # Governor refused admission - a retryable condition, not a fault.
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc), "retry": True})
        except Exception as exc:  # noqa: BLE001
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(exc).__name__}: {exc}"})


def make_http_server(app: AiServer, host: str, port: int) -> ThreadingHTTPServer:
    """Bind a threaded HTTP server bound to this application instance."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    _Handler.app = app
    _Handler.shutdown_hook = httpd.shutdown
    return httpd


def build_default(
    model_gguf: str | Path,
    model_dir: str | Path = str(paths.CHECKPOINT_DIR.parent / "gemma4b-text"),
    *,
    engine_name: str = "llamacpp",
    # Absolute, from `paths`. This was a path relative to the working
    # directory, and when `saledeed main/` moved under `models/` it stopped
    # resolving - silently, because a missing prompt file is an ordinary
    # condition here. The model then received OCR text with no instruction and
    # did what an instruction-tuned model does with an unlabelled wall of text:
    # wrote a prose summary. Every extraction failed with "no parseable JSON".
    # See R-040.
    prompt_file: str | Path = paths.PROMPT_FILE,
    binary: str | Path | None = None,
    port: int = 8077,
) -> AiServer:
    """Wire the standard stack: detect hardware, pick a profile, choose an engine."""
    governor = ResourceGovernor()
    profile = select_profile(model_dir, governor.hw)

    engine: InferenceEngine
    if engine_name == "mock":
        from .engines.mock import MockEngine

        engine = MockEngine()
    else:
        from .engines.llamacpp import LlamaCppEngine

        engine = LlamaCppEngine(
            model_gguf, profile, binary=binary, port=port + 1,
            idle_unload_s=0.0,
        )

    return AiServer(
        engine, profile, model_dir,
        governor=governor, prompt_file=prompt_file,
        workers=profile.n_parallel,
    )


def main() -> int:
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    # These defaults are absolute, from `paths`. They were paths relative to the
    # working directory, which meant `python -m ai_server.server` - the start
    # command in the setup instructions - died at startup on a correct install
    # unless it happened to be run from `src/`. `build_default` already used
    # `paths`; argparse then overrode it with the stale strings. Same restructure
    # fallout as R-040, found by the R-046 end-to-end run.
    ap = argparse.ArgumentParser(description="Sale Deed AI server")
    ap.add_argument("--model", default=str(paths.GGUF_DIR / "deeds-v6_7-Q4_K_M.gguf"))
    ap.add_argument("--model-dir", default=str(paths.AI_SERVER / "gemma4b-text"))
    ap.add_argument("--engine", default="llamacpp", choices=("llamacpp", "mock"))
    ap.add_argument("--binary", default=str(paths.TOOLS_DIR / "llamacpp" / "llama-server.exe"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()

    # The AI server is its own process and configures its own logging. No
    # database handler here: serving inference must never depend on the database
    # being reachable.
    try:
        from core import logging_setup

        logging_setup.configure(app_name="saledeed.ai", console=True)
    except Exception:  # noqa: BLE001 - logging must never block startup
        pass

    app = build_default(
        args.model, args.model_dir,
        engine_name=args.engine, binary=args.binary, port=args.port,
    )
    print(app.governor.hw.summary())
    print()
    print(app.profile.explain())
    print()
    print(f"starting engine ({args.engine}) ...", flush=True)
    try:
        app.start()
    except Exception as exc:  # noqa: BLE001
        # Full traceback, not just the message: a failure to bring up
        # the engine is the one moment where the stack is the diagnosis.
        log.critical("engine failed to start: %s: %s",
                     type(exc).__name__, exc, exc_info=True)
        return 1

    httpd = make_http_server(app, args.host, args.port)
    log.info("listening on http://%s:%d", args.host, args.port,
             extra={"engine": args.engine, "model": str(args.model),
                    "routes": "GET /health /hardware /profile /jobs; "
                              "POST /extract /extract/batch /model /shutdown"})
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
