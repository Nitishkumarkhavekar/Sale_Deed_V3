"""llama.cpp backend - the production path on constrained VRAM.

Chosen for this hardware because of one property no other runtime offers here:
**it preallocates**. Context and KV are sized at load and never grow, so once
the server is up VRAM is flat regardless of prompt length or request volume. A
33k-token deed cannot OOM the process mid-batch; it is rejected at admission.
That is what makes the "never OOM" requirement achievable on a 4 GB card.

The model runs as a separate process, deliberately:

  * a crash in the runtime cannot take down the desktop application
  * the UI process never links CUDA and never touches the GPU
  * the same HTTP contract is served by vLLM on larger hardware, so moving to
    the deploy box is a configuration change rather than a rewrite

Standard library only - urllib rather than httpx - so this module stays
importable and diagnosable before anything is installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from ..profiles import Profile
from .base import (
    EngineHealth,
    EngineNotReadyError,
    ExtractionRequest,
    ExtractionResult,
    FinishReason,
    InferenceEngine,
    ModelOutOfMemoryError,
)

#: Startup includes one-off CUDA graph and kernel compilation; be generous.
DEFAULT_START_TIMEOUT_S = 300.0

#: Substrings in llama.cpp output that mean "did not fit".
_OOM_MARKERS = (
    "out of memory",
    "failed to allocate",
    "cudamalloc failed",
    "unable to allocate backend buffer",
)

#: Backends that would enumerate the AMD integrated GPU. Selecting one is silent
#: and catastrophic for throughput, so it is treated as fatal.
_FORBIDDEN_BACKENDS = ("Vulkan", "OpenCL", "SYCL")


def _resolve_binary(binary: str | Path | None) -> str:
    """Absolute path to llama-server, or a bare name for PATH lookup.

    Relative paths are resolved because Windows CreateProcess raises WinError 2
    on a relative path with forward slashes, even when the file plainly exists.
    """
    if binary:
        candidate = Path(binary)
        if candidate.is_file():
            return str(candidate.resolve())
        found = shutil.which(str(binary))
        return str(Path(found).resolve()) if found else str(binary)
    found = shutil.which("llama-server")
    return str(Path(found).resolve()) if found else "llama-server"


class LlamaCppEngine(InferenceEngine):
    """Supervises a `llama-server` process and speaks its HTTP API."""

    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path,
        profile: Profile,
        *,
        binary: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8077,
        idle_unload_s: float = 0.0,
        extra_args: list[str] | None = None,
        log_lines: int = 400,
    ) -> None:
        # Absolute paths throughout. Windows CreateProcess fails with WinError 2
        # on a relative path containing forward slashes, and llama-server needs
        # its sibling DLLs found from its own directory.
        self.model_path = Path(model_path).resolve()
        self.profile = profile
        self.binary = _resolve_binary(binary)
        self.host = host
        self.port = port
        #: Unload after this many idle seconds; 0 keeps the model resident.
        #: Resident is correct for batch processing - reloading per request would
        #: dominate runtime - but an idle operator should not hold 3 GB of VRAM.
        self.idle_unload_s = idle_unload_s
        self.extra_args = extra_args or []

        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._log: deque[str] = deque(maxlen=log_lines)
        self._stopping = threading.Event()
        self._reaper: threading.Thread | None = None
        self._requests_served = 0
        self._last_activity = time.monotonic()
        self._detail = "not started"

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- command line -----------------------------------------------------

    def build_argv(self) -> list[str]:
        """Translate the selected Profile into llama-server arguments."""
        p = self.profile
        argv = [
            self.binary,
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(p.n_ctx),
            "-ngl", str(p.n_gpu_layers if p.device == "cuda" else 0),
            "-t", str(p.n_threads),
            "--parallel", str(p.n_parallel),
            # Continuous batching: overlap prefill of queued deeds with decode of
            # in-flight ones. This is where throughput comes from on a workload
            # of long, unique prompts.
            "--cont-batching",
            # Reuse KV for the shared instruction prefix across deeds. The ~330
            # token prompt block is identical for every document in a batch.
            "--cache-reuse", "256",
            # Takes an explicit value since b10184; bare --flash-attn is rejected.
            "--flash-attn", "on",
            "--cache-type-k", p.kv_type,
            "--cache-type-v", p.kv_type,
            # Reject an over-long deed rather than silently sliding the window
            # and dropping the start of the document.
            "--no-context-shift",
        ]
        return argv + self.extra_args

    def _environment(self) -> dict[str, str]:
        """Pin CUDA to the selected NVIDIA device and nothing else.

        Pinned by UUID rather than index: indices reorder across driver updates,
        UUIDs do not. The Vulkan and SYCL visibility variables are cleared
        defensively in case a non-CUDA build ends up on PATH.
        """
        env = dict(os.environ)
        if self.profile.device == "cuda" and self.profile.gpu_uuid:
            env["CUDA_VISIBLE_DEVICES"] = self.profile.gpu_uuid
            env["GGML_CUDA_VISIBLE_DEVICES"] = self.profile.gpu_uuid
            env["GGML_VK_VISIBLE_DEVICES"] = ""
            env["ONEAPI_DEVICE_SELECTOR"] = ""
        return env

    # -- lifecycle --------------------------------------------------------

    def start(self, timeout_s: float = DEFAULT_START_TIMEOUT_S) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._preflight()
            self._stopping.clear()
            self._log.clear()
            self._detail = "starting"
            self._proc = subprocess.Popen(
                self.build_argv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=self._environment(),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            threading.Thread(target=self._drain_log, name="llamacpp-log", daemon=True).start()

        self._await_ready(timeout_s)
        self._assert_cuda_backend()

        with self._lock:
            self._detail = "ready"
            self._last_activity = time.monotonic()
            if self.idle_unload_s > 0 and self._reaper is None:
                self._reaper = threading.Thread(
                    target=self._idle_reaper, name="llamacpp-idle", daemon=True
                )
                self._reaper.start()

    def _preflight(self) -> None:
        """Fail with instructions, not a stack trace, when something is missing."""
        if not self.model_path.is_file():
            q = self.profile.quant.name
            raise EngineNotReadyError(
                f"GGUF model not found: {self.model_path}\n"
                "Build it from the repacked checkpoint:\n"
                "  python convert_hf_to_gguf.py 'AI server/gemma4b-text' "
                "--outfile deeds-f16.gguf --outtype f16\n"
                f"  llama-quantize deeds-f16.gguf deeds-{q}.gguf {q}"
            )
        if shutil.which(self.binary) is None and not Path(self.binary).is_file():
            raise EngineNotReadyError(
                f"llama-server not found at {self.binary!r}. Install a CUDA build and put "
                "it on PATH, or pass binary=... explicitly."
            )

    def _drain_log(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log.append(line.rstrip())

    def _await_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._raise_startup_failure()
            if self._ping():
                return
            time.sleep(0.5)

        self.stop()
        raise EngineNotReadyError(
            f"llama-server did not become ready within {timeout_s:.0f}s.\n" + self.recent_log(20)
        )

    def _ping(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _raise_startup_failure(self) -> None:
        """Turn a dead process into a diagnosis rather than a return code."""
        lowered = "\n".join(self._log).lower()
        if any(marker in lowered for marker in _OOM_MARKERS):
            raise ModelOutOfMemoryError(
                f"llama-server ran out of VRAM loading {self.profile.quant.name} at "
                f"{self.profile.n_ctx:,} context "
                f"(estimated {self.profile.total_bytes / 1024**3:.2f} GiB).\n"
                "Lower the context, pick a more compact quantisation, or reduce "
                "MAX_BUDGET_UTILISATION in profiles.py.\n" + self.recent_log(20),
                required_bytes=self.profile.total_bytes,
                available_bytes=self.profile.budget_bytes,
            )
        raise EngineNotReadyError("llama-server exited during startup.\n" + self.recent_log(30))

    def _assert_cuda_backend(self) -> None:
        """Refuse to serve from the integrated GPU.

        A Vulkan or OpenCL build enumerates the AMD iGPU and may select it
        without complaint. The output is still correct, at a fraction of the
        speed - easy to miss and hard to attribute later.
        """
        if self.profile.device != "cuda":
            return
        log = "\n".join(self._log)
        for backend in _FORBIDDEN_BACKENDS:
            if re.search(rf"\b{backend}\b", log, re.IGNORECASE):
                self.stop()
                raise EngineNotReadyError(
                    f"llama-server reported the {backend} backend, which can bind the "
                    "integrated AMD GPU. Install a CUDA build.\n" + self.recent_log(20)
                )
        if not re.search(r"CUDA|cuBLAS|ggml-cuda", log, re.IGNORECASE):
            self._detail = "ready (warning: CUDA not confirmed in startup log)"

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            proc, self._proc = self._proc, None
            if proc is None:
                self._detail = "stopped"
                return
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            self._detail = "stopped"

    def _idle_reaper(self) -> None:
        """Release VRAM after a quiet period; reload lazily on the next request."""
        while not self._stopping.wait(5.0):
            with self._lock:
                running = self._proc is not None and self._proc.poll() is None
                idle = time.monotonic() - self._last_activity
            if running and idle >= self.idle_unload_s:
                self.stop()
                self._stopping.clear()  # stop() set it; keep the reaper alive
                self._detail = f"unloaded after {idle:.0f}s idle"

    # -- inference --------------------------------------------------------

    def generate(self, request: ExtractionRequest, timeout_s: float = 600.0) -> ExtractionResult:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
        if not running:
            self.start()  # lazy reload after an idle unload

        payload: dict[str, object] = {
            "messages": request.as_messages(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "repeat_penalty": request.repetition_penalty,
            "cache_prompt": True,
            "stream": False,
        }
        if request.grammar:
            payload["grammar"] = request.grammar
        if request.stop:
            payload["stop"] = list(request.stop)

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise EngineNotReadyError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EngineNotReadyError(f"llama-server unreachable: {exc}") from exc
        duration = time.monotonic() - started

        with self._lock:
            self._requests_served += 1
            self._last_activity = time.monotonic()

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        reason = (
            FinishReason.LENGTH
            if choice.get("finish_reason") == "length"
            else FinishReason.STOP
        )

        return ExtractionResult(
            text=(choice.get("message") or {}).get("content", "") or "",
            document_id=request.document_id,
            finish_reason=reason,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            duration_s=duration,
            engine=self.name,
            model=self.model_path.name,
            metadata={"quant": self.profile.quant.name, "n_ctx": self.profile.n_ctx},
        )

    def generate_batch(
        self, requests: list[ExtractionRequest], timeout_s: float = 600.0
    ) -> list[ExtractionResult]:
        """Submit concurrently so llama.cpp can continuously batch the queue.

        Concurrency is capped at the server's slot count: extra in-flight
        requests would queue inside the server anyway, and holding them here
        keeps the failure modes visible.
        """
        if not requests:
            return []
        if self.profile.n_parallel <= 1 or len(requests) == 1:
            return [self.generate(r, timeout_s) for r in requests]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.profile.n_parallel) as pool:
            return list(pool.map(lambda r: self.generate(r, timeout_s), requests))

    # -- status -----------------------------------------------------------

    def is_ready(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
        return self._ping()

    def health(self) -> EngineHealth:
        try:
            ready = self.is_ready()
        except Exception:  # noqa: BLE001 - health is the diagnostic of last resort
            ready = False
        with self._lock:
            loaded = self._proc is not None and self._proc.poll() is None
            return EngineHealth(
                ready=ready,
                engine=self.name,
                model=self.model_path.name,
                detail=self._detail,
                device=self.profile.device,
                loaded=loaded,
                vram_used_bytes=self.profile.total_bytes if loaded else 0,
                vram_total_bytes=self.profile.budget_bytes,
                # The model file size, as a deliberately low estimate of the
                # host RAM unloading would return. Measured on the development
                # machine: 2.65 GiB actually came back against a 2.33 GiB file.
                # Under-estimating keeps the governor on the strict side.
                reclaimable_bytes=(self._model_bytes() if loaded else 0),
                requests_served=self._requests_served,
                idle_seconds=time.monotonic() - self._last_activity,
            )

    def _model_bytes(self) -> int:
        try:
            return self.model_path.stat().st_size
        except OSError:
            return 0

    def recent_log(self, n: int = 40) -> str:
        lines = list(self._log)[-n:]
        return "\n".join(f"  | {line}" for line in lines) if lines else "  | (no output)"
