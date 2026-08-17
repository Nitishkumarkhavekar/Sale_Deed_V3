"""vLLM backend, for hardware with room to hold the model resident.

Same contract as `llamacpp`, and the same shape: vLLM runs as its **own
process** exposing an OpenAI-compatible HTTP API, and this class supervises it.
Two reasons that matters more here than convenience:

  * vLLM pins torch and CUDA versions that conflict with the ones Surya and
    NLLB need. A separate process is what lets them coexist on one machine
    instead of one dependency set winning.
  * The AI server keeps its promise never to link CUDA. A vLLM crash - and an
    OOM during PagedAttention warmup is a real one - takes down a subprocess,
    not the queue holding a thousand-document batch.

**What actually changes on bigger hardware.** vLLM serves the *unquantised*
checkpoint, so the Q4_K_M compression this project runs under disappears, and
with it the outstanding question of what quantisation costs on exact-copy
fields like Aadhaar and PAN. And `generate_batch` genuinely overlaps work here:
continuous batching keeps the GPU busy across deeds instead of idling between
them, which is where the throughput on a prefill-heavy workload comes from.

**What does not change.** Temperature stays 0 and the repetition penalty stays
disabled, for the reasons `base.ExtractionRequest` documents - a penalty
truncates the party list. A faster engine that returns different values is not
an upgrade.

Standard library only, so this module imports and can *explain itself* on a
machine where vLLM is not installed, rather than dying on an ImportError.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .base import (
    EngineHealth,
    EngineNotReadyError,
    ExtractionRequest,
    ExtractionResult,
    FinishReason,
    InferenceEngine,
    ModelOutOfMemoryError,
)

log = logging.getLogger("saledeed.ai.engine.vllm")

#: Loading an unquantised checkpoint and profiling the KV cache takes minutes on
#: first run - vLLM compiles CUDA graphs and measures free memory before it
#: serves. A short timeout here reads as a hang and gets killed halfway.
DEFAULT_START_TIMEOUT_S = 900.0

#: Fraction of VRAM vLLM may claim. Deliberately below 1.0: Surya still needs
#: the card on machines where both run, and vLLM allocates its KV cache pool up
#: front, so leaving nothing is how the OCR stage starts failing with OOM.
DEFAULT_GPU_FRACTION = 0.85


def _probe_vllm(python: str) -> tuple[bool, str]:
    """Is vLLM importable by the interpreter that would serve it?

    Reported rather than raised. A machine without vLLM should say so on the
    health endpoint, not fail at import and take the server with it.
    """
    try:
        proc = subprocess.run(
            [python, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run {python}: {exc}"
    if proc.returncode == 0:
        return True, (proc.stdout or "").strip() or "installed"
    tail = (proc.stderr or "").strip().splitlines()
    return False, tail[-1][:160] if tail else "vllm is not installed"


class VllmEngine(InferenceEngine):
    """Supervises a `vllm serve` process and talks to it over HTTP."""

    name = "vllm"

    def __init__(
        self,
        model_dir: str | Path,
        profile: Any = None,
        *,
        python: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8078,
        gpu_fraction: float = DEFAULT_GPU_FRACTION,
        max_model_len: int | None = None,
        served_name: str = "deeds",
        dtype: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.profile = profile
        #: The interpreter that has vLLM. Separate from this one by design: the
        #: torch it needs is not the torch Surya needs.
        self.python = python or os.environ.get("SALEDEED_VLLM_PYTHON") or sys.executable
        self.host = host
        self.port = port
        self.gpu_fraction = gpu_fraction
        self.max_model_len = max_model_len
        self.served_name = served_name
        self.dtype = dtype

        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._served = 0
        self._last_used = time.monotonic()
        self._detail = "not started"

    # -- plumbing ---------------------------------------------------------

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_argv(self) -> list[str]:
        """The command line, as its own method so a test can read it.

        `--max-model-len` is passed only when set. vLLM otherwise derives it
        from the checkpoint, and overriding it with a guess is how a deed that
        would have fit gets rejected.
        """
        argv = [
            self.python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_dir),
            "--served-model-name", self.served_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_fraction),
            "--dtype", self.dtype,
            # One process, one card. Multi-GPU is a deliberate decision with
            # its own failure modes, not a default.
            "--tensor-parallel-size", "1",
            "--disable-log-requests",
        ]
        if self.max_model_len:
            argv += ["--max-model-len", str(self.max_model_len)]
        return argv

    # -- lifecycle --------------------------------------------------------

    def start(self, timeout_s: float = DEFAULT_START_TIMEOUT_S) -> None:
        with self._lock:
            if self.is_ready():
                return

            if not self.model_dir.is_dir():
                raise EngineNotReadyError(
                    f"no model directory at {self.model_dir}. vLLM serves the "
                    "unquantised checkpoint, not the .gguf - point --model-dir "
                    "at the HF checkpoint.")

            ok, detail = _probe_vllm(self.python)
            if not ok:
                # Named precisely, because the usual cause is a torch version
                # mismatch and the error alone does not say so.
                raise EngineNotReadyError(
                    f"vLLM is not available to {self.python}: {detail}. "
                    "Install it into an interpreter whose torch and CUDA match "
                    "the wheel, and set SALEDEED_VLLM_PYTHON to it.")
            self._detail = f"vllm {detail}"

            log.info("starting vllm", extra={"model": str(self.model_dir),
                                             "port": self.port})
            self._process = subprocess.Popen(
                self.build_argv(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._await_ready(timeout_s)

    def _await_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                self._raise_startup_failure()
            if self._ping():
                self._detail = "ready"
                log.info("vllm ready", extra={"port": self.port})
                return
            time.sleep(2.0)
        self.stop()
        raise EngineNotReadyError(
            f"vLLM did not become ready within {timeout_s:.0f}s. Loading an "
            "unquantised checkpoint and profiling the KV cache is slow on a "
            "first run; raise the timeout before assuming a fault.")

    def _ping(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _raise_startup_failure(self) -> None:
        """Turn a dead subprocess into an error that names the cause."""
        output = ""
        if self._process is not None and self._process.stdout is not None:
            try:
                output = self._process.stdout.read(8000).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                output = ""
        lowered = output.lower()
        if "out of memory" in lowered or "kv cache" in lowered:
            # The characteristic vLLM failure: it allocates the KV pool up
            # front, so "it loaded then died" is nearly always this.
            raise ModelOutOfMemoryError(
                "vLLM could not fit the model and its KV cache in VRAM. Lower "
                f"--gpu-memory-utilization (currently {self.gpu_fraction}) or "
                "--max-model-len.")
        tail = [ln for ln in output.strip().splitlines() if ln.strip()]
        raise EngineNotReadyError(
            "vLLM exited during startup: "
            + (tail[-1][:200] if tail else "no output captured"))

    def stop(self) -> None:
        """Idempotent, and it waits. A vLLM process holding VRAM after this
        returns would starve the OCR stage that runs next."""
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            try:
                process.terminate()
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001 - stop must never raise
                pass
            self._detail = "stopped"

    def is_ready(self) -> bool:
        return (self._process is not None and self._process.poll() is None
                and self._ping())

    def health(self) -> EngineHealth:
        """Never raises: it is the diagnostic of last resort."""
        try:
            ready = self.is_ready()
        except Exception:  # noqa: BLE001
            ready = False
        return EngineHealth(
            ready=ready,
            engine=self.name,
            model=self.model_dir.name,
            detail=self._detail,
            device="cuda",
            loaded=ready,
            requests_served=self._served,
            idle_seconds=time.monotonic() - self._last_used,
        )

    # -- inference --------------------------------------------------------

    def generate(self, request: ExtractionRequest,
                 timeout_s: float = 600.0) -> ExtractionResult:
        if not self.is_ready():
            raise EngineNotReadyError("vLLM is not serving")

        payload = {
            "model": self.served_name,
            "messages": request.as_messages(),
            "max_tokens": request.max_tokens,
            # Zero, as everywhere else. A legal extraction has to be
            # reproducible, and it is what makes a rerun byte-identical.
            "temperature": request.temperature,
            # 1.0 = disabled. See base.ExtractionRequest: any penalty
            # truncates the repeated keys of a party list.
            "repetition_penalty": request.repetition_penalty,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.grammar:
            # vLLM expresses constrained decoding as a grammar in its own
            # extension field rather than as GBNF on the request root.
            payload["guided_grammar"] = request.grammar

        started = time.monotonic()
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise EngineNotReadyError(f"vLLM HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EngineNotReadyError(f"vLLM unreachable: {exc}") from exc

        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        reason = str(choice.get("finish_reason") or "stop")
        self._served += 1
        self._last_used = time.monotonic()

        return ExtractionResult(
            text=(choice.get("message") or {}).get("content") or "",
            document_id=request.document_id,
            finish_reason=(FinishReason.LENGTH if reason == "length"
                           else FinishReason.STOP),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            duration_s=round(time.monotonic() - started, 2),
            engine=self.name,
            model=self.served_name,
        )

    def generate_batch(self, requests: list[ExtractionRequest],
                       timeout_s: float = 600.0) -> list[ExtractionResult]:
        """Submit concurrently and let vLLM schedule.

        This override is the point of the engine. The base class runs requests
        one after another; vLLM's continuous batching keeps the GPU busy across
        deeds instead of idling between them, which is where throughput comes
        from on a workload of long, unique prompts.

        Order is preserved: the caller matches results to documents by
        position, and returning them in completion order would silently
        attribute one deed's extraction to another.
        """
        if not requests:
            return []
        workers = min(len(requests), 16)
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="vllm") as pool:
            return list(pool.map(lambda r: self.generate(r, timeout_s), requests))


#: Where the installer puts vLLM's own environment. A third interpreter is not
#: excess: vLLM pins `transformers>=5.5.3` and Surya pins `==4.57.1`, so they
#: cannot share one, and putting vLLM in the application's interpreter would
#: drag CUDA into the process that must never link it.
VENV_DIR = "models/vllm-env"


def resolve_vllm_python(root: Path | None = None) -> str:
    """The interpreter that should run vLLM, or "" if there is none.

    Order: an explicit `SALEDEED_VLLM_PYTHON`, then the environment the
    installer creates. Never falls back to `sys.executable` - that is the
    application's interpreter, and installing vLLM into it is the mistake this
    whole arrangement exists to prevent.
    """
    explicit = (os.environ.get("SALEDEED_VLLM_PYTHON") or "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    base = Path(root) if root else Path(__file__).resolve().parents[3]
    candidate = base / VENV_DIR / "Scripts" / "python.exe"
    if candidate.is_file():
        return str(candidate)
    candidate = base / VENV_DIR / "bin" / "python"
    return str(candidate) if candidate.is_file() else ""
