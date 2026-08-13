"""Inference engine interface.

One contract, several runtimes. The extraction pipeline depends only on what is
declared here, so the choice between llama.cpp on a 4 GB laptop and vLLM on a
16 GB server never reaches the calling code.

Standard library only: this module must be importable before any runtime is
installed, so the server can report *why* an engine is unavailable instead of
dying on an ImportError.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EngineError(RuntimeError):
    """Base class for every engine failure."""


class EngineNotReadyError(EngineError):
    """Raised when a request arrives before the model finished loading."""


class ModelOutOfMemoryError(EngineError):
    """Raised when a configuration cannot fit in available VRAM.

    Carries the shortfall so the caller can degrade deliberately rather than
    guess at a smaller configuration.
    """

    def __init__(self, message: str, *, required_bytes: int = 0, available_bytes: int = 0):
        super().__init__(message)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes


class FinishReason(str, Enum):
    """Why generation stopped. Drives retry and validation decisions upstream."""

    STOP = "stop"
    #: Hit the token ceiling. On this workload that usually means a repetition
    #: loop rather than a genuinely long answer: legitimate outputs average ~664
    #: tokens against a 2048 cap.
    LENGTH = "length"
    ERROR = "error"


@dataclass(frozen=True)
class ExtractionRequest:
    """One deed to extract.

    `prompt` is the instruction block and `ocr_text` the document body. They are
    kept separate so engines can cache the shared instruction prefix - it is
    identical across every deed in a batch.
    """

    ocr_text: str
    prompt: str
    document_id: str = ""

    max_tokens: int = 2048
    temperature: float = 0.0
    #: MUST stay 1.0 (disabled) for structured extraction. A penalty suppresses
    #: legitimate repetition: every element of buyer_details/seller_details
    #: repeats the same key tokens, so penalising them truncates the party list
    #: and drops trailing fields. Measured on deed 117 - at 1.1 the model emitted
    #: 3 of 5 persons and nulled paid_in_cash and registration_office; at 1.0 it
    #: matched the BF16 reference exactly. Runaway loops are bounded by
    #: max_tokens and eliminated properly by a GBNF grammar, not by this.
    repetition_penalty: float = 1.0
    #: Optional GBNF grammar constraining output to schema-valid JSON. Where the
    #: runtime supports it this makes malformed output structurally impossible.
    grammar: str | None = None
    stop: tuple[str, ...] = ()

    def as_messages(self) -> list[dict[str, str]]:
        """Render to the chat form the model was trained on.

        Single user turn, instruction followed by OCR - matching the finetuning
        format. Do not add a system turn: the model never saw one.
        """
        return [{"role": "user", "content": f"{self.prompt}\n\n{self.ocr_text}"}]


@dataclass(frozen=True)
class ExtractionResult:
    """Raw model output plus the accounting needed to diagnose a bad run."""

    text: str
    document_id: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    engine: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason is FinishReason.LENGTH

    @property
    def tokens_per_second(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.completion_tokens / self.duration_s


@dataclass(frozen=True)
class EngineHealth:
    """Snapshot for the dashboard's AI-server indicator."""

    ready: bool
    engine: str
    model: str
    detail: str = ""
    device: str = ""
    loaded: bool = False
    vram_used_bytes: int = 0
    vram_total_bytes: int = 0
    #: Host RAM that unloading this engine would return. Not VRAM: on a card
    #: this small the weights are offloaded, but the process still maps the
    #: model file, and that is what the memory governor sees.
    reclaimable_bytes: int = 0
    requests_served: int = 0
    idle_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "engine": self.engine,
            "model": self.model,
            "detail": self.detail,
            "device": self.device,
            "loaded": self.loaded,
            "vram_used_bytes": self.vram_used_bytes,
            "vram_total_bytes": self.vram_total_bytes,
            "reclaimable_bytes": self.reclaimable_bytes,
            "requests_served": self.requests_served,
            "idle_seconds": round(self.idle_seconds, 1),
        }


class InferenceEngine(abc.ABC):
    """Lifecycle and generation contract for a backend.

    Implementations must be safe to call from multiple threads: the pipeline
    runs OCR, extraction and translation stages concurrently.
    """

    name: str = "base"

    # -- lifecycle --------------------------------------------------------

    @abc.abstractmethod
    def start(self) -> None:
        """Load the model and block until it can serve.

        Raises ModelOutOfMemoryError if the configuration does not fit, rather
        than loading something degraded without saying so.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the model and all device memory. Must be idempotent."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when a request would be served immediately."""

    @abc.abstractmethod
    def health(self) -> EngineHealth:
        """Current state. Must never raise - it is the diagnostic of last resort."""

    # -- inference --------------------------------------------------------

    @abc.abstractmethod
    def generate(self, request: ExtractionRequest, timeout_s: float = 600.0) -> ExtractionResult:
        """Run one extraction."""

    def generate_batch(
        self, requests: list[ExtractionRequest], timeout_s: float = 600.0
    ) -> list[ExtractionResult]:
        """Run several extractions.

        The default is sequential. Backends with continuous batching override
        this and overlap the work - which is where the throughput actually comes
        from on a prefill-heavy workload of unique prompts.
        """
        return [self.generate(r, timeout_s) for r in requests]

    # -- context manager --------------------------------------------------

    def __enter__(self) -> InferenceEngine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
