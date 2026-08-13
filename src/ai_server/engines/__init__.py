"""Pluggable inference backends.

Every backend serves the same locally trained deeds model behind one interface,
so the pipeline never learns which runtime is underneath and switching is a
configuration change rather than a code change.

    llamacpp - production on constrained VRAM (GGUF, continuous batching)
    hf       - BF16 reference path; the accuracy baseline everything is measured against
    mock     - deterministic stub for pipeline tests, no GPU and no model
"""

from .base import (
    EngineError,
    EngineNotReadyError,
    ExtractionRequest,
    ExtractionResult,
    InferenceEngine,
    ModelOutOfMemoryError,
)

__all__ = [
    "EngineError",
    "EngineNotReadyError",
    "ExtractionRequest",
    "ExtractionResult",
    "InferenceEngine",
    "ModelOutOfMemoryError",
]
