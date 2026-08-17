"""The vLLM backend.

Tested without vLLM installed, on purpose. This machine is a 4 GB laptop that
cannot run it, and the engine has to behave correctly on exactly such a machine:
report *why* it is unavailable rather than dying at import, so the health
endpoint can say so and the server keeps running.

What cannot be tested here - that generation returns sensible text, that
continuous batching is faster - needs hardware with room for the unquantised
checkpoint.
"""

from __future__ import annotations

import pytest

from ai_server.engines.base import (
    EngineNotReadyError,
    ExtractionRequest,
    InferenceEngine,
)
from ai_server.engines.vllm import DEFAULT_GPU_FRACTION, VllmEngine, _probe_vllm


class TestTheContract:
    def test_it_is_an_engine(self):
        assert issubclass(VllmEngine, InferenceEngine)

    def test_every_abstract_method_is_implemented(self):
        for name in ("start", "stop", "is_ready", "health", "generate"):
            assert getattr(VllmEngine, name) is not getattr(InferenceEngine, name)

    def test_it_imports_without_vllm_installed(self):
        """The module must load on a machine that cannot run it, or the server
        dies at import instead of reporting the problem."""
        engine = VllmEngine("models/AI server/gemma4b-text")
        assert engine.name == "vllm"


class TestHealthNeverRaises:
    def test_health_answers_before_start(self, tmp_path):
        health = VllmEngine(tmp_path).health()
        assert health.ready is False
        assert health.engine == "vllm"

    def test_health_answers_when_the_model_is_absent(self):
        health = VllmEngine("no/such/directory").health()
        assert health.ready is False
        assert isinstance(health.as_dict(), dict)


class TestStartupRefusesClearly:
    def test_a_missing_model_directory_names_the_gguf_confusion(self, tmp_path):
        """The predictable mistake: pointing vLLM at the .gguf, which it cannot
        read. The error has to say so rather than fail deep inside vLLM."""
        with pytest.raises(EngineNotReadyError) as caught:
            VllmEngine(tmp_path / "absent").start(timeout_s=1)
        assert "gguf" in str(caught.value).lower()

    def test_missing_vllm_names_the_interpreter_and_the_fix(self, tmp_path):
        """The usual cause is a torch version mismatch, which the raw
        ImportError does not explain."""
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(EngineNotReadyError) as caught:
            VllmEngine(tmp_path, python="definitely-not-a-python").start(timeout_s=1)
        message = str(caught.value)
        assert "SALEDEED_VLLM_PYTHON" in message
        assert "definitely-not-a-python" in message

    def test_generating_before_start_is_refused(self, tmp_path):
        with pytest.raises(EngineNotReadyError):
            VllmEngine(tmp_path).generate(
                ExtractionRequest(ocr_text="deed", prompt="extract"))


class TestTheCommandLine:
    def test_it_serves_the_checkpoint_not_the_gguf(self, tmp_path):
        argv = VllmEngine(tmp_path).build_argv()
        assert "--model" in argv
        assert not any(a.endswith(".gguf") for a in argv)

    def test_vram_is_not_claimed_entirely(self, tmp_path):
        """vLLM allocates its KV pool up front. Claiming all of the card is how
        the OCR stage starts failing with OOM on a shared machine."""
        argv = VllmEngine(tmp_path).build_argv()
        fraction = float(argv[argv.index("--gpu-memory-utilization") + 1])
        assert 0 < fraction < 1.0
        assert fraction == DEFAULT_GPU_FRACTION

    def test_max_model_len_is_omitted_unless_set(self, tmp_path):
        """vLLM derives it from the checkpoint; overriding with a guess would
        reject deeds that would otherwise have fit."""
        assert "--max-model-len" not in VllmEngine(tmp_path).build_argv()
        with_len = VllmEngine(tmp_path, max_model_len=8192).build_argv()
        assert with_len[with_len.index("--max-model-len") + 1] == "8192"

    def test_a_single_card_unless_asked_otherwise(self, tmp_path):
        argv = VllmEngine(tmp_path).build_argv()
        assert argv[argv.index("--tensor-parallel-size") + 1] == "1"


class TestProbeReportsRatherThanRaises:
    def test_a_missing_interpreter_is_reported(self):
        ok, detail = _probe_vllm("no-such-interpreter")
        assert ok is False
        assert detail

    def test_this_interpreter_is_answered_either_way(self):
        import sys

        ok, detail = _probe_vllm(sys.executable)
        assert isinstance(ok, bool)
        assert detail


class TestItIsSelectable:
    def test_the_server_offers_vllm_as_an_engine(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src" / "ai_server"
                  / "server.py").read_text(encoding="utf-8")
        assert '"vllm"' in source, "the engine cannot be chosen"
        assert 'engine_name == "vllm"' in source

    def test_batch_order_is_preserved(self, tmp_path):
        """The caller matches results to documents by position; returning them
        in completion order would attribute one deed's extraction to another."""
        engine = VllmEngine(tmp_path)
        seen = []

        def fake(request, timeout_s=600.0):
            seen.append(request.document_id)
            return request.document_id

        engine.generate = fake  # type: ignore[method-assign]
        requests = [ExtractionRequest(ocr_text="x", prompt="p", document_id=str(n))
                    for n in range(8)]
        assert engine.generate_batch(requests) == [str(n) for n in range(8)]
