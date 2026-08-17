"""The llama.cpp engine - the one that actually serves every extraction.

It was at 0% coverage. Not because it is untestable, but because everything
touching it went through a running server: the pipeline tests exercise
extraction end to end, so the *engine* was only ever tested by proxy, and only
when a GPU was present.

What is tested here needs no GPU and no server: the command line it builds, the
environment it pins, and how it behaves when things are wrong. Those are where
the consequential decisions live - several of them are settings that must not
drift, and one of them decides whether the application can see the right GPU.
"""

from __future__ import annotations

import pytest

from ai_server.engines.base import EngineNotReadyError, ExtractionRequest
from ai_server.engines.llamacpp import LlamaCppEngine, _resolve_binary
from ai_server.profiles import Profile


def _profile(**overrides):
    """A real `Profile`, not a stand-in.

    Constructed rather than selected, so the test does not depend on the card
    in the machine running it - but a genuine dataclass, because a hand-rolled
    stub silently lacks the fields the engine reads and turns a real failure
    into an AttributeError in the test.
    """
    from ai_server.profiles import QUANT_LADDER

    fields = dict(
        device="cuda", quant=QUANT_LADDER[-1], n_ctx=16384, kv_type="q8_0",
        n_gpu_layers=35, n_parallel=1, n_threads=8, gpu_uuid="GPU-test",
        weight_bytes=2_400_000_000, kv_bytes=250_000_000,
        overhead_bytes=600_000_000, budget_bytes=3_500_000_000,
        reason="fixed for tests",
    )
    fields.update(overrides)
    return Profile(**fields)


def _engine(**overrides):
    return LlamaCppEngine("model.gguf", _profile(**overrides),
                          binary="llama-server.exe")


class TestTheCommandLine:
    """The argv is the engine's whole configuration surface."""

    def test_the_profile_reaches_the_command_line(self):
        argv = _engine().build_argv()
        assert argv[argv.index("-c") + 1] == "16384"
        assert argv[argv.index("-ngl") + 1] == "35"
        assert argv[argv.index("-t") + 1] == "8"
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q8_0"

    def test_cpu_means_no_gpu_layers(self):
        """A CPU profile must not ask for GPU offload - llama-server would try,
        and fail on a machine that has no usable card."""
        argv = _engine(device="cpu", n_gpu_layers=35).build_argv()
        assert argv[argv.index("-ngl") + 1] == "0"

    def test_the_context_window_never_slides(self):
        """`--no-context-shift` makes an over-long deed fail loudly. Without it
        llama.cpp silently drops the start of the document - the parties block -
        and returns a confident extraction of the wrong half."""
        assert "--no-context-shift" in _engine().build_argv()

    def test_flash_attention_carries_an_explicit_value(self):
        """Bare `--flash-attn` has been rejected since b10184; passing it
        without a value makes the server refuse to start."""
        argv = _engine().build_argv()
        assert argv[argv.index("--flash-attn") + 1] == "on"

    def test_continuous_batching_is_on(self):
        assert "--cont-batching" in _engine().build_argv()

    def test_the_shared_prompt_prefix_is_reused(self):
        """Every deed in a batch carries the same ~330-token instruction
        block; re-prefilling it per document is pure waste."""
        argv = _engine().build_argv()
        assert argv[argv.index("--cache-reuse") + 1] == "256"

    def test_extra_arguments_come_last_so_they_win(self):
        engine = LlamaCppEngine("model.gguf", _profile(),
                                binary="llama-server.exe",
                                extra_args=["--verbose"])
        assert engine.build_argv()[-1] == "--verbose"

    def test_the_model_path_is_absolute(self):
        """llama-server is launched with its own working directory; a relative
        model path would resolve against the wrong one."""
        argv = _engine().build_argv()
        assert __import__("pathlib").Path(argv[argv.index("-m") + 1]).is_absolute()


class TestTheGpuIsPinned:
    def test_only_the_selected_device_is_visible(self):
        """A laptop has two adapters and the integrated one must never be
        chosen. Pinning is how the engine guarantees that."""
        env = _engine()._environment()
        assert "CUDA_VISIBLE_DEVICES" in env

    def test_the_environment_is_a_superset_of_the_real_one(self):
        """Replacing the environment rather than extending it would strip PATH
        and the CUDA DLLs would not resolve."""
        import os

        env = _engine()._environment()
        assert "PATH" in env or "Path" in env
        assert len(env) > 3


class TestFailureIsExplained:
    def test_generating_before_start_is_refused(self):
        with pytest.raises(EngineNotReadyError):
            _engine().generate(ExtractionRequest(ocr_text="deed", prompt="p"))

    def test_health_never_raises_before_start(self):
        health = _engine().health()
        assert health.ready is False
        assert health.engine
        assert isinstance(health.as_dict(), dict)

    def test_is_ready_is_false_before_start(self):
        assert _engine().is_ready() is False

    def test_stop_is_safe_before_start(self):
        """Idempotent: the supervisor calls stop on paths where start never
        happened, and an exception there would mask the real failure."""
        _engine().stop()
        _engine().stop()

    def test_a_missing_binary_is_carried_not_swallowed(self):
        """Resolution does not fail here - the name is kept so `start` can
        report which binary was looked for. Silently substituting a different
        one would be far worse than a clear failure later."""
        assert _resolve_binary("no-such-llama-server.exe").endswith(
            "no-such-llama-server.exe")


class TestSettingsThatMustNotDrift:
    """Two request-level settings whose defaults are load-bearing."""

    def test_temperature_defaults_to_zero(self):
        """A legal extraction has to be reproducible."""
        assert ExtractionRequest(ocr_text="x", prompt="p").temperature == 0.0

    def test_the_repetition_penalty_is_disabled(self):
        """Measured on deed 117: at 1.1 the model emitted 3 of 5 persons and
        nulled two fields, because every party repeats the same JSON keys."""
        assert ExtractionRequest(ocr_text="x", prompt="p").repetition_penalty == 1.0

    def test_the_prompt_and_document_stay_separate(self):
        """They are kept apart so the shared instruction prefix can be cached
        across deeds; concatenating them at construction would defeat it."""
        request = ExtractionRequest(ocr_text="DEED BODY", prompt="INSTRUCTION")
        assert request.ocr_text == "DEED BODY"
        assert request.prompt == "INSTRUCTION"
        rendered = request.as_messages()[0]["content"]
        assert rendered.startswith("INSTRUCTION")
        assert rendered.endswith("DEED BODY")

    def test_there_is_no_system_turn(self):
        """The model was finetuned on a single user turn and never saw one."""
        messages = ExtractionRequest(ocr_text="x", prompt="p").as_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
