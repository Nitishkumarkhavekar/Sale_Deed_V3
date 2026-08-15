"""Why a deed failed, in words an operator can act on.

`failure_reason` held strings like
`EngineNotReadyError: llama-server HTTP 400: {"error":{"code":400,...` - which
names the component that gave up, not the problem anyone has to fix. These
tests pin the classification and, above all, the ordering: the same HTTP 400
means two completely different things depending on the text inside it.
"""

from __future__ import annotations

import pytest

from core import failure_codes as fc


class _Doc:
    """A document as the classifier reads one."""

    def __init__(self, reason="", stage=None, validation=None, retryable=None):
        self.failure_reason = reason
        self.validation_status = validation
        self.validation_error_code = None
        self.validation_error_message = None
        self.is_retryable = retryable
        self.overall_state = type("S", (), {"value": "failed"})()
        for name in ("ocr", "extract", "translate", "validate"):
            value = "failed" if name == stage else "pending"
            setattr(self, f"{name}_state", type("S", (), {"value": value})())


class TestTheThreeHeadlineCauses:
    def test_a_corrupt_pdf_says_so(self):
        out = fc.classify(_Doc(validation="CORRUPTED_PDF", stage="ocr"))
        assert out["code"] == fc.PDF_CORRUPTED
        assert out["reason"] == "PDF file is corrupted or cannot be read."
        assert out["stage"] == "PDF Validation"
        assert out["retryable"] is False

    def test_ocr_producing_nothing_says_so(self):
        out = fc.classify(_Doc("OCR produced no text", stage="ocr"))
        assert out["code"] == fc.OCR_NO_TEXT
        assert "no readable text" in out["reason"]

    def test_a_watermark_failure_says_so(self):
        out = fc.classify(_Doc("watermark removal did not produce a clean copy",
                               stage="ocr"))
        assert out["code"] == fc.WATERMARK_REMOVAL_FAILED
        assert out["reason"] == "Watermark was not removed."


class TestOrderingIsTheWholeDesign:
    def test_a_context_overflow_is_a_length_problem_not_a_server_fault(self):
        """Both arrive as HTTP 400 from llama-server. One means 'the deed is too
        long', the other means 'the server is down' - opposite actions."""
        out = fc.classify(_Doc(
            'llama-server HTTP 400: {"message":"request (16618 tokens) exceeds '
            'the available context size (16384 tokens)"}', stage="extract"))
        assert out["code"] == fc.AI_INPUT_TOO_LARGE
        assert out["code"] != fc.AI_SERVER_UNAVAILABLE

    def test_a_refused_connection_is_a_server_fault(self):
        out = fc.classify(_Doc(
            "EngineNotReadyError: URLError: [WinError 10061] No connection could "
            "be made because the target machine actively refused it",
            stage="extract"))
        assert out["code"] == fc.AI_SERVER_UNAVAILABLE
        assert out["reason"] == "The AI server is not reachable."

    def test_the_validator_verdict_beats_the_stage_text(self):
        """The validator examined the bytes; OCR only saw a symptom."""
        out = fc.classify(_Doc("PdfiumError: Failed to load document",
                               stage="ocr", validation="INCOMPLETE_PDF"))
        assert out["code"] == fc.PDF_INCOMPLETE


class TestRealStringsFromThisMachine:
    @pytest.mark.parametrize("reason,stage,expected", [
        ("response contained no parseable JSON", "extract", fc.AI_EXTRACTION_FAILED),
        ("OCR timed out after 900s", "ocr", fc.TIMEOUT),
        ("CUDA out of memory. Tried to allocate", "ocr", fc.MEMORY_ERROR),
        ("only 40 characters across 14 pages - the PDF has no usable text layer",
         "ocr", fc.OCR_NO_TEXT),
        ("PermissionError: [Errno 13] Permission denied", "ocr", fc.FILE_ACCESS_ERROR),
    ])
    def test_each_is_classified(self, reason, stage, expected):
        assert fc.classify(_Doc(reason, stage=stage))["code"] == expected

    def test_an_unrecognised_message_falls_back_to_the_stage(self):
        out = fc.classify(_Doc("something nobody anticipated", stage="translate"))
        assert out["code"] == fc.TRANSLATION_FAILED
        assert out["stage"] == "Translation"


class TestNothingLeaksToTheOperator:
    def test_a_stack_trace_is_stripped(self):
        trace = "\n".join([
            "boom",
            "Traceback (most recent call last):",
            '  File ' + repr("D:" + chr(92) + "secret" + chr(92) + "x.py"),
        ])
        out = fc.classify(_Doc(trace, stage="extract"))
        assert "Traceback" not in out["technical"]
        assert "secret" not in out["technical"]

    def test_the_reason_is_never_the_raw_exception(self):
        raw = 'EngineNotReadyError: llama-server HTTP 400: {"error":{"code":400}}'
        out = fc.classify(_Doc(raw, stage="extract"))
        assert out["reason"] != raw
        assert "EngineNotReadyError" not in out["reason"]
        # ...but it stays reachable for whoever debugs it.
        assert "llama-server" in out["technical"]

    def test_technical_text_is_bounded(self):
        out = fc.classify(_Doc("x" * 5000, stage="ocr"))
        assert len(out["technical"]) <= 400


class TestSuccessIsNotMisreported:
    def test_a_document_that_did_not_fail_returns_nothing(self):
        doc = _Doc()
        doc.overall_state = type("S", (), {"value": "processed"})()
        assert fc.classify(doc) is None

    def test_every_code_has_a_message_and_a_stage(self):
        for name, code in vars(fc).items():
            if name.isupper() and isinstance(code, str) and code == name:
                assert code in fc.MESSAGES, f"{code} has no message"
                assert code in fc.STAGES, f"{code} has no stage"

    def test_the_stored_retryable_flag_wins(self):
        """The PDF validator already decided; the classifier must not overrule."""
        out = fc.classify(_Doc("OCR failed", stage="ocr", retryable=False))
        assert out["retryable"] is False
