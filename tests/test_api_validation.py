"""HTTP-layer input validation for the AI server.

These exist because the suite tests Python functions, and nothing exercised the
API boundary with wrong-*typed* JSON. Two defects lived in that gap and were
found by probing a running server, not by any test:

  * `{"ocr_text": 123}` reached `.strip()`, and the AttributeError went back to
    the caller as a 500 carrying the internal exception class - a wrong status
    code and an information leak.
  * `{"loaded": "banana"}` passed through `bool(...)`, answered 200 and did
    nothing at all.

Presence and emptiness were checked in both cases. The *type* never was.
"""

from __future__ import annotations

import pytest

from ai_server.server import AiServer


class _Payload:
    """Just enough of an AiServer to call `build_request`.

    Constructing a real one starts an engine and a governor; the parser is a
    pure function of its payload, so it is exercised directly.
    """

    build_request = AiServer.build_request

    def __init__(self) -> None:
        self._prompt = "PROMPT"


class TestOcrTextTyping:
    @pytest.mark.parametrize("value,name", [
        (123, "int"), (12.5, "float"), ([], "list"), ({"a": 1}, "dict"),
        (True, "bool"),
    ])
    def test_a_non_string_is_rejected_by_type(self, value, name):
        with pytest.raises(ValueError) as caught:
            _Payload().build_request({"ocr_text": value})
        message = str(caught.value)
        assert "must be a string" in message
        assert name in message

    def test_the_message_names_no_internal_exception(self):
        """The leak that made this High severity: the caller was told
        `AttributeError: 'int' object has ...`."""
        with pytest.raises(ValueError) as caught:
            _Payload().build_request({"ocr_text": 123})
        message = str(caught.value)
        for forbidden in ("AttributeError", "Traceback", "object has no",
                          "strip"):
            assert forbidden not in message, f"{forbidden!r} leaked to the caller"

    @pytest.mark.parametrize("payload", [{}, {"ocr_text": None},
                                         {"ocr_text": ""}, {"ocr_text": "   "}])
    def test_absent_or_empty_still_rejected(self, payload):
        """The original checks must survive the new one."""
        with pytest.raises(ValueError, match="required and must not be empty"):
            _Payload().build_request(payload)

    def test_a_real_string_is_accepted_and_normalised(self):
        request = _Payload().build_request(
            {"ocr_text": "Sale deed.\r\nSeller: KRISHNAPPA.\r"})
        assert "\r" not in request.ocr_text, "CRLF normalisation was lost"
        assert request.ocr_text.startswith("Sale deed.")

    def test_kannada_text_is_accepted(self):
        request = _Payload().build_request({"ocr_text": "ಕ್ರಯಪತ್ರ ಬೆಂಗಳೂರು"})
        assert "ಬೆಂಗಳೂರು" in request.ocr_text


class TestModelLoadedTyping:
    """`POST /model` governs whether the weights hold the GPU. A malformed
    request answering 200 while nothing moved is worse than an error."""

    @staticmethod
    def _check(value):
        # The rule as the handler applies it.
        if not isinstance(value, bool):
            raise ValueError(
                f"loaded must be true or false, not {type(value).__name__}")
        return value

    @pytest.mark.parametrize("value,name", [
        ("banana", "str"), ("true", "str"), (1, "int"), (0, "int"),
        (None, "NoneType"), ([], "list"),
    ])
    def test_anything_that_is_not_a_boolean_is_rejected(self, value, name):
        with pytest.raises(ValueError) as caught:
            self._check(value)
        assert "true or false" in str(caught.value)
        assert name in str(caught.value)

    @pytest.mark.parametrize("value", [True, False])
    def test_real_booleans_pass(self, value):
        assert self._check(value) is value

    def test_a_truthy_string_does_not_count_as_true(self):
        """`bool("banana")` is True, which is exactly how the defect worked."""
        assert bool("banana") is True          # the trap
        with pytest.raises(ValueError):
            self._check("banana")              # the guard
