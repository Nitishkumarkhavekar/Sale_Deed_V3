"""Surya OCR wiring and output handling.

The fixture is `SuryaOCR/275_ocr.txt` - real output from the Surya build that
produced the finetuning corpus. It is the only sample that proves cleanup handles
what Surya actually emits rather than what it was assumed to emit, which is how
the markup problem went unnoticed: the training corpus contains no `<b>` or
`<math>` tags at all, so nothing in the pipeline expected them.

These tests never invoke Surya. Loading its three models takes minutes and needs
1.4 GB of downloaded weights; the parts worth testing are the interpreter
discovery and the text handling, both of which are pure functions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.ocr_cleanup import CleanupOptions, clean, page_texts
from core.pipeline.stages import OcrStage, find_surya

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "models" / "SuryaOCR" / "275_ocr.txt"

requires_sample = pytest.mark.skipif(
    not SAMPLE.is_file(), reason="Surya sample output not present")


@pytest.fixture(scope="module")
def surya_text() -> str:
    return SAMPLE.read_text(encoding="utf-8")


class TestMarkupStripping:
    """Surya emits inline HTML and LaTeX; the model was finetuned without it."""

    def test_bold_tags_removed_content_kept(self):
        out, report = clean("Consideration <b>Rs.30,00,000/-</b> paid")
        assert "<b>" not in out and "</b>" not in out
        assert "Rs.30,00,000/-" in out
        assert report.markup_tags_removed == 2

    def test_math_wrapper_unwrapped_not_deleted(self):
        # The survey number inside is real data on a property schedule.
        out, _ = clean("Survey <math>455/1</math> measuring")
        assert "455/1" in out
        assert "math" not in out.lower()

    def test_latex_fraction_becomes_readable(self):
        out, _ = clean(r"extent 42\frac{1}{2} guntas")
        assert "42 1/2" in out
        assert "frac" not in out

    def test_line_break_becomes_space_not_nothing(self):
        # <br> joins two words; deleting it outright would fuse them.
        out, _ = clean("first<br>second")
        assert "first second" in out

    def test_can_be_disabled(self):
        out, _ = clean("<b>x</b>", CleanupOptions(strip_markup=False))
        assert "<b>" in out


class TestPageRules:
    def test_equals_rule_removed(self):
        out, report = clean("===== PAGE 1 =====\ntext\n" + "=" * 40 + "\n")
        assert report.page_rules_removed >= 1
        assert "=" * 40 not in out

    def test_page_marker_itself_survives(self):
        out, report = clean("--- PAGE 1 (275) ---\nbody\n" + "=" * 30)
        assert report.pages_detected == 1
        assert "PAGE 1" in out


@requires_sample
class TestAgainstRealSuryaOutput:
    def test_every_page_detected(self, surya_text):
        _, report = clean(surya_text)
        assert report.pages_detected == 5

    def test_no_markup_survives(self, surya_text):
        out, _ = clean(surya_text)
        assert not re.search(r"</?(?:b|i|u|br|math|span)[ />]", out, re.IGNORECASE)

    def test_kannada_preserved(self, surya_text):
        """Cleanup must not touch the script the deed is written in."""
        def kannada(text: str) -> int:
            return sum(1 for c in text if 0x0C80 <= ord(c) <= 0x0CFF)

        out, _ = clean(surya_text)
        assert kannada(out) == kannada(surya_text)

    def test_identifiers_survive(self, surya_text):
        out, _ = clean(surya_text)
        assert re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", out), "PAN lost"
        assert re.search(r"\d{4}\s?\d{4}\s?\d{4}", out), "Aadhaar lost"

    def test_amounts_survive(self, surya_text):
        before = set(re.findall(r"[\d,]{6,}", surya_text))
        after = set(re.findall(r"[\d,]{6,}", clean(surya_text)[0]))
        assert before <= after or before == after, "an amount was altered"

    def test_page_split_returns_five(self, surya_text):
        out, _ = clean(surya_text)
        assert len(page_texts(out)) == 5


class TestDiscovery:
    def test_returns_paths_or_none(self):
        interpreter, script = find_surya()
        # Both or neither - a script without an interpreter is useless, and the
        # caller branches on the interpreter alone.
        assert (interpreter is None) == (script is None)
        if interpreter is not None:
            assert interpreter.is_file()
            assert script.is_file()

    def test_missing_root_is_not_an_error(self, tmp_path):
        assert find_surya(tmp_path) == (None, None)


class TestStageConfiguration:
    def test_surya_claims_gpu(self):
        assert OcrStage(engine="surya").uses_gpu is True

    def test_textlayer_does_not_claim_gpu(self):
        """Taking the GPU lease for CPU work would serialise the pipeline."""
        assert OcrStage(engine="textlayer").uses_gpu is False

    def test_device_defaults_to_auto(self):
        assert OcrStage(engine="surya").device == "auto"

    def test_missing_interpreter_reported_not_raised(self, tmp_path):
        stage = OcrStage(engine="surya", surya_python=tmp_path / "nope.exe",
                         surya_script=tmp_path / "nope.py")
        ok, detail = stage.available()
        assert ok is False
        assert "not found" in detail

    def test_unknown_engine_is_reported(self):
        ok, detail = OcrStage(engine="tesseract").available()
        assert ok is False
        assert "unknown" in detail.lower()
