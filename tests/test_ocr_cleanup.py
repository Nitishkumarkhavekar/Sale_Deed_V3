"""OCR cleanup.

The governing constraint: this model was finetuned on OCR text, so cleanup must
move the input *closer* to what it saw in training, never further away. CRLF
normalisation is the measured case - raw CRLF produced 6,758 tokens where the
training tokenizer produced 6,408, and normalising fixed it exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ocr_cleanup import (
    PAGE_MARKER,
    CleanupOptions,
    clean,
    page_texts,
)

pytestmark = pytest.mark.unit

CORPUS = Path(__file__).resolve().parents[1] / "tests" / "corpus" / "OCR saledeeds"


class TestLineEndings:
    def test_crlf_normalised(self):
        out, report = clean("line one\r\nline two\r\n")
        assert "\r" not in out
        assert report.crs_removed == 2

    def test_lone_cr_normalised(self):
        out, _ = clean("line one\rline two")
        assert "\r" not in out

    def test_already_lf_is_unchanged(self):
        out, report = clean("line one\nline two\n")
        assert report.crs_removed == 0
        assert "line one\nline two" in out


class TestPageMarkers:
    def test_existing_banner_rewritten_not_duplicated(self):
        raw = ("=" * 70 + "\nPAGE 1  (117)\n" + "=" * 70 + "\nbody text\n"
               + "=" * 70 + "\nPAGE 2  (117)\n" + "=" * 70 + "\nmore text\n")
        out, report = clean(raw)
        assert report.pages_detected == 2
        assert report.page_markers_rewritten == 2
        assert out.count("PAGE 1") == 1, "marker duplicated"
        assert PAGE_MARKER.format(n=1) in out
        assert "=" * 70 not in out, "old banner left behind"

    def test_form_feed_becomes_markers(self):
        out, report = clean("page one\fpage two")
        assert report.pages_detected == 2
        assert report.page_markers_inserted == 2

    def test_document_without_structure_becomes_one_page(self):
        out, report = clean("just some text")
        assert report.pages_detected == 1
        assert PAGE_MARKER.format(n=1) in out
        assert any("single page" in note for note in report.notes)

    def test_page_texts_splits_back_out(self):
        raw = ("=" * 70 + "\nPAGE 1  (x)\n" + "=" * 70 + "\nalpha\n"
               + "=" * 70 + "\nPAGE 2  (x)\n" + "=" * 70 + "\nbeta\n")
        out, _ = clean(raw)
        pages = page_texts(out)
        assert [n for n, _ in pages] == [1, 2]
        assert "alpha" in pages[0][1]
        assert "beta" in pages[1][1]


class TestWhitespace:
    def test_trailing_whitespace_stripped(self):
        out, _ = clean("text with trailing   \nnext line\t\t\n")
        assert "   \n" not in out

    def test_blank_runs_collapsed(self):
        out, report = clean("a\n\n\n\n\nb")
        assert "\n\n\n" not in out
        assert report.blank_runs_collapsed >= 1

    def test_column_padding_preserved_by_default(self):
        """Padding is a column separator; the model trained on it."""
        raw = "field one" + " " * 40 + "field two"
        out, report = clean(raw)
        assert " " * 40 in out
        assert report.padding_runs_collapsed == 0

    def test_padding_collapsed_only_when_requested(self):
        raw = "field one" + " " * 40 + "field two"
        out, report = clean(raw, CleanupOptions(collapse_wide_padding=True,
                                                max_space_run=8))
        assert " " * 40 not in out
        assert report.padding_runs_collapsed >= 1


class TestControlCharacters:
    def test_control_chars_removed(self):
        out, report = clean("good\x00text\x07here")
        assert "\x00" not in out and "\x07" not in out
        assert report.control_chars_removed == 2

    def test_tabs_and_newlines_kept(self):
        out, _ = clean("a\tb\nc")
        assert "\t" in out and "\n" in out


class TestAggressiveOptionsAreOff:
    def test_unicode_normalisation_off_by_default(self):
        assert CleanupOptions().unicode_normalise is False

    def test_header_dropping_off_by_default(self):
        assert CleanupOptions().drop_repeated_headers is False

    def test_masked_identifier_blanking_off_by_default(self):
        assert CleanupOptions().blank_masked_identifiers is False

    def test_masked_aadhaar_kept_by_default(self):
        """The model is trained to emit null for these; removing the evidence
        that a masked value was present would be a loss of information."""
        out, _ = clean("Aadhaar Card No : XXXX XXXX 0976")
        assert "XXXX XXXX 0976" in out

    def test_masked_blanking_when_requested(self):
        out, report = clean("Aadhaar XXXX XXXX 0976 here",
                            CleanupOptions(blank_masked_identifiers=True))
        assert "0976" not in out
        assert report.masked_identifiers_blanked == 1


class TestReport:
    def test_reports_what_changed(self):
        """Cleanup does not always shrink.

        A document with no page structure gains a `===== PAGE 1 =====` marker,
        which on a very short input outweighs the whitespace removed. Real deeds
        shrink (see TestOnCorpus) because their three-line banners collapse to
        one and trailing padding goes. The report records both directions.
        """
        out, report = clean("text   \r\n\r\n\r\n\r\nmore   \r\n")
        assert report.crs_removed == 5
        assert report.blank_runs_collapsed >= 1
        assert "   \n" not in out, "trailing whitespace survived"
        assert "->" in report.summary()

    def test_real_document_shrinks(self):
        raw = ("=" * 70 + "\r\nPAGE 1  (x)\r\n" + "=" * 70 + "\r\n"
               + "body line   \r\n" * 40)
        _, report = clean(raw)
        assert report.chars_after < report.chars_before
        assert report.chars_saved > 0

    def test_empty_input_is_safe(self):
        out, report = clean("")
        assert report.chars_before == 0
        assert out.strip() == PAGE_MARKER.format(n=1)


@pytest.mark.regression
class TestOnCorpus:
    def test_every_file_cleans_and_shrinks(self):
        if not CORPUS.is_dir():
            pytest.skip("corpus not present")
        files = sorted(CORPUS.glob("*.txt"))[:10]
        if not files:
            pytest.skip("no corpus files")
        for path in files:
            raw = path.read_text(encoding="utf-8", errors="replace", newline="")
            out, report = clean(raw)
            assert "\r" not in out, f"{path.name}: CR survived"
            assert report.chars_after <= report.chars_before, f"{path.name}: grew"
            assert report.pages_detected >= 1, f"{path.name}: no pages detected"

    def test_cleanup_does_not_break_grounding(self):
        """Validation must reach the same verdict on raw and cleaned text."""
        import json

        from core.validation import validate_extraction

        json_dir = (Path(__file__).resolve().parents[1] / "tests" / "corpus" / "test scripts"
                    / "outputs" / "vllm_ocr")
        if not CORPUS.is_dir() or not json_dir.is_dir():
            pytest.skip("corpus not present")

        checked = 0
        for ocr_path in sorted(CORPUS.glob("*.txt"))[:5]:
            reference = json_dir / f"{ocr_path.stem}.json"
            if not reference.is_file():
                continue
            raw = ocr_path.read_text(encoding="utf-8", errors="replace", newline="")
            cleaned, _ = clean(raw)
            pred = json.loads(reference.read_text(encoding="utf-8"))
            before = validate_extraction(pred, raw)
            after = validate_extraction(pred, cleaned)
            assert before.disposition is after.disposition, (
                f"{ocr_path.stem}: cleanup changed the verdict "
                f"{before.disposition} -> {after.disposition}")
            checked += 1
        if not checked:
            pytest.skip("no OCR/reference pairs")
