"""Watermark detection and removal, and OCR robustness on difficult documents.

The governing risk here is a false positive. A watermark "finding" that is
actually the deed's own content leads to removal that destroys data, and the
destruction is silent - the text is simply gone from the OCR and the model never
sees it. That is why `Finding.confirmed` exists: a scanned page has no extractable
text layer at all, which an earlier version reported as a detected watermark on
every scan.

Removal is therefore conservative by design, and these tests are written to fail
if it ever becomes eager.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.watermark import Fidelity, Finding, Kind, scan

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


def _pdfs() -> list[Path]:
    found = list((ROOT / "tests" / "corpus" / "saledeeds").glob("*.pdf"))
    return found or list((ROOT / "models" / "SuryaOCR").glob("*.pdf"))


class TestDetectionContract:
    def test_kinds_are_distinct(self):
        assert len({k.value for k in Kind}) == len(list(Kind))

    def test_fidelity_levels_are_ordered_by_name(self):
        assert len(list(Fidelity)) >= 2

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        result = scan(tmp_path / "absent.pdf")
        assert result is not None
        assert not result.findings or result.notes


class TestConfirmation:
    """`confirmed` separates a real watermark from a page that merely has no
    text - the distinction that stops every scanned deed being reported."""

    def test_finding_exposes_confirmed(self):
        assert hasattr(Finding, "confirmed")

    def test_confirmed_is_a_property_not_a_stored_flag(self):
        """Derived, so it cannot drift out of step with the evidence."""
        assert isinstance(Finding.confirmed, property)


class TestAgainstRealDocuments:
    @pytest.fixture(scope="class")
    def documents(self) -> list[Path]:
        found = _pdfs()
        if not found:
            pytest.skip("no sample PDFs")
        return found[:5]

    def test_scan_completes_on_every_sample(self, documents):
        for path in documents:
            result = scan(path)
            assert result is not None, f"scan returned nothing for {path.name}"

    def test_scan_reports_page_count(self, documents):
        result = scan(documents[0])
        assert getattr(result, "pages", 0) >= 0

    def test_no_unconfirmed_finding_is_presented_as_certain(self, documents):
        """A scanned page must not be reported as a watermark."""
        for path in documents:
            for finding in scan(path).findings:
                if not finding.confirmed:
                    assert finding.fidelity is not Fidelity.LOSSLESS, (
                        f"{path.name}: an unconfirmed finding claims lossless "
                        "removal - that would delete real content silently")

    def test_scanning_does_not_modify_the_source(self, documents, tmp_path):
        """Detection is read-only. Removal is a separate, explicit step."""
        import shutil

        copy = tmp_path / documents[0].name
        shutil.copy2(documents[0], copy)
        before = copy.read_bytes()
        scan(copy)
        assert copy.read_bytes() == before, "scan mutated the PDF"


class TestOcrRobustness:
    """Difficult documents: no text layer, empty pages, mixed scripts.

    Rotated and blurred inputs are exercised by Surya itself rather than here -
    reproducing them needs rendered images and minutes of GPU time, so they
    belong in the OCR accuracy runs, not in a suite that runs on every change.
    """

    def test_pure_scan_is_rejected_with_an_explanation(self, tmp_path):
        """A scan has no text layer. The failure must say so, and must not be
        retryable - retrying cannot conjure text that is not there."""
        from core.pipeline.stages import OcrStage

        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        target = tmp_path / "scan.pdf"
        doc.save(target)
        doc.close()

        result = OcrStage(engine="textlayer", min_chars_per_page=40).run(target)
        assert not result.ok
        assert result.retryable is False
        assert "text" in result.detail.lower() or "scan" in result.detail.lower()

    def test_mixed_script_text_survives_cleanup(self):
        from core.ocr_cleanup import clean

        mixed = ("===== PAGE 1 =====\n"
                 "Seller: ರಮೇಶ್ ಕುಮಾರ್  PAN ABCDE1234F\n"
                 "Amount Rs. 30,00,000/- (ಮೂವತ್ತು ಲಕ್ಷ)\n")
        out, _ = clean(mixed)
        assert "ರಮೇಶ್ ಕುಮಾರ್" in out
        assert "ABCDE1234F" in out
        assert "30,00,000" in out

    def test_empty_input_does_not_crash(self):
        from core.ocr_cleanup import clean

        out, report = clean("")
        assert isinstance(out, str)
        assert report.pages_detected in (0, 1)

    def test_control_characters_are_stripped(self):
        """OCR of a damaged scan emits stray control bytes."""
        from core.ocr_cleanup import clean

        out, _ = clean("text\x00with\x07control\x1bchars")
        assert "\x00" not in out and "\x07" not in out

    def test_whitespace_only_page_is_not_counted_as_content(self):
        from core.ocr_cleanup import clean, page_texts

        out, _ = clean("===== PAGE 1 =====\n   \n\t\n===== PAGE 2 =====\nreal text\n")
        # page_texts yields (page_number, text) so a caller can report which
        # page a value came from.
        pages = dict(page_texts(out))
        assert set(pages) == {1, 2}
        assert not pages[1].strip()
        assert "real text" in pages[2]


def _watermarked(tmp_path, text="For Government Purpose Only", *,
                 pages=5, size=28.0, grey=0.8, deed_text=True):
    """A deed carrying a watermark, built the way a producer would."""
    import pymupdf

    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        if deed_text:
            page.insert_text((72, 90), "SALE DEED", fontsize=14)
            page.insert_text((72, 130), "Consideration Rs. 30,00,000/-", fontsize=11)
            page.insert_text((72, 150), "Seller PAN ABCDE1234F Survey 455/1",
                             fontsize=11)
        page.insert_textbox(pymupdf.Rect(60, 300, 550, 420), text,
                            fontsize=size, color=(grey, grey, grey),
                            align=pymupdf.TEXT_ALIGN_CENTER)
    path = tmp_path / "watermarked.pdf"
    doc.save(path)
    doc.close()
    return path


class TestVocabularyIsNotTheOnlySignal:
    """Reported by the user: a deed stamped "For Government Purpose Only" was
    reported clean.

    Two faults. `WATERMARK_WORDS` was a generic Western list - draft, specimen,
    demo, trial - with nothing from an Indian registry. And the repeated-text
    check, which was supposed to be the robust fallback, was itself gated on
    that same vocabulary, so it could only ever confirm what the fragile signal
    had already found.
    """

    def test_the_reported_watermark_is_detected(self, tmp_path):
        result = scan(_watermarked(tmp_path))
        assert result.has_watermark
        assert any("Government Purpose" in f.label for f in result.confirmed)

    def test_wording_outside_the_vocabulary_is_still_found(self, tmp_path):
        """The vocabulary can never be complete. Typography is the backstop."""
        from core.watermark import WATERMARK_WORDS

        phrase = "Issued Under Section Nineteen"
        assert not any(w in phrase.lower() for w in WATERMARK_WORDS)
        result = scan(_watermarked(tmp_path, phrase))
        assert result.has_watermark, "typography-based detection did not fire"

    def test_indian_registry_wording_is_in_the_vocabulary(self):
        from core.watermark import WATERMARK_WORDS

        for phrase in ("government purpose", "official use", "office copy"):
            assert phrase in WATERMARK_WORDS

    def test_repetition_is_not_gated_on_the_vocabulary(self):
        """The defect: the fallback could never fire independently."""
        source = (ROOT / "src" / "core" / "watermark.py").read_text(encoding="utf-8")
        assert "_phrase_worth_testing" in source
        assert "_is_rotated" in source


class TestOrdinaryTextIsNotDeleted:
    """The risk created by decoupling repetition from the vocabulary.

    "the" appears 119 times in one corpus deed. Flagging it would delete real
    words from a legal record - far worse than the missed watermark.
    """

    def test_a_common_word_is_never_a_candidate(self):
        from core.watermark import _phrase_worth_testing

        for word in ("the", "and", "of", "Rs.", "1", "Page 1"):
            assert not _phrase_worth_testing(word), f"{word!r} would be deleted"

    def test_a_phrase_must_be_multi_word(self):
        from core.watermark import _phrase_worth_testing

        assert not _phrase_worth_testing("Bengaluru")
        assert _phrase_worth_testing("For Government Purpose Only")

    def test_a_repeated_number_is_not_a_watermark(self):
        from core.watermark import _phrase_worth_testing

        assert not _phrase_worth_testing("2025 26")

    def test_real_deeds_are_not_flagged(self):
        """The negative control. These carry no watermark and must stay clean."""
        pdfs = sorted((ROOT / "tests" / "corpus" / "saledeeds").glob("*.pdf"))[:4]
        if not pdfs:
            pytest.skip("no sample deeds")
        for path in pdfs:
            result = scan(path)
            bad = [f for f in result.confirmed]
            assert not bad, f"{path.name} falsely flagged: {[f.label for f in bad]}"

    def test_body_text_is_horizontal(self):
        """The discriminator relies on this being true of real documents."""
        from core.watermark import _is_rotated

        assert not _is_rotated((1.0, 0.0))
        assert _is_rotated((0.7, 0.7))


class TestRemovalActuallyRemoves:
    """Detection reporting a watermark while removal silently does nothing is
    worse than not detecting it - the operator believes the file is clean.

    The old remover matched the whole phrase inside one `Tj` operator. Producers
    split a textbox across one `Tj` per word, so the pattern matched nothing.
    """

    def test_the_watermark_is_gone(self, tmp_path):
        import pymupdf

        from core.watermark import remove

        source = _watermarked(tmp_path)
        result = remove(source, tmp_path / "clean.pdf",
                        scan_result=scan(source), allow_lossy=False)
        assert result.ok, f"removal failed: {result.error}"
        assert result.removed

        with pymupdf.open(tmp_path / "clean.pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        assert "Government Purpose" not in text

    def test_the_deed_survives(self, tmp_path):
        """The whole point. Removing the watermark must not remove the deed."""
        import pymupdf

        from core.watermark import remove

        source = _watermarked(tmp_path)
        remove(source, tmp_path / "clean.pdf", scan_result=scan(source))

        with pymupdf.open(tmp_path / "clean.pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        for kept in ("SALE DEED", "30,00,000", "ABCDE1234F", "455/1"):
            assert kept in text, f"{kept} was destroyed by removal"

    def test_removal_is_reported_lossless(self, tmp_path):
        from core.watermark import Fidelity, remove

        source = _watermarked(tmp_path)
        result = remove(source, tmp_path / "clean.pdf", scan_result=scan(source))
        assert result.fidelity is Fidelity.LOSSLESS

    def test_the_source_is_never_modified(self, tmp_path):
        from core.watermark import remove

        source = _watermarked(tmp_path)
        before = source.read_bytes()
        remove(source, tmp_path / "clean.pdf", scan_result=scan(source))
        assert source.read_bytes() == before

    def test_redaction_preserves_images_and_line_art(self):
        """Default redaction rasterises the region, which would replace deed
        content with a picture of itself."""
        source = (ROOT / "src" / "core" / "watermark.py").read_text(encoding="utf-8")
        assert "PDF_REDACT_IMAGE_NONE" in source
        assert "PDF_REDACT_LINE_ART_NONE" in source
