"""Document preparation: overlay removal, text layer, pipeline integration.

Preparation sits between upload and OCR and produces the file everything
downstream reads. Two properties matter more than anything else here:

* **The original is never modified.** It is the record.
* **Preparation is never a precondition.** A deed that cannot be cleaned must
  still be processed - an improvement to the input, not a gate on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.pdf_prepare import MIN_CHARS_PER_PAGE, pages_without_text, prepare

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit

WATERMARK = "For Government Purpose Only"
DEED = ("SALE DEED", "Consideration Rs. 30,00,000/-", "Seller PAN ABCDE1234F")


def _deed(tmp_path, *, watermark=True, pages=4, blank=False):
    import pymupdf

    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if not blank:
            for i, line in enumerate(DEED):
                page.insert_text((72, 90 + i * 22), line, fontsize=11)
        if watermark:
            page.insert_textbox(pymupdf.Rect(50, 300, 560, 430), WATERMARK,
                                fontsize=30, color=(0.78, 0.78, 0.78),
                                align=pymupdf.TEXT_ALIGN_CENTER)
    path = tmp_path / "deed.pdf"
    doc.save(path)
    doc.close()
    return path


def _text(path):
    import pymupdf

    with pymupdf.open(path) as doc:
        return "".join(page.get_text() for page in doc)


class TestPreparation:
    def test_the_overlay_is_removed(self, tmp_path):
        result = prepare(_deed(tmp_path), tmp_path / "out")
        assert result.ok and result.changed
        assert WATERMARK not in _text(result.output)

    def test_the_deed_survives(self, tmp_path):
        result = prepare(_deed(tmp_path), tmp_path / "out")
        cleaned = _text(result.output)
        for line in DEED:
            assert line in cleaned, f"{line!r} was destroyed"

    def test_the_original_is_never_modified(self, tmp_path):
        source = _deed(tmp_path)
        before = source.read_bytes()
        prepare(source, tmp_path / "out")
        assert source.read_bytes() == before

    def test_output_is_written_even_with_nothing_to_remove(self, tmp_path):
        """One file downstream either way - the caller should not need a
        conditional to know what it is reading."""
        result = prepare(_deed(tmp_path, watermark=False), tmp_path / "out")
        assert result.ok
        assert result.output.is_file()
        assert result.changed is False

    def test_a_clean_document_is_left_alone(self, tmp_path):
        source = _deed(tmp_path, watermark=False)
        result = prepare(source, tmp_path / "out")
        assert _text(result.output) == _text(source)

    def test_a_missing_file_is_reported_not_raised(self, tmp_path):
        result = prepare(tmp_path / "absent.pdf", tmp_path / "out")
        assert result.ok is False
        assert "not found" in result.error

    def test_timing_is_recorded(self, tmp_path):
        result = prepare(_deed(tmp_path), tmp_path / "out")
        assert result.seconds >= 0

    def test_the_result_serialises_for_logging(self, tmp_path):
        data = prepare(_deed(tmp_path), tmp_path / "out").as_dict()
        for key in ("watermarks_removed", "watermarks_skipped", "scanned_pages",
                    "fidelity", "seconds", "searchable"):
            assert key in data


class TestTextLayer:
    def test_a_digital_deed_is_already_selectable(self, tmp_path):
        result = prepare(_deed(tmp_path), tmp_path / "out")
        assert pages_without_text(result.output) == []
        assert result.searchable

    def test_a_page_without_text_is_identified(self, tmp_path):
        source = _deed(tmp_path, watermark=False, blank=True)
        assert pages_without_text(source), "a blank page was called selectable"

    def test_the_threshold_matches_the_ocr_stage(self):
        """Both decide what "scanned" means. If they disagree, one will OCR a
        page the other considers to have text."""
        from core.pipeline.stages import OcrStage

        assert MIN_CHARS_PER_PAGE == OcrStage(engine="textlayer").min_chars_per_page

    def test_an_invisible_layer_is_used(self):
        """Render mode 3 places selectable glyphs without drawing them, so a
        scan looks exactly as it did."""
        source = (ROOT / "src" / "core" / "pdf_prepare.py").read_text(encoding="utf-8")
        assert "_INVISIBLE = 3" in source
        assert "render_mode=_INVISIBLE" in source


class TestBurnedInOverlaysAreRefused:
    """A seal or signature in a scan is pixels. Removing it means inventing
    content on a document that establishes ownership of property."""

    def test_the_refusal_is_documented(self):
        source = (ROOT / "src" / "core" / "pdf_prepare.py").read_text(encoding="utf-8")
        assert "inventing content" in source
        assert "refuses" in source.lower()

    def test_lossy_removal_is_off_by_default(self):
        import inspect

        signature = inspect.signature(prepare)
        assert signature.parameters["allow_lossy"].default is False

    def test_scanned_pages_are_reported(self, tmp_path):
        result = prepare(_deed(tmp_path, watermark=False, blank=True),
                         tmp_path / "out")
        assert result.scanned_pages, "a scan was not reported as one"


class TestPipelineIntegration:
    RUNNER = ROOT / "src" / "core" / "pipeline" / "runner.py"

    def test_the_pipeline_reads_the_prepared_copy(self):
        source = self.RUNNER.read_text(encoding="utf-8")
        assert "doc.cleaned_path or doc.source_path" in source

    def test_preparation_runs_before_ocr(self):
        source = self.RUNNER.read_text(encoding="utf-8")
        prepare_at = source.index("self._prepare_document(")
        ocr_at = source.index("self._do_ocr(doc_pk, pdf_path)")
        assert prepare_at < ocr_at, "OCR would read the unprepared file"

    def test_a_retry_does_not_redo_the_work(self):
        """Re-running removal on an already-cleaned file would search for a
        watermark that is no longer there."""
        source = self.RUNNER.read_text(encoding="utf-8")
        assert "if doc.cleaned_path and Path(doc.cleaned_path).is_file():" in source

    def test_failure_falls_back_to_the_original(self):
        source = self.RUNNER.read_text(encoding="utf-8")
        block = source[source.index("def _prepare_document"):]
        block = block[:block.index("\n    def ")]
        assert "return pdf_path" in block

    def test_there_is_one_preparation_path(self):
        """Requirement: no duplicate processing flows."""
        source = self.RUNNER.read_text(encoding="utf-8")
        assert source.count("_prepare_document(") == 2  # definition + one call

    def test_the_column_exists(self):
        from core.db.models import Document

        assert hasattr(Document, "cleaned_path")


class TestViewer:
    def test_the_viewer_shows_the_prepared_copy(self):
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        block = source[source.index("def document_pdf"):]
        block = block[:block.index("\n    def ")]
        assert "doc.cleaned_path, doc.source_path" in block

    def test_the_viewer_takes_an_id_not_a_path(self):
        """A path parameter would turn a viewer into a file reader."""
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        block = source[source.index("def document_pdf"):]
        block = block[:block.index("\n    def ")]
        assert "isdigit()" in block

    def test_the_built_in_pdf_viewer_is_enabled(self):
        source = (ROOT / "src" / "app" / "main.py").read_text(encoding="utf-8")
        assert "PdfViewerEnabled, True" in source
        assert "PluginsEnabled, True" in source, \
            "the viewer is a plugin; PdfViewerEnabled alone does nothing"

    def test_the_view_button_is_wired(self):
        """It existed since the Data View was written with no handler - the
        dead-control audit only checked id= attributes, not data- ones."""
        js = (ROOT / "src" / "app" / "ui" / "assets" / "app.js").read_text(encoding="utf-8")
        assert "d.viewDocument" in js
        assert 'openModal("pdf_viewer"' in js

    def test_copy_text_serves_the_extracted_text(self):
        """What the operator copies must be what the model was given."""
        source = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        block = source[source.index("def document_text"):]
        block = block[:block.index("\n    def ")]
        assert "uow.ocr.full_text(doc)" in block

    def test_the_template_exists_and_parses(self):
        import pystache

        path = ROOT / "src" / "app" / "ui" / "templates" / "pdf_viewer.mustache"
        assert path.is_file()
        pystache.parse(path.read_text(encoding="utf-8"))


class TestScannedPagesBecomeSearchable:
    """The text layer, from the OCR boxes through to selectable text on a page.

    All of this existed and none of it ran: `add_text_layer` was referenced
    nowhere outside its own definition, so a scanned deed was cleaned but never
    made searchable and "Copy Text" returned nothing on exactly the pages that
    needed it. See R-032.
    """

    KANNADA = "ಶ್ರೀ ರಮೇಶ್ ಕುಮಾರ್ ಬೆಂಗಳೂರು ದಕ್ಷಿಣ"
    ENGLISH = "SALE DEED No. 3-2025-26 dated 14/05/2025"

    def _scanned(self, tmp_path):
        """A page with no text layer, as a scan has."""
        return _deed(tmp_path, watermark=False, pages=2, blank=True)

    def test_the_runner_actually_calls_it(self):
        """The defect was not a broken function - it was an uncalled one."""
        import inspect

        from core.pipeline.runner import BatchRunner

        source = inspect.getsource(BatchRunner)
        assert "add_text_layer" in source, (
            "nothing calls add_text_layer; scanned deeds are not searchable")
        assert "_make_searchable" in inspect.getsource(BatchRunner._do_ocr)

    def test_ocr_boxes_survive_the_subprocess_boundary(self):
        """Surya's line boxes exist only inside its own interpreter. If the
        runner does not emit them, the text layer has nothing to place."""
        runner = (ROOT / "src" / "tools" / "surya_runner.py").read_text(encoding="utf-8")
        assert "def line_boxes" in runner
        assert '"lines": boxes' in runner

        stages = (ROOT / "src" / "core" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        assert '"--json"' in stages, "the plain-text mode carries no boxes"
        assert "lines=lines" in stages

    def test_a_scanned_page_becomes_selectable(self, tmp_path):
        import pymupdf

        from core.pdf_prepare import add_text_layer

        path = self._scanned(tmp_path)
        assert pages_without_text(path) == [1, 2]

        with pymupdf.open(path) as doc:
            rect = doc[0].rect
        written = add_text_layer(path, {1: [
            ((0.08 * rect.width, 0.10 * rect.height,
              0.92 * rect.width, 0.13 * rect.height), self.ENGLISH)]})

        assert written == 1
        assert self.ENGLISH in _text(path).replace("\xa0", " ")

    def test_kannada_round_trips(self, tmp_path):
        """The whole point. `helv` cannot encode Kannada - it writes mojibake -
        so a font that can must be chosen per line."""
        import pymupdf

        from core.pdf_prepare import add_text_layer

        path = self._scanned(tmp_path)
        with pymupdf.open(path) as doc:
            rect = doc[0].rect
        add_text_layer(path, {1: [
            ((0.08 * rect.width, 0.10 * rect.height,
              0.92 * rect.width, 0.13 * rect.height), self.KANNADA)]})

        got = _text(path).replace("\xa0", " ")
        assert self.KANNADA in got, f"Kannada did not survive: {got!r}"
        assert "\x00" not in got and "\ufffd" not in got

    def test_the_page_count_returned_is_not_a_lie(self, tmp_path):
        """`insert_textbox` writes nothing and returns a negative number when
        the text does not fit; the count used to increment regardless, so it
        reported success over an empty page."""
        import pymupdf

        from core.pdf_prepare import add_text_layer

        path = self._scanned(tmp_path)
        with pymupdf.open(path) as doc:
            rect = doc[0].rect
        long_line = "SELLER " * 40
        written = add_text_layer(path, {1: [
            ((0, 0, rect.width, 12), long_line)]})

        if written:
            assert "SELLER" in _text(path), (
                "reported a page written but no text is selectable")

    def test_nothing_unencodable_is_ever_written(self, tmp_path, monkeypatch):
        """With no suitable font the correct output is *no text*. Question marks
        or null bytes would become the searchable content of a legal document."""
        import pymupdf

        from core import pdf_prepare

        monkeypatch.setattr(pdf_prepare, "_unicode_font", lambda: "")
        path = self._scanned(tmp_path)
        with pymupdf.open(path) as doc:
            rect = doc[0].rect
        pdf_prepare.add_text_layer(path, {1: [
            ((0, 0, rect.width, 20), self.KANNADA)]})

        got = _text(path)
        assert "\x00" not in got
        assert "?" not in got
        assert "\ufffd" not in got

    def test_pages_that_already_have_text_are_left_alone(self, tmp_path):
        """Writing a second layer over real text makes every selection double."""
        import pymupdf

        from core.pipeline.runner import BatchRunner

        path = _deed(tmp_path, watermark=False, pages=3)
        before = _text(path)
        with pymupdf.open(path) as doc:
            total = doc.page_count

        lines = [[[0.1, 0.1, 0.9, 0.13, self.ENGLISH]] for _ in range(total)]
        BatchRunner._make_searchable(
            BatchRunner.__new__(BatchRunner), str(path), lines)

        assert _text(path) == before, "an already-searchable page was rewritten"

    def test_a_failure_never_costs_the_document(self, tmp_path):
        """Searchability is a bonus; the OCR text is already in the database."""
        from core.pipeline.runner import BatchRunner

        runner = BatchRunner.__new__(BatchRunner)
        # Nonexistent path, malformed boxes - neither may raise.
        BatchRunner._make_searchable(runner, str(tmp_path / "gone.pdf"), [[]])
        BatchRunner._make_searchable(runner, str(self._scanned(tmp_path)),
                                     [[["not", "a", "box"]]])
