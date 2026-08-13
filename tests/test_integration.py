"""Integration and end-to-end testing.

Integration here means the seams: stage to stage, pipeline to database, service
to status layer. Each is exercised with real objects on both sides - a test that
mocks both sides of a seam proves only that the mock works.

The end-to-end tests run a real PDF through OCR, extraction, validation and CSV
export using a **stub extraction engine** rather than the language model. That is
deliberate. Model output quality belongs in AI model validation, where it is
measured against reference extractions; what these tests establish is that the
plumbing carries a document from file to CSV without losing or corrupting it, and
they must stay fast enough to run on every change.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.csv_export import CSV_COLUMNS, DocumentExport, write_csv
from core.ocr_cleanup import clean, page_texts
from core.pipeline.stages import (
    OcrStage,
    StageName,
    StageOutcome,
    TranslateStage,
    ValidateStage,
)
from core.validation import validate_extraction

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stage seams
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStageContract:
    """Every stage returns the same envelope, which is what lets the runner
    treat them uniformly."""

    def test_success_carries_data(self):
        out = StageOutcome.success(StageName.OCR, text="hello", pages=1)
        assert out.ok and out.data["pages"] == 1

    def test_failure_carries_a_reason(self):
        out = StageOutcome.failure(StageName.OCR, "no text")
        assert not out.ok and out.detail == "no text"

    def test_failures_are_not_retryable_by_default(self):
        """Retrying a deterministic failure only wastes GPU time."""
        assert StageOutcome.failure(StageName.OCR, "x").retryable is False

    def test_retryable_must_be_asked_for(self):
        assert StageOutcome.failure(StageName.OCR, "x", retryable=True).retryable


@pytest.mark.unit
class TestOcrToCleanupSeam:
    """OCR hands raw text to cleanup; cleanup must not lose page structure."""

    def test_page_markers_survive_into_page_texts(self):
        raw = "===== PAGE 1 =====\nfirst\n===== PAGE 2 =====\nsecond\n"
        cleaned, report = clean(raw)
        pages = dict(page_texts(cleaned))
        assert report.pages_detected == 2
        assert set(pages) == {1, 2}
        # Indexed by page number, not position: a caller reporting "value not
        # found on page 3" needs the real number.
        assert "first" in pages[1] and "second" in pages[2]

    def test_empty_ocr_is_not_silently_accepted(self, tmp_path):
        stage = OcrStage(engine="textlayer")
        result = stage.run(tmp_path / "does_not_exist.pdf")
        assert not result.ok
        assert "not found" in result.detail


@pytest.mark.unit
class TestExtractionToValidationSeam:
    def test_validation_accepts_extraction_shape(self):
        extraction = {
            "document_number": "1234/2024-25",
            "transaction_date": "2024-06-15",
            "consideration_amount": 3000000,
            "buyer_details": [{"name": "A", "pan_card_number": "ABCDE1234F"}],
            "seller_details": [{"name": "B"}],
        }
        report = validate_extraction(extraction, "Rs. 30,00,000 ABCDE1234F")
        assert report is not None
        assert hasattr(report, "document_flags")
        assert hasattr(report, "disposition")
        assert 0.0 <= report.confidence <= 1.0

    def test_validation_stage_wraps_the_report(self):
        stage = ValidateStage()
        out = stage.run({"buyer_details": [], "seller_details": []}, "some ocr text")
        assert out.stage is StageName.VALIDATE
        assert out.ok in (True, False)


@pytest.mark.unit
class TestTranslationSeam:
    def test_passthrough_does_not_alter_fields(self):
        extraction = {"buyer_details": [{"name": "ಕುಮಾರ"}], "seller_details": []}
        before = json.dumps(extraction, ensure_ascii=False, sort_keys=True)
        TranslateStage(engine="passthrough").run(extraction)
        after = json.dumps(extraction, ensure_ascii=False, sort_keys=True)
        assert before == after

    def test_availability_is_reported_either_way(self):
        """The stage must state plainly whether it can translate.

        A silent pass would put untranslated Kannada into an English column;
        a silent failure would hide that the model is missing. Both directions
        carry a reason.
        """
        ok, detail = TranslateStage().available()
        assert detail, "no explanation either way"
        if not ok:
            assert "no model weights" in detail or "not found" in detail                 or "disabled" in detail

    def test_disabled_translation_is_stated(self):
        ok, detail = TranslateStage(engine="passthrough").available()
        assert ok is False
        assert "disabled" in detail.lower()


# ---------------------------------------------------------------------------
# End to end: PDF -> CSV
# ---------------------------------------------------------------------------


#: Shaped like a real extraction: the model nests document metadata under
#: `document_details`, and the CSV writer reads it from there. A flat stub passes
#: the plumbing tests while writing an empty Transaction Date column, which is
#: exactly the kind of near-miss these tests exist to catch.
STUB_EXTRACTION = {
    "document_details": {
        "document_number": "275/2024-25",
        "transaction_date": "2024-06-15",
        "consideration_amount": 3000000,
        "registration_fee": 60000,
    },
    "buyer_details": [
        {"name": "Buyer One", "pan_card_number": "ABCDE1234F",
         "aadhaar_number": "123456789012", "address": "Bengaluru 560001"},
    ],
    "seller_details": [
        {"name": "Seller One", "pan_card_number": "ZYXWV9876E"},
    ],
    "property_details": {"survey_number": "455/1", "extent": "42 1/2 guntas"},
}


@pytest.mark.unit
class TestEndToEndWithoutModel:
    """PDF -> OCR -> cleanup -> (stub extraction) -> validate -> CSV."""

    @pytest.fixture()
    def a_pdf(self) -> Path:
        candidates = list((ROOT / "tests" / "corpus" / "saledeeds").glob("*.pdf"))
        if not candidates:
            candidates = list((ROOT / "models" / "SuryaOCR").glob("*.pdf"))
        if not candidates:
            pytest.skip("no sample PDF available")
        return candidates[0]

    def test_ocr_produces_text(self, a_pdf):
        result = OcrStage(engine="textlayer").run(a_pdf)
        if not result.ok:
            pytest.skip(f"no text layer in {a_pdf.name}: {result.detail}")
        assert result.data["chars"] > 0
        assert result.data["pages"] >= 1

    def test_full_chain_reaches_csv(self, a_pdf, tmp_path):
        ocr = OcrStage(engine="textlayer").run(a_pdf)
        if not ocr.ok:
            pytest.skip(f"no text layer in {a_pdf.name}")

        report = validate_extraction(STUB_EXTRACTION, ocr.data["text"])
        export = DocumentExport(
            transaction_identity="275/2024-25",
            extraction=STUB_EXTRACTION,
            report=report,
            source_filename=a_pdf.name)

        target = tmp_path / "export.csv"
        written = write_csv(target, [export])

        assert written >= 1, "no rows were written"
        with open(target, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == list(CSV_COLUMNS), "header drifted from the 42 columns"
        assert len(rows) == written + 1

    def test_values_survive_the_whole_chain(self, tmp_path):
        """The point of the pipeline is that data arrives intact."""
        export = DocumentExport(
            transaction_identity="275/2024-25",
            extraction=STUB_EXTRACTION,
            report=validate_extraction(STUB_EXTRACTION, "Rs. 30,00,000"),
            source_filename="275.pdf")
        target = tmp_path / "export.csv"
        write_csv(target, [export])

        body = target.read_text(encoding="utf-8-sig")
        for expected in ("Buyer One", "Seller One", "ABCDE1234F", "275/2024-25"):
            assert expected in body, f"{expected} was lost between stages"

    def test_date_is_written_in_indian_format(self, tmp_path):
        """DD-MM-YYYY, not ISO - the CSV is read by people, and 06-07 is
        ambiguous in the wrong order."""
        export = DocumentExport(
            transaction_identity="X", extraction=STUB_EXTRACTION,
            report=None, source_filename="x.pdf")
        target = tmp_path / "e.csv"
        write_csv(target, [export])
        body = target.read_text(encoding="utf-8-sig")
        assert "15-06-2024" in body
        assert "2024-06-15" not in body

    def test_kannada_survives_the_export(self, tmp_path):
        """utf-8-sig: without the BOM Excel renders Kannada as mojibake."""
        extraction = dict(STUB_EXTRACTION)
        extraction["buyer_details"] = [{"name": "ರಮೇಶ್ ಕುಮಾರ್"}]
        target = tmp_path / "kn.csv"
        write_csv(target, [DocumentExport(
            transaction_identity="K", extraction=extraction,
            source_filename="k.pdf")])
        assert target.read_bytes().startswith(b"\xef\xbb\xbf"), "BOM missing"
        assert "ರಮೇಶ್ ಕುಮಾರ್" in target.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Pipeline against the database
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPipelinePersistence:
    def test_ocr_text_round_trips_through_the_database(self, session_factory,
                                                       temp_batch):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        pages = [(1, "===== PAGE 1 =====\nfirst page"), (2, "second page")]
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch)
            if doc is None:
                pytest.skip("no claimable document")
            uow.ocr.save_pages(doc, pages)
            session.flush()
            stored = uow.ocr.full_text(doc)
        assert "first page" in stored and "second page" in stored

    def test_batch_progress_reflects_document_states(self, session_factory,
                                                     temp_batch):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        with session_scope(session_factory) as session:
            progress = UnitOfWork(session).batches.progress(temp_batch)
        assert progress is not None
        assert progress.total == 3
        assert 0.0 <= progress.percent <= 100.0


@pytest.mark.integration
class TestRunnerWiring:
    """The runner must assemble against a real session factory without a model."""

    def test_runner_constructs_and_reports_idle(self, session_factory):
        from core.pipeline.runner import BatchRunner, RunnerState, build_stages

        runner = BatchRunner(session_factory, build_stages(ocr_engine="textlayer"))
        assert runner.state is RunnerState.IDLE
        assert isinstance(runner.status(), dict)

    def test_recover_is_safe_to_call_on_a_clean_database(self, session_factory):
        from core.pipeline.runner import BatchRunner, build_stages

        runner = BatchRunner(session_factory, build_stages(ocr_engine="textlayer"))
        assert runner.recover() >= 0


class TestModelLifecycle:
    """Releasing and reloading the weights without stopping the service.

    R-035. On a 4 GiB card the language model and Surya cannot co-reside, and on
    a 7.4 GiB machine they cannot share host RAM either. OCR OOMed on the GPU,
    and forcing it to the CPU only moved the failure - there was no room there
    either. `llama-server` is a separate process, so no lock inside the
    application could make it let go. It has to be asked.
    """

    def test_the_server_exposes_a_release_route(self):
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        assert 'path == "/model"' in source, "no route to release the weights"
        assert "engine.stop()" in source and "engine.start()" in source

    def test_the_service_stays_up_while_unloaded(self):
        """Releasing weights must not stop the process. /health has to keep
        answering, reporting `loaded: false` - which the UI already renders as
        degraded rather than offline."""
        source = (ROOT / "src" / "ai_server" / "server.py").read_text(encoding="utf-8")
        model_route = source[source.index('path == "/model"'):]
        model_route = model_route[:model_route.index("if path ==", 10)]
        assert "shutdown" not in model_route, "releasing weights must not stop the server"

    def test_ocr_frees_the_card_before_running_surya(self):
        source = (ROOT / "src" / "core" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        assert "_free_the_gpu" in source
        surya = source[source.index("def _run_surya"):]
        assert "_free_the_gpu" in surya[:surya.index("subprocess.run")], (
            "the card is not freed before Surya is launched")

    def test_extraction_reloads_what_ocr_released(self):
        """Whoever needs the model brings it back. If extraction did not, the
        pipeline would release the weights and then report itself not ready."""
        source = (ROOT / "src" / "core" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        health = source[source.index("def health(self)"):]
        health = health[:health.index("\n    def ", 10)]
        assert '"loaded": True' in health, "extraction never reloads the model"

    def test_releasing_is_best_effort(self):
        """An old server without the route, or no server at all, must not stop
        a document being processed. This is an optimisation, not a precondition."""
        from core.pipeline.stages import OcrStage

        stage = OcrStage(engine="surya", ai_base_url="http://127.0.0.1:9")
        assert stage._free_the_gpu() is False      # nothing there; no exception


class TestTheExtractionPromptIsActuallyLoaded:
    """The model is useless without its prompt, and said so only in prose.

    R-040. `build_default` and `build_stages` both defaulted the prompt to a
    path *relative to the working directory*. When `saledeed main/` moved under
    `models/` the path stopped resolving, and both sites treat a missing prompt
    file as an ordinary condition - `prompt = ""` and carry on.

    The model then received OCR text with no instruction and did what an
    instruction-tuned model does with an unlabelled wall of text: wrote a
    Markdown summary of the deed, ending "Consult a lawyer for advice on your
    specific situation." Every extraction failed with "no parseable JSON" and
    every CSV column came out empty.
    """

    def test_the_prompt_file_exists_where_the_code_looks(self):
        from core import paths

        assert paths.PROMPT_FILE.is_file(), (
            f"no extraction prompt at {paths.PROMPT_FILE}")
        assert paths.PROMPT_FILE.read_text(encoding="utf-8").strip()

    def test_the_defaults_are_absolute(self):
        """A relative default resolves against whatever directory the process
        happened to start in, which is the bug in one sentence."""
        import inspect

        from ai_server.server import build_default
        from core.pipeline.runner import build_stages

        for func in (build_default, build_stages):
            default = inspect.signature(func).parameters["prompt_file"].default
            assert Path(default).is_absolute(), (
                f"{func.__name__} has a relative prompt default: {default!r}")

    def test_build_stages_loads_a_prompt(self):
        from core.pipeline.runner import build_stages

        prompt = build_stages().extract.prompt
        assert prompt, "the pipeline would send no instruction at all"
        assert "JSON" in prompt, "this does not look like the extraction prompt"

    def test_an_empty_prompt_stops_the_pipeline(self):
        """Refusing up front beats failing document by document with a message
        that describes the response rather than the cause."""
        from core.pipeline.stages import ExtractStage

        ok, detail = ExtractStage(prompt="").health()
        assert not ok
        assert "prompt" in detail.lower()

    def test_a_missing_prompt_is_logged(self):
        source = (ROOT / "src" / "core" / "pipeline" / "runner.py").read_text(
            encoding="utf-8")
        assert "extraction prompt is missing or empty" in source

    def test_the_prompt_asks_for_the_schema_the_exporter_expects(self):
        """A prompt that asks for different keys would parse and still export
        nothing, which is the same failure wearing a different hat."""
        from core import paths

        prompt = paths.PROMPT_FILE.read_text(encoding="utf-8")
        for key in ("buyer_details", "seller_details", "property_details",
                    "document_details"):
            assert key in prompt, f"the prompt never mentions {key}"
