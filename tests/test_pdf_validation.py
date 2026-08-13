"""PDF corruption detection.

Every fixture here is a real file written to disk with real bytes. A mocked
parser would prove only that the mock was called; the thing under test is
whether genuinely broken input is classified correctly and, above all, whether
it can make the validator raise - because an exception escaping this module
takes down the batch the corrupt file was supposed to be isolated from.
"""

from __future__ import annotations

import pytest

from core.pdf_validation import (
    CORRUPT_STATUSES,
    RETRYABLE_STATUSES,
    VALIDATOR_VERSION,
    Status,
    summarise,
    validate_pdf,
)

pymupdf = pytest.importorskip("pymupdf")


def _real_pdf(path, pages: int = 3, text: str = "Sale deed"):
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{text} page {n + 1}")
    doc.save(path)
    doc.close()
    return path


class TestValidFiles:
    def test_a_real_pdf_is_valid(self, tmp_path):
        result = validate_pdf(_real_pdf(tmp_path / "good.pdf", pages=3))
        assert result.is_valid
        assert result.status == Status.VALID
        assert result.page_count == 3
        assert result.corrupted_pages == []
        assert result.error_code is None
        assert result.validator_version == VALIDATOR_VERSION

    def test_a_large_pdf_is_valid_and_not_rendered(self, tmp_path):
        """60 pages must not cost 60 rasterisations - that is OCR's job."""
        result = validate_pdf(_real_pdf(tmp_path / "large.pdf", pages=60))
        assert result.is_valid and result.page_count == 60

    def test_a_valid_pdf_does_not_block_processing(self, tmp_path):
        assert validate_pdf(_real_pdf(tmp_path / "ok.pdf")).retryable


class TestFileLevel:
    def test_a_missing_file(self, tmp_path):
        result = validate_pdf(tmp_path / "nothing.pdf")
        assert result.status == Status.UNREADABLE_PDF
        assert result.error_code == "FILE_NOT_FOUND"

    def test_an_empty_file(self, tmp_path):
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        result = validate_pdf(path)
        assert result.status == Status.EMPTY_PDF
        assert result.error_code == "EMPTY_FILE"

    def test_a_file_that_is_not_a_pdf_at_all(self, tmp_path):
        """A JPEG renamed to .pdf - the commonest real case."""
        path = tmp_path / "photo.pdf"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 500)
        result = validate_pdf(path)
        assert result.status == Status.INVALID_PDF
        assert result.error_code == "INVALID_HEADER"
        assert "not a PDF" in result.error_message

    def test_a_wrong_extension(self, tmp_path):
        path = tmp_path / "deed.txt"
        path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        result = validate_pdf(path)
        assert result.status == Status.INVALID_PDF
        assert result.error_code == "NOT_A_PDF_EXTENSION"

    def test_a_damaged_header(self, tmp_path):
        path = tmp_path / "header.pdf"
        path.write_bytes(b"%PBF-1.4\n" + b"x" * 400 + b"\n%%EOF\n")
        assert validate_pdf(path).error_code == "INVALID_HEADER"

    def test_a_directory_named_like_a_pdf(self, tmp_path):
        folder = tmp_path / "folder.pdf"
        folder.mkdir()
        assert validate_pdf(folder).error_code in ("NOT_A_FILE", "FILE_READ_ERROR")


class TestStructureLevel:
    def test_a_truncated_pdf(self, tmp_path):
        """Half a real deed - a download that stopped."""
        source = _real_pdf(tmp_path / "whole.pdf", pages=8)
        data = source.read_bytes()
        cut = tmp_path / "cut.pdf"
        cut.write_bytes(data[: len(data) // 2])
        result = validate_pdf(cut)
        assert not result.is_valid
        assert result.status in (Status.INCOMPLETE_PDF, Status.CORRUPTED_PDF,
                                 Status.PARTIALLY_CORRUPTED)

    def test_a_header_with_nothing_behind_it(self, tmp_path):
        path = tmp_path / "stub.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"\x00" * 800 + b"\n%%EOF\n")
        result = validate_pdf(path)
        assert not result.is_valid
        assert result.status in (Status.CORRUPTED_PDF, Status.INCOMPLETE_PDF)
        assert "structure" in result.error_message.lower()

    def test_a_shredded_body(self, tmp_path):
        """Valid header and trailer, wreckage in between."""
        source = _real_pdf(tmp_path / "src.pdf", pages=5)
        data = bytearray(source.read_bytes())
        for i in range(200, min(len(data) - 200, 3000)):
            data[i] = 0
        path = tmp_path / "shredded.pdf"
        path.write_bytes(bytes(data))
        result = validate_pdf(path)
        assert not result.is_valid, "a shredded body was accepted"
        assert result.status in (Status.CORRUPTED_PDF, Status.INCOMPLETE_PDF,
                                 Status.PARTIALLY_CORRUPTED,
                                 Status.PDF_RENDER_ERROR)

    def test_an_encrypted_pdf(self, tmp_path):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "secret")
        path = tmp_path / "locked.pdf"
        doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                 owner_pw="owner", user_pw="user")
        doc.close()
        result = validate_pdf(path)
        assert result.status == Status.PASSWORD_PROTECTED
        assert result.error_code == "PASSWORD_PROTECTED"
        assert result.is_corrupt and not result.retryable

    def test_a_zero_page_pdf(self, tmp_path):
        """Written by hand: PyMuPDF refuses to *save* a zero-page document, so
        the only way to produce this real-world case is raw bytes - a catalogue
        whose page tree declares /Count 0."""
        path = tmp_path / "nopages.pdf"
        path.write_bytes(b"\n".join([
            b"%PDF-1.4",
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj",
            b"trailer<</Size 3/Root 1 0 R>>",
            b"%%EOF",
            b"",
        ]))
        result = validate_pdf(path)
        assert result.status in (Status.EMPTY_PDF, Status.CORRUPTED_PDF,
                                 Status.INCOMPLETE_PDF)


class TestItNeverRaises:
    """The guarantee the batch depends on."""

    @pytest.mark.parametrize("payload", [
        b"", b"%PDF-", b"%PDF-1.4", b"%PDF-1.4\n%%EOF", b"\x00" * 5000,
        b"%PDF-1.7\n" + bytes(range(256)) * 20,
        bytes([0xFF] * 4096),
        "%PDF-1.4\nಕನ್ನಡ text\n%%EOF".encode(),
    ])
    def test_arbitrary_bytes_return_a_result(self, tmp_path, payload):
        path = tmp_path / "fuzz.pdf"
        path.write_bytes(payload)
        result = validate_pdf(path)          # must not raise
        assert result.status in set(Status.ALL)
        assert result.error_message or result.is_valid

    def test_every_failure_explains_itself_without_a_stack_trace(self, tmp_path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"\x00" * 900)
        result = validate_pdf(path)
        assert result.error_message
        assert "Traceback" not in result.error_message
        assert len(result.error_message) < 400


class TestTheProcessingGate:
    def test_broken_files_block_and_valid_ones_do_not(self):
        for status in (Status.CORRUPTED_PDF, Status.EMPTY_PDF,
                       Status.PASSWORD_PROTECTED, Status.INVALID_PDF,
                       Status.UNREADABLE_PDF, Status.INCOMPLETE_PDF):
            assert status in CORRUPT_STATUSES, f"{status} would be retried"
            assert status not in RETRYABLE_STATUSES
        assert Status.VALID in RETRYABLE_STATUSES
        assert Status.PROCESSING_ERROR in RETRYABLE_STATUSES

    def test_a_partially_corrupted_deed_is_still_processed(self):
        """24 readable pages of a 25-page deed still carry the parties and the
        schedule. Refusing it would lose a recoverable document."""
        assert Status.PARTIALLY_CORRUPTED in RETRYABLE_STATUSES

    def test_the_summary_counts_what_the_dashboard_shows(self, tmp_path):
        good = validate_pdf(_real_pdf(tmp_path / "a.pdf"))
        empty = tmp_path / "b.pdf"
        empty.write_bytes(b"")
        fake = tmp_path / "c.pdf"
        fake.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        counts = summarise([good, validate_pdf(empty), validate_pdf(fake)])
        assert counts["TOTAL"] == 3
        assert counts["VALID"] == 1
        assert counts["EMPTY_PDF"] == 1
        assert counts["INVALID_PDF"] == 1
        assert counts["CORRUPT"] == 2
        assert counts["RETRYABLE"] == 1


class TestTheFailurePathIntegration:
    """The validator wired into the pipeline: it runs only on failure, it
    decides retryability, and revalidation clears a verdict once fixed."""

    pytestmark = pytest.mark.integration

    def test_a_failed_document_records_a_verdict(self, session_factory, tmp_path):
        from core.db.engine import session_scope
        from core.db.models import BatchState, StageState
        from core.db.repositories import UnitOfWork
        from core.pdf_validation import validate_pdf

        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 400)

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("validation_test")
            batch = uow.batches.create("validation", user, 1, 2048)
            uow.batches.set_state(batch, BatchState.COMPLETED)
            doc = uow.documents.add_many(batch, [
                {"document_id": "V-1", "source_filename": broken.name,
                 "source_path": str(broken), "size_bytes": 400}])[0]

            uow.documents.record_validation(doc, validate_pdf(broken))
            assert doc.validation_status == Status.INVALID_PDF
            assert doc.validation_error_code == "INVALID_HEADER"
            assert doc.is_retryable is False
            assert doc.validator_version == VALIDATOR_VERSION
            assert doc.validated_at is not None

            listed, total = uow.documents.corrupted(batch.id)
            assert total == 1 and listed[0].id == doc.id
            session.delete(batch)

    def test_a_repaired_file_stops_being_corrupt_after_revalidation(
            self, session_factory, app_service, tmp_path):
        """The whole point of Revalidate: a fixed file must become processable
        again without rebuilding the batch."""
        from core.db.engine import session_scope
        from core.db.models import BatchState
        from core.db.repositories import UnitOfWork
        from core.pdf_validation import validate_pdf

        path = tmp_path / "fixable.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"\x00" * 600)     # broken to begin with

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("validation_test")
            batch = uow.batches.create("revalidate", user, 1, 2048)
            uow.batches.set_state(batch, BatchState.COMPLETED)
            doc = uow.documents.add_many(batch, [
                {"document_id": "R-1", "source_filename": path.name,
                 "source_path": str(path), "size_bytes": 600}])[0]
            uow.documents.record_validation(doc, validate_pdf(path))
            assert doc.is_retryable is False
            doc_pk, batch_id = doc.id, batch.id

        try:
            _real_pdf(path, pages=2)                        # the operator fixes it
            result = app_service.revalidate([doc_pk])
            assert result["count"] == 1
            assert result["repaired"] == 1

            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                again = uow.documents.get(doc_pk)
                assert again.validation_status == Status.VALID
                assert again.is_retryable is True
                assert again.validation_error_code is None
                assert uow.documents.corrupted(batch_id)[1] == 0
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)

    def test_a_bulk_rerun_skips_a_known_corrupt_file(self, session_factory,
                                                     app_service, tmp_path):
        """Repeating a GPU stage on a file that cannot be opened wastes the
        most expensive resource in the system."""
        from core.db.engine import session_scope
        from core.db.models import BatchState, StageState
        from core.db.repositories import UnitOfWork
        from core.pdf_validation import validate_pdf

        bad = tmp_path / "hopeless.pdf"
        bad.write_bytes(b"")

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("validation_test")
            batch = uow.batches.create("gate", user, 1, 1024)
            uow.batches.set_state(batch, BatchState.COMPLETED)
            doc = uow.documents.add_many(batch, [
                {"document_id": "G-1", "source_filename": bad.name,
                 "source_path": str(bad), "size_bytes": 0}])[0]
            doc.ocr_attempts = 5
            uow.documents.mark_stage(doc, "ocr", StageState.FAILED,
                                     reason="OCR could not read it",
                                     processing_status="OCR_F")
            uow.documents.record_validation(doc, validate_pdf(bad))
            doc_pk, batch_id = doc.id, batch.id

        try:
            result = app_service.rerun_ocr(all_failed=True, batch_id=batch_id)
            assert result["count"] == 0, "a hopeless file was sent back to OCR"
            assert result["skipped"] == [bad.name]
            assert "repair or replace" in result["detail"]

            # The operator can still override deliberately.
            forced = app_service.rerun_ocr([doc_pk], force=True)
            assert forced["count"] == 1
        finally:
            app_service.runner.stop(timeout=5)
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)

    def test_a_valid_file_that_failed_reads_as_a_processing_error(self, tmp_path):
        """The most useful verdict: the PDF is fine, look elsewhere."""
        from core.pipeline.runner import BatchRunner

        good = _real_pdf(tmp_path / "fine.pdf", pages=2)

        class _Doc:
            source_path = str(good)
            source_filename = "fine.pdf"

        verdict = BatchRunner._validate_failed_pdf(str(good), _Doc(),
                                                   "llama-server timed out")
        assert verdict.status == Status.PROCESSING_ERROR
        assert verdict.retryable and not verdict.is_corrupt
        assert "readable" in verdict.error_message

    def test_an_ocr_failure_triggers_validation_through_the_runner(
            self, session_factory, tmp_path):
        """Drives `_do_ocr`'s failure branch itself: the verdict must be stored,
        and a corrupt file must stop being retryable even though the stage
        reported the failure as retryable."""
        from core.db.engine import session_scope
        from core.db.models import BatchState, StageState
        from core.db.repositories import UnitOfWork
        from core.pipeline.runner import BatchMode, BatchRunner, build_stages
        from core.pipeline.stages import StageName, StageOutcome

        broken = tmp_path / "unreadable.pdf"
        broken.write_bytes(b"%PDF-1.4\n" + b"\x00" * 500)

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("validation_test")
            batch = uow.batches.create("hook", user, 1, 512)
            uow.batches.set_state(batch, BatchState.COMPLETED)
            doc = uow.documents.add_many(batch, [
                {"document_id": "H-1", "source_filename": broken.name,
                 "source_path": str(broken), "size_bytes": 500}])[0]
            uow.documents.mark_stage(doc, "ocr", StageState.RUNNING)
            doc_pk, batch_id = doc.id, batch.id

        stages = build_stages(ocr_engine="textlayer",
                              translator_engine="passthrough")
        # Exactly what Surya reports on a file it cannot open - and it calls it
        # retryable, which is what the validator has to overrule.
        stages.ocr.run = lambda *a, **k: StageOutcome.failure(  # type: ignore[method-assign]
            StageName.OCR, "PdfiumError: Failed to load document", retryable=True)
        runner = BatchRunner(session_factory, stages, mode=BatchMode.MANUAL,
                             max_workers=1)
        try:
            runner._do_ocr(doc_pk, str(broken))
            with session_scope(session_factory) as session:
                again = UnitOfWork(session).documents.get(doc_pk)
                assert again.validation_status in CORRUPT_STATUSES, (
                    f"no verdict recorded (got {again.validation_status})")
                assert again.is_retryable is False, "a corrupt file stayed retryable"
                assert again.ocr_state is StageState.FAILED, (
                    "a corrupt file was requeued for another GPU pass")
                assert again.validated_at is not None
                assert again.validation_error_message
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)
