"""Append-only failure history.

A retry used to erase the previous verdict: `requeue_ocr` clears
`failure_reason`, so a document that failed three different ways showed only the
last one. The sequence is often the whole story - "watermark removal failed,
then OCR found no text" is a different problem from "OCR found no text" alone.
"""

from __future__ import annotations

import pytest

from core import failure_codes as fc
from core.db.engine import session_scope
from core.db.models import BatchState, StageState
from core.db.repositories import UnitOfWork

pytestmark = pytest.mark.integration


def _doc(uow: UnitOfWork, name: str):
    user = uow.users.get_or_create("failure_events_test")
    batch = uow.batches.create(name, user, 1, 1024)
    uow.batches.set_state(batch, BatchState.COMPLETED)
    doc = uow.documents.add_many(batch, [
        {"document_id": f"{name}-1", "source_filename": f"{name}.pdf",
         "size_bytes": 1024}])[0]
    return batch, doc


class TestAppendOnly:
    def test_each_failure_adds_a_row_rather_than_replacing(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "append")
            uow.documents.record_failure(
                doc, stage="watermark", code=fc.WATERMARK_REMOVAL_FAILED,
                message="Watermark was not removed.")
            uow.documents.record_failure(
                doc, stage="ocr", code=fc.OCR_NO_TEXT,
                message="OCR completed but no readable text was extracted.")

            history = uow.documents.failure_history(doc)
            assert len(history) == 2, "the second diagnosis replaced the first"
            assert [e.code for e in history] == [
                fc.WATERMARK_REMOVAL_FAILED, fc.OCR_NO_TEXT]
            session.delete(batch)

    def test_the_sequence_is_preserved_in_order(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "ordered")
            for code in (fc.WATERMARK_REMOVAL_FAILED, fc.OCR_NO_TEXT,
                         fc.AI_EXTRACTION_FAILED):
                uow.documents.record_failure(doc, stage="ocr", code=code)
            assert [e.code for e in uow.documents.failure_history(doc)] == [
                fc.WATERMARK_REMOVAL_FAILED, fc.OCR_NO_TEXT,
                fc.AI_EXTRACTION_FAILED]
            session.delete(batch)

    def test_repeating_the_same_stage_increments_the_attempt(self, session_factory):
        """Three identical failures must read as three attempts, not one."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "attempts")
            for _ in range(3):
                uow.documents.record_failure(doc, stage="ocr", code=fc.OCR_FAILED)
            assert [e.attempt for e in uow.documents.failure_history(doc)] == [1, 2, 3]
            session.delete(batch)

    def test_attempts_are_counted_per_stage(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "per_stage")
            uow.documents.record_failure(doc, stage="ocr", code=fc.OCR_FAILED)
            uow.documents.record_failure(doc, stage="extract",
                                         code=fc.AI_EXTRACTION_FAILED)
            uow.documents.record_failure(doc, stage="ocr", code=fc.OCR_FAILED)
            events = uow.documents.failure_history(doc)
            assert [(e.stage, e.attempt) for e in events] == [
                ("ocr", 1), ("extract", 1), ("ocr", 2)]
            session.delete(batch)


class TestRetryDoesNotEraseHistory:
    def test_requeueing_for_ocr_keeps_every_earlier_diagnosis(self, session_factory):
        """The defect this table exists to fix."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "retry_keeps")
            doc.ocr_attempts = 5
            uow.documents.mark_stage(doc, "ocr", StageState.FAILED,
                                     reason="Surya could not read it")
            uow.documents.record_failure(doc, stage="ocr", code=fc.OCR_FAILED,
                                         message="OCR was not completed.")
            assert len(uow.documents.failure_history(doc)) == 1

            uow.documents.requeue_ocr([doc.id])

            # The document's own verdict is cleared for the fresh attempt...
            assert doc.failure_reason is None
            # ...but the history is not.
            history = uow.documents.failure_history(doc)
            assert len(history) == 1, "the retry erased the history"
            assert history[0].code == fc.OCR_FAILED
            session.delete(batch)

    def test_history_dies_with_its_document(self, session_factory):
        """CASCADE: orphaned history helps nobody."""
        from sqlalchemy import func, select

        from core.db.models import FailureEvent

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "cascade")
            uow.documents.record_failure(doc, stage="ocr", code=fc.OCR_FAILED)
            doc_id = doc.id
            session.delete(batch)
            session.flush()
            left = session.scalar(
                select(func.count()).select_from(FailureEvent)
                .where(FailureEvent.document_id == doc_id))
            assert left == 0


class TestRendering:
    def test_describe_renders_the_sequence_for_display(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "render")
            uow.documents.record_failure(
                doc, stage="ocr", code=fc.OCR_NO_TEXT,
                technical='boom\nTraceback (most recent call last):\n  File "x"')
            rows = fc.describe(uow.documents.failure_history(doc))
            assert rows[0]["stage"] == "OCR"
            assert rows[0]["message"]
            assert "Traceback" not in rows[0]["technical"], "a stack trace leaked"
            session.delete(batch)

    def test_a_message_is_supplied_when_the_row_has_none(self, session_factory):
        """Older rows, or a caller that stored only a code."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, doc = _doc(uow, "nomsg")
            uow.documents.record_failure(doc, stage="ocr", code=fc.PDF_CORRUPTED)
            rows = fc.describe(uow.documents.failure_history(doc))
            assert rows[0]["message"] == "PDF file is corrupted or cannot be read."
            session.delete(batch)
