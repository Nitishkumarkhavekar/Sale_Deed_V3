"""Database repositories and the pipeline runner.

Requires PostgreSQL; skipped otherwise. These cover the behaviour that keeps a
thousand-document batch correct across crashes:

  * claim-and-swap so two workers never take the same document
  * stage ordering
  * idempotency, so a retry does not duplicate rows
  * continuous commit
"""

from __future__ import annotations

import pytest

from core.db.engine import session_scope
from core.db.models import BatchState, DocumentState, StageState
from core.db.repositories import (
    MAX_FILES_PER_BATCH,
    MAX_QUEUED_BATCHES,
    RepositoryError,
    UnitOfWork,
)

pytestmark = pytest.mark.integration


class TestBatchRules:
    def test_queue_cap(self, session_factory):
        made: list[int] = []
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("cap_test")
            try:
                for i in range(MAX_QUEUED_BATCHES + 2):
                    made.append(uow.batches.create(f"cap_{i}", user, 1, 1024).id)
                pytest.fail("queue cap was not enforced")
            except RepositoryError as exc:
                assert "maximum" in str(exc).lower()
            finally:
                for bid in made:
                    batch = uow.batches.get(bid)
                    if batch:
                        session.delete(batch)
                session.delete(uow.users.get_or_create("cap_test"))

    def test_file_count_limit(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("limit_test")
            with pytest.raises(RepositoryError, match="file limit"):
                uow.batches.create("too many", user, MAX_FILES_PER_BATCH + 1, 1024)
            session.delete(uow.users.get_or_create("limit_test"))

    def test_size_limit(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("size_test")
            with pytest.raises(RepositoryError, match="exceeds"):
                uow.batches.create("too big", user, 1, 30 * 1024**3)
            session.delete(uow.users.get_or_create("size_test"))

    def test_empty_name_rejected(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            with pytest.raises(RepositoryError):
                uow.batches.create("   ", None, 1, 1024)


class TestDocumentClaiming:
    def test_claim_marks_running(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch)
            assert doc is not None
            assert doc.ocr_state is StageState.RUNNING
            assert doc.ocr_attempts == 1

    def test_two_claims_take_different_documents(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            first = uow.documents.claim_next("ocr", temp_batch)
            second = uow.documents.claim_next("ocr", temp_batch)
            assert first is not None and second is not None
            assert first.id != second.id, "the same document was claimed twice"

    def test_stage_ordering_enforced(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.claim_next("extract", temp_batch) is None

    def test_extract_claimable_after_ocr_done(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch)
            uow.documents.mark_stage(doc, "ocr", StageState.DONE)
            assert uow.documents.claim_next("extract", temp_batch) is not None

    def test_retry_cap_respected(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.claim_next("ocr", temp_batch, max_attempts=1)
            doc.ocr_attempts = 5
            session.flush()
            uow.documents.mark_stage(doc, "ocr", StageState.PENDING)
            again = uow.documents.claim_next("ocr", temp_batch, max_attempts=1)
            assert again is None or again.id != doc.id

    def test_crash_recovery_returns_documents_to_pending(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.documents.claim_next("ocr", temp_batch)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.reset_running_to_pending(temp_batch) >= 1
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.claim_next("ocr", temp_batch) is not None


class TestIdempotency:
    def test_duplicate_registration_skipped(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(temp_batch)
            again = uow.documents.add_many(batch, [
                {"document_id": "PYTEST-1", "source_filename": "p1.pdf"}])
            assert again == []

    def test_save_pages_replaces_not_appends(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            assert uow.ocr.save_pages(doc, [(1, "one"), (2, "two")]) == 2
            assert uow.ocr.save_pages(doc, [(1, "one")]) == 1
            assert len(doc.ocr_pages) == 1, "pages accumulated across calls"

    def test_replace_persons_is_idempotent(self, session_factory, temp_batch):
        extraction = {"buyer_details": [{"name": "B1"}], "seller_details": [{"name": "S1"}]}
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            assert len(uow.results.replace_persons(doc, extraction)) == 2
            assert len(uow.results.replace_persons(doc, extraction)) == 2
            assert len(doc.persons) == 2, "persons accumulated across calls"

    def test_record_flags_replaces(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.results.record_flags(doc, [{"flag_code": "OCR_P", "person_id": None}])
            uow.results.record_flags(doc, [{"flag_code": "WSC", "person_id": None}])
            assert {v.flag_code for v in doc.validations} == {"WSC"}

    def test_extraction_attempt_overwrites(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.extractions.record(doc, attempt=1, raw_output="first",
                                   parsed_ok=False, pan_coverage=0.0)
            uow.extractions.record(doc, attempt=1, raw_output="second",
                                   parsed_ok=True, pan_coverage=1.0)
            assert len(doc.extractions) == 1
            assert uow.extractions.latest(doc).raw_output == "second"


class TestPersistence:
    def test_identifiers_stored_as_text(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            people = uow.results.replace_persons(doc, {
                "buyer_details": [{"name": "B", "aadhaar_number": "241391305374",
                                   "pan_card_number": "ADPPN2284H"}],
                "seller_details": []})
            assert people[0].aadhaar_number == "241391305374"
            assert people[0].pan_card_number == "ADPPN2284H"

    def test_malformed_identifiers_become_null(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            people = uow.results.replace_persons(doc, {
                "buyer_details": [{"name": "B", "aadhaar_number": "12345",
                                   "pan_card_number": "BLRPS9269"}],
                "seller_details": []})
            assert people[0].aadhaar_number is None
            assert people[0].pan_card_number is None

    def test_commit_visible_in_a_new_session(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.documents.mark_overall(doc, DocumentState.PROCESSED)
            pk = doc.id
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).documents.get(pk).overall_state \
                is DocumentState.PROCESSED

    def test_non_iso_date_becomes_null(self, session_factory, temp_batch):
        """Guessing a format risks silently transposing day and month."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            prop = uow.results.save_property(doc, {"sale_consideration": "100"})
            uow.results.save_document_meta(prop, {"transaction_date": "09-04-2025"})
            assert prop.transaction_date is None


class TestProgressAndPaging:
    def test_progress_counts(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            progress = UnitOfWork(session).batches.progress(temp_batch)
            assert progress is not None
            assert progress.total == 3
            assert set(progress.stages) == {"ocr", "extract", "translate", "validate"}

    def test_pagination_bounds(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            page1, total = uow.documents.list_for_batch(temp_batch, page=1, per_page=2)
            page2, _ = uow.documents.list_for_batch(temp_batch, page=2, per_page=2)
            assert total == 3
            assert len(page1) == 2 and len(page2) == 1
            assert {d.id for d in page1}.isdisjoint({d.id for d in page2})

    def test_reprocess_failed_requeues(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.documents.mark_overall(doc, DocumentState.FAILED, "test")
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.reprocess_failed(temp_batch) == 1
            refreshed = uow.documents.list_for_batch(temp_batch, per_page=3)[0]
            requeued = [d for d in refreshed if d.overall_state is DocumentState.PROCESSING]
            assert len(requeued) == 3

    def test_cascade_deletes_children(self, session_factory, temp_batch):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.ocr.save_pages(doc, [(1, "text")])
            uow.results.replace_persons(doc, {"buyer_details": [{"name": "B"}],
                                              "seller_details": []})
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(temp_batch)
            session.delete(batch)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.list_for_batch(temp_batch, per_page=1)[1] == 0


class TestSettings:
    def test_round_trip(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.settings.set("pytest_key", "value_one")
            assert uow.settings.get("pytest_key") == "value_one"
            uow.settings.set("pytest_key", "value_two")
            assert uow.settings.get("pytest_key") == "value_two"
            session.delete(session.get(type(uow.settings).__mro__[0] and
                                       __import__("core.db.models", fromlist=["Setting"]).Setting,
                                       "pytest_key"))

    def test_default_returned_for_missing(self, session_factory):
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).settings.get("nope", "fallback") == "fallback"
