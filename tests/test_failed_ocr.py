"""Failed OCR: listing what failed, and sending it back through the same stage.

The behaviour that matters is not "a list appears". It is that a rerun actually
becomes claimable again - `claim_next` enforces a retry cap, and a rerun that
leaves `ocr_attempts` spent is a button that reports success and does nothing.
That is the case most of these tests exist to pin down.
"""

from __future__ import annotations

import pytest

from core.db.engine import session_scope
from core.db.models import BatchState, DocumentState, StageState
from core.db.repositories import UnitOfWork

pytestmark = pytest.mark.integration


def _batch_with_documents(uow: UnitOfWork, name: str, count: int = 3):
    """A batch and its documents, not occupying a queue slot.

    Moved out of QUEUED immediately: `batches.create` enforces a four-batch
    queue cap, so on a machine that already has real batches waiting, a test
    creating two of its own would intermittently fail on the cap rather than on
    anything it was testing. Batch state is irrelevant to everything here -
    the failed-OCR list is a document-level query.
    """
    user = uow.users.get_or_create("failed_ocr_test")
    batch = uow.batches.create(name, user, count, 1024 * count)
    uow.batches.set_state(batch, BatchState.COMPLETED)
    docs = uow.documents.add_many(batch, [
        {"document_id": f"{name}-{i}", "source_filename": f"{name}-{i}.pdf",
         "page_count": 4, "size_bytes": 2048}
        for i in range(count)])
    return batch, docs


def _fail_ocr(uow: UnitOfWork, doc, reason: str = "Surya timed out") -> None:
    """Put a document in exactly the state `_do_ocr` leaves a failure in.

    `ocr_attempts` is set past any retry cap on purpose: `claim_next` admits a
    document while `attempts <= max_attempts`, so a count merely *equal* to the
    cap would leave the state reset as the only thing keeping the document out
    of the queue - and a test built on that cannot tell whether the attempt
    count was cleared.
    """
    doc.ocr_attempts = 5
    uow.documents.mark_stage(doc, "ocr", StageState.FAILED,
                             reason=reason, processing_status="OCR_F")


class TestListing:
    def test_a_failed_ocr_document_is_listed(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "list_one", 3)
            _fail_ocr(uow, docs[1])

            rows, total = uow.documents.failed_ocr(batch.id)
            assert total == 1
            assert rows[0].id == docs[1].id
            assert rows[0].failure_reason == "Surya timed out"
            session.delete(batch)

    def test_a_document_that_failed_a_later_stage_is_not_listed(self, session_factory):
        """Rerunning OCR on a document whose OCR succeeded would redo minutes of
        GPU work that was never the problem."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "list_extract", 2)
            uow.documents.mark_stage(docs[0], "ocr", StageState.DONE,
                                     processing_status="OCR_P")
            uow.documents.mark_stage(docs[0], "extract", StageState.FAILED,
                                     reason="no parseable JSON")

            assert uow.documents.failed_ocr_count(batch.id) == 0
            rows, total = uow.documents.failed_ocr(batch.id)
            assert (rows, total) == ([], 0)
            # It is still a failed *document* - the other list still has it.
            assert len(uow.documents.failed_for_batch(batch.id)) == 1
            session.delete(batch)

    def test_the_list_can_be_scoped_to_one_batch_or_span_all(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            first, docs_a = _batch_with_documents(uow, "scope_a", 2)
            second, docs_b = _batch_with_documents(uow, "scope_b", 2)
            _fail_ocr(uow, docs_a[0])
            _fail_ocr(uow, docs_b[0])
            _fail_ocr(uow, docs_b[1])

            assert uow.documents.failed_ocr_count(first.id) == 1
            assert uow.documents.failed_ocr_count(second.id) == 2
            assert uow.documents.failed_ocr_count() >= 3
            session.delete(first)
            session.delete(second)

    def test_the_list_is_paginated(self, session_factory):
        """A bad scanner run fails hundreds of files; rendering them all would
        freeze the window."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "paged", 7)
            for doc in docs:
                _fail_ocr(uow, doc)

            page_one, total = uow.documents.failed_ocr(batch.id, page=1, per_page=3)
            page_three, _ = uow.documents.failed_ocr(batch.id, page=3, per_page=3)
            assert total == 7
            assert len(page_one) == 3
            assert len(page_three) == 1
            assert {d.id for d in page_one}.isdisjoint({d.id for d in page_three})
            session.delete(batch)


class TestRerun:
    def test_a_rerun_makes_the_document_claimable_again(self, session_factory):
        """The whole point. `claim_next` refuses a document whose attempts are
        spent, so a rerun that does not clear them is a no-op that reports
        success - the worst possible outcome for a button pressed on purpose."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "claimable", 1)
            _fail_ocr(uow, docs[0])
            assert docs[0].ocr_attempts > 2
            assert uow.documents.claim_next("ocr", batch.id, max_attempts=2) is None

            uow.documents.requeue_ocr([docs[0].id])
            assert docs[0].ocr_attempts == 0, "the spent retry count survived"
            claimed = uow.documents.claim_next("ocr", batch.id, max_attempts=2)
            assert claimed is not None and claimed.id == docs[0].id
            session.delete(batch)

    def test_a_rerun_resets_every_later_stage(self, session_factory):
        """OCR text is the input to everything after it. A document that reruns
        OCR while keeping a DONE extraction would export results derived from
        text that no longer exists."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "reset_stages", 1)
            doc = docs[0]
            uow.documents.mark_stage(doc, "extract", StageState.DONE)
            uow.documents.mark_stage(doc, "translate", StageState.DONE)
            _fail_ocr(uow, doc)

            uow.documents.requeue_ocr([doc.id])
            for stage in ("ocr", "extract", "translate", "validate"):
                assert getattr(doc, f"{stage}_state") is StageState.PENDING, stage
            assert doc.overall_state is DocumentState.PROCESSING
            assert doc.failure_reason is None
            assert doc.processing_status is None
            session.delete(batch)

    def test_a_healthy_document_is_never_restarted(self, session_factory):
        """Reachable from a stale page. Quietly restarting a finished document
        would destroy work that was never in question."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "healthy", 2)
            good, bad = docs
            uow.documents.mark_stage(good, "ocr", StageState.DONE,
                                     processing_status="OCR_P")
            uow.documents.mark_overall(good, DocumentState.PROCESSED)
            _fail_ocr(uow, bad)

            touched = uow.documents.requeue_ocr([good.id, bad.id])
            assert [d.id for d in touched] == [bad.id]
            assert good.ocr_state is StageState.DONE
            assert good.overall_state is DocumentState.PROCESSED
            session.delete(batch)

    def test_requeuing_nothing_is_not_an_error(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.requeue_ocr([]) == []

    def test_a_rerun_removes_the_document_from_the_failed_list(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "leaves_list", 2)
            for doc in docs:
                _fail_ocr(uow, doc)
            assert uow.documents.failed_ocr_count(batch.id) == 2

            uow.documents.requeue_ocr([docs[0].id])
            assert uow.documents.failed_ocr_count(batch.id) == 1
            session.delete(batch)


class TestServiceLayer:
    def test_rerun_all_requeues_every_failure_and_starts_the_runner(
            self, session_factory, app_service):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "svc_all", 3)
            for doc in docs:
                _fail_ocr(uow, doc)
            # A batch is created QUEUED, so it must be moved off that state
            # first - otherwise "it is QUEUED afterwards" is true whether or not
            # the rerun did anything.
            uow.batches.set_state(batch, BatchState.COMPLETED)
            batch_id = batch.id

        try:
            result = app_service.rerun_ocr(all_failed=True, batch_id=batch_id)
            assert result["count"] == 3
            assert "3 documents queued" in result["detail"]
            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                assert uow.documents.failed_ocr_count(batch_id) == 0
                # Queued, not left waiting: a rerun that needs a second button
                # press looks exactly like a rerun that did nothing.
                assert uow.batches.get(batch_id).state in (
                    BatchState.QUEUED, BatchState.RUNNING)
        finally:
            app_service.runner.stop(timeout=5)
            with session_scope(session_factory) as session:
                batch = UnitOfWork(session).batches.get(batch_id)
                if batch:
                    session.delete(batch)

    def test_rerunning_a_stale_id_reports_it_rather_than_claiming_success(
            self, session_factory, app_service):
        """The page can be open while another operator clears the list."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "svc_stale", 1)
            batch_id, doc_pk = batch.id, docs[0].id

        try:
            result = app_service.rerun_ocr([doc_pk])
            assert result["count"] == 0
            assert "no longer" in result["detail"]
        finally:
            with session_scope(session_factory) as session:
                batch = UnitOfWork(session).batches.get(batch_id)
                if batch:
                    session.delete(batch)

    def test_the_page_model_carries_what_the_template_needs(
            self, session_factory, app_service):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "svc_model", 1)
            _fail_ocr(uow, docs[0], reason="page 3 rendered blank")
            batch_id = batch.id

        try:
            model = app_service.render_page("failed_ocr", {"batch_id": batch_id})
            assert "svc_model-0.pdf" in model
            # The reason is shown whole, not truncated.
            assert "page 3 rendered blank" in model
            assert "OCR failed" in model
            assert "Rerun OCR" in model
        finally:
            with session_scope(session_factory) as session:
                batch = UnitOfWork(session).batches.get(batch_id)
                if batch:
                    session.delete(batch)

    def test_the_status_poll_carries_the_failed_count(self, session_factory,
                                                      app_service):
        """The nav badge is the only place an operator learns a file failed
        without going looking for it."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "svc_badge", 2)
            for doc in docs:
                _fail_ocr(uow, doc)
            batch_id = batch.id

        try:
            assert app_service.status()["failed_ocr"] >= 2
        finally:
            with session_scope(session_factory) as session:
                batch = UnitOfWork(session).batches.get(batch_id)
                if batch:
                    session.delete(batch)


class TestOcrDrainsBeforeExtraction:
    """The runner finishes OCR for a batch before extracting any of it.

    Not cosmetic: the two GPU models cannot co-reside on a 4 GiB card, so
    alternating between them costs a measured ~5 s swap each way, twice per
    document. Draining OCR first makes it twice per batch. R-048.
    """

    def test_pending_ocr_is_counted_per_batch(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "pending_ocr", 3)
            assert uow.documents.pending_ocr_count(batch.id) == 3

            uow.documents.mark_stage(docs[0], "ocr", StageState.DONE)
            assert uow.documents.pending_ocr_count(batch.id) == 2
            session.delete(batch)

    def test_a_failed_document_is_not_counted_as_pending(self, session_factory):
        """It cannot be claimed, so counting it would make the runner wait for
        OCR that is never going to happen and starve extraction."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "pending_failed", 2)
            _fail_ocr(uow, docs[0])
            assert uow.documents.pending_ocr_count(batch.id) == 1
            session.delete(batch)

    def test_another_batch_does_not_hold_this_one_back(self, session_factory):
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            first, _ = _batch_with_documents(uow, "pending_a", 2)
            second, docs_b = _batch_with_documents(uow, "pending_b", 1)
            uow.documents.mark_stage(docs_b[0], "ocr", StageState.DONE)

            assert uow.documents.pending_ocr_count(second.id) == 0
            assert uow.documents.pending_ocr_count(first.id) == 2
            session.delete(first)
            session.delete(second)


class TestLongDeedsAndHonestStageStates:
    """Three faults found on two real documents that exported a blank city
    because extraction had produced nothing. R-051."""

    def test_a_deed_within_the_context_is_sent_untouched(self):
        from core.pipeline.stages import fit_to_context

        text = "Sale deed. " * 100
        assert fit_to_context(text) == (text, False)

    def test_a_long_deed_is_fitted_rather_than_rejected(self):
        """A 59,012-character deed came back `HTTP 400 ... exceeds the
        available context size` on all three attempts and extracted nothing.
        Sending less is worse than sending everything; sending nothing is worse
        than both."""
        from core.pipeline.stages import MAX_INPUT_CHARS, fit_to_context

        text = "PARTIES" + ("whereas the vendor " * 5000) + "SCHEDULE"
        sent, trimmed = fit_to_context(text)
        assert trimmed
        assert len(sent) <= MAX_INPUT_CHARS
        # Both ends survive: the parties open a deed and the schedule closes it,
        # and every field this application extracts is in one or the other.
        assert sent.startswith("PARTIES")
        assert sent.endswith("SCHEDULE")
        assert "omitted" in sent, "the gap is marked, not silent"

    def test_the_budget_is_honoured_exactly(self):
        from core.pipeline.stages import fit_to_context

        for size in (40_001, 60_000, 200_000):
            sent, trimmed = fit_to_context("x" * size, 40_000)
            assert trimmed and len(sent) <= 40_000

    def test_a_stage_that_produced_no_answer_is_not_recorded_done(self,
                                                                  session_factory):
        """`extract_state=done` with no output and no parties is a stage
        recorded as successful having produced nothing - it kept the document
        out of every failed-document report."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "no_answer", 1)
            doc = docs[0]
            uow.documents.mark_stage(doc, "extract", StageState.FAILED,
                                     reason="context exceeded")
            assert doc.extract_state is StageState.FAILED
            assert doc.overall_state is DocumentState.FAILED
            session.delete(batch)

    def test_a_document_pending_extraction_but_not_processing_is_unclaimable(
            self, session_factory):
        """The stranding this fix exists to prevent, pinned as a fact about
        `claim_next`: it admits only PROCESSING documents, so PENDING plus
        NEEDS_REVIEW can never be picked up by anything."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "stranded", 1)
            doc = docs[0]
            uow.documents.mark_stage(doc, "ocr", StageState.DONE)
            uow.documents.mark_overall(doc, DocumentState.NEEDS_REVIEW)
            assert doc.extract_state is StageState.PENDING

            claimed = uow.documents.claim_next("extract", batch.id, max_attempts=2)
            assert claimed is None, "a needs_review document must not be claimable"
            session.delete(batch)

    def test_the_stage_actually_sends_the_trimmed_text(self):
        """The helper being correct is not the same as the stage using it."""
        from core.pipeline.stages import ExtractStage

        stage = ExtractStage(max_input_chars=5_000)
        sent: dict[str, str] = {}

        def fake_submit(text, document_id, prompt):
            sent["text"] = text
            return {"state": "done", "result": "{}", "prompt_tokens": 1,
                    "completion_tokens": 1}

        stage._submit = fake_submit  # type: ignore[method-assign]
        stage.run("PARTIES" + ("x" * 50_000) + "SCHEDULE", "doc", 1)

        assert len(sent["text"]) <= 5_000, "the whole deed was sent regardless"
        assert sent["text"].startswith("PARTIES")
        assert sent["text"].endswith("SCHEDULE")

    def test_an_empty_answer_marks_the_stage_failed_through_the_runner(
            self, session_factory):
        """Drives `_do_extract` itself: the decision that matters is made
        there, not in the helper."""
        from core.pipeline.runner import BatchMode, BatchRunner, build_stages
        from core.pipeline.stages import StageName, StageOutcome

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "empty_answer", 1)
            doc = docs[0]
            uow.documents.mark_stage(doc, "ocr", StageState.DONE)
            doc_pk, batch_id = doc.id, batch.id

        stages = build_stages(ocr_engine="textlayer",
                              translator_engine="passthrough")
        # The server rejected the request: no output at all, not retryable.
        stages.extract.run = lambda *a, **k: StageOutcome.failure(  # type: ignore[method-assign]
            StageName.EXTRACT, "llama-server HTTP 400: exceeds context",
            retryable=False, raw_output="")
        runner = BatchRunner(session_factory, stages, mode=BatchMode.MANUAL,
                             max_workers=1)
        try:
            runner._do_extract(doc_pk, "some ocr text")
            with session_scope(session_factory) as session:
                again = UnitOfWork(session).documents.get(doc_pk)
                assert again.extract_state is StageState.FAILED, (
                    "a stage that produced no answer was recorded "
                    f"{again.extract_state.value}")
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)

    def test_a_failed_claim_leaves_the_document_claimable(self, session_factory):
        """`claim_next` returns the lowest-id claimable document, so asking for
        the second one gets the first back. That mismatch used to send the
        caller on to park *this* document in NEEDS_REVIEW with its extract stage
        still PENDING - a pair `claim_next` can never admit again."""
        from core.pipeline.runner import BatchMode, BatchRunner, REQUEUED, build_stages

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "claim_race", 2)
            for doc in docs:
                uow.documents.mark_stage(doc, "ocr", StageState.DONE)
            first_pk, second_pk, batch_id = docs[0].id, docs[1].id, batch.id

        stages = build_stages(ocr_engine="textlayer",
                              translator_engine="passthrough")
        runner = BatchRunner(session_factory, stages, mode=BatchMode.MANUAL,
                             max_workers=1)
        try:
            result = runner._do_extract(second_pk, "ocr text")
            assert result is REQUEUED, "the document was given up on"
            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                doc = uow.documents.get(second_pk)
                assert doc.overall_state is DocumentState.PROCESSING
                assert doc.extract_state is StageState.PENDING
                # The point: it can still be picked up.
                assert uow.documents.claim_next(
                    "extract", batch_id, max_attempts=2) is not None
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)

    def test_exhausted_retries_end_as_a_failure_not_an_endless_requeue(
            self, session_factory):
        """When the attempts are spent `claim_next` will never admit it again,
        so requeueing would spin the runner forever on a document it cannot
        take. It is marked FAILED, which is both true and actionable."""
        from core.pipeline.runner import BatchMode, BatchRunner, REQUEUED, build_stages

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "spent_retries", 1)
            doc = docs[0]
            uow.documents.mark_stage(doc, "ocr", StageState.DONE)
            doc.extract_attempts = 99
            session.flush()
            doc_pk, batch_id = doc.id, batch.id

        stages = build_stages(ocr_engine="textlayer",
                              translator_engine="passthrough")
        runner = BatchRunner(session_factory, stages, mode=BatchMode.MANUAL,
                             max_workers=1)
        try:
            result = runner._do_extract(doc_pk, "ocr text")
            assert result is not REQUEUED, "this would spin the runner"
            with session_scope(session_factory) as session:
                again = UnitOfWork(session).documents.get(doc_pk)
                assert again.extract_state is StageState.FAILED
                assert "exhausted" in (again.failure_reason or "")
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)

    def test_rerun_all_scoped_to_a_batch_leaves_other_batches_alone(
            self, session_factory, app_service):
        """The page can be scoped to one batch. Its count would then be that
        batch's while the action requeued every batch's failures."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            mine, mine_docs = _batch_with_documents(uow, "scope_mine", 2)
            other, other_docs = _batch_with_documents(uow, "scope_other", 2)
            for doc in mine_docs + other_docs:
                _fail_ocr(uow, doc)
            mine_id, other_id = mine.id, other.id

        try:
            result = app_service.rerun_ocr(all_failed=True, batch_id=mine_id)
            assert result["count"] == 2
            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                assert uow.documents.failed_ocr_count(mine_id) == 0
                assert uow.documents.failed_ocr_count(other_id) == 2, (
                    "another batch's failures were requeued")
        finally:
            app_service.runner.stop(timeout=5)
            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                for bid in (mine_id, other_id):
                    b = uow.batches.get(bid)
                    if b:
                        session.delete(b)

    def test_the_page_publishes_the_batch_it_is_scoped_to(self, session_factory,
                                                          app_service):
        """The template reads `batch_id` onto the Rerun All button; without it
        the frontend cannot scope the request."""
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch, docs = _batch_with_documents(uow, "scope_publish", 1)
            _fail_ocr(uow, docs[0])
            batch_id = batch.id

        try:
            html = app_service.render_page("failed_ocr", {"batch_id": batch_id})
            assert f'data-batch-id="{batch_id}"' in html
        finally:
            with session_scope(session_factory) as session:
                b = UnitOfWork(session).batches.get(batch_id)
                if b:
                    session.delete(b)
