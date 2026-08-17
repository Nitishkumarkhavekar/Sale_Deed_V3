"""Per-batch Run / Stop / Delete.

The runner's existing start/pause/stop are *global*: they govern the worker
threads. Nothing could act on one batch, so an operator with two batches queued
had no way to stop the wrong one, and a batch that had been stopped could not be
reached from the UI at all.

What is worth testing here is not that a state column changes - that is trivial -
but the four things that make the feature safe:

  * **Stop never discards work.** A document in flight finishes. That is the
    whole reason `STOPPING` exists as a state distinct from `STOPPED`.
  * **Resume continues, it does not restart.** Stage results already committed
    stay committed.
  * **Isolation.** Acting on one batch leaves every other batch untouched -
    including the one the runner is currently inside.
  * **Restart safety.** A process killed mid-stop must not leave a batch in a
    state with no exit.

`runner.start()` is stubbed in the service tests. The state machine is what is
under test; letting these tests spawn the real pipeline would start OCR on the
machine running them and make the result depend on a GPU.
"""

from __future__ import annotations

import pytest

from core.db.engine import session_scope
from core.db.models import BatchState, StageState
from core.db.repositories import RepositoryError, UnitOfWork

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_batch(session_factory):
    """Create disposable batches, cleaned up however the test ends."""
    created: list[int] = []

    def _make(name: str, files: int = 3) -> int:
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("pytest_mgmt")
            batch = uow.batches.create(name, user, files, files * 1024)
            uow.documents.add_many(batch, [
                {"document_id": f"{name}-{i}", "source_filename": f"{name}{i}.pdf",
                 "source_path": f"{name}{i}.pdf", "size_bytes": 1024}
                for i in range(files)])
            created.append(batch.id)
            return batch.id

    yield _make

    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        for batch_id in created:
            batch = uow.batches.get(batch_id)
            if batch is not None:
                session.delete(batch)


def _state(session_factory, batch_id: int) -> BatchState:
    with session_scope(session_factory) as session:
        batch = UnitOfWork(session).batches.get(batch_id)
        return batch.state


def _set_state(session_factory, batch_id: int, state: BatchState) -> None:
    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        uow.batches.set_state(uow.batches.get(batch_id), state)


def _hold_a_document(session_factory, batch_id: int) -> int:
    """Put one document into a RUNNING stage, as a worker mid-OCR would.

    Returns its primary key. This is what a stop has to wait for.
    """
    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        doc = uow.documents.list_for_batch(batch_id, per_page=1)[0][0]
        doc.ocr_state = StageState.RUNNING
        session.flush()
        return doc.id


def _release(session_factory, doc_pk: int, state=StageState.DONE) -> None:
    with session_scope(session_factory) as session:
        UnitOfWork(session).documents.get(doc_pk).ocr_state = state
        session.flush()


@pytest.fixture()
def service(app_service, monkeypatch):
    """The real service with the pipeline's ignition disconnected."""
    starts: list[int] = []
    monkeypatch.setattr(app_service.runner, "start",
                        lambda: starts.append(1))
    app_service.runner_starts = starts  # type: ignore[attr-defined]
    return app_service


# ---------------------------------------------------------------------------
# The happy path, in order
# ---------------------------------------------------------------------------


class TestTheFullLifecycle:
    """Create -> Queue -> Run -> Stop -> Resume -> Complete -> Delete."""

    def test_a_new_batch_is_queued(self, session_factory, make_batch):
        assert _state(session_factory, make_batch("LC-new")) is BatchState.QUEUED

    def test_running_a_queued_batch_starts_the_runner(self, service, make_batch,
                                                      session_factory):
        batch_id = make_batch("LC-run")
        result = service.batch_action(batch_id, "run")

        assert result["state"] == "queued"
        assert service.runner_starts, "Run did not start the worker threads"
        assert _state(session_factory, batch_id) is BatchState.QUEUED

    def test_run_puts_the_chosen_batch_at_the_head_of_the_queue(
            self, service, make_batch, session_factory):
        """"Run" on a specific batch has to mean *that* batch. Leaving it
        behind two others that were queued first would look like the button
        did nothing for the next hour."""
        first = make_batch("LC-head-1")
        second = make_batch("LC-head-2")
        service.batch_action(second, "run")

        with session_scope(session_factory) as session:
            nxt = UnitOfWork(session).batches.next_queued()
            assert nxt.id == second, f"queue head is {nxt.id}, wanted {second}"
        assert _state(session_factory, first) is BatchState.QUEUED

    def test_stopping_a_running_batch_waits_for_the_document_in_flight(
            self, service, make_batch, session_factory):
        batch_id = make_batch("LC-stop")
        _set_state(session_factory, batch_id, BatchState.RUNNING)
        doc_pk = _hold_a_document(session_factory, batch_id)

        result = service.batch_action(batch_id, "stop")

        assert result["state"] == "stopping"
        assert _state(session_factory, batch_id) is BatchState.STOPPING
        # The point of the whole design: the document was not touched.
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).documents.get(doc_pk).ocr_state \
                is StageState.RUNNING

    def test_the_stop_completes_once_the_document_is_released(
            self, service, make_batch, session_factory):
        batch_id = make_batch("LC-settle")
        _set_state(session_factory, batch_id, BatchState.RUNNING)
        doc_pk = _hold_a_document(session_factory, batch_id)
        service.batch_action(batch_id, "stop")

        _release(session_factory, doc_pk)
        with session_scope(session_factory) as session:
            settled = UnitOfWork(session).batches.settle_stopping()

        assert batch_id in settled
        assert _state(session_factory, batch_id) is BatchState.STOPPED

    def test_resuming_keeps_the_work_already_done(self, service, make_batch,
                                                  session_factory):
        """The reason a stopped batch is resumable rather than restartable.
        Stage results are committed as they complete; resuming must not reset
        them, or stopping a batch at 90% would throw away hours of OCR."""
        batch_id = make_batch("LC-resume", files=3)
        doc_pk = _hold_a_document(session_factory, batch_id)
        _release(session_factory, doc_pk, StageState.DONE)
        _set_state(session_factory, batch_id, BatchState.STOPPED)

        service.batch_action(batch_id, "run")

        assert _state(session_factory, batch_id) is BatchState.QUEUED
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).documents.get(doc_pk).ocr_state \
                is StageState.DONE, "resume reset a finished stage"

    def test_resuming_reports_how_much_is_left(self, service, make_batch,
                                               session_factory):
        batch_id = make_batch("LC-left", files=3)
        _set_state(session_factory, batch_id, BatchState.STOPPED)
        detail = service.batch_action(batch_id, "run")["detail"]
        assert "3 document(s) left" in detail, detail

    def test_resuming_clears_a_stale_finish_time(self, service, make_batch,
                                                 session_factory):
        """A batch that failed carries `finished_at`. Resumed, it has not
        finished, and leaving the timestamp would show a completion in the
        past for a batch about to run."""
        batch_id = make_batch("LC-stale")
        _set_state(session_factory, batch_id, BatchState.FAILED)
        service.batch_action(batch_id, "run")
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).batches.get(batch_id).finished_at is None

    def test_deleting_removes_the_batch_and_its_documents(
            self, service, make_batch, session_factory):
        batch_id = make_batch("LC-delete")
        service.batch_action(batch_id, "delete")

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.batches.get(batch_id) is None
            assert uow.documents.list_for_batch(batch_id, per_page=10)[1] == 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestInvalidActionsAreRefused:
    """Every guard is server-side. The dashboard refreshes on a timer and two
    windows can be open, so a visible button is never evidence that the action
    is still legal."""

    def test_a_running_batch_cannot_be_run_again(self, service, make_batch,
                                                 session_factory):
        batch_id = make_batch("RF-double")
        _set_state(session_factory, batch_id, BatchState.RUNNING)
        with pytest.raises(RepositoryError, match="already running"):
            service.batch_action(batch_id, "run")

    def test_a_stopping_batch_cannot_be_run(self, service, make_batch,
                                            session_factory):
        """Restarting mid-stop would race the worker that is finishing the
        document, and the batch would end up running with a stop pending."""
        batch_id = make_batch("RF-restart")
        _set_state(session_factory, batch_id, BatchState.STOPPING)
        with pytest.raises(RepositoryError, match="still stopping"):
            service.batch_action(batch_id, "run")

    def test_a_finished_batch_cannot_be_run(self, service, make_batch,
                                            session_factory):
        batch_id = make_batch("RF-done")
        _set_state(session_factory, batch_id, BatchState.COMPLETED)
        with pytest.raises(RepositoryError, match="already finished"):
            service.batch_action(batch_id, "run")

    def test_a_completed_batch_cannot_be_stopped(self, service, make_batch,
                                                 session_factory):
        batch_id = make_batch("RF-stopdone")
        _set_state(session_factory, batch_id, BatchState.COMPLETED)
        with pytest.raises(RepositoryError, match="nothing to stop"):
            service.batch_action(batch_id, "stop")

    def test_a_stopped_batch_cannot_be_stopped_again(self, service, make_batch,
                                                     session_factory):
        batch_id = make_batch("RF-restop")
        _set_state(session_factory, batch_id, BatchState.STOPPED)
        with pytest.raises(RepositoryError, match="already stopped"):
            service.batch_action(batch_id, "stop")

    def test_a_running_batch_is_not_deleted_without_confirmation(
            self, service, make_batch, session_factory):
        batch_id = make_batch("RF-delrun")
        _set_state(session_factory, batch_id, BatchState.RUNNING)

        with pytest.raises(RepositoryError, match="Stop it first"):
            service.batch_action(batch_id, "delete")
        assert _state(session_factory, batch_id) is BatchState.RUNNING, \
            "a refused delete still changed the batch"

    def test_an_unknown_batch_is_reported_not_crashed(self, service):
        for action in ("run", "stop", "delete"):
            with pytest.raises(RepositoryError, match="not found"):
                service.batch_action(2_000_000_000, action)

    def test_an_unknown_action_is_rejected(self, service, make_batch):
        with pytest.raises(ValueError, match="unknown batch action"):
            service.batch_action(make_batch("RF-bogus"), "obliterate")

    def test_no_refusal_leaks_a_python_type(self, service, make_batch,
                                            session_factory):
        """The project's standing rule, applied at this boundary."""
        batch_id = make_batch("RF-leak")
        _set_state(session_factory, batch_id, BatchState.COMPLETED)
        for action in ("run", "stop"):
            with pytest.raises(RepositoryError) as caught:
                service.batch_action(batch_id, action)
            message = str(caught.value)
            for leak in ("Traceback", "AttributeError", "NoneType", "object at 0x"):
                assert leak not in message, f"{leak!r} leaked: {message}"


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


class TestOneBatchDoesNotDisturbAnother:
    """The requirement that makes this feature non-trivial. Because the stop is
    recorded on the batch row rather than on the runner, the runner itself never
    learns about it - it simply finds a different batch to claim from."""

    def test_stopping_one_batch_leaves_the_other_queued(
            self, service, make_batch, session_factory):
        stopped = make_batch("ISO-stop")
        other = make_batch("ISO-keep")
        _set_state(session_factory, stopped, BatchState.RUNNING)

        service.batch_action(stopped, "stop")

        assert _state(session_factory, other) is BatchState.QUEUED

    def test_deleting_one_batch_leaves_the_others_documents_intact(
            self, service, make_batch, session_factory):
        doomed = make_batch("ISO-del", files=3)
        keeper = make_batch("ISO-safe", files=3)

        service.batch_action(doomed, "delete")

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.batches.get(keeper) is not None
            assert uow.documents.list_for_batch(keeper, per_page=10)[1] == 3

    def test_a_stop_only_counts_its_own_batchs_in_flight_work(
            self, service, make_batch, session_factory):
        """`in_flight` is scoped by batch. Unscoped, a batch could never stop
        while any other batch anywhere was mid-document."""
        quiet = make_batch("ISO-quiet")
        busy = make_batch("ISO-busy")
        _hold_a_document(session_factory, busy)
        _set_state(session_factory, quiet, BatchState.RUNNING)

        result = service.batch_action(quiet, "stop")

        assert result["state"] == "stopped", \
            "a stop waited on another batch's document"

    def test_settling_does_not_touch_a_healthy_running_batch(
            self, make_batch, session_factory):
        running = make_batch("ISO-run")
        _set_state(session_factory, running, BatchState.RUNNING)
        with session_scope(session_factory) as session:
            UnitOfWork(session).batches.settle_stopping()
        assert _state(session_factory, running) is BatchState.RUNNING


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_a_batch_caught_mid_stop_is_settled_at_startup(
            self, app_service, make_batch, session_factory):
        """`STOPPING` has no exit of its own. A process killed during a stop
        would strand the batch there for ever - no Run button, no Stop button -
        until someone edited the database by hand."""
        batch_id = make_batch("RS-stuck")
        _hold_a_document(session_factory, batch_id)
        _set_state(session_factory, batch_id, BatchState.STOPPING)

        # What the runner does on startup: release stranded documents, then
        # settle. The order matters and is asserted by the outcome.
        app_service.runner.recover()

        assert _state(session_factory, batch_id) is BatchState.STOPPED

    def test_recovery_is_idempotent(self, app_service, make_batch,
                                    session_factory):
        batch_id = make_batch("RS-twice")
        _set_state(session_factory, batch_id, BatchState.STOPPING)
        app_service.runner.recover()
        app_service.runner.recover()
        assert _state(session_factory, batch_id) is BatchState.STOPPED

    def test_a_stopping_batch_is_never_marked_completed(
            self, app_service, make_batch, session_factory):
        """A batch stopped with nothing left to claim looks 'finished' to the
        completion check. Marking it COMPLETED would report a full run of a
        batch the operator deliberately cut short."""
        batch_id = make_batch("RS-finalise")
        _set_state(session_factory, batch_id, BatchState.STOPPING)

        app_service.runner._finalise_if_complete(batch_id)

        assert _state(session_factory, batch_id) is not BatchState.COMPLETED


# ---------------------------------------------------------------------------
# Deletion of resources
# ---------------------------------------------------------------------------


class TestDeletionHandlesFiles:
    def test_prepared_copies_are_removed(self, service, make_batch,
                                         session_factory, tmp_path):
        prepared = tmp_path / "prepared.pdf"
        prepared.write_bytes(b"%PDF-1.4 prepared copy")
        batch_id = make_batch("FS-clean")
        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.list_for_batch(
                batch_id, per_page=1)[0][0]
            doc.cleaned_path = str(prepared)

        result = service.batch_action(batch_id, "delete")

        assert result["files_removed"] == 1
        assert not prepared.exists()

    def test_the_source_pdf_is_never_removed(self, service, make_batch,
                                             session_factory, tmp_path):
        """Often the operator's only copy, and never ours to delete."""
        source = tmp_path / "original.pdf"
        source.write_bytes(b"%PDF-1.4 the original")
        batch_id = make_batch("FS-source")
        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.list_for_batch(
                batch_id, per_page=1)[0][0]
            doc.source_path = str(source)

        service.batch_action(batch_id, "delete")

        assert source.exists(), "deleting a batch destroyed the operator's PDF"

    def test_a_missing_prepared_file_does_not_fail_the_delete(
            self, service, make_batch, session_factory, tmp_path):
        """The rows are committed before the files are touched. A delete that
        raised here would leave the operator unable to tell what happened."""
        batch_id = make_batch("FS-gone")
        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.list_for_batch(
                batch_id, per_page=1)[0][0]
            doc.cleaned_path = str(tmp_path / "never-existed.pdf")

        result = service.batch_action(batch_id, "delete")
        assert result["deleted"] == batch_id
        assert result["files_removed"] == 0

    def test_only_this_batchs_prepared_files_are_removed(
            self, service, make_batch, session_factory, tmp_path):
        """Scoped by batch id in SQL, not by sweeping the directory: two
        batches can hold a document of the same name."""
        mine = tmp_path / "a.pdf"
        theirs = tmp_path / "b.pdf"
        mine.write_bytes(b"%PDF mine")
        theirs.write_bytes(b"%PDF theirs")
        doomed = make_batch("FS-mine")
        keeper = make_batch("FS-theirs")
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.documents.list_for_batch(doomed, per_page=1)[0][0].cleaned_path = str(mine)
            uow.documents.list_for_batch(keeper, per_page=1)[0][0].cleaned_path = str(theirs)

        service.batch_action(doomed, "delete")

        assert not mine.exists()
        assert theirs.exists()


# ---------------------------------------------------------------------------
# What the dashboard is told
# ---------------------------------------------------------------------------


class TestTheDashboardShowsTheRightActions:
    """A button is offered only when pressing it would succeed - the flags come
    from the same state machine the service enforces."""

    @pytest.mark.parametrize("state,can_run,can_stop,needs_force", [
        (BatchState.QUEUED, True, True, False),
        (BatchState.RUNNING, False, True, True),
        (BatchState.STOPPING, False, False, True),
        (BatchState.STOPPED, True, False, False),
        (BatchState.PAUSED, True, False, False),
        (BatchState.COMPLETED, False, False, False),
        (BatchState.FAILED, True, False, False),
    ])
    def test_the_action_flags_match_the_state_machine(
            self, app_service, make_batch, session_factory,
            state, can_run, can_stop, needs_force):
        batch_id = make_batch(f"UI-{state.value}")
        _set_state(session_factory, batch_id, state)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            row = app_service._manage_row(uow.batches.get(batch_id),
                                          uow.batches.progress(batch_id))
        assert row["can_run"] is can_run
        assert row["can_stop"] is can_stop
        assert row["needs_force"] is needs_force

    def test_every_field_the_specification_asks_for_is_present(
            self, app_service, make_batch, session_factory):
        batch_id = make_batch("UI-fields")
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            row = app_service._manage_row(uow.batches.get(batch_id),
                                          uow.batches.progress(batch_id))
        for field in ("id", "name", "created_at", "file_count", "processed",
                      "pending", "completed", "failed", "percent", "state"):
            assert field in row, f"{field} missing from the management row"
        assert row["id"] == batch_id
        assert row["file_count"] == 3
        assert row["pending"] == 3, "nothing processed, so everything is pending"

    def test_the_run_button_says_resume_for_a_stopped_batch(
            self, app_service, make_batch, session_factory):
        """"Run" on a batch that is 90% done reads as "start over"."""
        batch_id = make_batch("UI-label")
        _set_state(session_factory, batch_id, BatchState.STOPPED)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            row = app_service._manage_row(uow.batches.get(batch_id), None)
        assert row["run_label"] == "Resume"

    def test_a_stopped_batch_appears_in_the_management_list(
            self, app_service, make_batch, session_factory):
        """It appeared nowhere before this feature: not queued, not active, not
        completed - so it could not be resumed or deleted from the UI at all."""
        batch_id = make_batch("UI-visible")
        _set_state(session_factory, batch_id, BatchState.STOPPED)
        model = app_service._dashboard({})
        assert batch_id in [r["id"] for r in model["manage"]]

    def test_the_dashboard_page_renders_with_every_state_present(
            self, app_service, make_batch, session_factory):
        """End to end through the template: a mustache section referring to a
        key the model does not carry fails silently as an empty cell, so the
        rendered HTML is the only real check."""
        for state in (BatchState.QUEUED, BatchState.RUNNING, BatchState.STOPPING,
                      BatchState.STOPPED):
            _set_state(session_factory, make_batch(f"UI-r-{state.value}"), state)

        html = app_service.render_page("dashboard", {}, shell_html=False)

        assert "Batch Management" in html
        assert "data-batch-run=" in html
        assert "data-batch-stop=" in html
        assert "data-batch-delete=" in html
        for label in ("Queued", "Running", "Stopping", "Stopped"):
            assert label in html, f"{label} is not shown anywhere on the page"


class TestTheQueueCannotWedge:
    """Three defects found while exercising this feature on the real machine.

    They were not introduced by it - two batches had been stuck behind them for
    some time - but per-batch control is unusable while they exist, because the
    queue never advances no matter what an operator presses.
    """

    def test_a_crash_gives_back_the_attempt_it_interrupted(
            self, app_service, make_batch, session_factory):
        """`claim_next` charges an attempt on the way in. A process killed
        mid-OCR has been charged for an attempt that produced nothing, and once
        the counter passes the cap the document is unclaimable while still
        PENDING - runnable by nothing, finished by nothing."""
        batch_id = make_batch("WD-attempt")
        doc_pk = _hold_a_document(session_factory, batch_id)
        with session_scope(session_factory) as session:
            UnitOfWork(session).documents.get(doc_pk).ocr_attempts = 2

        app_service.runner.recover()

        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.get(doc_pk)
            assert doc.ocr_state is StageState.PENDING
            assert doc.ocr_attempts == 1, "a crashed attempt was counted as a try"

    def test_the_attempt_counter_never_goes_negative(
            self, app_service, make_batch, session_factory):
        batch_id = make_batch("WD-floor")
        doc_pk = _hold_a_document(session_factory, batch_id)
        with session_scope(session_factory) as session:
            UnitOfWork(session).documents.get(doc_pk).ocr_attempts = 0
        app_service.runner.recover()
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).documents.get(doc_pk).ocr_attempts == 0

    def test_an_unclaimable_document_is_found(self, make_batch, session_factory):
        batch_id = make_batch("WD-find")
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(batch_id, per_page=1)[0][0]
            doc.ocr_attempts = 9  # far past any cap
            session.flush()
            found = uow.documents.stranded({"ocr": 1, "extract": 1}, batch_id)
        assert [pk for pk, _ in found] == [doc.id]

    def test_a_healthy_pending_document_is_not_swept_up(
            self, make_batch, session_factory):
        """The sweep marks documents FAILED. Catching a document that simply has
        not been processed yet would destroy work that was about to happen."""
        batch_id = make_batch("WD-healthy")
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.stranded({"ocr": 1, "extract": 1}, batch_id) == []

    def test_a_stranded_document_stops_blocking_its_batch(
            self, app_service, make_batch, session_factory):
        """The whole point. Before the sweep, a batch holding one of these could
        never satisfy `is_finished`, so it held RUNNING for ever and every
        queued batch waited behind it indefinitely."""
        batch_id = make_batch("WD-unblock", files=1)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.documents.list_for_batch(batch_id, per_page=1)[0][0].ocr_attempts = 9
            session.flush()
            assert uow.batches.is_finished(batch_id) is False

        failed = app_service.runner._fail_stranded(batch_id)

        assert failed == 1
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).batches.is_finished(batch_id) is True

    def test_the_sweep_says_which_stage_gave_up(self, app_service, make_batch,
                                                session_factory):
        """A bare "failed" on the Failed OCR page tells an operator nothing."""
        batch_id = make_batch("WD-reason", files=1)
        with session_scope(session_factory) as session:
            UnitOfWork(session).documents.list_for_batch(
                batch_id, per_page=1)[0][0].ocr_attempts = 9
        app_service.runner._fail_stranded(batch_id)
        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.list_for_batch(
                batch_id, per_page=1)[0][0]
            assert "ocr" in (doc.failure_reason or "")
            assert "attempts" in (doc.failure_reason or "")

    def test_the_sweep_is_scoped_to_one_batch(self, app_service, make_batch,
                                              session_factory):
        mine = make_batch("WD-mine", files=1)
        theirs = make_batch("WD-theirs", files=1)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            for bid in (mine, theirs):
                uow.documents.list_for_batch(bid, per_page=1)[0][0].ocr_attempts = 9

        app_service.runner._fail_stranded(mine)

        with session_scope(session_factory) as session:
            other = UnitOfWork(session).documents.list_for_batch(
                theirs, per_page=1)[0][0]
            assert other.ocr_state is not StageState.FAILED

    def test_a_finished_document_is_not_counted_as_in_flight(
            self, make_batch, session_factory):
        """A terminal document can still carry a stage column left at RUNNING.
        Counting those made a completed batch report work in flight for ever,
        and the delete interlock then waited on a document that had stopped
        existing."""
        from core.db.models import DocumentState

        batch_id = make_batch("WD-inflight")
        doc_pk = _hold_a_document(session_factory, batch_id)
        with session_scope(session_factory) as session:
            UnitOfWork(session).documents.get(doc_pk).overall_state = \
                DocumentState.PROCESSED

        with session_scope(session_factory) as session:
            assert UnitOfWork(session).batches.in_flight(batch_id) == 0

    def test_an_idle_worker_really_waits(self, app_service):
        """`_wake` is set by `start` and was never cleared, so every
        `wait(timeout=...)` in the loop returned instantly and each idle worker
        re-ran a locking claim query as fast as the database would answer."""
        import time

        runner = app_service.runner
        runner._wake.set()
        started = time.monotonic()
        runner._idle(0.4)
        first = time.monotonic() - started

        # The set flag is consumed by the first wait - that is what makes it a
        # wake-up rather than a permanent short-circuit.
        started = time.monotonic()
        runner._idle(0.4)
        second = time.monotonic() - started

        assert first < 0.3, "a pending wake-up should return immediately"
        assert second >= 0.35, f"the second wait did not wait ({second:.3f}s)"

    def test_a_wake_up_during_a_wait_is_not_swallowed(self, app_service):
        """The flag is cleared after the wait, never before: cleared first, a
        `set` racing in between would be lost and the worker would sleep
        through the start it was meant to react to."""
        import threading
        import time

        runner = app_service.runner
        runner._wake.clear()
        threading.Timer(0.1, runner._wake.set).start()
        started = time.monotonic()
        runner._idle(5.0)
        assert time.monotonic() - started < 1.0, "the wake-up was missed"


class TestTheBadgeColours:
    """Pure presentation, but wrong here means a healthy batch looks broken."""

    def test_a_stopped_batch_is_not_coloured_as_a_failure(self):
        from app.ui.renderer import state_badge

        assert state_badge("stopped") != "danger"
        assert state_badge("failed") == "danger"

    def test_stopping_reads_as_in_progress(self):
        from app.ui.renderer import state_badge

        assert state_badge("stopping") == state_badge("running")
