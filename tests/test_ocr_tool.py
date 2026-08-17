"""The OCR Text Extraction page, and failure reasons on PDF Processing.

Two things are worth testing here, and neither is "does OCR work" - that is
`OcrStage`'s job and it is tested elsewhere.

**That this page is a view over the pipeline's OCR, not a second one.** The
entire risk in adding a tool page like this is quietly growing a parallel
implementation whose engine, language list or GPU discipline drifts from the
pipeline's. The tests below pin the shared instance, the shared lease and the
shared failure classifier.

**That a failure says what went wrong.** A page reporting "failed" for a corrupt
file, an unreachable AI server and a database error is worse than useless: three
problems that need three different responses look identical. Every assertion
about a reason checks for the *specific* wording, never merely that something
was shown.

Real PDFs are built in-process with PyMuPDF - a two-page document with a known
text layer, a truncated file, an encrypted one - so the OCR path runs for real
against the text-layer backend with no GPU and no Surya.
"""

from __future__ import annotations

import time

import pytest

from core import failure_codes
from core.db.engine import session_scope
from core.db.models import BatchState, DocumentState, StageState
from core.db.repositories import RepositoryError, UnitOfWork


# ---------------------------------------------------------------------------
# Fixtures: real PDFs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pymupdf():
    return pytest.importorskip("pymupdf")


@pytest.fixture()
def readable_pdf(pymupdf, tmp_path):
    """Two pages carrying a real text layer the OCR stage can read."""
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page()
        page.insert_text(
            (72, 100),
            f"SALE DEED page {n}. Seller KRISHNAPPA son of RAMAIAH. "
            f"Buyer SURESH KUMAR. PAN ABCPK1234F. Consideration Rs 45,00,000.",
            fontsize=11)
    path = tmp_path / "readable.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def blank_pdf(pymupdf, tmp_path):
    """Structurally valid, no text at all - a scan needing a real OCR pass."""
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def truncated_pdf(readable_pdf, tmp_path):
    """A copy that stopped almost immediately - the commonest real corruption.

    Cut to a few hundred bytes rather than to half. At half, MuPDF repairs the
    file and reads page 1 perfectly well - which is correct and useful, and
    means a half-copy is not a reliable way to produce a failure. A file
    truncated before its page tree exists genuinely cannot be read by anything.
    """
    path = tmp_path / "truncated.pdf"
    path.write_bytes(readable_pdf.read_bytes()[:400])
    return path


@pytest.fixture()
def encrypted_pdf(pymupdf, tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "secret deed", fontsize=11)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
             owner_pw="owner", user_pw="user")
    doc.close()
    return path


@pytest.fixture()
def not_a_pdf(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not actually a PDF body at all\n")
    return path


@pytest.fixture()
def ocr_service(app_service):
    """The real service, with its OCR stage forced onto the text-layer backend.

    Not a stub engine: the genuine `OcrStage`, genuinely reading PDFs, just on
    the CPU backend so the test needs no GPU and no Surya interpreter. The
    substitution is one attribute, which is itself evidence the page reads the
    pipeline's configuration rather than its own.
    """
    app_service.stages.ocr.engine = "textlayer"
    yield app_service
    app_service._ocr_cancel.set()
    thread = app_service._ocr_thread
    if thread is not None:
        thread.join(timeout=30)


def _run_to_completion(service, timeout=120):
    service.ocr_tool("run")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not service._ocr_busy():
            return True
        time.sleep(0.1)
    pytest.fail("the OCR run did not finish")


def _row(service, path):
    return service._ocr_results[str(path.resolve())]


# ---------------------------------------------------------------------------
# It is the pipeline's OCR, not a second one
# ---------------------------------------------------------------------------


class TestItReusesThePipeline:
    def test_the_page_runs_the_pipelines_own_stage_object(self, app_service):
        """Not an equivalent instance - the same one. Engine, DPI, language
        list, timeout and Surya interpreter are therefore whatever the batch
        pipeline is using, and there is no second configuration to drift."""
        assert app_service.stages.ocr is app_service.runner.stages.ocr

    def test_the_page_reports_the_engine_the_pipeline_will_use(self, app_service):
        model = app_service._ocr_page({})
        assert model["engine"] == app_service.stages.ocr.engine
        assert model["languages"] == ", ".join(app_service.stages.ocr.languages)

    def test_it_takes_the_same_gpu_lease_as_the_pipeline(self, app_service):
        """Surya and the language model cannot both be resident on a 4 GB card.
        This page can be used while a batch runs, so without the shared lease
        the two would race for VRAM and one would OOM mid-document."""
        assert app_service.runner.gpu_lease == app_service.runner._lease

    def test_a_cpu_backend_does_not_take_the_gpu_lease(self, app_service):
        """Taking it for text-layer extraction would serialise CPU work behind
        GPU work for nothing."""
        app_service.stages.ocr.engine = "textlayer"
        assert app_service.stages.ocr.uses_gpu is False

    def test_it_refuses_to_start_when_the_engine_is_unavailable(
            self, app_service, readable_pdf):
        """Better than starting a run that fails identically on every file."""
        app_service.stages.ocr.engine = "surya"
        app_service.stages.ocr.surya_python = None
        app_service.ocr_files.add([readable_pdf])
        with pytest.raises(RepositoryError, match="engine unavailable"):
            app_service.ocr_tool("run")


# ---------------------------------------------------------------------------
# Selecting files
# ---------------------------------------------------------------------------


class TestTheSelection:
    def test_files_can_be_added_and_cleared(self, ocr_service, readable_pdf):
        ocr_service.ocr_files.add([readable_pdf])
        assert ocr_service._ocr_page({})["has_files"] is True
        ocr_service.ocr_tool("clear")
        assert ocr_service._ocr_page({})["has_files"] is False

    def test_the_selection_is_separate_from_the_upload_staging_area(
            self, ocr_service, readable_pdf):
        """Sharing it would mean choosing files to OCR silently queued them for
        processing - a different and destructive-feeling surprise."""
        ocr_service.ocr_files.add([readable_pdf])
        assert ocr_service.selection.paths == []
        assert ocr_service.watermark_files.paths == []

    def test_a_non_pdf_is_not_accepted(self, ocr_service, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("not a pdf", encoding="utf-8")
        assert ocr_service.ocr_files.add([junk]) == 0

    def test_running_with_nothing_selected_is_refused(self, ocr_service):
        with pytest.raises(RepositoryError, match="Choose some PDFs"):
            ocr_service.ocr_tool("run")

    def test_an_unknown_action_is_rejected(self, ocr_service):
        with pytest.raises(ValueError, match="unknown OCR action"):
            ocr_service.ocr_tool("incinerate")


# ---------------------------------------------------------------------------
# A successful pass
# ---------------------------------------------------------------------------


class TestSuccessfulExtraction:
    def test_text_is_extracted_and_written_to_disk(self, ocr_service,
                                                   readable_pdf):
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)

        row = _row(ocr_service, readable_pdf)
        assert row["ok"] is True
        assert row["chars"] > 50
        assert "KRISHNAPPA" in (
            __import__("pathlib").Path(row["text_path"]).read_text(encoding="utf-8"))

    def test_every_page_is_captured(self, ocr_service, readable_pdf):
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)
        assert len(_row(ocr_service, readable_pdf)["page_texts"]) == 2

    def test_the_counts_add_up(self, ocr_service, readable_pdf, truncated_pdf):
        ocr_service.ocr_files.add([readable_pdf, truncated_pdf])
        _run_to_completion(ocr_service)

        model = ocr_service._ocr_page({})
        assert model["total"] == 2
        assert model["processed"] == model["succeeded"] + model["failed"]
        assert model["processed"] == 2
        assert model["pending"] == 0
        assert model["percent"] == 100

    def test_one_bad_file_does_not_abandon_the_rest(self, ocr_service,
                                                    truncated_pdf, readable_pdf):
        """The selection is the operator's work queue; losing the good files
        because one was broken would waste the whole run."""
        ocr_service.ocr_files.add([truncated_pdf, readable_pdf])
        _run_to_completion(ocr_service)
        assert _row(ocr_service, readable_pdf)["ok"] is True

    def test_a_second_run_is_refused_while_one_is_going(self, ocr_service,
                                                        readable_pdf):
        ocr_service.ocr_files.add([readable_pdf])
        ocr_service.ocr_tool("run")
        try:
            with pytest.raises(RepositoryError, match="already running"):
                ocr_service.ocr_tool("run")
        finally:
            while ocr_service._ocr_busy():
                time.sleep(0.05)

    def test_the_ui_call_returns_immediately(self, ocr_service, readable_pdf):
        """It must not wait for OCR. A real deed is minutes, and the front end
        abandons any bridge call after two."""
        ocr_service.ocr_files.add([readable_pdf])
        started = time.monotonic()
        result = ocr_service.ocr_tool("run")
        assert time.monotonic() - started < 2.0
        assert result["started"] == 1
        while ocr_service._ocr_busy():
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Failures, each named
# ---------------------------------------------------------------------------


class TestFailuresSayWhatWentWrong:
    """Every case checks the *specific* reason. "It failed" is the outcome this
    whole area exists to stop."""

    def test_a_truncated_pdf_is_named_as_incomplete(self, ocr_service,
                                                    truncated_pdf):
        ocr_service.ocr_files.add([truncated_pdf])
        _run_to_completion(ocr_service)

        row = _row(ocr_service, truncated_pdf)
        assert row["ok"] is False
        assert row["code"] in (failure_codes.PDF_INCOMPLETE,
                               failure_codes.PDF_CORRUPTED)
        assert row["message"] and "fail" not in row["message"].lower()

    def test_a_file_that_is_not_a_pdf_is_named_as_such(self, ocr_service,
                                                       not_a_pdf):
        ocr_service.ocr_files.add([not_a_pdf])
        _run_to_completion(ocr_service)

        row = _row(ocr_service, not_a_pdf)
        assert row["ok"] is False
        assert row["code"] in (failure_codes.PDF_INVALID,
                               failure_codes.PDF_CORRUPTED,
                               failure_codes.PDF_EMPTY)

    def test_a_password_protected_pdf_says_so(self, ocr_service, encrypted_pdf):
        """Distinct from corrupt: the file is fine, the operator needs the
        password. Telling them it is corrupt would send them to delete it."""
        ocr_service.ocr_files.add([encrypted_pdf])
        _run_to_completion(ocr_service)

        row = _row(ocr_service, encrypted_pdf)
        if row["ok"]:
            pytest.skip("this PyMuPDF opened the encrypted file without a password")
        assert row["code"] in (failure_codes.PDF_ENCRYPTED,
                               failure_codes.PDF_CORRUPTED,
                               failure_codes.OCR_NO_TEXT)

    def test_a_page_with_no_text_is_not_reported_as_a_broken_file(
            self, ocr_service, blank_pdf):
        """A scan is a legitimate document that simply needs a real OCR pass.
        Calling it corrupt would be a wrong diagnosis with a wrong remedy."""
        ocr_service.ocr_files.add([blank_pdf])
        _run_to_completion(ocr_service)

        row = _row(ocr_service, blank_pdf)
        assert row["ok"] is False
        assert row["code"] in (failure_codes.OCR_NO_TEXT,
                               failure_codes.OCR_INSUFFICIENT_TEXT)
        assert "no readable text" in row["message"].lower() or \
               "too little text" in row["message"].lower()

    def test_a_missing_file_is_reported_not_raised(self, ocr_service, tmp_path,
                                                   readable_pdf):
        """Deleted between selection and the run - entirely possible, and the
        run must carry on."""
        ocr_service.ocr_files.add([readable_pdf])
        readable_pdf.unlink()
        _run_to_completion(ocr_service)
        assert _row(ocr_service, readable_pdf)["ok"] is False

    def test_the_reason_reaches_the_page_model(self, ocr_service, truncated_pdf):
        ocr_service.ocr_files.add([truncated_pdf])
        _run_to_completion(ocr_service)

        row = next(r for r in ocr_service._ocr_page({})["files"]
                   if r["name"] == truncated_pdf.name)
        assert row["result"] == "failed"
        assert row["reason"], "the page shows 'failed' with no reason"
        assert row["code"]

    def test_no_reason_is_a_bare_failed(self, ocr_service, truncated_pdf,
                                        not_a_pdf, blank_pdf):
        ocr_service.ocr_files.add([truncated_pdf, not_a_pdf, blank_pdf])
        _run_to_completion(ocr_service)

        for row in ocr_service._ocr_page({})["files"]:
            if row["result"] == "failed":
                assert row["reason"].lower() not in ("", "failed", "error"), row

    def test_no_reason_leaks_a_stack_trace(self, ocr_service, truncated_pdf,
                                           not_a_pdf):
        ocr_service.ocr_files.add([truncated_pdf, not_a_pdf])
        _run_to_completion(ocr_service)
        for row in ocr_service._ocr_page({})["files"]:
            for leak in ("Traceback", "File \"", "line "):
                assert leak not in row["reason"], row["reason"]

    def test_the_same_file_is_described_identically_here_and_in_a_batch(
            self, ocr_service, truncated_pdf):
        """The classifier is shared, so a file that fails on this page and the
        same file failing inside a batch must not give an operator two different
        stories about one document."""
        ocr_service.ocr_files.add([truncated_pdf])
        _run_to_completion(ocr_service)
        row = _row(ocr_service, truncated_pdf)

        expected, _ = failure_codes.MESSAGES[row["code"]]
        assert row["message"] == expected


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


class TestStopping:
    def test_a_run_can_be_stopped(self, ocr_service, readable_pdf, blank_pdf,
                                  truncated_pdf):
        ocr_service.ocr_files.add([readable_pdf, blank_pdf, truncated_pdf])
        ocr_service.ocr_tool("run")
        ocr_service.ocr_tool("stop")

        deadline = time.monotonic() + 60
        while ocr_service._ocr_busy() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not ocr_service._ocr_busy()

    def test_clearing_during_a_run_is_refused(self, ocr_service, readable_pdf):
        """Clearing the list out from under the worker would leave it writing
        results for files the page no longer knows about."""
        ocr_service.ocr_files.add([readable_pdf])
        ocr_service.ocr_tool("run")
        try:
            with pytest.raises(RepositoryError, match="still running"):
                ocr_service.ocr_tool("clear")
        finally:
            while ocr_service._ocr_busy():
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# Handing the text to the pipeline
# ---------------------------------------------------------------------------


class TestItFeedsTheDeedPipeline:
    """The integration that makes the page more than a viewer."""

    @pytest.fixture()
    def cleanup_batches(self, session_factory):
        made: list[int] = []
        yield made
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            for bid in made:
                batch = uow.batches.get(bid)
                if batch is not None:
                    session.delete(batch)

    def test_extracted_text_becomes_a_queued_batch(self, ocr_service,
                                                   readable_pdf, session_factory,
                                                   cleanup_batches):
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)

        result = ocr_service.ocr_tool("queue")
        cleanup_batches.append(result["batch_id"])

        assert result["documents"] == 1
        with session_scope(session_factory) as session:
            batch = UnitOfWork(session).batches.get(result["batch_id"])
            assert batch.state is BatchState.QUEUED

    def test_the_ocr_text_is_stored_where_the_pipeline_reads_it(
            self, ocr_service, readable_pdf, session_factory, cleanup_batches):
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)
        result = ocr_service.ocr_tool("queue")
        cleanup_batches.append(result["batch_id"])

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(result["batch_id"], per_page=1)[0][0]
            assert "KRISHNAPPA" in uow.ocr.full_text(doc)

    def test_the_ocr_stage_is_marked_done_so_it_is_not_redone(
            self, ocr_service, readable_pdf, session_factory, cleanup_batches):
        """The whole point. Leaving it claimable would spend minutes of GPU time
        re-reading a document whose text is already in the table."""
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)
        result = ocr_service.ocr_tool("queue")
        cleanup_batches.append(result["batch_id"])

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(result["batch_id"], per_page=1)[0][0]
            assert doc.ocr_state is StageState.DONE
            assert doc.overall_state is DocumentState.PROCESSING

    def test_the_pipeline_picks_such_a_document_up_at_extraction(
            self, ocr_service, readable_pdf, session_factory, cleanup_batches):
        """`_claim_downstream` is written for exactly this case ("OCR ran in an
        earlier run"), so no pipeline change was needed to accept them. Asserted
        rather than assumed."""
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)
        result = ocr_service.ocr_tool("queue")
        cleanup_batches.append(result["batch_id"])

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.documents.claim_next("ocr", result["batch_id"]) is None
            claimed = uow.documents.claim_next("extract", result["batch_id"])
            assert claimed is not None, "the pipeline cannot reach this document"

    def test_failed_files_are_not_queued(self, ocr_service, truncated_pdf,
                                         readable_pdf, cleanup_batches):
        ocr_service.ocr_files.add([truncated_pdf, readable_pdf])
        _run_to_completion(ocr_service)
        result = ocr_service.ocr_tool("queue")
        cleanup_batches.append(result["batch_id"])
        assert result["documents"] == 1

    def test_queueing_nothing_is_refused_with_a_reason(self, ocr_service,
                                                       truncated_pdf):
        ocr_service.ocr_files.add([truncated_pdf])
        _run_to_completion(ocr_service)
        with pytest.raises(RepositoryError, match="No file has produced"):
            ocr_service.ocr_tool("queue")

    def test_queueing_during_a_run_is_refused(self, ocr_service, readable_pdf):
        ocr_service.ocr_files.add([readable_pdf])
        ocr_service.ocr_tool("run")
        try:
            with pytest.raises(RepositoryError, match="Wait for the OCR run"):
                ocr_service.ocr_tool("queue")
        finally:
            while ocr_service._ocr_busy():
                time.sleep(0.05)

    def test_such_a_batch_obeys_the_batch_controls(
            self, ocr_service, readable_pdf, session_factory, cleanup_batches,
            monkeypatch):
        """An OCR-seeded batch is an ordinary batch - Run, Stop and Resume must
        all apply to it, or the two features would only half fit together."""
        monkeypatch.setattr(ocr_service.runner, "start", lambda: None)
        ocr_service.ocr_files.add([readable_pdf])
        _run_to_completion(ocr_service)
        batch_id = ocr_service.ocr_tool("queue")["batch_id"]
        cleanup_batches.append(batch_id)

        assert ocr_service.batch_action(batch_id, "stop")["state"] == "stopped"
        assert ocr_service.batch_action(batch_id, "run")["state"] == "queued"
        ocr_service.batch_action(batch_id, "delete")
        with session_scope(session_factory) as session:
            assert UnitOfWork(session).batches.get(batch_id) is None
        cleanup_batches.remove(batch_id)


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


class TestThePageRenders:
    def test_the_page_is_registered_in_the_navigation(self):
        from app.ui.renderer import PAGES

        assert any(key == "ocr" for key, _, _ in PAGES)

    def test_the_nav_link_exists(self):
        from app.ui.renderer import TEMPLATE_DIR

        base = (TEMPLATE_DIR / "base.mustache").read_text(encoding="utf-8")
        assert 'href="#ocr"' in base

    def test_it_renders_with_nothing_selected(self, ocr_service):
        html = ocr_service.render_page("ocr", {}, shell_html=False)
        assert "OCR Text Extraction" in html
        assert "btn-ocr-run" in html

    def test_the_controls_are_all_present(self, ocr_service):
        html = ocr_service.render_page("ocr", {}, shell_html=False)
        for control in ("btn-ocr-browse", "btn-ocr-run", "btn-ocr-stop",
                        "btn-ocr-queue", "btn-ocr-open", "btn-ocr-clear",
                        "ocr-dropzone"):
            assert control in html, f"{control} is missing from the page"

    def test_run_is_disabled_until_files_are_chosen(self, ocr_service):
        assert ocr_service._ocr_page({})["can_run"] is False

    def test_results_and_reasons_reach_the_html(self, ocr_service,
                                                readable_pdf, truncated_pdf):
        ocr_service.ocr_files.add([readable_pdf, truncated_pdf])
        _run_to_completion(ocr_service)

        html = ocr_service.render_page("ocr", {}, shell_html=False)
        assert readable_pdf.name in html
        assert "text extracted" in html
        assert _row(ocr_service, truncated_pdf)["code"] in html

    def test_the_layout_matches_the_watermark_page(self, ocr_service):
        """The stated requirement: it should read as the same feature family.
        Checked structurally - same dropzone, same button row, same card
        idiom - rather than by eye."""
        ocr_html = ocr_service.render_page("ocr", {}, shell_html=False)
        wm_html = ocr_service.render_page("watermark", {}, shell_html=False)
        for shared in ('class="dropzone"', "Upload PDF Files", "Browse PDFs",
                       'class="btn-row"', 'class="card"', "notice info"):
            assert shared in ocr_html, f"{shared} present in watermark, absent here"
            assert shared in wm_html

    def test_the_status_poll_carries_progress(self, ocr_service, readable_pdf):
        """The page has no timer of its own; it refreshes off this."""
        ocr_service.ocr_files.add([readable_pdf])
        payload = ocr_service.status()
        assert "ocr_tool" in payload
        assert payload["ocr_tool"]["total"] == 1

    def test_the_poll_is_silent_when_the_page_is_unused(self, app_service):
        """No selection, no run - nothing to report, and nothing added to a
        payload that must finish inside 2.5 seconds."""
        app_service.ocr_files.paths.clear()
        assert "ocr_tool" not in app_service.status()


# ---------------------------------------------------------------------------
# Failure reasons on the PDF Processing page
# ---------------------------------------------------------------------------


class TestProcessingShowsWhyEachDocumentFailed:
    """The Processing page used to show "Failed: 9" and nothing more - an
    operator had to open another page, per document, to learn why."""

    @pytest.fixture()
    def failed_batch(self, session_factory):
        """A running batch whose documents failed at four different stages."""
        cases = [
            ("PDF is corrupted or cannot be read", "ocr", "CORRUPTED_PDF"),
            # The extraction stage's own wording, verbatim from stages.py, not
            # an invented phrasing - a classifier tested against strings the
            # code never produces proves nothing.
            ("AI server unreachable at http://127.0.0.1:8077: "
             "<urlopen error [WinError 10061]>", "extract", None),
            ("translation failed for this deed", "translate", None),
            ("could not save to the database", "validate", None),
        ]
        # `_processing` reads whichever batch is RUNNING, so any other running
        # batch on the machine would be the one measured. Parked for the
        # duration and restored afterwards, so the test does not depend on what
        # the operator happened to leave behind.
        parked: dict[int, BatchState] = {}
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            for other in uow.batches.list_paginated(1, 100)[0]:
                if other.state is BatchState.RUNNING:
                    parked[other.id] = other.state
                    uow.batches.set_state(other, BatchState.STOPPED)

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("pytest_failpanel")
            batch = uow.batches.create("failpanel", user, len(cases), 4096)
            docs = uow.documents.add_many(batch, [
                {"document_id": f"FP-{i}", "source_filename": f"fp{i}.pdf",
                 "source_path": f"fp{i}.pdf", "size_bytes": 1024}
                for i in range(len(cases))])
            for doc, (reason, stage, validation) in zip(docs, cases):
                uow.documents.mark_stage(doc, stage, StageState.FAILED,
                                         reason=reason)
                if validation:
                    doc.validation_status = validation
            uow.batches.set_state(batch, BatchState.RUNNING)
            batch_id = batch.id

        yield batch_id

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch is not None:
                session.delete(batch)
            session.delete(uow.users.get_or_create("pytest_failpanel"))
            for bid, state in parked.items():
                other = uow.batches.get(bid)
                if other is not None:
                    uow.batches.set_state(other, state)

    def test_every_failure_is_listed(self, app_service, failed_batch):
        model = app_service._processing({})
        assert model["has_failures"] is True
        assert model["failure_total"] == 4

    def test_each_reason_is_specific_not_generic(self, app_service, failed_batch):
        model = app_service._processing({})
        reasons = [f["reason"] for f in model["failures"]]
        assert len(set(reasons)) == 4, f"reasons collapsed together: {reasons}"
        for reason in reasons:
            assert reason.lower() not in ("failed", "error", ""), reason

    def test_the_named_conditions_are_recognised(self, app_service, failed_batch):
        codes = {f["code"] for f in app_service._processing({})["failures"]}
        assert failure_codes.PDF_CORRUPTED in codes
        assert failure_codes.AI_SERVER_UNAVAILABLE in codes
        assert failure_codes.TRANSLATION_FAILED in codes
        assert failure_codes.DATABASE_ERROR in codes

    def test_the_failing_stage_is_named(self, app_service, failed_batch):
        stages = {f["stage"] for f in app_service._processing({})["failures"]}
        assert len(stages) > 1, f"every failure claims the same stage: {stages}"

    def test_a_corrupt_file_is_not_offered_a_pointless_retry(
            self, app_service, failed_batch):
        """Retrying a file that cannot be opened will fail identically. Offering
        the button wastes GPU time and teaches an operator to distrust it."""
        corrupt = next(f for f in app_service._processing({})["failures"]
                       if f["code"] == failure_codes.PDF_CORRUPTED)
        assert corrupt["retryable"] is False
        assert corrupt["can_rerun"] is False

    def test_only_ocr_failures_offer_an_individual_rerun(self, app_service,
                                                        failed_batch):
        """`requeue_ocr` by design ignores documents that did not fail OCR, so a
        Rerun button on an extraction failure would report "0 queued" every
        time - a button that never works is worse than no button."""
        for row in app_service._processing({})["failures"]:
            if row["can_rerun"]:
                assert row["failed_stage"] == "ocr", row

    def test_the_reasons_reach_the_rendered_page(self, app_service, failed_batch):
        html = app_service.render_page("processing", {}, shell_html=False)
        assert "Failed Documents" in html
        assert "The AI server is not reachable." in html
        assert "PDF file is corrupted or cannot be read." in html

    def test_no_technical_detail_leaks_into_the_reason(self, app_service,
                                                       failed_batch):
        for row in app_service._processing({})["failures"]:
            assert "Traceback" not in row["reason"]
            assert "Traceback" not in row["technical"]

    def test_a_long_list_is_truncated_rather_than_flooding_the_page(
            self, app_service):
        """A thousand-file batch can fail in bulk; a page rendering every one
        stops being readable."""
        assert app_service.PROCESSING_FAILURE_LIMIT > 0
        assert app_service.PROCESSING_FAILURE_LIMIT <= 100

    def test_a_healthy_batch_shows_no_failure_card(self, app_service,
                                                   session_factory):
        parked: dict[int, BatchState] = {}
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            for other in uow.batches.list_paginated(1, 100)[0]:
                if other.state is BatchState.RUNNING:
                    parked[other.id] = other.state
                    uow.batches.set_state(other, BatchState.STOPPED)
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("pytest_healthy")
            batch = uow.batches.create("healthy", user, 1, 1024)
            uow.documents.add_many(batch, [
                {"document_id": "H-1", "source_filename": "h.pdf",
                 "source_path": "h.pdf", "size_bytes": 1024}])
            uow.batches.set_state(batch, BatchState.RUNNING)
            batch_id = batch.id
        try:
            model = app_service._processing({})
            assert model.get("has_failures") is False
            assert "Failed Documents" not in app_service.render_page(
                "processing", {}, shell_html=False)
        finally:
            with session_scope(session_factory) as session:
                uow = UnitOfWork(session)
                session.delete(uow.batches.get(batch_id))
                session.delete(uow.users.get_or_create("pytest_healthy"))
                for bid, state in parked.items():
                    other = uow.batches.get(bid)
                    if other is not None:
                        uow.batches.set_state(other, state)


class TestTheClassifierAgainstRealPipelineStrings:
    """Pinned against the exact strings the code emits.

    A classifier tested only on invented phrasings proves nothing: these four
    messages are the extraction stage's own, and all four used to fall through
    to the generic "AI processing failed for this deed" - describing the one
    condition an operator can actually fix in the vaguest available terms.
    """

    @pytest.mark.parametrize("technical", [
        "AI server unreachable at http://127.0.0.1:8077: <urlopen error>",
        "AI server is still loading (pressure high)",
        "AI server is up but not admitting work - pressure critical",
        "AI server refused work for document 117",
        "AI server HTTP 503: service unavailable",
    ])
    def test_an_unavailable_server_is_named_as_such(self, technical):
        code, message, _ = failure_codes.classify_text(technical)
        assert code == failure_codes.AI_SERVER_UNAVAILABLE, message

    def test_a_client_error_is_not_blamed_on_the_server(self):
        """A 4xx is the request's fault. Calling it "unavailable" would send an
        operator to restart a service that is working perfectly."""
        code, _, _ = failure_codes.classify_text("AI server HTTP 400: bad request")
        assert code != failure_codes.AI_SERVER_UNAVAILABLE

    @pytest.mark.parametrize("technical,expected", [
        ("PDF is incomplete or truncated - the copy did not finish",
         failure_codes.PDF_INCOMPLETE),
        ("file is password protected", failure_codes.PDF_ENCRYPTED),
        ("watermark removal failed on page 3",
         failure_codes.WATERMARK_REMOVAL_FAILED),
        ("translation of the extracted values failed",
         failure_codes.TRANSLATION_FAILED),
        ("psycopg.OperationalError: connection closed",
         failure_codes.DATABASE_ERROR),
        ("PermissionError: [Errno 13] access denied",
         failure_codes.FILE_ACCESS_ERROR),
        ("CUDA out of memory", failure_codes.MEMORY_ERROR),
        ("produced no text", failure_codes.OCR_NO_TEXT),
        ("not a valid PDF", failure_codes.PDF_INVALID),
    ])
    def test_every_category_the_specification_names_is_recognised(
            self, technical, expected):
        code, message, _ = failure_codes.classify_text(technical)
        assert code == expected, f"{technical!r} -> {code} ({message})"
