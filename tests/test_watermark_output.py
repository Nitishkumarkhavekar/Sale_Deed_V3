"""Where cleaned deeds are written.

Cleaned copies used to land in a shared runtime folder under the installation
directory, named `<stem>_clean.pdf`. They now go to a `Cleaned Watermark`
subfolder beside the deeds themselves, under the original filename.

The property that matters most here is negative: **the input must survive
untouched**. Everything else is recoverable by running the removal again; an
overwritten original is not. Several tests below check the source bytes rather
than only the output, because "the clean copy exists" and "the original is
intact" are different claims and only one of them is unrecoverable if wrong.

Real PDFs with real overlay watermarks are built in-process, so `wm.remove`
genuinely runs and genuinely writes files.
"""

from __future__ import annotations

import errno
import os
import stat
import sys

import pytest

from core import watermark as wm
from core.db.repositories import RepositoryError


@pytest.fixture(scope="module")
def pymupdf():
    return pytest.importorskip("pymupdf")


def _watermarked(pymupdf, path, pages=2):
    """A PDF carrying a removable text-overlay watermark on every page."""
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((72, 120), f"SALE DEED page {n + 1}. Seller KRISHNAPPA.",
                         fontsize=11)
        # What the text-overlay detector looks for: a phrase from
        # WATERMARK_WORDS, repeated on most pages, set much larger than body
        # text and in a faint grey. Separable, so removal is lossless.
        # No rotation - PyMuPDF only accepts multiples of 90 and the detector
        # does not care either way.
        page.insert_text((120, 400), "DUPLICATE COPY", fontsize=48,
                         color=(0.8, 0.8, 0.8))
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def deeds(pymupdf, tmp_path):
    """A folder of three watermarked deeds, as an operator would have."""
    folder = tmp_path / "deeds"
    folder.mkdir()
    return [_watermarked(pymupdf, folder / f"deed-{n}.pdf") for n in range(1, 4)]


@pytest.fixture()
def service(app_service):
    return app_service


def _clean(service, paths):
    """Select, scan and remove - the operator's three clicks."""
    service.watermark_files.add(list(paths))
    service.watermark("scan")
    return service.watermark("remove")


# ---------------------------------------------------------------------------
# Where the files go
# ---------------------------------------------------------------------------


class TestTheOutputFolder:
    def test_cleaned_copies_land_beside_the_deeds(self, service, deeds):
        result = _clean(service, deeds)
        folder = deeds[0].parent / "Cleaned Watermark"

        assert result["removed"] == 3, result
        assert folder.is_dir()
        assert sorted(p.name for p in folder.glob("*.pdf")) == [
            "deed-1.pdf", "deed-2.pdf", "deed-3.pdf"]

    def test_the_folder_is_named_exactly_as_specified(self, service, deeds):
        _clean(service, deeds)
        names = [p.name for p in deeds[0].parent.iterdir() if p.is_dir()]
        assert names == ["Cleaned Watermark"], names

    def test_the_original_filename_is_kept(self, service, deeds):
        """No `_clean` suffix. The copy is in a different directory, so the
        original name cannot collide with the input."""
        _clean(service, deeds)
        for source in deeds:
            assert (source.parent / "Cleaned Watermark" / source.name).is_file()

    def test_an_existing_folder_is_reused_not_duplicated(self, service, deeds):
        folder = deeds[0].parent / "Cleaned Watermark"
        folder.mkdir()
        (folder / "from-an-earlier-run.pdf").write_bytes(b"%PDF-1.4 earlier")

        _clean(service, deeds)

        dirs = [p.name for p in deeds[0].parent.iterdir() if p.is_dir()]
        assert dirs == ["Cleaned Watermark"], dirs
        assert (folder / "from-an-earlier-run.pdf").is_file(), \
            "reusing the folder destroyed an earlier run's output"

    def test_the_folder_is_created_when_absent(self, service, deeds):
        assert not (deeds[0].parent / "Cleaned Watermark").exists()
        _clean(service, deeds)
        assert (deeds[0].parent / "Cleaned Watermark").is_dir()

    def test_each_source_folder_gets_its_own(self, pymupdf, service, tmp_path):
        """A selection can span folders. One shared output directory would mix
        two operators' work and collide on any repeated filename."""
        a = tmp_path / "north"
        b = tmp_path / "south"
        a.mkdir()
        b.mkdir()
        # Same filename in both, which is exactly the collision case.
        first = _watermarked(pymupdf, a / "deed.pdf")
        second = _watermarked(pymupdf, b / "deed.pdf")

        result = _clean(service, [first, second])

        assert result["removed"] == 2
        assert (a / "Cleaned Watermark" / "deed.pdf").is_file()
        assert (b / "Cleaned Watermark" / "deed.pdf").is_file()
        assert len(result["output_dirs"]) == 2

    def test_rendering_the_page_creates_no_directories(self, service, deeds):
        """Selecting files must not scatter empty folders across a disk for
        deeds the operator has not chosen to clean."""
        service.watermark_files.add(deeds)
        service._watermark_page({})
        assert not (deeds[0].parent / "Cleaned Watermark").exists()

    def test_a_second_run_replaces_its_own_previous_output(self, service, deeds):
        """Idempotent. Piling up `deed-1 (2).pdf` after each run would leave an
        operator unable to tell which copy is current."""
        _clean(service, deeds[:1])
        folder = deeds[0].parent / "Cleaned Watermark"
        first = (folder / deeds[0].name).read_bytes()

        service.watermark("clear")
        _clean(service, deeds[:1])

        assert len(list(folder.glob("*.pdf"))) == 1
        assert (folder / deeds[0].name).read_bytes()[:8] == first[:8]


# ---------------------------------------------------------------------------
# The input is not touched
# ---------------------------------------------------------------------------


class TestTheOriginalSurvives:
    """The one unrecoverable failure. Checked by bytes, not by existence."""

    def test_the_source_bytes_are_unchanged(self, service, deeds):
        before = {p: p.read_bytes() for p in deeds}
        _clean(service, deeds)
        for source, original in before.items():
            assert source.read_bytes() == original, f"{source.name} was modified"

    def test_the_source_still_has_its_watermark(self, service, deeds):
        """Stronger than a byte comparison in one way: it proves the removal
        happened to the copy and not to the input."""
        _clean(service, deeds)
        assert wm.scan(deeds[0]).confirmed, \
            "the watermark was removed from the original"

    def test_the_output_is_a_different_file_from_the_input(self, service, deeds):
        _clean(service, deeds)
        cleaned = deeds[0].parent / "Cleaned Watermark" / deeds[0].name
        assert cleaned.resolve() != deeds[0].resolve()

    def test_the_watermark_really_is_gone_from_the_copy(self, service, deeds):
        """Otherwise every other test here would pass over a plain copy."""
        _clean(service, deeds)
        cleaned = deeds[0].parent / "Cleaned Watermark" / deeds[0].name
        assert not wm.scan(cleaned).confirmed

    def test_a_file_already_in_the_output_folder_is_refused_not_overwritten(
            self, service, deeds):
        """The one case where the subfolder rule points at the source's own
        directory. It refuses with a reason rather than decorating the name."""
        from app.services import _AlreadyInOutputFolder

        _clean(service, deeds[:1])
        cleaned = deeds[0].parent / "Cleaned Watermark" / deeds[0].name
        before = cleaned.read_bytes()

        with pytest.raises(_AlreadyInOutputFolder, match="already inside"):
            service._cleaned_target(cleaned)
        assert cleaned.read_bytes() == before

    def test_that_refusal_reaches_the_page_as_a_reason(self, pymupdf, service,
                                                       tmp_path):
        """It must fail that one file, not the run - and say why."""
        folder = tmp_path / "deeds" / "Cleaned Watermark"
        folder.mkdir(parents=True)
        stray = _watermarked(pymupdf, folder / "stray.pdf")

        result = _clean(service, [stray])

        assert result["removed"] == 0
        assert result["failed"] == 1
        row = service._watermark_page({})["files"][0]
        assert row["result"] == "failed"
        assert "already inside" in row["reason"]

    def test_no_output_name_ever_carries_a_clean_suffix(self, service, deeds):
        """The decoration is gone everywhere, not just in the common case."""
        _clean(service, deeds)
        folder = deeds[0].parent / "Cleaned Watermark"
        for produced in folder.glob("*.pdf"):
            assert "_clean" not in produced.stem, produced.name


# ---------------------------------------------------------------------------
# Filesystem failures
# ---------------------------------------------------------------------------


class TestFilesystemFailuresAreExplained:
    """`[Errno 28]` is not an answer. Each of these has a different remedy, and
    naming which is the entire point."""

    def test_a_full_disk_says_so(self, service, tmp_path):
        reason = service._filesystem_reason(
            OSError(errno.ENOSPC, "No space left on device"), tmp_path / "x.pdf")
        assert "full" in reason.lower()
        assert "Errno" not in reason

    def test_a_locked_file_says_to_close_it(self, service, tmp_path):
        exc = PermissionError(errno.EACCES, "in use")
        exc.winerror = 32
        reason = service._filesystem_reason(exc, tmp_path / "deed.pdf")
        assert "open in another program" in reason
        assert "deed.pdf" in reason

    def test_a_permission_problem_is_distinguished_from_a_lock(self, service,
                                                               tmp_path):
        """Both are PermissionError on Windows and the remedies are completely
        different - close a viewer, or obtain write access."""
        exc = PermissionError(errno.EACCES, "denied")
        exc.winerror = 5
        reason = service._filesystem_reason(exc, tmp_path / "deed.pdf")
        assert "permission" in reason.lower()
        assert "open in another program" not in reason

    def test_a_missing_folder_says_so(self, service, tmp_path):
        reason = service._filesystem_reason(
            OSError(errno.ENOENT, "not found"), tmp_path / "gone" / "x.pdf")
        assert "no longer exists" in reason

    def test_an_unrecognised_error_still_says_where(self, service, tmp_path):
        reason = service._filesystem_reason(
            OSError(errno.EIO, "I/O error"), tmp_path / "x.pdf")
        assert str(tmp_path) in reason
        assert reason

    def test_no_reason_is_a_bare_error_code(self, service, tmp_path):
        for code in (errno.ENOSPC, errno.EACCES, errno.EROFS, errno.ENOENT,
                     errno.ENAMETOOLONG, errno.EIO):
            reason = service._filesystem_reason(OSError(code, "x"),
                                                tmp_path / "d.pdf")
            assert len(reason) > 20, reason
            assert "Errno" not in reason

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="a read-only directory bit does not stop writes "
                               "for the owner on Windows")
    def test_an_unwritable_folder_fails_the_file_not_the_run(
            self, pymupdf, service, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        deed = _watermarked(pymupdf, locked / "deed.pdf")
        open_dir = tmp_path / "open"
        open_dir.mkdir()
        fine = _watermarked(pymupdf, open_dir / "deed.pdf")
        os.chmod(locked, stat.S_IREAD | stat.S_IEXEC)
        try:
            result = _clean(service, [deed, fine])
            assert result["failed"] >= 1
            assert result["removed"] == 1, "one bad folder aborted the whole run"
        finally:
            os.chmod(locked, stat.S_IRWXU)

    def test_a_failure_is_reported_per_file_not_as_an_abort(self, pymupdf,
                                                            service, tmp_path,
                                                            monkeypatch):
        """A selection can span folders and only one may be the problem. Losing
        the whole run because of it would waste the operator's time."""
        folder = tmp_path / "deeds"
        folder.mkdir()
        bad = _watermarked(pymupdf, folder / "bad.pdf")
        good = _watermarked(pymupdf, folder / "good.pdf")

        real = wm.remove

        def flaky(source, target=None, **kw):
            if "bad" in str(source):
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(source, target, **kw)

        monkeypatch.setattr(wm, "remove", flaky)
        result = _clean(service, [bad, good])

        assert result["removed"] == 1
        assert result["failed"] == 1
        rows = {r["name"]: r for r in service._watermark_page({})["files"]}
        assert "full" in rows["bad.pdf"]["reason"].lower()
        assert rows["good.pdf"]["result"] == "lossless"


# ---------------------------------------------------------------------------
# What the page shows
# ---------------------------------------------------------------------------


class TestThePageShowsTheDestination:
    def test_the_destination_is_shown_before_the_run(self, service, deeds):
        """An operator should not have to clean a folder of deeds to discover
        where the copies will land."""
        service.watermark_files.add(deeds)
        model = service._watermark_page({})
        assert model["output_dir"].endswith("Cleaned Watermark")

    def test_both_destinations_reach_the_rendered_page(self, service, deeds):
        service.watermark_files.add(deeds)
        html = service.render_page("watermark", {}, shell_html=False)
        assert "Cleaned Watermark" in html
        assert "Failed" in html
        assert "Input folder" in html

    def test_the_input_folder_is_named_on_the_page(self, service, deeds):
        """Files are picked through a dialog, so nothing else on this screen
        tells an operator which folder the app treats as the input."""
        service.watermark_files.add(deeds)
        model = service._watermark_page({})
        assert model["input_dir"] == str(deeds[0].parent)

    def test_several_source_folders_are_all_listed(self, pymupdf, service,
                                                   tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        service.watermark_files.add([_watermarked(pymupdf, a / "x.pdf"),
                                     _watermarked(pymupdf, b / "y.pdf")])
        model = service._watermark_page({})
        assert model["many_outputs"] is True
        assert len(model["output_dirs"]) == 2

    def test_the_counts_are_shown_after_the_run(self, service, deeds):
        _clean(service, deeds)
        model = service._watermark_page({})
        assert model["has_run"] is True
        assert model["cleaned"] == 3
        assert model["failed"] == 0
        assert model["cleaned"] + model["failed"] == model["done"]

    def test_each_row_names_where_its_copy_went(self, service, deeds):
        _clean(service, deeds)
        rows = service._watermark_page({})["files"]
        for row in rows:
            assert row["output"].endswith(row["name"])
            assert "Cleaned Watermark" in row["output"]

    def test_a_failed_row_carries_a_reason_not_a_bare_failed(self, service,
                                                             deeds, monkeypatch):
        monkeypatch.setattr(wm, "remove", lambda *a, **k: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "No space left on device")))
        _clean(service, deeds[:1])

        row = service._watermark_page({})["files"][0]
        assert row["result"] == "failed"
        assert row["reason"]
        assert row["reason"].lower() not in ("", "failed", "error")

    def test_the_counts_reach_the_rendered_page(self, service, deeds):
        _clean(service, deeds)
        html = service.render_page("watermark", {}, shell_html=False)
        assert "Cleaned" in html
        assert "Failed" in html

    def test_open_output_folder_points_at_the_folder_actually_used(
            self, service, deeds):
        """Not the old shared runtime directory - that is now a fallback for a
        session in which nothing has been cleaned."""
        _clean(service, deeds)
        opened = service.watermark("open")["path"]
        assert opened == str(deeds[0].parent / "Cleaned Watermark")

    def test_open_output_folder_never_leads_to_the_legacy_directory(
            self, service, deeds):
        """The pre-change shared folder still holds copies under the old
        `_clean` names. Opening it to look for today's output is exactly how an
        operator concludes the rename never happened - which is how this was
        reported."""
        from app.services import WATERMARK_DIR

        service.watermark_files.add(deeds)
        assert service._watermark_output_dir() != WATERMARK_DIR

        _clean(service, deeds)
        assert service._watermark_output_dir() != WATERMARK_DIR

    def test_the_button_is_disabled_when_there_is_nothing_to_open(self, service):
        """Rather than enabled and pointing somewhere misleading."""
        assert service._watermark_page({})["has_output"] is False


# ---------------------------------------------------------------------------
# Nothing else changed
# ---------------------------------------------------------------------------


class TestTheRestOfTheFeatureStillWorks:
    def test_scanning_is_unaffected(self, service, deeds):
        service.watermark_files.add(deeds)
        assert service.watermark("scan")["scanned"] == 3
        assert service._watermark_page({})["can_remove"] is True

    def test_a_document_with_no_watermark_produces_no_cleaned_copy(
            self, pymupdf, service, tmp_path):
        """It is not damaged, but it was not cleaned either - so it belongs in
        `Failed`, with a reason that says exactly that rather than implying
        the file is broken."""
        plain = tmp_path / "plain.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "an ordinary deed", fontsize=11)
        doc.save(plain)
        doc.close()

        result = _clean(service, [plain])

        assert result["removed"] == 0
        assert result["failed"] == 1
        assert not (tmp_path / "Cleaned Watermark").exists(), \
            "a cleaned folder was created for a file that was never cleaned"
        assert (tmp_path / "Failed" / "plain.pdf").is_file()
        row = service._watermark_page({})["files"][0]
        assert "No watermark was detected" in row["reason"]
        assert "corrupt" not in row["reason"].lower()

    def test_clearing_resets_the_destinations_too(self, service, deeds):
        _clean(service, deeds)
        service.watermark("clear")
        model = service._watermark_page({})
        assert model["has_files"] is False
        assert model["output_dirs"] == []

    def test_a_single_file_works_as_well_as_a_batch(self, service, deeds):
        result = _clean(service, deeds[:1])
        assert result["removed"] == 1
        assert (deeds[0].parent / "Cleaned Watermark" / deeds[0].name).is_file()

    def test_an_unknown_action_is_still_rejected(self, service):
        with pytest.raises(ValueError, match="unknown watermark action"):
            service.watermark("obliterate")

    def test_the_ocr_page_is_untouched_by_this(self, service):
        """The two tool pages share a shape but not a selection."""
        assert service.ocr_files.paths == []
        assert "OCR Text Extraction" in service.render_page(
            "ocr", {}, shell_html=False)


# ---------------------------------------------------------------------------
# The Failed folder
# ---------------------------------------------------------------------------


class TestFailedDeedsAreFiled:
    """Every input ends in exactly one output folder. That completeness is what
    lets an operator account for 200 deeds without a screen open."""

    @pytest.fixture()
    def mixed(self, pymupdf, tmp_path):
        """One deed that cleans, one with no watermark, one unreadable."""
        folder = tmp_path / "deeds"
        folder.mkdir()
        good = _watermarked(pymupdf, folder / "good.pdf")

        plain = folder / "plain.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "an ordinary deed", fontsize=11)
        doc.save(plain)
        doc.close()

        broken = folder / "broken.pdf"
        broken.write_bytes(good.read_bytes()[:400])
        return folder, good, plain, broken

    def test_every_input_lands_in_exactly_one_folder(self, service, mixed):
        folder, good, plain, broken = mixed
        result = _clean(service, [good, plain, broken])

        cleaned = {p.name for p in (folder / "Cleaned Watermark").glob("*.pdf")}
        failed = {p.name for p in (folder / "Failed").glob("*.pdf")}

        assert cleaned == {"good.pdf"}
        assert failed == {"plain.pdf", "broken.pdf"}
        assert not (cleaned & failed), "a deed was filed in both folders"
        assert result["removed"] == 1
        assert result["failed"] == 2

    def test_the_folder_is_named_exactly_Failed(self, service, mixed):
        folder, _good, plain, _broken = mixed
        _clean(service, [plain])
        assert [d.name for d in folder.iterdir() if d.is_dir()] == ["Failed"]

    def test_failed_deeds_keep_their_filename(self, service, mixed):
        folder, _good, plain, broken = mixed
        _clean(service, [plain, broken])
        for source in (plain, broken):
            assert (folder / "Failed" / source.name).is_file()

    def test_a_failed_deed_is_copied_not_moved(self, service, mixed):
        """Moving would take the deed out of the folder the operator is working
        through - a change to their filing, made by a tool asked only to clean
        a copy."""
        folder, _good, plain, _broken = mixed
        before = plain.read_bytes()

        _clean(service, [plain])

        assert plain.is_file(), "the original was moved out of the input folder"
        assert plain.read_bytes() == before
        assert (folder / "Failed" / "plain.pdf").read_bytes() == before

    def test_an_existing_failed_folder_is_reused(self, service, mixed):
        folder, _good, plain, _broken = mixed
        failed = folder / "Failed"
        failed.mkdir()
        (failed / "from-before.pdf").write_bytes(b"%PDF-1.4 earlier")

        _clean(service, [plain])

        assert [d.name for d in folder.iterdir() if d.is_dir()] == ["Failed"]
        assert (failed / "from-before.pdf").is_file()

    def test_no_failed_folder_when_nothing_fails(self, service, deeds):
        _clean(service, deeds)
        assert not (deeds[0].parent / "Failed").exists()

    def test_the_reason_is_recorded_beside_the_deeds(self, service, mixed):
        """A screen closes with the application; someone handed this folder a
        week later still needs to know why each file is in it."""
        folder, _good, plain, broken = mixed
        _clean(service, [plain, broken])

        note = folder / "Failed" / "why-these-failed.txt"
        assert note.is_file()
        text = note.read_text(encoding="utf-8")
        assert "plain.pdf" in text and "broken.pdf" in text
        assert "No watermark was detected" in text
        assert "untouched" in text, "the note should say these are copies"

    def test_each_reason_is_specific_to_its_deed(self, service, mixed):
        _folder, _good, plain, broken = mixed
        _clean(service, [plain, broken])

        rows = {r["name"]: r["reason"] for r in service._watermark_page({})["files"]}
        assert "No watermark was detected" in rows["plain.pdf"]
        assert rows["broken.pdf"] != rows["plain.pdf"], \
            "an unreadable file and an unwatermarked one got the same reason"
        assert rows["broken.pdf"]

    def test_no_reason_is_a_bare_failure(self, service, mixed):
        _folder, _good, plain, broken = mixed
        _clean(service, [plain, broken])
        for row in service._watermark_page({})["files"]:
            if row["result"] == "failed":
                assert len(row["reason"]) > 15, row
                assert row["reason"].lower() not in ("failed", "error")

    def test_the_row_says_where_the_failed_copy_went(self, service, mixed):
        folder, _good, plain, _broken = mixed
        _clean(service, [plain])
        row = service._watermark_page({})["files"][0]
        assert row["output"] == str(folder / "Failed" / "plain.pdf")

    def test_each_source_folder_gets_its_own_failed_folder(self, pymupdf,
                                                           service, tmp_path):
        a, b = tmp_path / "north", tmp_path / "south"
        a.mkdir()
        b.mkdir()
        for folder in (a, b):
            doc = pymupdf.open()
            doc.new_page().insert_text((72, 100), "plain deed", fontsize=11)
            doc.save(folder / "deed.pdf")
            doc.close()

        _clean(service, [a / "deed.pdf", b / "deed.pdf"])

        assert (a / "Failed" / "deed.pdf").is_file()
        assert (b / "Failed" / "deed.pdf").is_file()

    def test_a_deed_in_the_failed_folder_is_not_nested_deeper(self, service,
                                                              mixed):
        """Re-selecting from `Failed` must not build Failed/Failed/Failed."""
        folder, _good, plain, _broken = mixed
        _clean(service, [plain])
        stray = folder / "Failed" / "plain.pdf"

        assert service._failed_dir(stray) == folder / "Failed"

    def test_the_counts_match_the_folders(self, service, mixed):
        folder, good, plain, broken = mixed
        result = _clean(service, [good, plain, broken])
        model = service._watermark_page({})

        assert model["cleaned"] == result["removed"] == len(
            list((folder / "Cleaned Watermark").glob("*.pdf")))
        assert model["failed"] == result["failed"] == len(
            list((folder / "Failed").glob("*.pdf")))
        assert model["cleaned"] + model["failed"] == 3

    def test_both_folders_are_reported_to_the_caller(self, service, mixed):
        _folder, good, plain, _broken = mixed
        result = _clean(service, [good, plain])
        assert result["output_dir"].endswith("Cleaned Watermark")
        assert result["failed_dir"].endswith("Failed")

    def test_opening_the_failed_folder_is_refused_when_there_is_none(
            self, service, deeds):
        _clean(service, deeds)
        with pytest.raises(RepositoryError, match="no .*Failed.* folder"):
            service.watermark("open_failed")

    def test_opening_the_failed_folder_points_at_it(self, service, mixed):
        folder, _good, plain, _broken = mixed
        _clean(service, [plain])
        assert service.watermark("open_failed")["path"] == str(folder / "Failed")

    def test_a_single_file_works_the_same_as_a_batch(self, service, mixed):
        folder, _good, plain, _broken = mixed
        result = _clean(service, [plain])
        assert result["failed"] == 1
        assert (folder / "Failed" / "plain.pdf").is_file()

    def test_an_unscanned_file_is_told_to_scan_first(self, service, mixed):
        """Pressing Remove without Detect used to skip the file in silence."""
        _folder, good, _plain, _broken = mixed
        service.watermark_files.add([good])
        result = service.watermark("remove")

        assert result["failed"] == 1
        row = service._watermark_page({})["files"][0]
        assert "Detect Watermarks" in row["reason"]


class TestReasonsAreOperatorFacing:
    """PyMuPDF's text is not a reason.

    The live run first reported "FileDataError: Failed to open file
    'C:/Users/.../RMN-1-00155-2024-25.pdf' as type pdf" and "ValueError:
    document closed or encrypted" - an exception class and an absolute path,
    which name the component that gave up rather than the problem to fix. Both
    are conditions this project already has words for, so they are classified
    through the same table the pipeline uses.
    """

    @pytest.fixture()
    def broken(self, pymupdf, tmp_path):
        folder = tmp_path / "deeds"
        folder.mkdir()
        good = _watermarked(pymupdf, folder / "good.pdf")
        # Deliberately *not* named "truncated": the classifier reads the error
        # text, which contains the path, and a filename carrying the answer
        # would let this pass while the code matched the wrong thing.
        truncated = folder / "RMN-1-00155.pdf"
        truncated.write_bytes(good.read_bytes()[:400])

        locked = folder / "locked.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "secret", fontsize=11)
        doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                 owner_pw="o", user_pw="u")
        doc.close()
        return folder, truncated, locked

    def _reasons(self, service, paths):
        _clean(service, paths)
        return {r["name"]: r for r in service._watermark_page({})["files"]}

    def test_a_corrupt_file_is_described_as_corrupt(self, service, broken):
        _folder, truncated, _locked = broken
        row = self._reasons(service, [truncated])[truncated.name]
        assert row["reason"] == "PDF file is corrupted or cannot be read."

    def test_a_locked_file_is_described_as_protected(self, service, broken):
        """Distinct from corrupt: the file is fine and the operator needs a
        password. Telling them it is corrupt sends them to delete it."""
        _folder, _truncated, locked = broken
        row = self._reasons(service, [locked])[locked.name]
        if not row["reason"]:
            pytest.skip("this PyMuPDF opened the encrypted file without a password")
        assert row["reason"] == "PDF is password protected and cannot be opened."

    def test_no_reason_names_a_python_exception(self, service, broken):
        _folder, truncated, locked = broken
        for row in self._reasons(service, [truncated, locked]).values():
            for leak in ("Error:", "Exception", "Traceback", "ValueError",
                         "FileDataError"):
                assert leak not in row["reason"], row["reason"]

    def test_no_reason_leaks_a_filesystem_path(self, service, broken):
        """The full path of a file the operator selected tells them nothing and
        is most of the line."""
        _folder, truncated, locked = broken
        for row in self._reasons(service, [truncated, locked]).values():
            assert ":\\" not in row["reason"], row["reason"]
            assert "/" not in row["reason"], row["reason"]

    def test_the_library_text_is_kept_behind_a_disclosure(self, service, broken):
        """Not discarded - it is what a support question needs - just not shown
        in front of a clerk."""
        _folder, truncated, _locked = broken
        row = self._reasons(service, [truncated])[truncated.name]
        assert row["technical"], "the original error was thrown away"
        assert "FileDataError" in row["technical"]

    def test_the_note_on_disk_carries_the_readable_reason(self, service, broken):
        folder, truncated, locked = broken
        _clean(service, [truncated, locked])
        text = (folder / "Failed" / "why-these-failed.txt").read_text(
            encoding="utf-8")
        assert "corrupted" in text
        assert "FileDataError" not in text

    def test_an_unrecognised_error_is_still_shown_sanitised(self, service,
                                                           broken, monkeypatch):
        """Better than a shrug, but bounded and traceback-free."""
        folder, truncated, _locked = broken
        monkeypatch.setattr(wm, "remove", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("something wholly unexpected\nTraceback (most recent "
                         "call last):\n  File \"x.py\"")))
        good = _watermarked(__import__("pymupdf"), folder / "another.pdf")
        row = self._reasons(service, [good])[good.name]
        assert row["reason"]
        assert "Traceback" not in row["reason"]
