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

    def test_the_destination_reaches_the_rendered_page(self, service, deeds):
        service.watermark_files.add(deeds)
        html = service.render_page("watermark", {}, shell_html=False)
        assert "Cleaned Watermark" in html
        assert "will be saved to" in html

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

    def test_a_document_with_no_watermark_is_skipped_not_failed(
            self, pymupdf, service, tmp_path):
        plain = tmp_path / "plain.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "an ordinary deed", fontsize=11)
        doc.save(plain)
        doc.close()

        result = _clean(service, [plain])

        assert result["removed"] == 0
        assert not (tmp_path / "Cleaned Watermark").exists(), \
            "a folder was created for a file that was never cleaned"

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
