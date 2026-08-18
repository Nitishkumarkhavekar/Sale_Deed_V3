"""Transaction Identity: on every row of a deed, and never on the wrong one.

Reported as blank in some Excel rows. There are two separate questions here and
they have different answers:

**Within a deed it cannot be partially blank.** The value goes into `base`,
which `build_rows` applies to every person row, so all rows of a deed carry the
same value or none do. The tests below pin that, because it is the shape the
report described.

**A whole deed could come out blank**, and did. `extract()` is handed the file
name as `source=` and used it only for logging - so a deed whose OCR text
yielded no candidate was left blank even when the file was named after its own
registration number, `RMN-1-02264-2024-25.pdf`. That was the root cause.

The fix is deliberately narrow. R-043 was this same fallback without a check: a
deed whose number could not be read exported "275", its file stem, as the
Transaction Identity. The file name is used **only** when it is itself a valid
registration number, checked by the same pattern a text candidate must pass.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.csv_export import DocumentExport, build_rows, write_csv
from core.transaction_id import CANONICAL, extract, from_source_name

# No `logging.disable` here. It is module-level global state: pytest imports
# every test file before running anything, so disabling logging in one of them
# silences the whole session and fails any test that asserts on log output -
# which is exactly what it did to `TestTerminalLogging`. The identity extractor
# logs at INFO and WARNING; pytest captures that and shows it only on failure.

NO_NUMBER = ("This deed of absolute sale is executed between the parties "
             "described below. The schedule property is described hereunder.")


def _person(tag: str) -> dict:
    digits = "".join(c for c in tag if c.isdigit()).rjust(4, "0")
    return {"name": tag,
            "pan_card_number": f"ABCP{digits[0]}{digits[1:]}F",
            "aadhaar_number": f"66{digits}2345678"[:12]}


def _deed(identity: str, sellers: int, buyers: int) -> DocumentExport:
    return DocumentExport(
        transaction_identity=identity,
        extraction={
            "seller_details": [_person(f"SELLER {n}") for n in range(1, sellers + 1)],
            "buyer_details": [_person(f"BUYER {n}") for n in range(1, buyers + 1)],
            "property_details": {"sale_consideration": "10000"},
            "document_details": {}})


class TestEveryRowOfADeedCarriesIt:
    @pytest.mark.parametrize("sellers,buyers", [
        (1, 1), (3, 1), (1, 4), (2, 2), (5, 5), (1, 0), (0, 3),
    ])
    def test_no_row_is_left_blank(self, sellers, buyers):
        rows = build_rows([_deed("YPR-1-00001-2024-25", sellers, buyers)])
        assert rows, "the deed produced no rows at all"
        assert all(r["Transaction Identity"] == "YPR-1-00001-2024-25"
                   for r in rows), [r["Transaction Identity"] for r in rows]

    def test_sellers_and_buyers_get_the_same_value(self):
        rows = build_rows([_deed("YPR-1-00001-2024-25", 2, 2)])
        by_side = {r["Transaction Relation (PC)"]: r["Transaction Identity"]
                   for r in rows}
        assert by_side["S"] == by_side["B"] == "YPR-1-00001-2024-25"

    def test_a_document_with_no_parties_still_carries_it(self):
        """One row is emitted so the deed stays visible in the export; it must
        be identified like any other."""
        rows = build_rows([_deed("YPR-1-00009-2024-25", 0, 0)])
        assert len(rows) == 1
        assert rows[0]["Transaction Identity"] == "YPR-1-00009-2024-25"

    def test_a_dropped_duplicate_does_not_blank_the_survivors(self):
        """Duplicate removal changes which rows exist, never what they say
        about their deed."""
        twin = _person("BUYER 1")
        doc = DocumentExport(
            transaction_identity="YPR-1-00010-2024-25",
            extraction={"seller_details": [_person("SELLER 1")],
                        "buyer_details": [twin, dict(twin)],
                        "property_details": {}, "document_details": {}})
        rows = build_rows([doc])
        assert len(rows) == 2
        assert all(r["Transaction Identity"] == "YPR-1-00010-2024-25"
                   for r in rows)


class TestDeedsNeverBorrowEachOtherIdentity:
    def test_four_deeds_in_one_batch_stay_separate(self):
        shapes = [("YPR-1-00001-2024-25", 1, 1), ("YPR-1-00002-2024-25", 3, 1),
                  ("YPR-1-00003-2024-25", 1, 4), ("YPR-1-00004-2024-25", 2, 2)]
        rows = build_rows([_deed(i, s, b) for i, s, b in shapes])

        by_serial: dict[str, set[str]] = {}
        for row in rows:
            by_serial.setdefault(row["Report Serial Number"], set()).add(
                row["Transaction Identity"])
        assert all(len(v) == 1 for v in by_serial.values()), by_serial
        assert len(by_serial) == len(shapes)

    def test_a_blank_deed_does_not_take_a_neighbour_value(self):
        """The dangerous repair: filling a blank from the row above would
        attribute one deed's parties to a different deed. Blank is correct."""
        rows = build_rows([_deed("YPR-1-00001-2024-25", 2, 0),
                           _deed("", 2, 0),
                           _deed("YPR-1-00003-2024-25", 2, 0)])
        blanks = [r for r in rows if not r["Transaction Identity"]]
        assert len(blanks) == 2, "the middle deed's rows should stay blank"
        assert {r["Transaction Identity"] for r in rows} == {
            "YPR-1-00001-2024-25", "", "YPR-1-00003-2024-25"}


class TestTheFileNameFallback:
    """The root cause, and the narrow fix for it."""

    def test_a_canonical_file_name_supplies_a_missing_identity(self):
        result = extract(NO_NUMBER, source="RMN-1-02264-2024-25.pdf",
                         ocr_used=True)
        assert result.found
        assert result.value == "RMN-1-02264-2024-25"
        assert "file name" in result.reason

    @pytest.mark.parametrize("name", [
        "275.pdf", "deed.pdf", "2025-26-1457.pdf", "scan001.pdf",
        "RMN-1-02264.pdf", "final copy.pdf",
    ])
    def test_a_non_canonical_file_name_supplies_nothing(self, name):
        """R-043: a deed whose number could not be read exported "275" - its
        file stem - as the Transaction Identity. A blank is correct here."""
        assert not extract(NO_NUMBER, source=name, ocr_used=True).found

    def test_the_fallback_is_less_confident_than_a_text_match(self):
        """The name is evidence about the file, not about what the deed says."""
        from_name = extract(NO_NUMBER, source="RMN-1-02264-2024-25.pdf")
        from_text = extract("Document No. RMN-1-02264-2024-25 registered on ...",
                            source="RMN-1-02264-2024-25.pdf")
        assert from_name.confidence < from_text.confidence

    def test_the_text_still_wins_when_it_disagrees(self):
        """The deed's own contents outrank what someone called the file."""
        result = extract("Document No. YPR-1-00777-2024-25 registered at ...",
                         source="RMN-1-02264-2024-25.pdf")
        assert result.value == "YPR-1-00777-2024-25"

    def test_from_source_name_validates_with_the_canonical_pattern(self):
        """The same check a text candidate has to pass - not a looser one."""
        assert from_source_name("YPR-1-00001-2024-25.pdf") == "YPR-1-00001-2024-25"
        assert from_source_name("275.pdf") == ""
        assert from_source_name("") == ""
        for name in ("YPR-1-00001-2024-25", "ABC-9-99999-1999-00"):
            assert CANONICAL.match(from_source_name(f"{name}.pdf"))

    def test_a_full_path_is_handled(self):
        assert from_source_name("D:/deeds/RMN-1-02264-2024-25.pdf") \
            == "RMN-1-02264-2024-25"

    def test_nothing_anywhere_still_yields_a_blank(self):
        """The rule that keeps this safe: no evidence means blank, never a
        guess. A wrong registration number attributes a transaction to the
        wrong deed."""
        assert not extract(NO_NUMBER, source="", ocr_used=True).found


class TestTheWrittenFile:
    def test_every_row_in_the_csv_has_its_own_deed_identity(self, tmp_path):
        shapes = [("YPR-1-00001-2024-25", 2, 2), ("YPR-1-00002-2024-25", 3, 1)]
        target = tmp_path / "identities.csv"
        write_csv(target, [_deed(i, s, b) for i, s, b in shapes])

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 4 + 4
        assert not [r for r in rows if not r["Transaction Identity"]]
        assert {r["Transaction Identity"] for r in rows} == {i for i, _, _ in shapes}


class TestTheOfficeCodeIsNotHardCoded:
    """The pattern accepts any 2-4 character office code.

    Raised as a concern that only YPR, BGP and MDG were supported. They are
    not special anywhere - `PATTERN` matches `([A-Z0-9]{2,4})` and `CANONICAL`
    validates `[A-Z]{2,4}`. These tests exist so that stays true.
    """

    CERTIFICATE = (
        "Kaveri Online Services\n"
        "1 \u0ca8\u0cc7 \u0caa\u0cc1\u0cb8\u0ccd\u0ca4\u0c95\u0ca6 "
        "\u0ca6\u0cb8\u0ccd\u0ca4\u0cbe\u0cb5\u0cc7\u0c9c\u0cc1\n"
        "\u0ca8\u0c82\u0cac\u0cb0 {ident} \u0c86\u0c97\u0cbf\n"
        "\u0ca6\u0cbf\u0ca8\u0cbe\u0c82\u0c95 19/04/2025 \u0cb0\u0c82\u0ca6\u0cc1 "
        "\u0ca8\u0ccb\u0c82\u0ca6\u0cbe\u0caf\u0cbf\u0cb8\u0cbf\n")

    @pytest.mark.parametrize("identity", [
        "RMN-1-02264-2024-25", "YPR-1-00001-2024-25",
        "BGP-1-00275-2025-26", "MDG-1-00146-2025-26",
        "BES-1-01151-2023-24", "SRJ-1-00003-2025-26",
        "AB-1-00001-2024-25", "ABCD-1-00001-2024-25",
    ])
    def test_any_office_code_is_read(self, identity):
        """Including two- and four-letter codes, which none of the reported
        examples covers."""
        result = extract(self.CERTIFICATE.format(ident=identity),
                         source="scan.pdf", ocr_used=True)
        assert result.value == identity
        assert result.confidence >= 0.7

    @pytest.mark.parametrize("raw,expected", [
        ("RMN - 1 - 02264 - 2024 - 25", "RMN-1-02264-2024-25"),
        ("RMN\u20131\u201302264\u20132024\u201325", "RMN-1-02264-2024-25"),
        ("RMN.1.02264.2024.25", "RMN-1-02264-2024-25"),
        ("RMN-1-2264-2024-25", "RMN-1-02264-2024-25"),
        ("RMN-1-002264-2024-25", "RMN-1-02264-2024-25"),
    ])
    def test_ocr_damage_is_normalised(self, raw, expected):
        """Spacing, dash shape and a serial one digit short or long. The scan
        is of a printed certificate, so these are the failures that occur."""
        result = extract(self.CERTIFICATE.format(ident=raw),
                         source="scan.pdf", ocr_used=True)
        assert result.value == expected

    def test_a_digit_misread_in_the_office_code_is_repaired(self):
        """`0` for `O`, `1` for `I`, `5` for `S`, `8` for `B` - inside the
        office code only, where a digit cannot legitimately appear."""
        result = extract(self.CERTIFICATE.format(ident="RM0-1-02264-2024-25"),
                         source="scan.pdf", ocr_used=True)
        assert result.value == "RMO-1-02264-2024-25"
        assert CANONICAL.match(result.value)

    def test_the_certificate_wording_is_recognised_as_a_label(self):
        """The block in the scanned certificate - book number, document number,
        registered on - is what marks this as the deed's own number rather than
        a citation."""
        result = extract(self.CERTIFICATE.format(ident="BGP-1-00275-2025-26"),
                         source="scan.pdf", ocr_used=True)
        assert "registration label" in result.reason or result.confidence >= 0.9

    def test_a_prior_deed_in_the_same_text_does_not_win(self):
        """A deed recites its chain of title. The cited number is older and is
        introduced as a prior document; both count against it."""
        text = (self.CERTIFICATE.format(ident="RMN-1-02264-2024-25")
                + "\nThe vendor acquired the property vide document No. "
                  "RMN-1-00111-2018-19 registered earlier.\n")
        assert extract(text, source="scan.pdf").value == "RMN-1-02264-2024-25"


@pytest.mark.integration
class TestADeedWithNoIdentityGoesToReview:
    """Requirement: flag it rather than insert an incorrect value.

    The extractor already refuses to choose between two candidates, because a
    previous owner's document number is worse than a blank. That refusal used to
    be silent - the deed exported clean with an empty column. It now routes the
    document to review, so the refusal is visible before the file is filed.
    """

    def test_the_runner_checks_the_identity_before_finishing(self):
        """Asserted on the source: exercising it needs OCR, a GPU and the AI
        server, and the defect was precisely that this check was absent."""
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._process_one)
        assert "_has_identity" in body
        assert body.index("_has_identity") < body.index("DocumentState.PROCESSED")

    def test_the_review_reason_names_the_problem(self):
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._process_one)
        assert "registration number could not be read" in body

    def test_has_identity_reads_the_stored_value(self, app_service, temp_batch,
                                                 session_factory):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        with session_scope(session_factory) as session:
            doc = UnitOfWork(session).documents.list_for_batch(
                temp_batch, per_page=1)[0][0]
            doc_pk = doc.id
            doc.transaction_identity = None

        assert app_service.runner._has_identity(doc_pk) is False

        with session_scope(session_factory) as session:
            UnitOfWork(session).documents.get(doc_pk).transaction_identity = \
                "RMN-1-02264-2024-25"

        assert app_service.runner._has_identity(doc_pk) is True

    def test_the_flag_code_exists_for_the_condition(self):
        from core.validation import Flag

        assert Flag.NO_TXN_IDENTITY.value == "WTI"


# ---------------------------------------------------------------------------
# Meaningless file names, and reading the number out of the deed instead
# ---------------------------------------------------------------------------


#: Names real scans arrive with. None is a registration number, and two of them
#: are close enough to look like one at a glance: `8369-2024-25` carries a
#: serial and a financial year but no office code or book number, and
#: `2025-26-1457` carries the year first. Both would pass a looser check.
MEANINGLESS_NAMES = (
    "275.pdf",
    "deed.pdf",
    "2025-26-1457.pdf",
    "8369-2024-25.pdf",
    "6542-24-25.pdf",
    "1367.pdf",
    "scan0001.pdf",
    "Document (3).pdf",
    "BGP-1-00275.pdf",          # office and serial, no financial year
    "BGP-00275-2025-26.pdf",    # no book number
)

#: The certificate Kaveri stamps on the scan, as OCR returns it.
CERTIFICATE = (
    "===== PAGE 1 =====\n"
    "Kaveri Online Services\n"
    "1 \u0ca8\u0cc7 \u0caa\u0cc1\u0cb8\u0ccd\u0ca4\u0c95\u0ca6 "
    "\u0ca6\u0cb8\u0ccd\u0ca4\u0cbe\u0cb5\u0cc7\u0c9c\u0cc1\n"
    "\u0ca8\u0c82\u0cac\u0cb0 {value} \u0c86\u0c97\u0cbf\n"
    "\u0ca6\u0cbf\u0ca8\u0cbe\u0c82\u0c95 19/04/2025 "
    "\u0cb0\u0c82\u0ca6\u0cc1 "
    "\u0ca8\u0ccb\u0c82\u0ca6\u0cbe\u0caf\u0cbf\u0cb8\u0cbf\n"
    "\u0c89\u0caa \u0ca8\u0ccb\u0c82\u0ca6\u0ca3\u0cbe"
    "\u0ca7\u0cbf\u0c95\u0cbe\u0cb0\u0cbf\n"
)


class TestAMeaninglessFileNameIsNeverTheIdentity:
    """R-043 in its original form: `275.pdf` exported "275"."""

    @pytest.mark.parametrize("name", MEANINGLESS_NAMES)
    def test_the_name_alone_yields_nothing(self, name):
        assert from_source_name(name) == ""

    @pytest.mark.parametrize("name", MEANINGLESS_NAMES)
    def test_a_deed_with_no_number_stays_blank(self, name):
        """Not "275", not "deed" - blank, so the gap is visible."""
        assert extract(NO_NUMBER, source=name).value == ""

    @pytest.mark.parametrize("name", MEANINGLESS_NAMES)
    def test_the_deed_text_wins_over_the_name(self, name):
        """The point of the whole exercise: open the file, read the number."""
        result = extract(CERTIFICATE.format(value="BGP-1-00275-2025-26"),
                         source=name)
        assert result.value == "BGP-1-00275-2025-26"
        assert result.confidence >= 0.9

    def test_a_numeric_name_does_not_become_a_candidate(self):
        """`8369-2024-25.pdf` must not be read as a serial and a year."""
        result = extract(CERTIFICATE.format(value="RMN-1-02264-2024-25"),
                         source="8369-2024-25.pdf")
        assert result.value == "RMN-1-02264-2024-25"

    @pytest.mark.parametrize("name", MEANINGLESS_NAMES)
    def test_every_row_of_such_a_deed_carries_the_read_value(self, name):
        result = extract(CERTIFICATE.format(value="MDG-1-00146-2025-26"),
                         source=name)
        doc = DocumentExport(
            transaction_identity=result.value, source_filename=name,
            extraction={"seller_details": [{"name": "Ramesh"}, {"name": "Sita"}],
                       "buyer_details": [{"name": "Anil"}, {"name": "Kavya"},
                                         {"name": "Deepak"}]})
        rows = build_rows([doc])
        assert len(rows) == 5
        assert {r["Transaction Identity"] for r in rows} == {"MDG-1-00146-2025-26"}


class TestTheExtractionPriority:
    """Embedded text, then OCR, then - only as a reference - the file name."""

    def test_text_outranks_a_correct_file_name(self):
        """Both agree here; what is asserted is which one was consulted."""
        result = extract(CERTIFICATE.format(value="YPR-1-00001-2024-25"),
                         source="YPR-1-00001-2024-25.pdf")
        assert result.value == "YPR-1-00001-2024-25"
        assert "file name" not in result.reason

    def test_the_name_is_used_only_when_the_text_has_nothing(self):
        result = extract(NO_NUMBER, source="RMN-1-02264-2024-25.pdf")
        assert result.value == "RMN-1-02264-2024-25"
        assert result.confidence < 0.95, "weaker than anything read off the deed"

    def test_a_conflicting_name_does_not_override_the_deed(self):
        """The scan was misfiled under another deed's number. The deed wins."""
        result = extract(CERTIFICATE.format(value="BGP-1-00275-2025-26"),
                         source="RMN-1-02264-2024-25.pdf")
        assert result.value == "BGP-1-00275-2025-26"


class TestReReadingTheScannedCertificate:
    """A digital deed whose certificate is a pasted image.

    `_run_textlayer` returns nothing for a picture, so the one field that
    identifies the deed is the one field PyMuPDF cannot see. The runner reads
    the pages the certificate lands on back through the OCR engine.
    """

    def test_it_reads_the_front_and_the_back_only(self):
        from core.transaction_id import identity_pages

        assert identity_pages(20) == [1, 2, 19, 20]

    def test_a_short_deed_is_read_whole(self):
        from core.transaction_id import identity_pages

        assert identity_pages(3) == [1, 2, 3]
        assert identity_pages(1) == [1]

    def test_an_unknown_page_count_asks_for_nothing(self):
        from core.transaction_id import identity_pages

        assert identity_pages(0) == []

    def test_a_build_with_no_ocr_engine_returns_empty_not_an_error(self, tmp_path):
        """The recovery is an optimisation. It must never fail a document."""
        from core.pipeline.stages import OcrStage

        pdf = tmp_path / "deed.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        assert OcrStage(engine="textlayer").ocr_pages(pdf, [1, 2]) == ""

    def test_a_missing_file_returns_empty(self, tmp_path):
        from core.pipeline.stages import OcrStage

        assert OcrStage(engine="textlayer").ocr_pages(tmp_path / "gone.pdf", [1]) == ""

    def test_the_runner_retries_through_ocr_when_the_text_layer_had_nothing(self):
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._do_ocr)
        assert "_identity_from_ocr" in body
        assert "not identity.found and not ocr_used" in body

    def test_the_retry_keeps_the_original_result_when_it_finds_nothing(self):
        """A deed that genuinely has no number behaves exactly as before."""
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._identity_from_ocr)
        assert body.count("return current") == 3

    def test_the_recovered_value_is_validated_like_any_other(self):
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._identity_from_ocr)
        assert "extract_transaction_id(text" in body


class TestExtrasCannotBlankIt:
    """`extras` is a caller override. It does not reach this column."""

    def test_an_override_is_ignored(self):
        doc = DocumentExport(
            transaction_identity="BGP-1-00275-2025-26",
            source_filename="275.pdf",
            extraction={"seller_details": [{"name": "Ramesh"}],
                       "buyer_details": [{"name": "Anil"}]},
            extras={"Transaction Identity": ""})
        rows = build_rows([doc])
        assert {r["Transaction Identity"] for r in rows} == {"BGP-1-00275-2025-26"}


class TestABatchOfMixedDeeds:
    """Different offices, different years, different file-name quality."""

    def test_each_deed_keeps_its_own(self, tmp_path):
        batch = [
            ("275.pdf", "BGP-1-00275-2025-26", 2, 3),
            ("RMN-1-02264-2024-25.pdf", "RMN-1-02264-2024-25", 1, 1),
            ("deed.pdf", "YPR-1-00001-2024-25", 3, 2),
            ("2025-26-1457.pdf", "MDG-1-00146-2025-26", 1, 4),
        ]
        documents = []
        for name, value, sellers, buyers in batch:
            result = extract(CERTIFICATE.format(value=value), source=name)
            assert result.value == value, name
            documents.append(DocumentExport(
                transaction_identity=result.value, source_filename=name,
                extraction={
                    "seller_details": [{"name": f"S{i}{value}"}
                                       for i in range(sellers)],
                    "buyer_details": [{"name": f"B{i}{value}"}
                                      for i in range(buyers)]}))

        target = tmp_path / "batch.csv"
        write_csv(target, documents)
        with target.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == sum(s + b for _, _, s, b in batch)
        for _, value, sellers, buyers in batch:
            mine = [r for r in rows if value in r["Person Name (PC)"]]
            assert len(mine) == sellers + buyers, value
            assert {r["Transaction Identity"] for r in mine} == {value}
        assert "" not in {r["Transaction Identity"] for r in rows}


class TestWhichPagesAreActuallyRendered:
    """`ocr_pages` renders and hands off. Surya is stubbed; the rendering is real.

    Worth testing directly: the recovery is worthless if it rasterises the wrong
    pages, and worse than worthless if it rasterises all twenty of them.
    """

    @staticmethod
    def _pdf(tmp_path, pages):
        import pymupdf

        doc = pymupdf.open()
        for number in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {number + 1}")
        target = tmp_path / "deed.pdf"
        doc.save(target)
        doc.close()
        return target

    @staticmethod
    def _stage(tmp_path):
        from core.pipeline.stages import OcrStage

        # Any existing files will do - the subprocess is stubbed out below, so
        # these are only checked for existence.
        fake = tmp_path / "surya.py"
        fake.write_text("", encoding="utf-8")
        return OcrStage(engine="surya", surya_python=fake, surya_script=fake)

    def test_it_renders_exactly_the_pages_asked_for(self, tmp_path, monkeypatch):
        import subprocess

        from core.pipeline import stages as stages_module

        pdf = self._pdf(tmp_path, 20)
        stage = self._stage(tmp_path)
        seen = {}

        def fake_run(cmd, **kwargs):
            directory = Path(cmd[cmd.index("--images") + 1])
            seen["names"] = sorted(p.name for p in directory.glob("*.png"))
            Path(cmd[cmd.index("--out") + 1]).write_text(
                '{"text": "recovered"}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(stages_module.subprocess, "run", fake_run)

        text = stage.ocr_pages(pdf, [1, 2, 19, 20])
        assert text == "recovered"
        assert seen["names"] == ["0001.png", "0002.png", "0019.png", "0020.png"]

    def test_it_does_not_rasterise_the_whole_deed(self, tmp_path, monkeypatch):
        """Four pages, not twenty. This is the entire cost argument."""
        import subprocess

        from core.pipeline import stages as stages_module
        from core.transaction_id import identity_pages

        pdf = self._pdf(tmp_path, 20)
        stage = self._stage(tmp_path)
        counted = {}

        def fake_run(cmd, **kwargs):
            directory = Path(cmd[cmd.index("--images") + 1])
            counted["n"] = len(list(directory.glob("*.png")))
            Path(cmd[cmd.index("--out") + 1]).write_text("plain text",
                                                         encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(stages_module.subprocess, "run", fake_run)

        stage.ocr_pages(pdf, identity_pages(20))
        assert counted["n"] == 4

    def test_a_page_number_past_the_end_is_skipped(self, tmp_path, monkeypatch):
        import subprocess

        from core.pipeline import stages as stages_module

        pdf = self._pdf(tmp_path, 3)
        stage = self._stage(tmp_path)
        seen = {}

        def fake_run(cmd, **kwargs):
            directory = Path(cmd[cmd.index("--images") + 1])
            seen["names"] = sorted(p.name for p in directory.glob("*.png"))
            Path(cmd[cmd.index("--out") + 1]).write_text("x", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(stages_module.subprocess, "run", fake_run)

        stage.ocr_pages(pdf, [1, 2, 99])
        assert seen["names"] == ["0001.png", "0002.png"]

    def test_a_crashing_ocr_engine_returns_empty(self, tmp_path, monkeypatch):
        import subprocess

        from core.pipeline import stages as stages_module

        pdf = self._pdf(tmp_path, 5)
        stage = self._stage(tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 6, "", "OCR failed: OOM")

        monkeypatch.setattr(stages_module.subprocess, "run", fake_run)
        assert stage.ocr_pages(pdf, [1]) == ""

    def test_the_recovered_text_yields_the_identity(self, tmp_path, monkeypatch):
        """The whole point, end to end: image-only certificate -> Excel value."""
        import subprocess

        from core.pipeline import stages as stages_module

        pdf = self._pdf(tmp_path, 12)
        stage = self._stage(tmp_path)
        certificate = CERTIFICATE.format(value="BGP-1-00275-2025-26")

        def fake_run(cmd, **kwargs):
            Path(cmd[cmd.index("--out") + 1]).write_text(
                json.dumps({"text": certificate}), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(stages_module.subprocess, "run", fake_run)

        from core.transaction_id import identity_pages

        text = stage.ocr_pages(pdf, identity_pages(12))
        assert extract(text, source="275.pdf").value == "BGP-1-00275-2025-26"
