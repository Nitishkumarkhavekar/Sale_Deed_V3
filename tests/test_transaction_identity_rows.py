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
