"""Splitting a deed's consideration between the parties on each side.

Every row used to carry the deed's full consideration, so a ₹1,000 deed with
four buyers and two sellers reported ₹6,000 of transactions.

The rule is that **each side is divided on its own**: the buyers share the whole
consideration between themselves, and so do the sellers, because each side is a
complete account of the same transaction seen from one end. Dividing by the
combined head count would report a transaction that never took place.

Two properties carry most of the weight here, and both are about money rather
than arithmetic:

  * **Each side sums back to the deed total.** ₹1,000 over three parties as
    ₹333.33 each reports ₹999.99 - a shortfall in a tax return, arrived at by
    rounding. Tested at many awkward divisors, not just the tidy ones.
  * **The divisor is the number of rows actually written.** Duplicates and
    unusable names are dropped before the split, so the shares cannot be
    computed against a party count the export does not produce.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.csv_export import DocumentExport, build_rows, person_shares

AMOUNT_COLUMN = "Transaction Amount related to the person (PC)"


def _person(tag: str, **extra) -> dict:
    """A distinct party.

    The identifiers have to be genuinely distinct, not merely look it: the
    export drops parties whose name and identifiers match one already seen, so a
    fixture that reuses an Aadhaar silently produces fewer rows than it asked
    for. An earlier version keyed the Aadhaar on the tag's last character and
    quietly lost S11 as a duplicate of S1.
    """
    digits = "".join(c for c in tag if c.isdigit()).rjust(4, "0")
    person = {"name": f"PERSON {tag}",
              "pan_card_number": f"ABCP{digits[0]}{digits[1:]}F",
              "aadhaar_number": f"66{digits}2345678"[:12]}
    person.update(extra)
    return person


def _deed(sellers: int, buyers: int, amount="1000", **extra) -> DocumentExport:
    return DocumentExport(
        transaction_identity="DEED-1",
        extraction={
            "seller_details": [_person(f"S{n}") for n in range(1, sellers + 1)],
            "buyer_details": [_person(f"B{n}") for n in range(1, buyers + 1)],
            "property_details": {"sale_consideration": amount},
            "document_details": {},
            **extra,
        })


def _by_side(rows) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"S": [], "B": []}
    for row in rows:
        out[row["Transaction Relation (PC)"]].append(row[AMOUNT_COLUMN])
    return out


# ---------------------------------------------------------------------------
# The rule as specified
# ---------------------------------------------------------------------------


class TestTheWorkedExample:
    """₹1,000, four buyers, two sellers - the case in the specification."""

    def test_each_buyer_gets_a_quarter(self):
        shares = _by_side(build_rows([_deed(sellers=2, buyers=4)]))
        assert shares["B"] == ["250"] * 4

    def test_each_seller_gets_a_half(self):
        shares = _by_side(build_rows([_deed(sellers=2, buyers=4)]))
        assert shares["S"] == ["500"] * 2

    def test_the_sides_are_not_pooled(self):
        """Six parties dividing ₹1,000 would be ₹166.67 each. That is the
        mistake this rule exists to prevent."""
        shares = _by_side(build_rows([_deed(sellers=2, buyers=4)]))
        assert "166.67" not in shares["B"] + shares["S"]

    def test_the_deed_total_is_unchanged_on_every_row(self):
        """The per-person column is derived; the deed's own column still
        carries what the deed says."""
        rows = build_rows([_deed(sellers=2, buyers=4)])
        assert {r["Transaction Amount"] for r in rows} == {"1000"}


class TestEverySideCountUpTo12:
    @pytest.mark.parametrize("count", range(1, 13))
    def test_one_side_divides_by_its_own_count(self, count):
        rows = build_rows([_deed(sellers=count, buyers=1, amount="1200")])
        sellers = _by_side(rows)["S"]
        assert len(sellers) == count
        assert sum(Decimal(s) for s in sellers) == Decimal("1200")

    @pytest.mark.parametrize("sellers,buyers", [
        (1, 1), (1, 2), (2, 1), (3, 4), (4, 3), (5, 2), (7, 11), (10, 10),
    ])
    def test_both_sides_each_sum_to_the_whole(self, sellers, buyers):
        rows = build_rows([_deed(sellers=sellers, buyers=buyers, amount="4473271")])
        split = _by_side(rows)
        assert sum(Decimal(s) for s in split["S"]) == Decimal("4473271")
        assert sum(Decimal(s) for s in split["B"]) == Decimal("4473271")

    def test_a_lone_party_receives_the_whole_amount(self):
        split = _by_side(build_rows([_deed(sellers=1, buyers=1, amount="56700000")]))
        assert split["S"] == ["56700000"]
        assert split["B"] == ["56700000"]


class TestOneSidedDeeds:
    def test_a_deed_with_only_sellers(self):
        split = _by_side(build_rows([_deed(sellers=3, buyers=0, amount="900")]))
        assert split["S"] == ["300"] * 3
        assert split["B"] == []

    def test_a_deed_with_only_buyers(self):
        split = _by_side(build_rows([_deed(sellers=0, buyers=4, amount="900")]))
        assert split["B"] == ["225"] * 4
        assert split["S"] == []

    def test_a_deed_with_no_parties_still_produces_a_row(self):
        """Unchanged behaviour: the document stays visible in the export."""
        rows = build_rows([_deed(sellers=0, buyers=0)])
        assert len(rows) == 1
        assert rows[0]["Transaction Amount"] == "1000"


# ---------------------------------------------------------------------------
# Money, not arithmetic
# ---------------------------------------------------------------------------


class TestNothingIsLostToRounding:
    @pytest.mark.parametrize("total,count", [
        ("1000", 3), ("1000", 7), ("1000", 6), ("100", 3), ("1", 3),
        ("4473271", 3), ("56700000", 7), ("0.03", 2), ("10", 8),
        ("999999999", 7), ("0.01", 3),
    ])
    def test_the_shares_add_back_to_the_total(self, total, count):
        """Dropping the remainder understates the deed - ₹1,000 over three as
        ₹333.33 each reports ₹999.99."""
        shares = person_shares(total, count)
        assert sum(Decimal(s) for s in shares) == Decimal(total), shares

    def test_the_remainder_goes_to_the_earliest_parties(self):
        """Deterministic, so two exports of one deed agree."""
        assert person_shares("1000", 3) == ["333.34", "333.33", "333.33"]

    def test_the_split_is_reproducible(self):
        assert person_shares("1000", 7) == person_shares("1000", 7)

    def test_no_share_differs_from_another_by_more_than_a_paisa(self):
        """Equal division, to the smallest unit that exists."""
        shares = [Decimal(s) for s in person_shares("1000", 7)]
        assert max(shares) - min(shares) <= Decimal("0.01")

    def test_whole_rupees_keep_their_plain_form(self):
        """What the column held before this change, and what an evenly divided
        deed should still produce - not "250.00"."""
        assert person_shares("1000", 4) == ["250"] * 4
        assert person_shares("56700000", 3) == ["18900000"] * 3

    def test_part_rupees_carry_exactly_two_decimals(self):
        for share in person_shares("1000", 3):
            assert len(share.split(".")[1]) == 2

    def test_a_large_amount_is_exact(self):
        """`float` would lose this. `Decimal` does not, which is why the whole
        calculation is done in it."""
        shares = person_shares("999999999999999999", 7)
        assert sum(Decimal(s) for s in shares) == Decimal("999999999999999999")

    def test_an_amount_already_carrying_paise_is_respected(self):
        shares = person_shares("100.50", 2)
        assert sum(Decimal(s) for s in shares) == Decimal("100.50")
        assert shares == ["50.25", "50.25"]


class TestUnreadableAmounts:
    """A wrong figure in this column is worse than the original text: the text
    at least shows what was found in the deed."""

    def test_a_blank_amount_stays_blank(self):
        assert person_shares("", 3) == ["", "", ""]

    def test_a_missing_amount_stays_blank(self):
        assert person_shares(None, 2) == ["", ""]

    def test_text_that_is_not_a_number_is_passed_through(self):
        assert person_shares("as per schedule", 2) == ["as per schedule"] * 2

    def test_no_parties_means_no_shares(self):
        assert person_shares("1000", 0) == []

    def test_a_deed_with_an_unreadable_amount_still_exports(self):
        rows = build_rows([_deed(sellers=2, buyers=2, amount="not stated")])
        assert len(rows) == 4
        assert all(r[AMOUNT_COLUMN] == "not stated" for r in rows)


class TestAmountsAsTheyArriveFromTheDatabase:
    """`Property.sale_consideration` is a `Decimal`; a CSV round trip makes it
    text; the model sometimes writes separators."""

    @pytest.mark.parametrize("value", [
        Decimal("1000"), "1000", 1000, " 1000 ", "1,000", "₹1,000", "Rs.1000",
        "Rs 1000", "INR1000", "1,000.00",
    ])
    def test_every_shape_divides_the_same(self, value):
        assert person_shares(value, 4) == ["250"] * 4

    def test_an_indian_grouped_amount_is_read(self):
        assert person_shares("45,00,000", 4) == ["1125000"] * 4


# ---------------------------------------------------------------------------
# The divisor is the number of rows written
# ---------------------------------------------------------------------------


class TestTheDivisorMatchesTheRows:
    def test_a_dropped_duplicate_does_not_shrink_everyone_else(self):
        """The export drops a party listed twice. Splitting by the raw list
        would hand out three quarters of the deed's value and leave the rest
        unaccounted for."""
        twin = _person("B1")
        doc = DocumentExport(
            transaction_identity="DEED-DUP",
            extraction={
                "seller_details": [_person("S1")],
                "buyer_details": [twin, dict(twin), _person("B2")],
                "property_details": {"sale_consideration": "900"},
                "document_details": {},
            })
        rows = build_rows([doc])
        buyers = _by_side(rows)["B"]

        assert len(buyers) == 2, "the duplicate was not dropped"
        assert buyers == ["450", "450"]
        assert sum(Decimal(b) for b in buyers) == Decimal("900")

    def test_a_party_with_no_usable_name_does_not_take_a_share(self):
        doc = DocumentExport(
            transaction_identity="DEED-NONAME",
            extraction={
                "seller_details": [_person("S1"), {"name": "---"}],
                "buyer_details": [_person("B1")],
                "property_details": {"sale_consideration": "800"},
                "document_details": {},
            })
        split = _by_side(build_rows([doc]))
        assert split["S"] == ["800"], "the unnamed party took a share"

    def test_a_non_dictionary_entry_is_ignored(self):
        doc = DocumentExport(
            transaction_identity="DEED-JUNK",
            extraction={
                "seller_details": [_person("S1"), "a string", None],
                "buyer_details": [],
                "property_details": {"sale_consideration": "600"},
                "document_details": {},
            })
        assert _by_side(build_rows([doc]))["S"] == ["600"]

    def test_someone_on_both_sides_is_counted_once_per_side(self):
        """A person selling and buying in one deed keeps a row on each side,
        and each row is split against that side's own count."""
        both = _person("X1")
        doc = DocumentExport(
            transaction_identity="DEED-BOTH",
            extraction={
                "seller_details": [both],
                "buyer_details": [dict(both), _person("B2")],
                "property_details": {"sale_consideration": "1000"},
                "document_details": {},
            })
        split = _by_side(build_rows([doc]))
        assert split["S"] == ["1000"]
        assert split["B"] == ["500", "500"]


class TestSeveralDeedsAtOnce:
    def test_each_deed_is_split_against_its_own_parties(self):
        """The rule is per transaction identity. A shared divisor across deeds
        would be wrong for all but one of them."""
        first = _deed(sellers=2, buyers=4, amount="1000")
        second = DocumentExport(
            transaction_identity="DEED-2",
            extraction={
                "seller_details": [_person("S9")],
                "buyer_details": [_person("B8"), _person("B9")],
                "property_details": {"sale_consideration": "600"},
                "document_details": {},
            })
        rows = build_rows([first, second])

        one = [r for r in rows if r["Transaction Identity"] == "DEED-1"]
        two = [r for r in rows if r["Transaction Identity"] == "DEED-2"]
        assert _by_side(one) == {"S": ["500", "500"], "B": ["250"] * 4}
        assert _by_side(two) == {"S": ["600"], "B": ["300", "300"]}


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------


class TestTheRestOfTheRowIsUntouched:
    def test_the_column_order_is_unchanged(self):
        from core.csv_export import CSV_COLUMNS

        rows = build_rows([_deed(sellers=1, buyers=1)])
        assert list(rows[0].keys()) == list(CSV_COLUMNS)

    def test_sellers_are_still_written_before_buyers(self):
        rows = build_rows([_deed(sellers=2, buyers=2)])
        assert [r["Transaction Relation (PC)"] for r in rows] == ["S", "S", "B", "B"]

    def test_the_row_count_is_unchanged(self):
        assert len(build_rows([_deed(sellers=3, buyers=5)])) == 8

    def test_names_and_identifiers_still_arrive(self):
        rows = build_rows([_deed(sellers=1, buyers=1)])
        assert rows[0]["Person Name (PC)"] == "PERSON S1"
        # The shape, not a literal: the fixture generates a distinct PAN per
        # party, and pinning one here only records what the fixture happens to
        # produce today.
        assert rows[0]["PAN (PC)"] == _person("S1")["pan_card_number"]
        assert rows[0]["Aadhaar Number (PC)"] == _person("S1")["aadhaar_number"]

    def test_the_written_file_carries_the_split(self, tmp_path):
        """End to end through the CSV writer: the value must survive quoting,
        escaping and the round trip to disk."""
        import csv

        from core.csv_export import write_csv

        target = tmp_path / "export.csv"
        write_csv(target, [_deed(sellers=2, buyers=4)])

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        split = _by_side(rows)
        assert split["S"] == ["500", "500"]
        assert split["B"] == ["250"] * 4

    def test_the_excel_safe_export_carries_the_split(self, tmp_path):
        """Identifier columns are wrapped as formulas in this mode; the amount
        must not be."""
        import csv

        from core.csv_export import write_csv

        target = tmp_path / "export-excel.csv"
        write_csv(target, [_deed(sellers=2, buyers=4)], excel_safe=True)

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert not row[AMOUNT_COLUMN].startswith("=")
        assert _by_side(rows)["B"] == ["250"] * 4
