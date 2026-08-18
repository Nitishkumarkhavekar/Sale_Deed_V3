"""Splitting a deed's consideration between the people on it.

Every row used to carry the deed's full consideration, so a ₹1,000 deed with
four buyers and two sellers reported ₹6,000 of transactions.

The rule is that **all parties on the deed share one pool**: sellers and buyers
together. ₹10,000 with two sellers and two buyers gives ₹2,500 to each of the
four, and the four shares add back to ₹10,000 once.

This replaced a per-side split, in which each side divided the whole
consideration between itself - ₹5,000 to every seller *and* ₹5,000 to every
buyer, so each side's rows summed to the deed total separately. Both are
defensible and they answer different questions. The combined rule is the one
specified, and the tests below pin it so the two cannot be confused again.

Two properties carry most of the weight, and both are about money:

  * **The shares add back to the total.** ₹1,000 over three parties as ₹333.33
    each reports ₹999.99 - a shortfall in a tax return, arrived at by rounding.
    Tested at many awkward divisors, not just the tidy ones.
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


def _shares(rows) -> list[str]:
    return [r[AMOUNT_COLUMN] for r in rows]


def _by_side(rows) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"S": [], "B": []}
    for row in rows:
        out[row["Transaction Relation (PC)"]].append(row[AMOUNT_COLUMN])
    return out


# ---------------------------------------------------------------------------
# The rule as specified
# ---------------------------------------------------------------------------


class TestTheWorkedExample:
    """₹10,000, two sellers, two buyers - the case in the specification."""

    def test_every_person_gets_a_quarter(self):
        rows = build_rows([_deed(sellers=2, buyers=2, amount="10000")])
        assert _shares(rows) == ["2500"] * 4

    def test_sellers_and_buyers_are_treated_alike(self):
        split = _by_side(build_rows([_deed(sellers=2, buyers=2, amount="10000")]))
        assert split["S"] == split["B"] == ["2500", "2500"]

    def test_the_deed_total_is_shared_once_across_everyone(self):
        rows = build_rows([_deed(sellers=2, buyers=2, amount="10000")])
        assert sum(Decimal(s) for s in _shares(rows)) == Decimal("10000")

    def test_it_is_not_split_per_side(self):
        """The rule this replaced would give ₹5,000 to each of the four, and
        each side would sum to the deed total separately."""
        assert "5000" not in _shares(
            build_rows([_deed(sellers=2, buyers=2, amount="10000")]))

    def test_the_deed_total_column_is_unchanged(self):
        """The per-person column is derived; the deed's own column still
        carries what the deed says."""
        rows = build_rows([_deed(sellers=2, buyers=2, amount="10000")])
        assert {r["Transaction Amount"] for r in rows} == {"10000"}


class TestEveryPartyCountUpTo12:
    @pytest.mark.parametrize("count", range(1, 13))
    def test_the_pool_divides_by_the_whole_party_count(self, count):
        rows = build_rows([_deed(sellers=count, buyers=1, amount="1200")])
        shares = _shares(rows)
        assert len(shares) == count + 1
        assert sum(Decimal(s) for s in shares) == Decimal("1200")

    @pytest.mark.parametrize("sellers,buyers", [
        (1, 1), (1, 2), (2, 1), (3, 4), (4, 3), (5, 2), (7, 11), (10, 10),
    ])
    def test_the_shares_always_add_back(self, sellers, buyers):
        rows = build_rows([_deed(sellers=sellers, buyers=buyers,
                                 amount="4473271")])
        assert sum(Decimal(s) for s in _shares(rows)) == Decimal("4473271")
        assert len(_shares(rows)) == sellers + buyers

    def test_a_lone_party_receives_the_whole_amount(self):
        rows = build_rows([_deed(sellers=1, buyers=0, amount="56700000")])
        assert _shares(rows) == ["56700000"]

    def test_one_of_each_halves_it(self):
        rows = build_rows([_deed(sellers=1, buyers=1, amount="56700000")])
        assert _shares(rows) == ["28350000", "28350000"]


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
        ("1000", 3), ("1000", 6), ("1000", 7), ("100", 3), ("1", 3),
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

    def test_the_earlier_worked_example_now_splits_six_ways(self):
        """₹1,000 with two sellers and four buyers. Under the previous rule
        this gave ₹500 and ₹250; under this one all six share the pool."""
        rows = build_rows([_deed(sellers=2, buyers=4, amount="1000")])
        shares = _shares(rows)
        assert len(shares) == 6
        assert sum(Decimal(s) for s in shares) == Decimal("1000")
        assert shares[0] == "166.67"

    def test_the_split_is_reproducible(self):
        assert person_shares("1000", 7) == person_shares("1000", 7)

    def test_no_share_differs_from_another_by_more_than_a_paisa(self):
        shares = [Decimal(s) for s in person_shares("1000", 7)]
        assert max(shares) - min(shares) <= Decimal("0.01")

    def test_whole_rupees_keep_their_plain_form(self):
        """Not "2500.00" - what the column held before this change, and what
        every evenly divided deed still produces."""
        assert person_shares("10000", 4) == ["2500"] * 4
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
        Decimal("10000"), "10000", 10000, " 10000 ", "10,000", "₹10,000",
        "Rs.10000", "Rs 10000", "INR10000", "10,000.00",
    ])
    def test_every_shape_divides_the_same(self, value):
        assert person_shares(value, 4) == ["2500"] * 4

    def test_an_indian_grouped_amount_is_read(self):
        assert person_shares("45,00,000", 4) == ["1125000"] * 4


# ---------------------------------------------------------------------------
# The divisor is the number of rows written
# ---------------------------------------------------------------------------


class TestTheDivisorMatchesTheRows:
    def test_a_dropped_duplicate_does_not_shrink_everyone_else(self):
        """The export drops a party listed twice. Splitting by the raw lists
        would hand out more than the deed's value and leave the rest
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
        shares = _shares(build_rows([doc]))

        assert len(shares) == 3, "the duplicate was not dropped"
        assert shares == ["300", "300", "300"]
        assert sum(Decimal(s) for s in shares) == Decimal("900")

    def test_a_party_with_no_usable_name_does_not_take_a_share(self):
        doc = DocumentExport(
            transaction_identity="DEED-NONAME",
            extraction={
                "seller_details": [_person("S1"), {"name": "---"}],
                "buyer_details": [_person("B1")],
                "property_details": {"sale_consideration": "800"},
                "document_details": {},
            })
        shares = _shares(build_rows([doc]))
        assert shares == ["400", "400"], "the unnamed party took a share"

    def test_a_non_dictionary_entry_is_ignored(self):
        doc = DocumentExport(
            transaction_identity="DEED-JUNK",
            extraction={
                "seller_details": [_person("S1"), "a string", None],
                "buyer_details": [],
                "property_details": {"sale_consideration": "600"},
                "document_details": {},
            })
        assert _shares(build_rows([doc])) == ["600"]

    def test_someone_on_both_sides_counts_twice(self):
        """A person selling and buying in one deed keeps a row on each side, so
        they hold two shares of the pool - which is what the rows say."""
        both = _person("X1")
        doc = DocumentExport(
            transaction_identity="DEED-BOTH",
            extraction={
                "seller_details": [both],
                "buyer_details": [dict(both), _person("B2")],
                "property_details": {"sale_consideration": "900"},
                "document_details": {},
            })
        split = _by_side(build_rows([doc]))
        assert split["S"] == ["300"]
        assert split["B"] == ["300", "300"]


class TestSeveralDeedsAtOnce:
    def test_each_deed_is_split_against_its_own_parties(self):
        """The rule is per transaction identity. A shared divisor across deeds
        would be wrong for all but one of them."""
        first = _deed(sellers=2, buyers=2, amount="10000")
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
        assert _shares(one) == ["2500"] * 4
        assert _shares(two) == ["200", "200", "200"]


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
        assert rows[0]["PAN (PC)"] == _person("S1")["pan_card_number"]
        assert rows[0]["Aadhaar Number (PC)"] == _person("S1")["aadhaar_number"]

    def test_the_written_file_carries_the_split(self, tmp_path):
        """End to end through the CSV writer: the value must survive quoting,
        escaping and the round trip to disk."""
        import csv

        from core.csv_export import write_csv

        target = tmp_path / "export.csv"
        write_csv(target, [_deed(sellers=2, buyers=2, amount="10000")])

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert _shares(rows) == ["2500"] * 4

    def test_the_excel_safe_export_carries_the_split(self, tmp_path):
        """Identifier columns are wrapped as formulas in this mode; the amount
        must not be."""
        import csv

        from core.csv_export import write_csv

        target = tmp_path / "export-excel.csv"
        write_csv(target, [_deed(sellers=2, buyers=2, amount="10000")],
                  excel_safe=True)

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert not row[AMOUNT_COLUMN].startswith("=")
        assert _shares(rows) == ["2500"] * 4
