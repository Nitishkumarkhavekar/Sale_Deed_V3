"""Validation engine.

The regression cases here are not invented. Each one encodes a defect that was
found by running real data, and would otherwise be easy to reintroduce:

  * deed 07  - Q4_K_M returned 1500000 where the OCR reads 1,50,000
  * deed 1316 - a decimal point was read as a thousands separator, so a wrong
    registration fee reported as grounded
  * deed 1359 - a two-PAN document failed coverage forever on a metric artefact
"""

from __future__ import annotations

import pytest

from core.validation import (
    Disposition,
    Flag,
    RuleToggles,
    aadhaar_format_valid,
    aadhaar_in_ocr,
    amount_format_valid,
    amount_in_ocr,
    date_in_ocr,
    date_iso_valid,
    derive_stamp_value,
    extract_json,
    indian_grouping,
    name_in_ocr,
    normalise_for_match,
    ocr_pans,
    pan_aadhaar_proximity,
    pan_coverage,
    pan_format_valid,
    pan_in_ocr,
    reg_fee_candidates,
    validate_extraction,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Layer 2 - JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_strips_code_fences(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_ignores_trailing_prose(self):
        assert extract_json('{"a": 1}\n\nThat is the answer.') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"a": "a { brace } inside"}') == {"a": "a { brace } inside"}

    def test_escaped_quote_inside_string(self):
        assert extract_json(r'{"a": "say \"hi\""}') == {"a": 'say "hi"'}

    def test_duplicated_object_returns_the_first(self):
        # Observed in the benchmark corpus: deed 2725 emitted a second object.
        assert extract_json('{"a": 1}{"b": 2}') == {"a": 1}

    def test_truncated_json_is_none(self):
        assert extract_json('{"a": 1') is None

    def test_empty_is_none(self):
        assert extract_json("") is None


# ---------------------------------------------------------------------------
# Layer 3 - amounts
# ---------------------------------------------------------------------------


class TestAmounts:
    @pytest.mark.parametrize("digits,expected", [
        ("150000", "1,50,000"), ("1500000", "15,00,000"),
        ("3300000", "33,00,000"), ("500", "500"), ("52500", "52,500"),
    ])
    def test_indian_grouping(self, digits, expected):
        assert indian_grouping(digits) == expected

    @pytest.mark.parametrize("value,ok", [
        ("150000", True), ("1,50,000", False), ("150000.00", False),
        ("150 000", False), ("", False), (None, False),
    ])
    def test_format_requires_plain_digits(self, value, ok):
        assert amount_format_valid(value) is ok

    def test_finds_amount_in_indian_grouping(self):
        ocr = "consideration of Rs.1,50,000=00 (Rupees One Lakh Fifty thousand)"
        assert amount_in_ocr("150000", normalise_for_match(ocr), ocr)

    def test_deed_07_ten_times_error_is_rejected(self):
        """The exact Q4_K_M failure: 1500000 is absent, 150000 is present."""
        ocr = "a sum of Rs.1,50,000=00 (Rupees One Lakh Fifty thousand) as stated"
        norm = normalise_for_match(ocr)
        assert amount_in_ocr("150000", norm, ocr) is True
        assert amount_in_ocr("1500000", norm, ocr) is False

    def test_deed_1316_decimal_point_is_not_a_separator(self):
        """`200.00` must not satisfy a query for `20000`.

        Treating the decimal point as a thousands separator made a genuinely
        wrong registration fee report as grounded.
        """
        ocr = "ನೋಂದಣಿ ಶುಲ್ಕ    200.00"
        norm = normalise_for_match(ocr)
        assert amount_in_ocr("200", norm, ocr) is True
        assert amount_in_ocr("20000", norm, ocr) is False

    def test_does_not_match_inside_a_longer_number(self):
        ocr = "total 6360000 rupees"
        assert amount_in_ocr("636000", normalise_for_match(ocr), ocr) is False

    def test_line_wrap_support_permits_bridging_adjacent_numbers(self):
        """Documents a known, accepted trade-off.

        Whitespace has to be allowed between digits so an amount split across a
        line break still matches - verified on real deeds. The cost is that a
        long enough query can bridge two separate numbers separated by a space.
        Harmless in practice: real amounts are at most ~9 digits, so no genuine
        value could span two numbers this way.
        """
        ocr = "63,60,000 13,00,000"
        assert amount_in_ocr("63600001300000", normalise_for_match(ocr), ocr) is True
        # What actually matters: a real amount does not match a longer neighbour.
        assert amount_in_ocr("636000", normalise_for_match(ocr), ocr) is False

    def test_matches_across_a_line_break(self):
        ocr = "Rs. 33,00,\n000 only"
        assert amount_in_ocr("3300000", normalise_for_match(ocr), ocr)

    def test_trailing_decimal_suffix_does_not_block_the_match(self):
        ocr = "amount 12,30,000.00 paid"
        assert amount_in_ocr("1230000", normalise_for_match(ocr), ocr)


# ---------------------------------------------------------------------------
# Layer 3 - identifiers
# ---------------------------------------------------------------------------


class TestPan:
    @pytest.mark.parametrize("pan,ok", [
        ("ADPPN2284H", True), ("BLRPS9269", False),
        # Case is normalised: the spec asks for PAN normalisation, and OCR
        # routinely lowercases. The value is upper-cased before storage.
        ("adppn2284h", True),
        ("ADPP12284H", False), ("", False), (None, False),
    ])
    def test_format(self, pan, ok):
        assert pan_format_valid(pan) is ok

    @pytest.mark.parametrize("written", [
        "ALQPP8332F", "ALQPP 8332F", "ALQPP-8332F", "ALQPP.8332F",
    ])
    def test_found_despite_separators(self, written):
        assert pan_in_ocr("ALQPP8332F", normalise_for_match(f"PAN {written} of"))

    def test_ocr_pans_finds_all(self):
        text = "AZMPA8189K and BUVPB8312G, plus truncated BLRPS9269"
        assert ocr_pans(text) == {"AZMPA8189K", "BUVPB8312G"}


class TestAadhaar:
    @pytest.mark.parametrize("value,ok", [
        ("241391305374", True), ("2413 9130 5374", True),
        ("24139130537", False), ("", False), (None, False),
    ])
    def test_format(self, value, ok):
        assert aadhaar_format_valid(value) is ok

    def test_found_when_wrapped_across_lines(self):
        ocr = "Aadhaar 9414 5063\n                 3293"
        assert aadhaar_in_ocr("941450633293", normalise_for_match(ocr), ocr)

    def test_absent_number_is_rejected(self):
        ocr = "Aadhaar 2413 9130 5374"
        assert aadhaar_in_ocr("856712232477", normalise_for_match(ocr), ocr) is False


class TestProximity:
    def test_near_pair_passes(self):
        ocr = "Name X  PAN ADPPN2284H  Aadhaar 2413 9130 5374"
        assert pan_aadhaar_proximity("ADPPN2284H", "241391305374", ocr, 250) is True

    def test_distant_pair_flags(self):
        ocr = "PAN ADPPN2284H" + " filler" * 200 + " Aadhaar 2413 9130 5374"
        assert pan_aadhaar_proximity("ADPPN2284H", "241391305374", ocr, 250) is False

    def test_missing_value_is_indeterminate(self):
        assert pan_aadhaar_proximity(None, "241391305374", "text", 250) is None


# ---------------------------------------------------------------------------
# Names and dates
# ---------------------------------------------------------------------------


class TestNamesAndDates:
    def test_name_matches_when_contiguous(self):
        assert name_in_ocr("SATISH V PATHAK", normalise_for_match("seller SATISH V PATHAK aged"))

    def test_name_matches_on_non_contiguous_tokens(self):
        # Kannada clusters survive OCR but get separated; strict equality fails.
        ocr = normalise_for_match("ಅಚಲ ಎಲ್ ... other text ... ನರಗುಂದ")
        assert name_in_ocr("ಅಚಲ ಎಲ್ ನರಗುಂದ", ocr)

    def test_unrelated_name_rejected(self):
        assert name_in_ocr("ZZZZ QQQQ", normalise_for_match("nothing alike here")) is False

    @pytest.mark.parametrize("value,ok", [
        ("2025-04-09", True), ("09-04-2025", False), ("2025-13-01", False),
        ("", False), (None, False),
    ])
    def test_iso_format(self, value, ok):
        assert date_iso_valid(value) is ok

    @pytest.mark.parametrize("written", ["9/4/2025", "09/04/2025", "09-04-2025"])
    def test_date_found_in_indian_forms(self, written):
        assert date_in_ocr("2025-04-09", normalise_for_match(f"dated {written} at"))


# ---------------------------------------------------------------------------
# Layer 5 / 6
# ---------------------------------------------------------------------------


class TestRegFeeAndCoverage:
    def test_regex_finds_labelled_fee(self):
        assert 33000 in reg_fee_candidates("Registration Fee: Rs. 33,000")

    def test_regex_finds_kannada_label(self):
        assert 200 in reg_fee_candidates("ನೋಂದಣಿ ಶುಲ್ಕ 200.00")

    def test_out_of_range_values_ignored(self):
        assert reg_fee_candidates("Registration Fee Rs. 99,99,99,999") == []

    def test_coverage_full(self):
        pred = {"buyer_details": [{"pan_card_number": "AZMPA8189K"}],
                "seller_details": [{"pan_card_number": "BUVPB8312G"}]}
        assert pan_coverage(pred, "AZMPA8189K BUVPB8312G") == 1.0

    def test_coverage_half(self):
        pred = {"buyer_details": [{"pan_card_number": "AZMPA8189K"}], "seller_details": []}
        assert pan_coverage(pred, "AZMPA8189K BUVPB8312G") == 0.5

    def test_no_pans_in_source_is_no_signal(self):
        assert pan_coverage({"buyer_details": [], "seller_details": []}, "no pans") == 1.0


# ---------------------------------------------------------------------------
# Stamp value
# ---------------------------------------------------------------------------


class TestStampValue:
    @pytest.mark.parametrize("fee,date_,expected", [
        (33000, "2025-04-09", 16500),   # before cutoff -> halved
        (33000, "2025-08-30", 16500),   # day before -> halved
        (33000, "2025-08-31", 33000),   # on cutoff -> unchanged
        (33000, "2025-09-15", 33000),   # after -> unchanged
        (71500, "2025-07-10", 35750),
        (33001, "2025-01-01", 16501),   # rounds half up
    ])
    def test_halving_rule(self, fee, date_, expected):
        assert derive_stamp_value(fee, date_, RuleToggles()) == expected

    def test_no_date_derives_nothing(self):
        """Guessing would be wrong by 2x half the time."""
        assert derive_stamp_value(33000, None, RuleToggles()) is None

    def test_unparseable_date_derives_nothing(self):
        assert derive_stamp_value(33000, "09-04-2025", RuleToggles()) is None

    def test_multiplier_applies_after_halving(self):
        rules = RuleToggles(stamp_value_multiplier=2.0)
        assert derive_stamp_value(33000, "2025-04-09", rules) == 33000


# ---------------------------------------------------------------------------
# Layer 7 - disposition
# ---------------------------------------------------------------------------


def _pred(**overrides):
    base = {
        "buyer_details": [], "seller_details": [],
        "property_details": {"sale_consideration": None, "registration_fee": None,
                             "paid_in_cash": "no"},
        "document_details": {"transaction_date": "2025-04-09"},
    }
    base.update(overrides)
    return base


class TestDisposition:
    def test_unparseable_output_retries(self):
        report = validate_extraction("not json at all", "some ocr")
        assert report.disposition is Disposition.RETRY
        assert report.parsed is False

    def test_clean_document_accepts(self):
        ocr = "sale for Rs.3,30,000 dated 09/04/2025"
        report = validate_extraction(
            _pred(property_details={"sale_consideration": "330000",
                                    "registration_fee": None, "paid_in_cash": "no"}), ocr)
        assert report.disposition is Disposition.ACCEPT

    def test_ungrounded_amount_goes_to_review_not_retry(self):
        """Retry cannot fix a grounding failure at temperature 0."""
        ocr = "sale for Rs.1,50,000 dated 09/04/2025"
        report = validate_extraction(
            _pred(property_details={"sale_consideration": "1500000",
                                    "registration_fee": None, "paid_in_cash": "no"}), ocr)
        assert Flag.WRONG_CONSIDERATION in report.document_flags
        assert report.disposition is Disposition.REVIEW

    def test_two_pan_document_is_not_retried_on_coverage_alone(self):
        """Deed 1359: 0.5 coverage with only one PAN unmatched is not evidence."""
        ocr = "AZMPA8189K seller, BUVPB8312G witness, dated 09/04/2025"
        report = validate_extraction(
            _pred(seller_details=[{"name": "S", "pan_card_number": "AZMPA8189K"}]), ocr)
        assert report.pan_coverage == 0.5
        assert report.disposition is not Disposition.RETRY

    def test_many_unmatched_pans_does_retry(self):
        pans = " ".join(f"AZMPA818{i}K" for i in range(5))
        report = validate_extraction(_pred(), pans + " dated 09/04/2025")
        assert report.disposition is Disposition.RETRY

    def test_malformed_pan_is_discarded_and_flagged(self):
        ocr = "PAN BLRPS9269 truncated, dated 09/04/2025"
        report = validate_extraction(
            _pred(buyer_details=[{"name": "B", "pan_card_number": "BLRPS9269"}]), ocr)
        person = report.persons[0]
        assert Flag.HALLUCINATED_PAN in person.flags
        assert "pan_card_number" in person.discarded

    def test_missing_date_flags_wtd(self):
        report = validate_extraction(
            _pred(document_details={"transaction_date": None}), "no date here")
        assert Flag.WRONG_TXN_DATE in report.document_flags

    def test_paid_in_cash_must_not_be_null(self):
        report = validate_extraction(
            _pred(property_details={"sale_consideration": None,
                                    "registration_fee": None, "paid_in_cash": None}),
            "dated 09/04/2025")
        checks = {c.field: c for c in report.document_checks}
        assert checks["paid_in_cash"].format_ok is False

    def test_truncated_output_is_flagged(self):
        report = validate_extraction(_pred(), "dated 09/04/2025", truncated=True)
        assert Flag.TRUNCATED in report.document_flags
        assert report.disposition is Disposition.REVIEW

    def test_ocr_failure_recorded(self):
        report = validate_extraction(_pred(), "text", ocr_succeeded=False)
        assert Flag.OCR_FAILED in report.document_flags

    def test_disabled_rule_is_not_evaluated(self):
        ocr = "nothing matches"
        rules = RuleToggles(sale_consideration=False)
        report = validate_extraction(
            _pred(property_details={"sale_consideration": "999999",
                                    "registration_fee": None, "paid_in_cash": "no"}),
            ocr, rules)
        assert Flag.WRONG_CONSIDERATION not in report.document_flags


# ---------------------------------------------------------------------------
# Regression across the whole corpus
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestCorpus:
    def test_every_reference_parses_and_validates(self, corpus_pairs):
        """Uses extract_json, not json.loads.

        Deed 2725 in the corpus emitted a valid object followed by a duplicate,
        which json.loads rejects as "Extra data". Tolerating that is precisely
        what extract_json exists for, so the test exercises the real path.
        """
        for stem, ocr_path, reference in corpus_pairs:
            ocr = ocr_path.read_text(encoding="utf-8", errors="replace")
            pred = extract_json(reference.read_text(encoding="utf-8"))
            if pred is None:
                continue  # genuinely unparseable output is a known corpus case
            report = validate_extraction(pred, ocr)
            assert report.parsed, f"{stem}: reference failed to parse"
            assert 0.0 <= report.confidence <= 1.0, f"{stem}: confidence out of range"

    def test_known_genuine_errors_are_caught(self, corpus_pairs):
        """gemma6.8 score.md records two real errors; both must be flagged."""
        by_stem = {stem: (o, r) for stem, o, r in corpus_pairs}
        expected = {
            "1316": Flag.WRONG_STAMP_VALUE,   # reg_fee 20000, OCR says 200.00
            "2231": Flag.WRONG_AADHAAR,       # hallucinated seller Aadhaar
        }
        for stem, flag in expected.items():
            if stem not in by_stem:
                pytest.skip(f"deed {stem} not in corpus")
            ocr_path, reference = by_stem[stem]
            report = validate_extraction(
                extract_json(reference.read_text(encoding="utf-8")),
                ocr_path.read_text(encoding="utf-8", errors="replace"))
            found = set(report.document_flags) | {
                f for p in report.persons for f in p.flags}
            assert flag in found, f"deed {stem}: expected {flag.value}, got {found}"
