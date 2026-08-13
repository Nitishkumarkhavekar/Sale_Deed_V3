"""CSV export.

The reference file is the specification. Two conventions were read off it rather
than assumed, and both would have been silent failures:

  * dates are DD-MM-YYYY, while the model emits ISO
  * Aadhaar in the reference arrived as `6.63E+11` - a spreadsheet coerced the
    12-digit string to a float and destroyed it
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core.csv_export import (
    ADDRESS_TYPES,
    CSV_COLUMNS,
    IDENTIFICATION_CODES,
    FAILED_COLUMNS,
    CODED_COLUMNS,
    STRUCTURALLY_ABSENT,
    DocumentExport,
    FailedDocument,
    address_type,
    build_rows,
    city_town,
    coded_column_violations,
    extract_pincode,
    identification_type,
    iso_to_ddmmyyyy,
    property_type,
    verify_against_reference,
    write_csv,
    write_failed_csv,
)

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.unit

REFERENCE = Path(__file__).resolve().parents[1] / "models" / "saledeed main" / "refering docs" / "example.csv"


def _extraction(**overrides):
    base = {
        "buyer_details": [{
            "name": "BUYER ONE", "gender": "Female", "father_name": "FATHER B",
            "aadhaar_number": "241391305374", "pan_card_number": "ADPPN2284H",
            "address": "House 1, Bengaluru - 560010", "state": "Karnataka",
        }],
        "seller_details": [{
            "name": "SELLER ONE", "gender": "Male", "father_name": "FATHER S",
            "aadhaar_number": "234968440246", "pan_card_number": "AIMPP2121R",
            "address": "House 2, Mysuru - 570001", "state": "Karnataka",
        }],
        "property_details": {
            "schedule_c_property_address": "Survey 12, Village X - 562162",
            "state": "Karnataka", "sale_consideration": "3300000",
            "registration_fee": "33000", "paid_in_cash": "no",
        },
        "document_details": {"transaction_date": "2025-04-09",
                             "registration_office": "Sub Registrar, Test"},
    }
    base.update(overrides)
    return base


def _buyer(rows):
    """The buyer row, by relation rather than by position.

    Indexing `rows[0]` encoded the emission order into tests that were about
    something else entirely, so changing the order broke five unrelated checks.
    """
    return next(r for r in rows if r["Transaction Relation (PC)"] == "B")


class TestSchema:
    def test_matches_reference_exactly(self):
        if not REFERENCE.is_file():
            pytest.skip("example.csv not present")
        assert verify_against_reference(REFERENCE) == []

    def test_has_42_columns(self):
        assert len(CSV_COLUMNS) == 42

    def test_remarks_columns_present(self):
        assert "Remarks" in CSV_COLUMNS
        assert "Person Details Remarks (PC)" in CSV_COLUMNS


class TestConversions:
    @pytest.mark.parametrize("iso,expected", [
        ("2025-07-19", "19-07-2025"), ("2025-04-09", "09-04-2025"),
        ("2025-12-31", "31-12-2025"),
    ])
    def test_iso_to_csv_date(self, iso, expected):
        assert iso_to_ddmmyyyy(iso) == expected

    def test_non_iso_passes_through(self):
        assert iso_to_ddmmyyyy("19-07-2025") == "19-07-2025"

    def test_empty_date(self):
        assert iso_to_ddmmyyyy(None) == ""

    @pytest.mark.parametrize("text,expected", [
        ("House 67, Karnataka - 562123", "562123"),
        ("no pin here", ""),
        ("Village X - 562162, Bengaluru", "562162"),
    ])
    def test_pincode_extraction(self, text, expected):
        assert extract_pincode(text) == expected

    def test_pincode_ignores_leading_zero(self):
        # Indian PINs never start with 0.
        assert extract_pincode("ref 012345 only") == ""

    def test_pincode_takes_the_last_match(self):
        # Addresses put the PIN at the end.
        assert extract_pincode("560010 something 570001") == "570001"


class TestRowExpansion:
    def test_one_row_per_person(self):
        rows = build_rows([DocumentExport("D-1", _extraction())])
        assert len(rows) == 2

    def test_relation_codes(self):
        rows = build_rows([DocumentExport("D-1", _extraction())])
        # Sellers first, then buyers - the order the reference report uses
        # and the order a deed reads in.
        assert [r["Transaction Relation (PC)"] for r in rows] == ["S", "B"]
        # Transaction Type carries the same code: S for sale, B for buy.
        assert [r["Transaction Type"] for r in rows] == ["S", "B"]

    def test_document_fields_repeat_on_every_row(self):
        rows = build_rows([DocumentExport("D-1", _extraction())])
        for column in ("Transaction Identity", "Transaction Amount",
                       "Transaction Date", "Property Address"):
            assert len({r[column] for r in rows}) == 1, f"{column} differs across rows"

    def test_serial_shared_within_a_document(self):
        rows = build_rows([DocumentExport("D-1", _extraction())])
        assert len({r["Report Serial Number"] for r in rows}) == 1

    def test_serial_increments_across_documents(self):
        rows = build_rows([DocumentExport("D-1", _extraction()),
                           DocumentExport("D-2", _extraction())])
        assert sorted({r["Report Serial Number"] for r in rows}) == ["1", "2"]

    def test_document_with_no_parties_still_appears(self):
        """A deed that extracted nothing must be visible, not silently absent."""
        rows = build_rows([DocumentExport(
            "D-EMPTY", _extraction(buyer_details=[], seller_details=[]))])
        assert len(rows) == 1
        # The row still appears, so the document is visible in the export.
        # The explanation now goes to the log: Remarks must stay empty (R-042).
        assert rows[0]["Remarks"] == ""
        assert rows[0]["Transaction Identity"]

    def test_translated_values_preferred(self):
        extraction = _extraction()
        extraction["buyer_details"][0]["name_translated"] = "TRANSLITERATED"
        extraction["buyer_details"][0]["address_translated"] = "Translated address"
        rows = build_rows([DocumentExport("D-1", extraction)])
        assert _buyer(rows)["Person Name (PC)"] == "TRANSLITERATED"
        assert _buyer(rows)["Address (PC-L)"] == "Translated address"

    def test_newlines_collapsed_within_a_cell(self):
        extraction = _extraction()
        extraction["property_details"]["schedule_c_property_address"] = "Line A\nLine B"
        rows = build_rows([DocumentExport("D-1", extraction)])
        assert "\n" not in rows[0]["Property Address"]


class TestIdentifierSafety:
    def test_aadhaar_written_as_text(self):
        rows = build_rows([DocumentExport("D-1", _extraction())])
        assert _buyer(rows)["Aadhaar Number (PC)"] == "241391305374"

    def test_excel_safe_wraps_identifiers(self):
        rows = build_rows([DocumentExport("D-1", _extraction())], excel_safe=True)
        assert _buyer(rows)["Aadhaar Number (PC)"] == '="241391305374"'
        assert _buyer(rows)["PAN (PC)"] == '="ADPPN2284H"'

    def test_aadhaar_survives_a_write_read_cycle(self, tmp_path):
        target = tmp_path / "out.csv"
        write_csv(target, [DocumentExport("D-1", _extraction())])
        with target.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert _buyer(rows)["Aadhaar Number (PC)"] == "241391305374"
        assert not rows[0]["Aadhaar Number (PC)"].startswith("6.")


class TestWriting:
    def test_header_and_row_count(self, tmp_path):
        target = tmp_path / "out.csv"
        count = write_csv(target, [DocumentExport("D-1", _extraction())])
        assert count == 2
        with target.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            assert next(reader) == list(CSV_COLUMNS)
            assert len(list(reader)) == 2

    def test_kannada_survives_round_trip(self, tmp_path):
        extraction = _extraction()
        extraction["buyer_details"][0]["name"] = "ಅಚಲ ಎಲ್ ನರಗುಂದ"
        target = tmp_path / "kn.csv"
        write_csv(target, [DocumentExport("D-1", extraction)])
        with target.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert _buyer(rows)["Person Name (PC)"] == "ಅಚಲ ಎಲ್ ನರಗುಂದ"

    def test_bom_present_for_excel(self, tmp_path):
        target = tmp_path / "bom.csv"
        write_csv(target, [DocumentExport("D-1", _extraction())])
        assert target.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_creates_missing_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "out.csv"
        write_csv(target, [DocumentExport("D-1", _extraction())])
        assert target.is_file()

    def test_failed_export(self, tmp_path):
        target = tmp_path / "failed.csv"
        count = write_failed_csv(target, [FailedDocument(
            "D-9", "d9.pdf", "ocr", "OCR_F", "no text layer", "OCR_F", 0.0)])
        assert count == 1
        with target.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            assert next(reader) == list(FAILED_COLUMNS)
            assert next(reader)[0] == "D-9"


@pytest.mark.regression
class TestAgainstReference:
    def test_reference_row_shape_matches_ours(self):
        """The reference repeats document fields per person; so must we."""
        if not REFERENCE.is_file():
            pytest.skip("example.csv not present")
        with REFERENCE.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["Report Serial Number"], []).append(row)

        multi = [g for g in grouped.values() if len(g) > 1]
        assert multi, "reference should contain multi-party documents"
        for group in multi:
            assert len({r["Transaction Identity"] for r in group}) == 1
            assert len({r["Transaction Amount"] for r in group}) == 1
            assert {r["Transaction Relation (PC)"] for r in group} <= {"B", "S"}


class TestAgainstTheReferenceReport:
    """The format the user supplied as `example.csv`, checked against ours.

    Every assertion here comes from that file rather than from my reading of
    what the report ought to look like.
    """

    ADDRESS = ("Residential Site bearing No.01 One, Katha No.0601, Formed in "
               "converted Survey No.06, situated at Kammagondanahalli Village, "
               "Bangalore North Taluk, jurisdiction of BBMP, Ward No.12")

    def _rows(self):
        return build_rows([DocumentExport(
            transaction_identity="1365-2025-26", serial=2,
            source_text=self.ADDRESS,
            extraction={
                "seller_details": [{"name": "K. ANAND", "father_name": "K Vemanna",
                                    "address": "Jalahalli West, BENGALURU-560 015"},
                                   {"name": "REVATHAMMA",
                                    "pan_card_number": "BXEPR5306Q",
                                    "address": "Jalahalli West, BENGALURU-560 015"}],
                "buyer_details": [{"name": "CHOWDURI VENKATA SUBBAIAH",
                                   "pan_card_number": "BJZPC9051L",
                                   "address": "Jalahalli, BENGALURU-560 013."}],
                "property_details": {"schedule_c_property_address": self.ADDRESS,
                                     "state": "KARNATAKA",
                                     "sale_consideration": "6700000"},
                "document_details": {"transaction_date": "2025-07-11"}})])

    def test_transaction_type_is_the_s_or_b_code(self):
        """The reference uses the same single letter as the relation: S for a
        sale, B for a buy. It is per row, because a row is one party and one
        deed has both."""
        rows = self._rows()
        assert [r["Transaction Type"] for r in rows] == ["S", "S", "B"]
        for row in rows:
            assert row["Transaction Type"] == row["Transaction Relation (PC)"]

    def test_sellers_are_listed_before_buyers(self):
        assert [r["Transaction Relation (PC)"] for r in self._rows()] == ["S", "S", "B"]

    def test_one_serial_per_document_repeated_across_its_parties(self):
        assert {r["Report Serial Number"] for r in self._rows()} == {"2"}

    def test_a_pin_written_with_a_space_is_read(self):
        """`BENGALURU-560 015`. Requiring six consecutive digits missed these,
        and that is most of why Postal Code was filled on a third of rows."""
        assert extract_pincode("Jalahalli West, BENGALURU-560 015") == "560015"
        assert extract_pincode("Bengaluru-560 090.") == "560090"
        assert self._rows()[0]["Pin Code (PC-L)"] == "560015"

    def test_a_pin_must_still_be_six_digits(self):
        assert extract_pincode("12345") == ""
        assert extract_pincode("1234567") == ""
        assert extract_pincode("no pin here") == ""

    def test_a_converted_site_is_residential_not_agricultural(self):
        """"Residential Site ... formed in converted Survey No.06" - the survey
        number says where the land came from. Matching it first made the
        reference document come out Agricultural."""
        assert self._rows()[0]["Property Type"] == "Residential"

    def test_real_agricultural_land_is_still_agricultural(self):
        """The precedence fix must not simply relabel everything residential."""
        assert property_type("Survey No 42 measuring 2 acres of agricultural land") \
            == "Agricultural"

    def test_bbmp_is_within_municipal_limits(self):
        assert self._rows()[0]["Whether property is within municipal limits"] == "Yes"

    def test_columns_match_the_reference_header_exactly(self):
        expected = [
            "Report Serial Number", "Original Report Serial Number",
            "Transaction Date", "Transaction Identity", "Transaction Type",
            "Transaction Amount", "Property Type",
            "Whether property is within municipal limits", "Property Address",
            "City / Town", "Postal Code", "State Code", "Country Code",
            "Stamp Value", "Remarks", "Transaction Relation (PC)",
            "Transaction Amount related to the person (PC)", "Person Name (PC)",
            "Person Type (PC)", "Gender (PC)", "Father's Name (PC)", "PAN (PC)",
            "Aadhaar Number (PC)", "Form 60 Acknowledgement (PC)",
            "Identification Type (PC)", "Identification Number (PC)",
            "Date of Birth/ Incorporation (PC)",
            "Nationality/Country of Incorporation (PC)", "Address Type (PC-L)",
            "Address (PC-L)", "City/Town (PC-L)", "Pin Code (PC-L)",
            "State (PC-L)", "Country (PC-L)", "Primary STD Code (PC)",
            "Primary Phone Number (PC)", "Primary Mobile Number (PC)",
            "Secondary STD Code (PC)", "Secondary Phone Number (PC)",
            "Secondary Mobile Number (PC)", "Email (PC)",
            "Person Details Remarks (PC)"]
        assert list(CSV_COLUMNS) == expected


class TestAadhaarIsTextNotANumber:
    """Aadhaar is an identifier, and identifiers are text.

    R-041. Twelve digits is one more than Excel will show in a general cell, so
    it switches to scientific notation: `465267227316` becomes `4.65E+11`. That
    is not a display quirk. Save the file and **three significant digits are all
    that remain** - the other nine are gone, not mangled. The reference
    `example.csv` supplied by the user contains `6.63E+11` in the Aadhaar
    column, so this had already happened to real data.

    Nobody adds up Aadhaar numbers. There is no reason for the cell to be
    numeric and one very good reason for it not to be.
    """

    AADHAAR = "465267227316"

    def _row(self, excel_safe):
        return build_rows([DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [{"name": "A",
                                           "aadhaar_number": self.AADHAAR,
                                           "pan_card_number": "BJZPC9051L"}],
                        "seller_details": [], "property_details": {},
                        "document_details": {}})], excel_safe=excel_safe)[0]

    def test_all_twelve_digits_are_written(self):
        for safe in (False, True):
            cell = self._row(safe)["Aadhaar Number (PC)"]
            assert sum(c.isdigit() for c in cell) == 12, cell
            assert self.AADHAAR in cell

    def test_excel_safe_marks_the_cell_as_text(self):
        """`="..."` is how a CSV tells Excel "this is a string". Non-standard,
        and the alternative is silent data loss."""
        assert self._row(True)["Aadhaar Number (PC)"] == f'="{self.AADHAAR}"'

    def test_pan_is_protected_the_same_way(self):
        assert self._row(True)["PAN (PC)"] == '="BJZPC9051L"'

    def test_it_survives_a_write_and_read(self, tmp_path):
        import csv

        target = tmp_path / "export.csv"
        write_csv(target, [DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [{"name": "A",
                                           "aadhaar_number": self.AADHAAR}],
                        "seller_details": [], "property_details": {},
                        "document_details": {}})], excel_safe=True)
        row = next(csv.DictReader(target.open(encoding="utf-8-sig")))
        assert self.AADHAAR in row["Aadhaar Number (PC)"]

    def test_scientific_notation_is_never_produced(self):
        for safe in (False, True):
            cell = self._row(safe)["Aadhaar Number (PC)"]
            assert "E+" not in cell.upper(), f"Excel notation reached the file: {cell}"

    def test_the_file_holds_plain_twelve_digits(self):
        """The exported CSV carries the bare number - `465267227316`, no
        wrapper, no notation. That is the requested format and it is also the
        correct one for any machine reading the file.

        What Excel *displays* when the file is double-clicked is a separate
        question, and not one the file can answer: see the class docstring.
        """
        cell = self._row(False)["Aadhaar Number (PC)"]
        assert cell == self.AADHAAR
        assert cell.isdigit() and len(cell) == 12

    def test_three_significant_digits_is_what_the_damage_looks_like(self):
        """Documents the failure mode, so nobody 'simplifies' this away later."""
        assert f"{float('6.63E+11'):.0f}" == "663000000000"
        assert float("6.63E+11") != float(self.AADHAAR)


class TestNamesAddressesAndRemarks:
    """The five corrections requested against the reference report. R-042."""

    # -- 1. names ---------------------------------------------------------

    def test_an_english_name_is_returned_exactly(self):
        """`KRISHNAPPA` must stay `KRISHNAPPA` - not Krishnappa, not Krishnapa.
        The deed's spelling of a name is the record."""
        from core.translation.detect import detect
        from core.translation.transliterate import transliterate_mixed

        for name in ("KRISHNAPPA", "K. ANAND", "REVATHAMMA", "Smt.Parvathi",
                     "GOPAL.M", "BALAJI B MAKA"):
            out = transliterate_mixed(name, script=detect(name).script)
            assert out == name, f"{name!r} became {out!r}"

    def test_an_english_name_is_never_queued_for_translation(self):
        from core.translation.detect import needs_translation

        for name in ("KRISHNAPPA", "K. ANAND", "REVATHAMMA"):
            assert not needs_translation(name)

    def test_a_mixed_name_keeps_its_english_half_byte_for_byte(self):
        """`KRISHNAPPA ರಾಜು`: the Kannada is rendered, the English is not.

        Measured honestly: the transliteration library already leaves Latin
        runs alone, so splitting the value per script does not change today's
        output. It is kept because it makes the guarantee explicit and cheap to
        check - `_title_case` is one edit away from restyling a name that the
        deed spells in capitals. The defect that actually reached the CSV was
        upstream, in what got queued at all - see the next test.
        """
        from core.translation.detect import detect
        from core.translation.transliterate import transliterate_mixed

        out = transliterate_mixed("KRISHNAPPA ರಾಜು", script=detect("KRISHNAPPA ರಾಜು").script)
        assert out == "KRISHNAPPA Raju"
        assert out.startswith("KRISHNAPPA")

    def test_a_mixed_name_is_queued_at_all(self):
        """Ten Latin characters against four Kannada made the dominant-script
        test call it English, so the Kannada reached the CSV untranslated."""
        from core.translation.detect import needs_translation

        assert needs_translation("KRISHNAPPA ರಾಜು")

    def test_the_exporter_does_not_restyle_names(self):
        row = build_rows([DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [{"name": "KRISHNAPPA",
                                           "father_name": "VENKATESH"}],
                        "seller_details": [], "property_details": {},
                        "document_details": {}})])[0]
        assert row["Person Name (PC)"] == "KRISHNAPPA"
        assert row["Father's Name (PC)"] == "VENKATESH"

    # -- 2. the person's own address --------------------------------------

    def test_the_address_column_comes_from_the_party_not_the_property(self):
        """`Address (PC-L)` is the party's address. It must never fall back to
        the schedule, the registration office, or anything else on the deed."""
        rows = build_rows([DocumentExport(
            transaction_identity="T",
            extraction={
                "buyer_details": [{"name": "A", "address": "PARTY ADDRESS 560001"}],
                "seller_details": [],
                "property_details": {
                    "schedule_c_property_address": "PROPERTY SCHEDULE 560002"},
                "document_details": {"registration_office": "OFFICE ADDRESS"}})])
        assert rows[0]["Address (PC-L)"] == "PARTY ADDRESS 560001"
        assert "PROPERTY" not in rows[0]["Address (PC-L)"]
        assert "OFFICE" not in rows[0]["Address (PC-L)"]

    def test_an_absent_party_address_stays_empty(self):
        """Rather than borrowing the property address to fill the gap."""
        rows = build_rows([DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [{"name": "A"}], "seller_details": [],
                        "property_details": {
                            "schedule_c_property_address": "PROPERTY SCHEDULE"},
                        "document_details": {}})])
        assert rows[0]["Address (PC-L)"] == ""

    # -- 3. property city and PIN -----------------------------------------

    def test_a_measurement_is_not_a_postal_code(self):
        """A schedule states `64.660488 square metres`. The fractional part of
        every measurement is six digits, and reading those as postal codes put
        660488 in the Postal Code column of a real document."""
        assert extract_pincode("ಅಳತೆ 7.315215 ಮೀ = 64.660488 ಚದರಮೀಟರ್") == ""
        assert extract_pincode("total 800 sq.ft or 74.322 sq.mtrs") == ""

    def test_a_real_postal_code_is_still_found(self):
        assert extract_pincode("Chitradurga - 577001") == "577001"
        assert extract_pincode("Kudumalakunte Village ... - 561208") == "561208"

    def test_city_is_taken_from_the_administrative_name(self):
        from core.csv_export import city_town

        assert city_town("Kudumalakunte Village, Kasaba Hobli, "
                         "Gowribidanur Taluk, Chikkaballapur District") == "Gowribidanur"
        assert city_town("Kammagondanahalli Village, Bangalore North Taluk, "
                         "BBMP") == "Bangalore"

    def test_a_direction_is_not_a_town(self):
        """`Bengaluru North Taluk` is Bengaluru."""
        from core.csv_export import city_town

        assert city_town("Bengaluru North Taluk") == "Bengaluru"

    def test_no_city_is_better_than_a_guessed_one(self):
        from core.csv_export import city_town

        assert city_town("Survey No 42, land with no place named") == ""

    def test_the_property_columns_are_populated_when_present(self):
        address = ("Kudumalakunte Village, Kasaba Hobli, Gowribidanur Taluk, "
                   "Chikkaballapur District. - 561208")
        row = build_rows([DocumentExport(
            transaction_identity="T", source_text=address,
            extraction={"buyer_details": [{"name": "A"}], "seller_details": [],
                        "property_details": {
                            "schedule_c_property_address": address,
                            "state": "Karnataka"},
                        "document_details": {}})])[0]
        assert row["Property Address"] == address
        assert row["Postal Code"] == "561208"
        assert row["City / Town"] == "Gowribidanur"
        assert row["State Code"] == "Karnataka"

    # -- 5. remarks -------------------------------------------------------

    def test_remarks_is_empty_even_when_validation_has_something_to_say(self):
        """Confidence scores and dispositions are this application's opinion of
        its own work, not a fact about the transaction. They stay in the
        database and on the Validation screen."""
        source = (ROOT / "src" / "core" / "csv_export.py").read_text(encoding="utf-8")
        assert 'f"conf={doc.report.confidence:.2f}"' not in source

        row = build_rows([DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [{"name": "A"}], "seller_details": [],
                        "property_details": {}, "document_details": {}})])[0]
        assert row["Remarks"] == ""
        assert row["Person Details Remarks (PC)"] == ""

    def test_a_document_with_no_parties_has_empty_remarks_too(self):
        rows = build_rows([DocumentExport(
            transaction_identity="T",
            extraction={"buyer_details": [], "seller_details": [],
                        "property_details": {}, "document_details": {}})])
        assert len(rows) == 1
        assert rows[0]["Remarks"] == ""


class TestCodedColumns:
    """Address Type is 1-5, Identification Type is a letter, Identification
    Number is always empty. R-046."""

    # -- Identification Type ----------------------------------------------

    def test_every_code_is_a_single_letter_from_the_permitted_set(self):
        permitted = set("ABCDEGHZ")
        assert set(IDENTIFICATION_CODES.values()) <= permitted
        assert len(set(IDENTIFICATION_CODES.values())) == len(IDENTIFICATION_CODES)

    def test_a_pan_gives_c(self):
        assert identification_type({"pan_card_number": "ABCPK1234F"}) == "C"

    def test_an_aadhaar_alone_gives_g(self):
        assert identification_type({"aadhaar_number": "663212345678"}) == "G"

    def test_pan_outranks_aadhaar(self):
        """Both present: the transaction is reported against the PAN."""
        person = {"pan_card_number": "ABCPK1234F", "aadhaar_number": "663212345678"}
        assert identification_type(person) == "C"

    def test_neither_leaves_the_cell_empty_rather_than_claiming_z(self):
        """`Z` means "some other document was seen". None was."""
        assert identification_type({}) == ""
        assert identification_type({"pan_card_number": "  ", "aadhaar_number": ""}) == ""

    # -- Identification Number --------------------------------------------

    def test_the_identification_number_column_is_empty_in_every_row(self):
        rows = build_rows([_export_with_both_identifiers()])
        assert rows, "no rows built"
        for row in rows:
            assert row["Identification Number (PC)"] == ""

    def test_a_party_carrying_identifiers_still_leaves_the_number_blank(self):
        """The identifiers reach the report through the PAN and Aadhaar
        columns; this column is not a second copy of them."""
        rows = build_rows([_export_with_both_identifiers()])
        row = rows[0]
        assert row["Identification Number (PC)"] == ""
        assert "663212345678" in row["Aadhaar Number (PC)"]
        assert "ABCPK1234F" in row["PAN (PC)"]

    # -- Address Type -----------------------------------------------------

    def test_every_address_code_is_one_of_one_to_five(self):
        assert set(ADDRESS_TYPES.values()) == {"1", "2", "3", "4", "5"}

    @pytest.mark.parametrize("address,expected", [
        ("No.126, 4 Cross, Raghavendra Layout, BENGALURU-560 015", "2"),
        ("House No 67, Gangondanahalli Village", "2"),
        ("Flat 402, Purva Westend Apartment", "2"),
        ("Shop No. 12, Commercial Complex, MG Road", "3"),
        ("Godown behind the factory, Peenya Industrial Area", "3"),
        ("House No 5 and office premises, MG Road", "1"),
        ("Registered Office: Prestige Tower, Bengaluru", "4"),
        ("Regd. Off: 4th Floor, Brigade Road", "4"),
    ])
    def test_the_address_says_what_it_is(self, address, expected):
        assert address_type(address, {}) == expected

    def test_an_address_that_says_nothing_about_its_use_is_unspecified(self):
        """5 is not a failure value - a deed records where somebody lives
        without stating what the premises are used for, and the format
        provides 5 for exactly that."""
        assert address_type("Plot 42, Sy No 118/2", {}) == "5"
        assert address_type("", {}) == "5"
        assert address_type(None, {}) == "5"

    def test_a_company_is_at_a_business_address_by_construction(self):
        """The PAN's fourth character identifies the holder; a company does
        not have a residence."""
        company = {"pan_card_number": "AAACK1234F"}   # C = company
        assert address_type("No.126, 4 Cross, Raghavendra Layout", company) == "3"

    def test_an_individual_pan_does_not_force_a_business_address(self):
        individual = {"pan_card_number": "ABCPK1234F"}  # P = individual
        assert address_type("No.126, 4 Cross, Raghavendra Layout", individual) == "2"

    def test_a_registered_office_outranks_the_pan(self):
        company = {"pan_card_number": "AAACK1234F"}
        assert address_type("Registered Office: Prestige Tower", company) == "4"

    # -- through the writer ------------------------------------------------

    def test_the_written_file_carries_only_permitted_codes(self, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(path, [_export_with_both_identifiers()])
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert row["Address Type (PC-L)"] in {"1", "2", "3", "4", "5"}
            assert row["Identification Type (PC)"] in set("ABCDEGHZ") | {""}
            assert row["Identification Number (PC)"] == ""


class TestCodedColumnsHoldCodesOnly:
    """`Address Type` sits beside `Property Type`, which legitimately holds
    "Residential" and "Commercial". A description written into the coded column
    looks plausible in a spreadsheet and is only rejected much later, by the
    system that reads the file. R-049."""

    def test_address_type_is_never_a_word(self):
        """The exact confusion reported: the descriptive text belongs to
        `Property Type` and must never reach `Address Type`."""
        rows = build_rows([_export_with_both_identifiers()])
        for row in rows:
            assert row["Address Type (PC-L)"] in {"1", "2", "3", "4", "5"}
            assert row["Address Type (PC-L)"] not in {
                "Residential", "Commercial", "Business", "Agricultural"}

    def test_property_type_keeps_its_words_and_address_type_its_codes(self):
        """Both columns are correct at once; they are not the same field."""
        rows = build_rows([_export_with_both_identifiers()])
        row = rows[0]
        assert row["Property Type"] == "Residential"
        assert row["Address Type (PC-L)"] == "2"

    def test_a_clean_export_reports_no_violations(self):
        assert coded_column_violations(build_rows([_export_with_both_identifiers()])) == []

    def test_a_description_in_a_coded_column_is_detected(self):
        rows = build_rows([_export_with_both_identifiers()])
        rows[0]["Address Type (PC-L)"] = "Residential"
        problems = coded_column_violations(rows)
        assert problems and "Address Type" in problems[0]
        assert "'Residential'" in problems[0]

    def test_a_blank_is_permitted_in_the_other_coded_columns(self):
        """Identification Type is blank when no document identified the party,
        and a party-less row carries no relation. Address Type is the exception
        and is covered below."""
        rows = build_rows([_export_with_both_identifiers()])
        for column in CODED_COLUMNS:
            if column == "Address Type (PC-L)":
                continue
            rows[0][column] = ""
        assert coded_column_violations(rows) == []

    def test_address_type_is_a_code_even_when_no_party_was_extracted(self):
        """Requested explicitly: the column holds one of 1-5 in every row, with
        no exception. `5` is "Unspecified", which is exactly the situation when
        nothing about the party or the address could be read."""
        empty = DocumentExport(
            transaction_identity="BGP-1-00999-2025-26",
            source_filename="unreadable.pdf",
            extraction={"seller_details": [], "buyer_details": []},
        )
        rows = build_rows([empty])
        assert len(rows) == 1, "a document with no parties still gets a row"
        assert rows[0]["Person Name (PC)"] == "", "the person half stays empty"
        assert rows[0]["Address Type (PC-L)"] == "5"

    def test_no_exported_row_ever_leaves_address_type_blank(self, tmp_path):
        path = tmp_path / "mixed.csv"
        write_csv(path, [_export_with_both_identifiers(),
                         DocumentExport(transaction_identity="X-1-00001-2025-26",
                                        source_filename="none.pdf",
                                        extraction={})])
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) >= 3
        for row in rows:
            assert row["Address Type (PC-L)"] in {"1", "2", "3", "4", "5"}, row

    def test_every_coded_column_is_checked(self):
        rows = build_rows([_export_with_both_identifiers()])
        for column in CODED_COLUMNS:
            broken = [dict(r) for r in rows]
            broken[0][column] = "Residential"
            assert coded_column_violations(broken), f"{column} is not checked"

    def test_the_written_file_carries_only_codes(self, tmp_path):
        path = tmp_path / "coded.csv"
        write_csv(path, [_export_with_both_identifiers()])
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert coded_column_violations(rows) == []


class TestBothCityTownColumns:
    """Both City/Town columns name the property's city, and neither is read
    from anywhere else in the deed. R-050."""

    @pytest.mark.parametrize("address,expected", [
        # kind after the name - the only shape the old version handled
        ("Begur Village, Begur Hobli, Bangalore South Taluk, Bangalore.", "Bangalore"),
        ("Basapura Village, Doddathogur Village Panchayath, Begur Hobli, "
         "Bangalore South Taluk.", "Bangalore"),
        # kind before the name, which returned nothing at all before
        ("Angol Village, Taluka & Dist: Belgaum.", "Belgaum"),
        ("Dist: Belgaum", "Belgaum"),
        ("District - Dharwad", "Dharwad"),
        # a bare city closing the address, with and without a PIN
        ("4th 'C' Block, Koramangala, Bangalore - 560 034", "Bangalore"),
        ("Sector - 7, Hosur Sarjapur Road Layout, Bangalore", "Bangalore"),
        # a named city outranks the taluk and district around it
        ("Hubli City, Dharwad District", "Hubli"),
        ("Anekal Taluk, Bangalore District", "Anekal"),
    ])
    def test_the_city_is_read_from_the_property_address(self, address, expected):
        assert city_town(address) == expected

    @pytest.mark.parametrize("address", [
        "Sy No 118/2, Gangondanahalli Village, Kasaba Hobli",
        "Site No 45, 3rd Main Road, 2nd Block",
        "", None,
    ])
    def test_a_village_or_a_structure_is_never_returned_as_a_city(self, address):
        """Agricultural land names a village and a hobli and no town. Inventing
        the nearest one would be a guess about jurisdiction, and putting
        "Kasaba Hobli" in a city column would be plainly wrong."""
        assert city_town(address) == ""

    def test_a_state_is_not_a_city(self):
        assert city_town("Some Layout, Karnataka") == ""

    @pytest.mark.parametrize("noise", [
        "1 H. mosama 2 R. Kavalaky 3. Indbox 4. Valalalder Yhst. Dhamadallehis",
        "THE SEAL OF THE SEAL OF THE SEAL OF THE SEAL OF THE SEAL",
        "ON WIND WAS TO WEST: 9.14+12.04/2 METERS",
        "Page 24 of 30",
        "BBMP PID NO. 68-8-106",
        "K. R",
        "###########",
        # Isolates the word-count bound: four words, all letters, no
        # punctuation - every other guard passes it.
        "THE SEAL OF THE",
        # Isolates the length bound: OCR runs words together into one long
        # alphabetic token, which nothing else rejects.
        "Bengalurusouthtalukbangaloreurbandistrictkarnataka",
        # Isolates the short-word rule: initials without their full stops pass
        # every other guard - letters only, two words, four characters.
        "K R",
    ])
    def test_ocr_noise_never_becomes_a_city(self, noise):
        """Real signature-page scrawl from BMH-1-00049. A cell holding this is
        worse than a blank one, because it looks like data. A place name is
        short, alphabetic and at most three words."""
        assert city_town(noise) == ""

    def test_both_columns_carry_the_same_property_city(self):
        """The requirement: one property, one city, in both columns."""
        rows = build_rows([_export_with_both_identifiers()])
        assert rows
        for row in rows:
            # The schedule says "Yelahanka Town" outright, so it outranks the
            # "Bangalore North Taluk" that follows it - a town named as such is
            # a closer answer than the taluk administering it.
            assert row["City / Town"] == "Yelahanka"
            assert row["City/Town (PC-L)"] == "Yelahanka"
            assert row["City / Town"] == row["City/Town (PC-L)"]

    def test_the_party_address_does_not_decide_the_city(self):
        """A seller's Aadhaar address is routinely in another town. The column
        is asked for the property's city, so the party's must not win."""
        export = DocumentExport(
            transaction_identity="BGP-1-00275-2025-26",
            source_filename="proof.pdf",
            extraction={
                "property_details": {
                    "schedule_c_property_address":
                        "Site No 45, Anekal Taluk, Bengaluru - 562106"},
                "seller_details": [{
                    "name": "KRISHNAPPA",
                    "address": "No 12, Hubli City, Dharwad District - 580020"}],
                "buyer_details": [],
            })
        rows = build_rows([export])
        for row in rows:
            assert row["City / Town"] == "Anekal"
            assert row["City/Town (PC-L)"] == "Anekal", "the party's Hubli won"

    def test_an_unrelated_place_elsewhere_in_the_deed_is_ignored(self):
        """`City / Town` used to search the whole OCR text, which is how a post
        office and a registration office reached it."""
        export = DocumentExport(
            transaction_identity="BGP-1-00275-2025-26",
            source_filename="proof.pdf",
            source_text=("Registered at the Sub Registrar office, Pavagada Town. "
                         "Presented at Madhugiri Taluk."),
            extraction={
                "property_details": {
                    "schedule_c_property_address":
                        "Site No 45, Anekal Taluk, Bengaluru"},
                "seller_details": [{"name": "KRISHNAPPA"}],
                "buyer_details": [],
            })
        rows = build_rows([export])
        for row in rows:
            assert row["City / Town"] == "Anekal"
            assert "Pavagada" not in row["City / Town"]
            assert "Madhugiri" not in row["City / Town"]

    def test_the_city_is_not_left_empty_when_the_deed_states_it(self):
        """Every address here names its city in one shape or another."""
        for address in ("Angol Village, Taluka & Dist: Belgaum.",
                        "4th 'C' Block, Koramangala, Bangalore - 560 034",
                        "Sector - 7, Hosur Sarjapur Road Layout, Bangalore"):
            assert city_town(address), f"{address!r} produced nothing"


class TestTheAbsentColumnListIsHonest:
    """`STRUCTURALLY_ABSENT` is the record of which columns a sale deed cannot
    fill, and `extraction_report` subtracts it from the coverage denominator.
    A stale entry therefore does not just mislead a reader - it inflates the
    reported coverage of every run. `City / Town` sat here after R-042 taught
    the exporter to populate it."""

    def test_nothing_declared_absent_is_actually_populated(self):
        rows = build_rows([_export_with_both_identifiers()])
        for row in rows:
            for column, reason in STRUCTURALLY_ABSENT.items():
                assert not str(row[column]).strip(), (
                    f"{column!r} is declared absent ({reason}) but carries "
                    f"{row[column]!r}"
                )

    def test_every_entry_names_a_real_column(self):
        for column in STRUCTURALLY_ABSENT:
            assert column in CSV_COLUMNS, f"{column!r} is not a report column"


def _export_with_both_identifiers() -> DocumentExport:
    """A deed with every column a deed can fill actually filled.

    The document half matters as much as the person half: `STRUCTURALLY_ABSENT`
    is checked against this, and a claim about a column can only be falsified by
    a fixture that would populate it."""
    return DocumentExport(
        transaction_identity="BGP-1-00275-2025-26",
        source_filename="proof.pdf",
        stamp_value="72500",
        source_text=(
            "The schedule property is a residential site situated within the "
            "limits of the Bruhat Bengaluru Mahanagara Palike, Yelahanka Town, "
            "Bangalore North Taluk."
        ),
        extraction={
            "document_details": {"transaction_date": "2025-06-14"},
            "property_details": {
                "sale_consideration": "4500000",
                "registration_fee": "72500",
                "state": "Karnataka",
                "schedule_c_property_address": (
                    "Site No 45, Yelahanka Town, Bangalore North Taluk, "
                    "BENGALURU-560 064"
                ),
            },
            "seller_details": [{
                "name": "KRISHNAPPA",
                "aadhaar_number": "663212345678",
                "pan_card_number": "ABCPK1234F",
                "address": "No.126, 4 Cross, Raghavendra Layout, BENGALURU-560 015",
            }],
            "buyer_details": [{
                "name": "RAMESH",
                "aadhaar_number": "551298765432",
                "address": "Shop No. 12, Commercial Complex, MG Road",
            }],
        },
    )
